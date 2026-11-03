import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import onnxruntime as ort
import torch

from edgeai.benchmark import _detection_matching
from edgeai.detector import ImageDetector
from edgeai.errors import ImageInputError, ModelLoadError
from edgeai.onnx_detector import OnnxDetector, _decode_yolo_output, _letterbox, prepare_onnx_input
from edgeai.onnx_export import validate_onnx_artifact
from edgeai.postprocess import extract_detections
from edgeai.video import VideoProcessor


ROOT = Path(__file__).resolve().parents[1]
ONNX_MODEL = ROOT / "models" / "yolo11n.onnx"
PT_WEIGHTS = ROOT / "yolo11n.pt"
SAMPLE_IMAGE = ROOT / "outputs" / "smoke_input.jpg"


def make_video(path: Path, frame_count=4, width=160, height=96, fps=10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV MJPG writer unavailable")
    for index in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10 + index, 15), (45 + index, 70), (20, 180, 240), -1)
        writer.write(frame)
    writer.release()


class OnnxPostprocessTests(unittest.TestCase):
    def test_class_aware_nms_filters_overlaps_and_decodes_names(self):
        # [cx, cy, w, h, person score, bicycle score], three predictions.
        raw = np.asarray(
            [
                [50, 50, 40, 40, 0.90, 0.01],
                [51, 51, 40, 40, 0.80, 0.01],
                [120, 50, 30, 30, 0.01, 0.88],
            ],
            dtype=np.float32,
        )
        detections = _decode_yolo_output(
            raw[None],
            scale=1.0,
            pad_left=0,
            pad_top=0,
            image_width=160,
            image_height=100,
            names={0: "person", 1: "bicycle"},
            confidence=0.25,
            iou_threshold=0.5,
        )
        self.assertEqual(len(detections), 2)
        self.assertEqual({item.class_name for item in detections}, {"person", "bicycle"})

    def test_preprocessing_matches_previous_normalization_and_output_contract(self):
        image = np.zeros((90, 150, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(150, dtype=np.uint8)
        image[:, :, 1] = 127
        image[:, :, 2] = 240
        padded, expected_scale, expected_left, expected_top = _letterbox(image, 640)
        previous_tensor = np.ascontiguousarray(
            padded[:, :, ::-1].transpose(2, 0, 1)[None],
            dtype=np.float32,
        ) / np.float32(255.0)

        tensor, scale, pad_left, pad_top = prepare_onnx_input(image, 640)

        self.assertTrue(np.array_equal(tensor, previous_tensor))
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.isfinite(tensor).all())
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual((scale, pad_left, pad_top), (expected_scale, expected_left, expected_top))


@unittest.skipUnless(ONNX_MODEL.is_file(), "Export models/yolo11n.onnx before integration tests")
class OnnxArtifactTests(unittest.TestCase):
    def test_export_artifact_has_fixed_expected_io_shapes(self):
        report = validate_onnx_artifact(ONNX_MODEL)
        self.assertEqual(report["input"]["shape"], [1, 3, 640, 640])
        self.assertEqual(report["outputs"][0]["shape"], [1, 84, 8400])
        self.assertGreater(report["file_size_bytes"], 1_000_000)

    def test_invalid_model_path_is_reported(self):
        with self.assertRaisesRegex(ModelLoadError, "does not exist"):
            OnnxDetector(ONNX_MODEL.parent / "missing.onnx", device="cpu")

    def test_auto_device_reports_missing_cuda_provider_and_uses_cpu(self):
        with patch("edgeai.onnx_detector.torch.cuda.is_available", return_value=True), patch(
            "edgeai.onnx_detector.ort.get_available_providers", return_value=["CPUExecutionProvider"]
        ):
            detector = OnnxDetector(ONNX_MODEL, device="auto")
        self.assertEqual(detector.device, "cpu")
        self.assertEqual(detector.execution_provider, "CPUExecutionProvider")
        self.assertIn("CUDAExecutionProvider unavailable", detector.fallback_error)

    def test_cpu_inference_returns_structured_detections(self):
        detector = OnnxDetector(ONNX_MODEL, device="cpu")
        image = cv2.imread(str(SAMPLE_IMAGE))
        if image is None:
            image = np.zeros((96, 160, 3), dtype=np.uint8)
        result, elapsed_ms = detector.predict_frame(image, 0.25, iou_threshold=0.7)
        self.assertEqual(detector.execution_provider, "CPUExecutionProvider")
        self.assertEqual(detector.device, "cpu")
        self.assertGreaterEqual(elapsed_ms, 0)
        self.assertEqual(len(result.boxes.xyxy.shape), 2)
        self.assertEqual(len(result.boxes), len(detector.last_detections))

    def test_invalid_image_path_is_reported(self):
        detector = OnnxDetector(ONNX_MODEL, device="cpu")
        with self.assertRaises(ImageInputError):
            detector.detect_image(ROOT / "outputs" / "missing.jpg")

    def test_onnx_cpu_predictions_match_pytorch(self):
        if not PT_WEIGHTS.is_file() or not SAMPLE_IMAGE.is_file():
            self.skipTest("Local YOLO checkpoint or comparison image unavailable")
        image = cv2.imread(str(SAMPLE_IMAGE))
        onnx_detector = OnnxDetector(ONNX_MODEL, device="cpu")
        pytorch_detector = ImageDetector(model_name=str(PT_WEIGHTS), device="cpu")
        onnx_result, _ = onnx_detector.predict_frame(image, 0.25, iou_threshold=0.7)
        pytorch_result, _ = pytorch_detector.predict_frame(
            image, 0.25, image_size=640, iou_threshold=0.7
        )
        report = _detection_matching(
            extract_detections(pytorch_result),
            extract_detections(onnx_result),
        )
        self.assertGreaterEqual(report["match_rate_reference"], 0.8)
        self.assertLessEqual(report["mean_absolute_confidence_difference"], 0.03)
        self.assertLessEqual(report["mean_absolute_box_coordinate_difference_px"], 3.0)

    def test_video_inference_writes_all_frames_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "synthetic.avi"
            make_video(source)
            detector = OnnxDetector(ONNX_MODEL, device="cpu")
            run = VideoProcessor(detector).process_video(
                source, output_dir=Path(directory) / "out", warmup_frames=1,
                image_size=640, iou_threshold=0.7,
            )
            self.assertEqual(run.performance.processed_frames, 4)
            self.assertEqual(run.device, "cpu")
            output = cv2.VideoCapture(run.annotated_video_path)
            self.assertTrue(output.isOpened())
            self.assertEqual(int(round(output.get(cv2.CAP_PROP_FRAME_COUNT))), 4)
            output.release()

    @unittest.skipUnless(
        torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers(),
        "ONNX Runtime CUDA provider unavailable",
    )
    def test_cuda_provider_is_selected_and_runs(self):
        detector = OnnxDetector(ONNX_MODEL, device="cuda")
        self.assertEqual(detector.device, "cuda:0")
        self.assertEqual(detector.execution_provider, "CUDAExecutionProvider")
        image = cv2.imread(str(SAMPLE_IMAGE))
        if image is None:
            image = np.zeros((96, 160, 3), dtype=np.uint8)
        detector.predict_frame(image, 0.25)


if __name__ == "__main__":
    unittest.main()
