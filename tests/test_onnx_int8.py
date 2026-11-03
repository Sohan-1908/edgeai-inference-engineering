import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from edgeai.benchmark import _detection_matching
from edgeai.errors import ImageInputError, VideoInputError
from edgeai.image_io import load_image
from edgeai.onnx_detector import OnnxDetector, prepare_onnx_input
from edgeai.onnx_int8 import (
    DEFAULT_CALIBRATION_DIR,
    ImageCalibrationReader,
    quantize_yolo11n_static,
    sample_calibration_frames,
)

ROOT = Path(__file__).resolve().parents[1]
FP32_MODEL = ROOT / "models" / "yolo11n.onnx"
VIDEO = ROOT / "outputs" / "cuda_video_smoke.mp4"


class Int8CalibrationTests(unittest.TestCase):
    def test_deterministic_video_frame_sampling_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "synthetic.avi"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 64))
            self.assertTrue(writer.isOpened())
            for index in range(8):
                frame = np.full((64, 80, 3), index * 20, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            images, metadata = sample_calibration_frames(video_path, output_dir=Path(temp_dir) / "frames", sample_count=4)
            self.assertEqual(metadata["selected_frame_indices"], [0, 2, 5, 7])
            self.assertEqual(metadata["calibration_image_count"], 4)
            self.assertTrue(all(path.is_file() for path in images))

    def test_invalid_calibration_video_is_rejected(self):
        with self.assertRaises(VideoInputError):
            sample_calibration_frames("outputs/missing-calibration.mp4")

    def test_reader_uses_detector_preprocessing_and_reports_bad_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.jpg"
            image = np.full((70, 120, 3), (12, 90, 210), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), image))
            reader = ImageCalibrationReader([path], "images")
            actual = reader.get_next()["images"]
            expected, _, _, _ = prepare_onnx_input(image)
            np.testing.assert_array_equal(actual, expected)
            self.assertIsNone(reader.get_next())
            reader.rewind()
            self.assertEqual(reader.get_next()["images"].shape, (1, 3, 640, 640))
            with self.assertRaises(ImageInputError):
                ImageCalibrationReader([Path(temp_dir) / "missing.jpg"], "images").get_next()

    @unittest.skipUnless(FP32_MODEL.is_file() and VIDEO.is_file(), "FP32 model or local calibration video missing")
    def test_static_int8_model_creation_cpu_inference_and_consistency(self):
        calibration_images = sorted(DEFAULT_CALIBRATION_DIR.glob("calibration_*.jpg"))[:4]
        if len(calibration_images) < 4:
            self.skipTest("Run --quantize-int8 first to prepare local representative calibration images")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "yolo11n_int8.onnx"
            report = quantize_yolo11n_static(calibration_images, output_path=output_path)
            self.assertTrue(output_path.is_file())
            self.assertEqual(report["graph_analysis"]["quantized_conv_count"], 56)
            self.assertEqual(report["validation"]["execution_provider"], "CPUExecutionProvider")
            detector_fp32 = OnnxDetector(FP32_MODEL, device="cpu")
            detector_int8 = OnnxDetector(output_path, device="cpu", precision="int8")
            self.assertEqual(detector_int8.device, "cpu")
            self.assertEqual(detector_int8.execution_provider, "CPUExecutionProvider")
            image, _ = load_image(calibration_images[0])
            detector_fp32.predict_frame(image, 0.25)
            detector_int8.predict_frame(image, 0.25)
            consistency = _detection_matching(detector_fp32.last_detections, detector_int8.last_detections)
            self.assertIn("missed_reference_count", consistency)
            self.assertIn("additional_candidate_count", consistency)
            self.assertEqual(len(detector_fp32.session.get_outputs()), len(detector_int8.session.get_outputs()))

    @unittest.skipUnless(FP32_MODEL.is_file(), "FP32 model artifact missing")
    def test_int8_path_is_cpu_only_and_fp32_model_remains_loadable(self):
        with self.assertRaises(ValueError):
            OnnxDetector(FP32_MODEL, device="cuda", precision="int8")
        baseline = OnnxDetector(FP32_MODEL, device="cpu", precision="fp32")
        self.assertEqual(baseline.precision, "FP32")


if __name__ == "__main__":
    unittest.main()
