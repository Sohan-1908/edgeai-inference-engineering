import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from edgeai.detector_types import Detection
from edgeai.errors import DeviceUnavailableError, ImageInputError, VideoInputError
from edgeai.onnx_detector import OnnxDetector, prepare_onnx_input
from edgeai.resolution_benchmark import (
    RESOLUTION_SIZES,
    _aggregate_consistency,
    _require_cuda,
    run_image_trials,
    run_resolution_benchmark,
    sample_video_frames,
    validate_resolutions,
)


class _FakeOnnxDetector:
    runtime_name = "onnxruntime-gpu"
    last_detections = []
    last_timings = {"preprocess_ms": 1.0, "runtime_ms": 2.0, "postprocess_ms": 0.5}

    def predict_frame(self, image, confidence, *, image_size, iou_threshold):
        self.last_detections = []
        return SimpleNamespace(speed={"inference": 2.0}), 3.5


class ResolutionConfigurationTests(unittest.TestCase):
    def test_configurations_and_invalid_resolutions(self):
        self.assertEqual(RESOLUTION_SIZES, (640, 480, 320))
        self.assertEqual(validate_resolutions([640, 480, 320]), (640, 480, 320))
        for values in ([480, 320], [640, 640], [640, 400]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_resolutions(values)

    def test_onnx_preprocessing_respects_each_square_letterbox_size(self):
        image = np.full((320, 640, 3), (20, 100, 220), dtype=np.uint8)
        for size in RESOLUTION_SIZES:
            with self.subTest(size=size):
                tensor, scale, pad_left, pad_top = prepare_onnx_input(image, size)
                self.assertEqual(tensor.shape, (1, 3, size, size))
                self.assertEqual(tensor.dtype, np.float32)
                self.assertGreater(scale, 0)
                self.assertGreaterEqual(pad_left, 0)
                self.assertGreater(pad_top, 0)

    def test_invalid_video_or_image_path_is_reported(self):
        with self.assertRaises(VideoInputError):
            sample_video_frames("outputs/no-such-resolution-video.mp4")
        with patch("edgeai.resolution_benchmark._require_cuda"):
            with self.assertRaises(ImageInputError):
                run_resolution_benchmark(["outputs/no-such-resolution-image.jpg"])

    def test_cuda_unavailable_or_missing_provider_fails_without_fallback(self):
        with patch("edgeai.resolution_benchmark.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(DeviceUnavailableError, "requires CUDA"):
                _require_cuda()
        with patch("edgeai.resolution_benchmark.torch.cuda.is_available", return_value=True), \
             patch("edgeai.resolution_benchmark.ort.get_available_providers", return_value=["CPUExecutionProvider"]):
            with self.assertRaisesRegex(DeviceUnavailableError, "CUDAExecutionProvider"):
                _require_cuda()

    def test_benchmark_trial_runner_collects_raw_samples_and_stages(self):
        with patch("edgeai.resolution_benchmark._sync_cuda"):
            report = run_image_trials(
                _FakeOnnxDetector(), [np.zeros((32, 32, 3), dtype=np.uint8)],
                resolution=320, warmups=1, runs_per_trial=2, trials=2,
            )
        self.assertEqual(len(report["raw_latency_samples_ms"]), 4)
        self.assertEqual(report["mean_latency_ms"], 3.5)
        self.assertEqual(report["stage_mean_ms"]["runtime"], 2.0)
        self.assertIsNone(report["gpu_memory"]["peak_allocated_mb"])

    def test_detection_consistency_counts_reference_misses_and_additions(self):
        reference = [
            Detection(0, "person", 0.9, (0, 0, 20, 20)),
            Detection(2, "car", 0.8, (40, 40, 80, 80)),
        ]
        candidate = [
            Detection(0, "person", 0.85, (1, 1, 21, 21)),
            Detection(5, "bus", 0.75, (100, 100, 130, 130)),
        ]
        from edgeai.benchmark import _detection_matching

        item = _detection_matching(reference, candidate, iou_threshold=0.5)
        aggregate = _aggregate_consistency([item])
        self.assertEqual(aggregate["matched_count"], 1)
        self.assertEqual(aggregate["missed_reference_count"], 1)
        self.assertEqual(aggregate["additional_candidate_count"], 1)
        self.assertGreater(aggregate["mean_matched_iou"], 0.5)

    def test_static_onnx_variants_accept_their_exported_resolution(self):
        root = __import__("pathlib").Path(__file__).resolve().parents[1]
        variants = {size: root / "models" / f"yolo11n_{size}.onnx" for size in (480, 320)}
        if not all(path.is_file() for path in variants.values()):
            self.skipTest("Resolution-specific ONNX variants are created by the actual benchmark run")
        image = np.zeros((96, 160, 3), dtype=np.uint8)
        for size, path in variants.items():
            with self.subTest(size=size):
                detector = OnnxDetector(path, device="cpu")
                self.assertEqual(detector.input_size, size)
                self.assertEqual(detector._prepare(image)[0].shape, (1, 3, size, size))
                detector.predict_frame(image, image_size=size)


if __name__ == "__main__":
    unittest.main()
