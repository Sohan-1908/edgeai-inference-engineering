import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from edgeai.detector import ImageDetector
from edgeai.device import resolve_device
from edgeai.errors import ImageInputError, InferenceError
from edgeai.image_io import load_image


class FakeBoxes:
    xyxy = np.asarray([[10.0, 12.0, 60.0, 70.0]], dtype=np.float32)
    conf = np.asarray([0.875], dtype=np.float32)
    cls = np.asarray([0.0], dtype=np.float32)

    def __len__(self):
        return 1


class FakeModel:
    names = {0: "person"}

    def predict(self, **kwargs):
        self.last_device = kwargs["device"]
        self.last_confidence = kwargs["conf"]
        return [SimpleNamespace(boxes=FakeBoxes(), names=self.names)]


class ImageInputTests(unittest.TestCase):
    def test_loads_valid_jpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jpg"
            image = np.zeros((40, 60, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), image))
            decoded, elapsed_ms = load_image(path)
            self.assertEqual(decoded.shape, image.shape)
            self.assertGreaterEqual(elapsed_ms, 0.0)

    def test_missing_path_is_reported(self):
        with self.assertRaisesRegex(ImageInputError, "does not exist"):
            load_image("missing-image.jpg")

    def test_unsupported_extension_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.gif"
            path.write_bytes(b"not an image")
            with self.assertRaisesRegex(ImageInputError, "Unsupported image format"):
                load_image(path)

    def test_empty_and_corrupt_images_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "empty.png"
            empty.write_bytes(b"")
            corrupt = Path(directory) / "corrupt.png"
            corrupt.write_bytes(b"not a png")
            with self.assertRaisesRegex(ImageInputError, "empty"):
                load_image(empty)
            with self.assertRaisesRegex(ImageInputError, "corrupt"):
                load_image(corrupt)


class DeviceTests(unittest.TestCase):
    def test_auto_falls_back_to_cpu_when_cuda_is_unavailable(self):
        with patch("edgeai.device.torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), "cpu")

    def test_explicit_cpu_is_supported(self):
        self.assertEqual(resolve_device("cpu"), "cpu")

    def test_explicit_cuda_fails_clearly_if_unavailable(self):
        with patch("edgeai.device.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                resolve_device("cuda")


class DetectionStructureTests(unittest.TestCase):
    def test_detection_output_is_structured_and_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.jpg"
            cv2.imwrite(str(image_path), np.zeros((80, 100, 3), dtype=np.uint8))
            model = FakeModel()
            detector = ImageDetector(model=model, device="cpu", model_name="test-model")
            result = detector.detect_image(
                image_path,
                output_dir=Path(directory) / "outputs",
                confidence=0.4,
                warmup_runs=0,
                measured_runs=2,
            )

            payload = result.to_dict()
            self.assertEqual(payload["device"], "cpu")
            self.assertEqual(payload["number_of_detections"], 1)
            self.assertEqual(payload["detections"][0]["class_name"], "person")
            self.assertAlmostEqual(payload["detections"][0]["confidence"], 0.875, places=3)
            self.assertEqual(payload["detections"][0]["bbox_xyxy"], [10.0, 12.0, 60.0, 70.0])
            self.assertTrue(Path(result.annotated_image_path).is_file())
            self.assertEqual(model.last_device, "cpu")
            self.assertEqual(model.last_confidence, 0.4)
            self.assertEqual(len(result.performance.inference_samples_ms), 2)
            self.assertIsNone(result.performance.gpu_memory_peak_mb)

    def test_auto_device_runs_cpu_inference_when_cuda_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "cpu-fallback.png"
            cv2.imwrite(str(image_path), np.zeros((80, 100, 3), dtype=np.uint8))
            with patch("edgeai.device.torch.cuda.is_available", return_value=False):
                detector = ImageDetector(model=FakeModel(), device="auto")
                result = detector.detect_image(
                    image_path,
                    output_dir=Path(directory) / "outputs",
                    warmup_runs=0,
                    measured_runs=1,
                )
            self.assertEqual(result.device, "cpu")
            self.assertTrue(Path(result.annotated_image_path).is_file())

    def test_inference_errors_are_reported_cleanly(self):
        class BrokenModel:
            def predict(self, **kwargs):
                raise RuntimeError("synthetic prediction failure")

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.jpg"
            cv2.imwrite(str(image_path), np.zeros((40, 60, 3), dtype=np.uint8))
            detector = ImageDetector(model=BrokenModel(), device="cpu")
            with self.assertRaisesRegex(InferenceError, "YOLO inference failed"):
                detector.detect_image(image_path, warmup_runs=0, measured_runs=1)

    def test_invalid_run_settings_are_rejected(self):
        detector = ImageDetector(model=FakeModel(), device="cpu")
        with self.assertRaisesRegex(ValueError, "confidence"):
            detector.detect_image("anything.jpg", confidence=1.2)
        with self.assertRaisesRegex(ValueError, "measured_runs"):
            detector.detect_image("anything.jpg", measured_runs=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class CudaInferenceTests(unittest.TestCase):
    def test_pretrained_model_runs_on_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "cuda-smoke.jpg"
            cv2.imwrite(str(image_path), np.zeros((160, 160, 3), dtype=np.uint8))
            detector = ImageDetector(device="cuda")
            result = detector.detect_image(
                image_path,
                output_dir=Path(directory) / "outputs",
                warmup_runs=0,
                measured_runs=1,
            )
            self.assertEqual(result.device, "cuda:0")
            self.assertIsNotNone(result.performance.gpu_memory_peak_mb)
            self.assertTrue(Path(result.annotated_image_path).is_file())


if __name__ == "__main__":
    unittest.main()
