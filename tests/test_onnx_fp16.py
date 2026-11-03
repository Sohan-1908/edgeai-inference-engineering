import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

from edgeai.errors import ModelLoadError
from edgeai.onnx_detector import OnnxDetector
from edgeai.onnx_fp16 import DEFAULT_FP16_ONNX_PATH, load_onnx_detector

FP32_MODEL = Path("models/yolo11n.onnx")


class OnnxFp16Tests(unittest.TestCase):
    def test_invalid_fp16_model_path(self):
        with self.assertRaises(ModelLoadError):
            OnnxDetector("models/does-not-exist-fp16.onnx", device="cuda", precision="fp16", require_cuda=True)

    def test_fp16_requires_strict_cuda(self):
        with self.assertRaises(ValueError):
            OnnxDetector(DEFAULT_FP16_ONNX_PATH, device="cpu", precision="fp16")

    def test_fp32_fallback_is_opt_in(self):
        marker = SimpleNamespace()
        failure = RuntimeError("CUDA EP operator unsupported")
        with patch("edgeai.onnx_detector.OnnxDetector", side_effect=[failure, marker]) as detector_cls:
            with self.assertRaisesRegex(RuntimeError, "CUDA EP operator unsupported"):
                load_onnx_detector("fp16", allow_fp32_fallback=False)
            self.assertEqual(detector_cls.call_count, 1)
        with patch("edgeai.onnx_detector.OnnxDetector", side_effect=[failure, marker]) as detector_cls:
            detector, notice = load_onnx_detector("fp16", allow_fp32_fallback=True)
            self.assertIs(detector, marker)
            self.assertIn("Explicit FP16-to-FP32 fallback", notice)
            self.assertEqual(detector_cls.call_count, 2)

    @unittest.skipUnless(
        torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers()
        and FP32_MODEL.is_file() and DEFAULT_FP16_ONNX_PATH.is_file(),
        "FP16 artifact or CUDA provider unavailable",
    )
    def test_fp16_cuda_execution_and_output(self):
        detector = OnnxDetector(DEFAULT_FP16_ONNX_PATH, device="cuda", precision="fp16", require_cuda=True)
        self.assertEqual(detector.execution_provider, "CUDAExecutionProvider")
        self.assertEqual(detector.session.get_providers()[0], "CUDAExecutionProvider")
        image = np.zeros((96, 160, 3), dtype=np.uint8)
        result, _ = detector.predict_frame(image, 0.25)
        self.assertEqual(len(result.boxes.xyxy.shape), 2)
        self.assertEqual(detector.precision, "FP16")

    @unittest.skipUnless(FP32_MODEL.is_file(), "FP32 baseline artifact missing")
    def test_fp32_baseline_still_loads_on_cpu(self):
        detector = OnnxDetector(FP32_MODEL, device="cpu")
        self.assertEqual(detector.device, "cpu")


if __name__ == "__main__":
    unittest.main()
