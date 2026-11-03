"""Tests for lazy package exports and the dashboard's lightweight import path."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PackageImportTests(unittest.TestCase):
    def test_api_client_import_does_not_load_inference_modules_in_fresh_process(self):
        script = """
import sys
import edgeai.api_client

for module in (
    "torch",
    "onnxruntime",
    "edgeai.detector",
    "edgeai.detector_types",
    "edgeai.video",
    "edgeai.video_types",
    "edgeai.onnx_detector",
):
    assert module not in sys.modules, f"{module} was imported"
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_package_level_exports_remain_importable(self):
        script = """
from edgeai import (
    Detection,
    DetectionRun,
    ImageDetector,
    PerformanceMetrics,
    VideoDetectionRun,
    VideoPerformanceMetrics,
    VideoProcessor,
)
from edgeai.detector import ImageDetector as ExpectedImageDetector
from edgeai.detector_types import Detection as ExpectedDetection
from edgeai.video import VideoProcessor as ExpectedVideoProcessor

assert Detection is ExpectedDetection
assert ImageDetector is ExpectedImageDetector
assert VideoProcessor is ExpectedVideoProcessor
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
