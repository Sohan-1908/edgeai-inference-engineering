"""Static checks for the API container build configuration."""

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (PROJECT_ROOT / "Dockerfile.api").read_text(encoding="utf-8")
DOCKERIGNORE = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
API_REQUIREMENTS = (PROJECT_ROOT / "requirements-api.txt").read_text(encoding="utf-8")


class ApiDockerConfigurationTests(unittest.TestCase):
    def test_pins_verified_cuda_base_and_single_worker_server(self):
        self.assertIn(
            "nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04@sha256:",
            DOCKERFILE,
        )
        self.assertIn('EDGEAI_API_HOST=0.0.0.0', DOCKERFILE)
        self.assertIn('EDGEAI_API_PORT=8000', DOCKERFILE)
        self.assertIn('--workers 1', DOCKERFILE)
        self.assertIn("EDGEAI_API_RUNTIME=pytorch", DOCKERFILE)
        self.assertIn("EDGEAI_API_DEVICE=cuda", DOCKERFILE)

    def test_healthcheck_requires_ready_initialized_detector(self):
        self.assertIn("/health", DOCKERFILE)
        self.assertIn("${EDGEAI_API_PORT}", DOCKERFILE)
        self.assertIn("health.get('status') == 'ready'", DOCKERFILE)
        self.assertIn("health.get('detector_initialized') is True", DOCKERFILE)
        self.assertIn("--start-period=300s", DOCKERFILE)
        healthcheck = next(
            line.strip()
            for line in DOCKERFILE.splitlines()
            if line.strip().startswith("CMD")
        )
        self.assertTrue(healthcheck.startswith("CMD python -c "))
        self.assertNotIn("CMD-SHELL", DOCKERFILE)

    def test_container_runs_non_root_with_writable_temp_and_no_model_copy(self):
        self.assertIn("USER edgeai:edgeai", DOCKERFILE)
        self.assertIn("TMPDIR=/tmp", DOCKERFILE)
        self.assertIn("chmod 1777 /tmp", DOCKERFILE)
        self.assertIn("COPY edgeai/ /app/edgeai/", DOCKERFILE)
        self.assertNotIn("COPY models/", DOCKERFILE)
        self.assertNotIn("COPY yolo11n.pt", DOCKERFILE)
        self.assertIn("*.pt", DOCKERIGNORE)
        self.assertIn("*.onnx", DOCKERIGNORE)
        self.assertIn("outputs/", DOCKERIGNORE)
        self.assertIn("benchmarks/", DOCKERIGNORE)
        self.assertIn(".venv/", DOCKERIGNORE)

    def test_backend_requirements_keep_pins_and_exclude_dashboard_tools(self):
        for requirement in (
            "torch==2.14.0+cu130",
            "torchvision==0.29.0+cu130",
            "numpy==2.5.4",
            "opencv-python==5.0.0.93",
            "ultralytics==8.4.197",
            "lap==0.5.13",
            "onnx==1.23.0",
            "onnxruntime-gpu==1.30.0",
            "fastapi==0.162.0",
            "pydantic==2.14.2",
            "starlette==1.10.0",
            "uvicorn==0.54.0",
            "python-multipart==0.0.33",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, API_REQUIREMENTS)
        for excluded in ("streamlit", "requests", "httpx2", "onnxconverter-common", "psutil"):
            with self.subTest(excluded=excluded):
                self.assertNotIn(excluded, API_REQUIREMENTS)


if __name__ == "__main__":
    unittest.main()
