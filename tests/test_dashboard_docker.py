"""Static and subprocess checks for the lightweight dashboard image."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (PROJECT_ROOT / "Dockerfile.dashboard").read_text(encoding="utf-8")
DASHBOARD_REQUIREMENTS = (PROJECT_ROOT / "requirements-dashboard.txt").read_text(
    encoding="utf-8"
)


class DashboardDockerTests(unittest.TestCase):
    def test_dashboard_dependencies_are_pinned_and_inference_free(self):
        self.assertIn("streamlit==1.66.1", DASHBOARD_REQUIREMENTS)
        self.assertIn("requests==2.34.2", DASHBOARD_REQUIREMENTS)
        for package in (
            "torch",
            "torchvision",
            "onnxruntime",
            "ultralytics",
            "cuda",
            "onnx==",
            "opencv",
        ):
            with self.subTest(package=package):
                self.assertNotIn(package, DASHBOARD_REQUIREMENTS.lower())

    def test_image_uses_python_312_non_root_and_only_dashboard_port(self):
        self.assertIn(
            "FROM python:3.12-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487",
            DOCKERFILE,
        )
        self.assertIn("USER dashboard:dashboard", DOCKERFILE)
        self.assertIn("EXPOSE 8501", DOCKERFILE)
        self.assertIn("--server.address=0.0.0.0", DOCKERFILE)
        self.assertIn("--server.port=8501", DOCKERFILE)
        self.assertNotIn("EXPOSE 8000", DOCKERFILE)
        self.assertIn("http://api:8000", DOCKERFILE)
        self.assertNotIn("http://localhost", DOCKERFILE)
        healthcheck = next(
            line.strip()
            for line in DOCKERFILE.splitlines()
            if line.strip().startswith("CMD")
        )
        self.assertTrue(healthcheck.startswith('CMD ["python",'))
        self.assertNotIn("CMD-SHELL", DOCKERFILE)

    def test_image_copies_only_dashboard_sources_and_config(self):
        self.assertIn("COPY --chown=dashboard:dashboard app.py", DOCKERFILE)
        self.assertIn(
            "edgeai/__init__.py edgeai/api_client.py /app/edgeai/",
            DOCKERFILE,
        )
        self.assertIn(".streamlit/config.toml", DOCKERFILE)
        self.assertNotIn("COPY edgeai/ /app/edgeai/", DOCKERFILE)
        self.assertNotIn("COPY models/", DOCKERFILE)
        dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
        self.assertIn(".env", dockerignore)
        self.assertIn("credentials.json", dockerignore)
        self.assertIn("secrets.toml", dockerignore)

    def test_dashboard_import_respects_api_url_without_loading_inference(self):
        script = """
import os
import sys
os.environ["EDGEAI_API_BASE_URL"] = "http://api:8000"
import app
import edgeai.api_client
assert app.DEFAULT_API_URL == "http://api:8000"
for module in (
    "torch",
    "torchvision",
    "onnxruntime",
    "edgeai.detector",
    "edgeai.video",
    "edgeai.onnx_detector",
):
    assert module not in sys.modules, f"{module} was imported"
"""
        environment = os.environ.copy()
        environment["EDGEAI_API_BASE_URL"] = "http://api:8000"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_local_api_url_default_is_preserved(self):
        script = """
import app
assert app.DEFAULT_API_URL == "http://127.0.0.1:8000"
"""
        environment = os.environ.copy()
        environment.pop("EDGEAI_API_BASE_URL", None)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
