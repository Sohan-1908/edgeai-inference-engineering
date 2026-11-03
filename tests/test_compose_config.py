"""Static checks for Docker Compose service orchestration."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")
API_DOCKERFILE = (PROJECT_ROOT / "Dockerfile.api").read_text(encoding="utf-8")
DASHBOARD_DOCKERFILE = (PROJECT_ROOT / "Dockerfile.dashboard").read_text(
    encoding="utf-8"
)


def _service_block(service_name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(service_name)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        COMPOSE,
    )
    if match is None:
        raise AssertionError(f"Compose service {service_name!r} is missing.")
    return match.group(1)


class ComposeConfigurationTests(unittest.TestCase):
    def test_api_and_dashboard_build_their_own_dockerfiles(self):
        api = _service_block("api")
        dashboard = _service_block("dashboard")
        self.assertIn("dockerfile: Dockerfile.api", api)
        self.assertIn("dockerfile: Dockerfile.dashboard", dashboard)
        self.assertIn("context: .", api)
        self.assertIn("context: .", dashboard)

    def test_only_api_reserves_one_nvidia_gpu(self):
        api = _service_block("api")
        dashboard = _service_block("dashboard")
        self.assertIn("driver: nvidia", api)
        self.assertIn("count: 1", api)
        self.assertIn("- gpu", api)
        self.assertNotIn("devices:", dashboard)
        self.assertNotIn("driver: nvidia", dashboard)
        self.assertIn("--workers 1", API_DOCKERFILE)

    def test_model_mounts_are_read_only_at_application_paths(self):
        api = _service_block("api")
        self.assertRegex(
            api,
            r"(?ms)source: \./yolo11n\.pt\n\s+target: /app/yolo11n\.pt\n\s+read_only: true",
        )
        self.assertRegex(
            api,
            r"(?ms)source: \./models\n\s+target: /app/models\n\s+read_only: true",
        )
        self.assertIn("DEFAULT_MODEL = \"yolo11n.pt\"", (PROJECT_ROOT / "edgeai" / "config.py").read_text(encoding="utf-8"))
        self.assertIn("DEFAULT_ONNX_PATH = Path(\"models/yolo11n.onnx\")", (PROJECT_ROOT / "edgeai" / "onnx_export.py").read_text(encoding="utf-8"))
        self.assertNotIn("COPY yolo11n.pt", API_DOCKERFILE)
        self.assertNotIn("COPY models/", API_DOCKERFILE)
        self.assertNotIn("COPY models/", DASHBOARD_DOCKERFILE)

    def test_loopback_ports_and_dashboard_service_url(self):
        api = _service_block("api")
        dashboard = _service_block("dashboard")
        self.assertIn('127.0.0.1:${EDGEAI_API_HOST_PORT:-8000}:8000', api)
        self.assertIn(
            '127.0.0.1:${EDGEAI_DASHBOARD_HOST_PORT:-8501}:8501',
            dashboard,
        )
        self.assertIn("EDGEAI_API_BASE_URL: http://api:8000", dashboard)
        self.assertNotIn("localhost", dashboard)
        self.assertIn("EXPOSE 8000", API_DOCKERFILE)
        self.assertIn("EXPOSE 8501", DASHBOARD_DOCKERFILE)

    def test_api_defaults_to_onnx_cuda_640_and_supports_runtime_override(self):
        api = _service_block("api")
        self.assertIn("EDGEAI_API_RUNTIME: ${EDGEAI_API_RUNTIME:-onnx}", api)
        self.assertIn("EDGEAI_API_DEVICE: ${EDGEAI_API_DEVICE:-cuda}", api)
        self.assertIn("EDGEAI_API_IMGSZ: ${EDGEAI_API_IMGSZ:-640}", api)
        self.assertIn(
            "EDGEAI_API_ONNX_MODEL: ${EDGEAI_API_ONNX_MODEL:-models/yolo11n.onnx}",
            api,
        )
        self.assertIn("source: ./models\n        target: /app/models\n        read_only: true", api)

    def test_dashboard_waits_for_api_health_and_has_no_models_or_gpu(self):
        api = _service_block("api")
        dashboard = _service_block("dashboard")
        self.assertIn("condition: service_healthy", dashboard)
        self.assertNotIn("volumes:", dashboard)
        self.assertNotIn("devices:", dashboard)
        self.assertIn("HEALTHCHECK", API_DOCKERFILE)
        self.assertIn("HEALTHCHECK", DASHBOARD_DOCKERFILE)
        self.assertIn("health.get('status') == 'ready'", API_DOCKERFILE)
        self.assertIn("health.get('detector_initialized') is True", API_DOCKERFILE)

    def test_compose_does_not_add_credentials_or_host_networking(self):
        self.assertNotIn("privileged:", COMPOSE)
        self.assertNotIn("network_mode: host", COMPOSE)
        self.assertNotIn("secrets:", COMPOSE)
        self.assertNotIn("devices:", _service_block("dashboard"))


if __name__ == "__main__":
    unittest.main()
