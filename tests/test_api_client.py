"""Tests for the dashboard's HTTP-only EdgeAI API client."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import requests

from edgeai.api_client import ApiClientError, EdgeAIApiClient, parse_video_metadata


def _response(
    *,
    status=200,
    body=None,
    content=b"video-bytes",
    headers=None,
):
    return SimpleNamespace(
        status_code=status,
        headers=headers or {},
        content=content,
        json=Mock(return_value={} if body is None else body),
    )


class ApiClientTests(unittest.TestCase):
    def setUp(self):
        self.session = Mock()
        self.client = EdgeAIApiClient(" http://127.0.0.1:8000/ ", session=self.session)

    def test_health_request_returns_backend_status(self):
        self.session.request.return_value = _response(body={"status": "ready", "detector_initialized": True})
        value = self.client.get_health()
        self.assertEqual(value["status"], "ready")
        self.session.request.assert_called_once_with(
            "GET", "http://127.0.0.1:8000/health", timeout=(5.0, 60.0)
        )

    def test_health_request_failure_raises_api_error(self):
        self.session.request.return_value = _response(
            status=503, body={"detail": "Detector is not initialized."}
        )
        with self.assertRaisesRegex(ApiClientError, "Detector is not initialized"):
            self.client.get_health()

    def test_image_upload_request_uses_multipart_and_requests_annotation(self):
        image_payload = {"number_of_detections": 1, "annotated_image_base64": "abc"}
        self.session.request.return_value = _response(body=image_payload)
        result = self.client.detect_image(b"image-bytes", "folder/person.jpg", content_type="image/jpeg")
        self.assertEqual(result, image_payload)
        kwargs = self.session.request.call_args.kwargs
        self.assertEqual(self.session.request.call_args.args[:2], ("POST", "http://127.0.0.1:8000/api/v1/detect"))
        self.assertEqual(kwargs["files"]["file"], ("person.jpg", b"image-bytes", "image/jpeg"))
        self.assertEqual(kwargs["data"], {"include_annotated_image": "true"})

    def test_video_upload_request_passes_tracking_and_parses_metadata(self):
        metadata = {"processed_frame_count": 9, "runtime": "pytorch"}
        self.session.request.return_value = _response(
            headers={
                "Content-Type": "video/mp4",
                "Content-Disposition": 'attachment; filename="annotated.mp4"',
                "X-EdgeAI-Metadata": json.dumps(metadata),
            }
        )
        result = self.client.detect_video(
            b"video-bytes", "input.mp4", content_type="video/mp4", track=True, runtime="pytorch"
        )
        self.assertEqual(result.metadata, metadata)
        self.assertIsNone(result.metadata_error)
        self.assertEqual(result.content, b"video-bytes")
        self.assertEqual(result.file_name, "annotated.mp4")
        kwargs = self.session.request.call_args.kwargs
        self.assertEqual(self.session.request.call_args.args[:2], ("POST", "http://127.0.0.1:8000/api/v1/detect/video"))
        self.assertEqual(kwargs["files"]["file"], ("input.mp4", b"video-bytes", "video/mp4"))
        self.assertEqual(kwargs["data"], {"track": "true"})
        self.assertEqual(kwargs["timeout"], (5.0, 600.0))

    def test_onnx_tracking_is_rejected_before_sending_a_request(self):
        with self.assertRaisesRegex(ApiClientError, "only when the API runtime is PyTorch"):
            self.client.detect_video(b"video", "clip.avi", track=True, runtime="onnx")
        self.session.request.assert_not_called()

    def test_metadata_parser_handles_missing_malformed_and_unexpected_values(self):
        self.assertEqual(parse_video_metadata(None)[0], None)
        self.assertIn("did not include", parse_video_metadata(None)[1])
        self.assertIn("malformed", parse_video_metadata("{")[1])
        self.assertIn("unexpected format", parse_video_metadata("[]")[1])
        self.assertEqual(parse_video_metadata('{"frames": 2}'), ({"frames": 2}, None))

    def test_connection_failure_becomes_clear_dashboard_error(self):
        self.session.request.side_effect = requests.ConnectionError("low-level details")
        with self.assertRaisesRegex(ApiClientError, "Could not connect.*127.0.0.1:8000"):
            self.client.get_health()

    def test_timeout_becomes_clear_dashboard_error(self):
        self.session.request.side_effect = requests.Timeout("low-level details")
        with self.assertRaisesRegex(ApiClientError, "request timed out"):
            self.client.get_health()

    def test_http_error_uses_safe_api_detail(self):
        self.session.request.return_value = _response(status=413, body={"detail": "Video exceeds the configured limit."})
        with self.assertRaisesRegex(ApiClientError, "Video exceeds the configured limit"):
            self.client.get_health()

    def test_video_response_without_metadata_or_video_type_is_handled(self):
        self.session.request.return_value = _response(headers={"Content-Type": "video/avi"})
        result = self.client.detect_video(b"video", "clip.avi")
        self.assertIsNone(result.metadata)
        self.assertIn("did not include", result.metadata_error)
        self.assertEqual(result.file_name, "edgeai_annotated.avi")

        self.session.request.return_value = _response(headers={"Content-Type": "application/json"})
        with self.assertRaisesRegex(ApiClientError, "did not contain an annotated video"):
            self.client.detect_video(b"video", "clip.avi")


if __name__ == "__main__":
    unittest.main()
