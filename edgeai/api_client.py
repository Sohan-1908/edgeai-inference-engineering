"""Small HTTP client for the existing EdgeAI FastAPI service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


class ApiClientError(RuntimeError):
    """A user-safe error returned by or raised while calling the EdgeAI API."""


@dataclass(frozen=True)
class VideoApiResult:
    content: bytes
    metadata: dict[str, Any] | None
    metadata_error: str | None
    content_type: str
    file_name: str


def parse_video_metadata(raw_header: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the compact metadata header without trusting its contents."""
    if not raw_header:
        return None, "The API response did not include video processing metadata."
    try:
        value = json.loads(raw_header)
    except (TypeError, json.JSONDecodeError):
        return None, "The API returned malformed video processing metadata."
    if not isinstance(value, dict):
        return None, "The API returned video metadata in an unexpected format."
    return value, None


class EdgeAIApiClient:
    """Calls health, config, image detection, and video detection endpoints."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        session: requests.Session | Any | None = None,
        connect_timeout: float = 5.0,
        request_timeout: float = 60.0,
        video_timeout: float = 600.0,
    ):
        self.base_url = base_url.strip().rstrip("/")
        if not self.base_url:
            raise ValueError("API base URL cannot be empty.")
        self.session = session or requests.Session()
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self.video_timeout = video_timeout

    def _request(self, method: str, path: str, *, timeout: tuple[float, float], **kwargs):
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                timeout=timeout,
                **kwargs,
            )
        except requests.Timeout as exc:
            raise ApiClientError("The EdgeAI API request timed out. Check that the backend is running and try again.") from exc
        except requests.ConnectionError as exc:
            raise ApiClientError(
                f"Could not connect to the EdgeAI API at {self.base_url}. Start the backend or check the API URL."
            ) from exc
        except requests.RequestException as exc:
            raise ApiClientError(f"The EdgeAI API request failed ({type(exc).__name__}).") from exc

        if not 200 <= response.status_code < 300:
            detail = None
            try:
                body = response.json()
                if isinstance(body, dict):
                    detail = body.get("detail")
            except (ValueError, requests.RequestException):
                pass
            message = str(detail) if detail else f"The EdgeAI API returned HTTP {response.status_code}."
            raise ApiClientError(message)
        return response

    def get_health(self) -> dict[str, Any]:
        response = self._request(
            "GET", "/health", timeout=(self.connect_timeout, self.request_timeout)
        )
        try:
            value = response.json()
        except ValueError as exc:
            raise ApiClientError("The API health response was not valid JSON.") from exc
        if not isinstance(value, dict):
            raise ApiClientError("The API health response had an unexpected format.")
        return value

    def get_config(self) -> dict[str, Any]:
        response = self._request(
            "GET", "/api/v1/config", timeout=(self.connect_timeout, self.request_timeout)
        )
        try:
            value = response.json()
        except ValueError as exc:
            raise ApiClientError("The API configuration response was not valid JSON.") from exc
        if not isinstance(value, dict):
            raise ApiClientError("The API configuration response had an unexpected format.")
        return value

    def detect_image(
        self,
        content: bytes,
        filename: str,
        *,
        content_type: str = "application/octet-stream",
        include_annotated_image: bool = True,
    ) -> dict[str, Any]:
        response = self._request(
            "POST",
            "/api/v1/detect",
            timeout=(self.connect_timeout, self.request_timeout),
            files={"file": (Path(filename).name, content, content_type)},
            data={"include_annotated_image": str(include_annotated_image).lower()},
        )
        try:
            value = response.json()
        except ValueError as exc:
            raise ApiClientError("The API image response was not valid JSON.") from exc
        if not isinstance(value, dict):
            raise ApiClientError("The API image response had an unexpected format.")
        return value

    def detect_video(
        self,
        content: bytes,
        filename: str,
        *,
        content_type: str = "application/octet-stream",
        track: bool = False,
        runtime: str | None = None,
    ) -> VideoApiResult:
        if track and runtime == "onnx":
            raise ApiClientError("ByteTrack is available only when the API runtime is PyTorch.")
        response = self._request(
            "POST",
            "/api/v1/detect/video",
            timeout=(self.connect_timeout, self.video_timeout),
            files={"file": (Path(filename).name, content, content_type)},
            data={"track": str(track).lower()},
        )
        content_type_header = response.headers.get("Content-Type", "application/octet-stream")
        if not content_type_header.lower().startswith("video/"):
            raise ApiClientError("The API response did not contain an annotated video stream.")
        if not response.content:
            raise ApiClientError("The API returned an empty video file.")
        metadata, metadata_error = parse_video_metadata(
            response.headers.get("X-EdgeAI-Metadata")
        )
        disposition = response.headers.get("Content-Disposition", "")
        filename_out = _download_filename(disposition, Path(filename).suffix.lower())
        return VideoApiResult(
            content=response.content,
            metadata=metadata,
            metadata_error=metadata_error,
            content_type=content_type_header.split(";", 1)[0].strip(),
            file_name=filename_out,
        )


def _download_filename(content_disposition: str, fallback_suffix: str) -> str:
    """Extract a safe basename from Content-Disposition or generate one."""
    candidate = None
    for part in content_disposition.split(";"):
        key, separator, value = part.strip().partition("=")
        if separator and key.lower() == "filename":
            candidate = value.strip().strip('"')
            break
    if candidate:
        name = Path(candidate.replace("\\", "/")).name
        if name and name not in {".", ".."}:
            return name
    suffix = fallback_suffix if fallback_suffix in {".mp4", ".avi"} else ".mp4"
    return f"edgeai_annotated{suffix}"
