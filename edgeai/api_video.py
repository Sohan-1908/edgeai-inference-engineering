"""Temporary upload, validation, and streaming helpers for the video API."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
from fastapi import HTTPException, UploadFile

from edgeai.video import FALLBACK_OUTPUT_FPS

logger = logging.getLogger("edgeai.api.video")
VIDEO_MIME_TYPES = {
    ".mp4": {"video/mp4", "application/mp4", "application/octet-stream"},
    ".avi": {"video/x-msvideo", "video/avi", "application/octet-stream"},
}


def validate_video_filename_and_type(filename: str, content_type: str | None) -> str:
    """Return an allowed suffix; MIME hints are checked but never used as proof of validity."""
    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Only MP4 and AVI video uploads are supported.")
    mime = (content_type or "application/octet-stream").split(";", 1)[0].strip().lower()
    if mime not in VIDEO_MIME_TYPES[suffix]:
        raise HTTPException(status_code=415, detail="Video content type does not match the supported file extension.")
    return suffix


def copy_upload_to_path(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    """Copy a spooled upload to a random temporary path with a hard byte limit."""
    total = 0
    try:
        with destination.open("wb") as output:
            while True:
                chunk = upload.file.read(min(1024 * 1024, max_bytes + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="Uploaded video exceeds the configured size limit.")
                output.write(chunk)
    except HTTPException:
        raise
    except OSError:
        raise HTTPException(status_code=400, detail="Uploaded video could not be stored temporarily.") from None
    if total == 0:
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")
    return total


def probe_video(path: Path, *, max_frames: int, max_duration_seconds: float) -> dict:
    """Open the real video stream and validate decoded frame metadata before inference."""
    capture = None
    try:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="Video cannot be opened by the installed OpenCV codecs.")
        ok, first_frame = capture.read()
        if not ok or first_frame is None or first_frame.size == 0:
            raise HTTPException(status_code=400, detail="Video contains no readable frames.")
        height, width = first_frame.shape[:2]
        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="Video has invalid frame dimensions.")

        raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_fps = raw_fps if raw_fps > 0 and raw_fps < float("inf") else None
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_frames = int(round(raw_count)) if raw_count > 0 and raw_count < float("inf") else None
        if source_frames is not None:
            if source_frames > max_frames:
                raise HTTPException(status_code=413, detail="Video exceeds the configured frame limit.")
            effective_fps = source_fps or FALLBACK_OUTPUT_FPS
            if source_frames / effective_fps > max_duration_seconds:
                raise HTTPException(status_code=413, detail="Video exceeds the configured duration limit.")
        # Missing FPS or frame-count metadata is tolerated. The processor uses a
        # 30 FPS output fallback and enforces both limits while reading frames.
        return {
            "width": width,
            "height": height,
            "source_fps": source_fps,
            "source_frame_count": source_frames,
        }
    except HTTPException:
        raise
    except (cv2.error, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Video metadata could not be read.") from None
    finally:
        if capture is not None:
            capture.release()


def cleanup_temp_directory(directory: Path) -> None:
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Temporary video cleanup failed (%s).", type(exc).__name__)


async def stream_file_and_cleanup(path: Path, directory: Path, chunk_size: int = 1024 * 1024):
    """Stream bounded chunks and remove request-scoped files even after cancellation."""
    from starlette.concurrency import run_in_threadpool

    try:
        with path.open("rb") as source:
            while True:
                chunk = await run_in_threadpool(source.read, chunk_size)
                if not chunk:
                    break
                yield chunk
    finally:
        await run_in_threadpool(cleanup_temp_directory, directory)
