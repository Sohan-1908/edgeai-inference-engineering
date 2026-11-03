"""Video path validation, capture setup, and output naming."""

from pathlib import Path
import re

from edgeai.errors import VideoInputError

SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".avi"}


def validate_video_path(video_path: str | Path) -> Path:
    path = Path(video_path).expanduser()
    if path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_VIDEO_SUFFIXES))
        raise VideoInputError(f"Unsupported video format '{path.suffix}'. Use {allowed}.")
    if not path.exists() or not path.is_file():
        raise VideoInputError(f"Video file does not exist: {path}")
    try:
        if path.stat().st_size == 0:
            raise VideoInputError(f"Video file is empty: {path}")
    except OSError as exc:
        raise VideoInputError(f"Could not inspect video file: {path} ({exc})") from exc
    return path


def output_video_path(output_dir: str | Path, video_path: str | Path) -> Path:
    source = Path(video_path)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem).strip("._") or "video"
    suffix = source.suffix.lower()
    return Path(output_dir) / f"{stem}_annotated{suffix}"
