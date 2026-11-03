"""Image validation and decoding."""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from edgeai.config import MAX_IMAGE_BYTES, SUPPORTED_IMAGE_SUFFIXES
from edgeai.errors import ImageInputError


def validate_image_path(image_path: str | Path) -> Path:
    """Check path, supported suffix, and size without decoding the image."""
    path = Path(image_path).expanduser()
    if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise ImageInputError(f"Unsupported image format '{path.suffix}'. Use {allowed}.")
    if not path.exists() or not path.is_file():
        raise ImageInputError(f"Image file does not exist: {path}")

    try:
        size = path.stat().st_size
        if size == 0:
            raise ImageInputError(f"Image file is empty: {path}")
        if size > MAX_IMAGE_BYTES:
            limit_mb = MAX_IMAGE_BYTES // (1024 * 1024)
            raise ImageInputError(f"Image exceeds the {limit_mb} MB input limit: {path}")

    except ImageInputError:
        raise
    except OSError as exc:
        raise ImageInputError(f"Could not inspect image file: {path} ({exc})") from exc
    return path


def load_image(image_path: str | Path) -> tuple[np.ndarray, float]:
    """Validate and decode a JPG/JPEG/PNG image, returning BGR pixels and ms."""
    started = perf_counter()
    path = validate_image_path(image_path)
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except OSError as exc:
        raise ImageInputError(f"Could not read image file: {path} ({exc})") from exc

    if image is None or image.size == 0:
        raise ImageInputError(f"Image is corrupt or cannot be decoded: {path}")
    return image, (perf_counter() - started) * 1000.0
