"""Default settings for the first EdgeAI image-detection milestone."""

from pathlib import Path

DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_WARMUP_RUNS = 1
DEFAULT_MEASURED_RUNS = 5
DEFAULT_OUTPUT_DIR = Path("outputs")
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024
