"""Core image-detection components for EdgeAI."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from edgeai.detector import ImageDetector
    from edgeai.detector_types import Detection, DetectionRun, PerformanceMetrics
    from edgeai.video import VideoProcessor
    from edgeai.video_types import VideoDetectionRun, VideoPerformanceMetrics

_EXPORT_MODULES = {
    "Detection": "edgeai.detector_types",
    "DetectionRun": "edgeai.detector_types",
    "ImageDetector": "edgeai.detector",
    "PerformanceMetrics": "edgeai.detector_types",
    "VideoDetectionRun": "edgeai.video_types",
    "VideoPerformanceMetrics": "edgeai.video_types",
    "VideoProcessor": "edgeai.video",
}

__all__ = [
    "Detection",
    "DetectionRun",
    "ImageDetector",
    "PerformanceMetrics",
    "VideoDetectionRun",
    "VideoPerformanceMetrics",
    "VideoProcessor",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
