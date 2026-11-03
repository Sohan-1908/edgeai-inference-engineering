"""Plain-data result types shared by the inference and post-processing code."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    track_id: int | None = None

    def to_dict(self) -> dict:
        data = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 6),
            "bbox_xyxy": [round(value, 2) for value in self.bbox_xyxy],
        }
        if self.track_id is not None:
            data["track_id"] = self.track_id
        return data


@dataclass(frozen=True)
class PerformanceMetrics:
    model_load_ms: float
    image_load_ms: float
    warmup_ms: float
    inference_mean_ms: float
    inference_median_ms: float
    inference_samples_ms: tuple[float, ...]
    warmup_runs: int
    measured_runs: int
    annotation_save_ms: float
    total_ms: float
    gpu_memory_baseline_mb: float | None
    gpu_memory_peak_mb: float | None
    gpu_memory_peak_reserved_mb: float | None
    runtime_name: str = "pytorch"
    runtime_version: str | None = None
    execution_provider: str | None = None
    input_size: int | None = None
    iou_threshold: float | None = None
    preprocessing_mean_ms: float | None = None
    runtime_execution_mean_ms: float | None = None
    postprocessing_mean_ms: float | None = None
    p95_inference_ms: float | None = None
    throughput_per_second: float | None = None
    model_size_bytes: int | None = None
    fallback_error: str | None = None
    precision: str = "FP32"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["inference_samples_ms"] = list(self.inference_samples_ms)
        return data


@dataclass(frozen=True)
class DetectionRun:
    image_path: str
    annotated_image_path: str
    model: str
    device: str
    confidence_threshold: float
    detections: tuple[Detection, ...]
    performance: PerformanceMetrics

    @property
    def count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "annotated_image_path": self.annotated_image_path,
            "model": self.model,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "number_of_detections": self.count,
            "detections": [item.to_dict() for item in self.detections],
            "performance": self.performance.to_dict(),
        }
