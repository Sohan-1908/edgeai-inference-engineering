"""Plain-data results for video inference and performance reporting."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VideoPerformanceMetrics:
    model_load_ms: float
    video_decoding_ms: float
    warmup_inference_ms: float
    inference_mean_ms: float
    inference_median_ms: float
    inference_samples_ms: tuple[float, ...]
    model_inference_mean_ms: float
    model_inference_median_ms: float
    model_inference_samples_ms: tuple[float, ...]
    tracking_processing_mean_ms: float | None
    tracking_processing_median_ms: float | None
    tracking_processing_samples_ms: tuple[float, ...]
    annotation_ms: float
    output_writing_ms: float
    total_processing_ms: float
    measured_pipeline_ms: float
    input_video_fps: float | None
    output_video_fps: float
    model_only_fps: float
    end_to_end_fps: float
    whole_video_processing_fps: float
    warmup_frames: int
    measured_frames: int
    processed_frames: int
    reported_input_frames: int | None
    gpu_memory_baseline_mb: float | None
    gpu_memory_peak_allocated_mb: float | None
    gpu_memory_peak_reserved_mb: float | None
    tracking_enabled: bool
    tracker_name: str | None
    tracking_inference_confidence: float | None
    runtime_name: str = "pytorch"
    runtime_version: str | None = None
    execution_provider: str | None = None
    model_size_bytes: int | None = None
    fallback_error: str | None = None
    runtime_preprocessing_mean_ms: float | None = None
    runtime_execution_mean_ms: float | None = None
    runtime_postprocessing_mean_ms: float | None = None
    precision: str = "FP32"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["inference_samples_ms"] = list(self.inference_samples_ms)
        data["model_inference_samples_ms"] = list(self.model_inference_samples_ms)
        data["tracking_processing_samples_ms"] = list(self.tracking_processing_samples_ms)
        return data


@dataclass(frozen=True)
class VideoDetectionRun:
    video_path: str
    annotated_video_path: str
    model: str
    device: str
    confidence_threshold: float
    frame_width: int
    frame_height: int
    total_detections: int
    unique_tracking_ids: int
    current_visible_tracked_objects: int
    tracking_enabled: bool
    performance: VideoPerformanceMetrics

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "annotated_video_path": self.annotated_video_path,
            "model": self.model,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "frame_dimensions": {"width": self.frame_width, "height": self.frame_height},
            "total_detections": self.total_detections,
            "unique_tracking_ids": self.unique_tracking_ids,
            "current_visible_tracked_objects": self.current_visible_tracked_objects,
            "tracking_enabled": self.tracking_enabled,
            "performance": self.performance.to_dict(),
        }
