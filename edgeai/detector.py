"""Image object-detection service with warm-up and repeated measurements."""

from pathlib import Path
from statistics import mean, median
from threading import RLock
from time import perf_counter

import torch
import numpy as np

from edgeai.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MEASURED_RUNS,
    DEFAULT_MODEL,
    DEFAULT_WARMUP_RUNS,
)
from edgeai.detector_types import DetectionRun, PerformanceMetrics
from edgeai.device import resolve_device, ultralytics_device
from edgeai.errors import InferenceError, ModelLoadError
from edgeai.image_io import load_image
from edgeai.model_loader import load_yolo_model
from edgeai.postprocess import annotate_image, extract_detections, output_image_path, save_annotated_image
from edgeai.config import DEFAULT_CONFIDENCE
from edgeai.onnx_detector import DEFAULT_IOU_THRESHOLD


class ImageDetector:
    """Load one YOLO model and reuse it for subsequent image requests."""

    gpu_memory_supported = True
    runtime_name = "pytorch"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        *,
        model=None,
        model_load_ms: float | None = None,
        model_loader=load_yolo_model,
    ):
        self.model_name = model_name
        self.inference_lock = RLock()
        self.device = resolve_device(device)
        if model is None:
            try:
                self.model, measured_load_ms = model_loader(model_name, self.device)
            except Exception as exc:
                if isinstance(exc, ModelLoadError):
                    raise
                raise InferenceError(str(exc)) from exc
            self.model_load_ms = measured_load_ms
        else:
            self.model = model
            self.model_load_ms = float(model_load_ms or 0.0)
        self.runtime_version = torch.__version__
        self.execution_provider = "CUDAExecutionProvider" if self.device == "cuda:0" else "CPUExecutionProvider"
        candidate = Path(model_name).expanduser()
        self.model_size_bytes = candidate.stat().st_size if candidate.is_file() else None
        self.last_timings = {"preprocess_ms": 0.0, "runtime_ms": 0.0, "postprocess_ms": 0.0}

    @staticmethod
    def _validate_confidence(confidence: float):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

    @staticmethod
    def _validate_run_settings(confidence: float, warmup_runs: int, measured_runs: int):
        ImageDetector._validate_confidence(confidence)
        if warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative")
        if measured_runs < 1:
            raise ValueError("measured_runs must be at least 1")

    def _predict(self, image, confidence: float, image_size=None, iou_threshold=None):
        kwargs = {
            "source": image,
            "conf": confidence,
            "device": ultralytics_device(self.device),
            "verbose": False,
        }
        if image_size is not None:
            kwargs.update(imgsz=int(image_size), rect=False)
        if iou_threshold is not None:
            kwargs["iou"] = float(iou_threshold)
        return self.model.predict(**kwargs)[0]

    def predict_frame(
        self,
        frame,
        confidence: float = DEFAULT_CONFIDENCE,
        *,
        image_size: int | None = None,
        iou_threshold: float | None = None,
    ):
        """Run one timed prediction on a decoded BGR frame.

        CUDA synchronization brackets the call so the returned duration
        includes asynchronous GPU work. The time includes Ultralytics
        preprocessing and prediction post-processing, but not file decoding
        or drawing.
        """
        self._validate_confidence(confidence)
        try:
            with self.inference_lock:
                if self.device == "cuda:0":
                    torch.cuda.synchronize(0)
                started = perf_counter()
                result = self._predict(frame, confidence, image_size, iou_threshold)
                if self.device == "cuda:0":
                    torch.cuda.synchronize(0)
                speed = getattr(result, "speed", {}) or {}
                self.last_timings = {
                    "preprocess_ms": float(speed.get("preprocess", 0.0)),
                    "runtime_ms": float(speed.get("inference", (perf_counter() - started) * 1000.0)),
                    "postprocess_ms": float(speed.get("postprocess", 0.0)),
                }
                return result, (perf_counter() - started) * 1000.0
        except Exception as exc:
            raise InferenceError(f"YOLO inference failed: {exc}") from exc

    def detect_image(
        self,
        image_path: str | Path,
        output_dir: str | Path = "outputs",
        confidence: float = DEFAULT_CONFIDENCE,
        warmup_runs: int = DEFAULT_WARMUP_RUNS,
        measured_runs: int = DEFAULT_MEASURED_RUNS,
        *,
        image_size: int | None = None,
        iou_threshold: float | None = None,
    ) -> DetectionRun:
        """Detect, annotate, save, and return one image's structured results.

        Inference latency covers the Ultralytics prediction call, including
        its preprocessing, model forward pass, and prediction post-processing.
        File decoding and annotation/writing are timed separately.
        """
        self._validate_run_settings(confidence, warmup_runs, measured_runs)
        total_started = perf_counter()
        source = Path(image_path).expanduser().resolve()
        image, image_load_ms = load_image(source)

        try:
            warmup_started = perf_counter()
            for _ in range(warmup_runs):
                with self.inference_lock:
                    self._predict(image, confidence, image_size, iou_threshold)
            if self.device == "cuda:0":
                torch.cuda.synchronize(0)
            warmup_ms = (perf_counter() - warmup_started) * 1000.0
            if self.device == "cuda:0":
                torch.cuda.reset_peak_memory_stats(0)
                gpu_memory_baseline_mb = torch.cuda.memory_allocated(0) / (1024**2)
            else:
                gpu_memory_baseline_mb = None

            samples = []
            preprocess_samples = []
            runtime_samples = []
            postprocess_samples = []
            last_result = None
            for _ in range(measured_runs):
                last_result, inference_ms = self.predict_frame(
                    image,
                    confidence,
                    image_size=image_size,
                    iou_threshold=iou_threshold,
                )
                samples.append(inference_ms)
                preprocess_samples.append(self.last_timings["preprocess_ms"])
                runtime_samples.append(self.last_timings["runtime_ms"])
                postprocess_samples.append(self.last_timings["postprocess_ms"])

            if self.device == "cuda:0":
                gpu_memory_peak_mb = torch.cuda.max_memory_allocated(0) / (1024**2)
                gpu_memory_peak_reserved_mb = torch.cuda.max_memory_reserved(0) / (1024**2)
            else:
                gpu_memory_peak_mb = None
                gpu_memory_peak_reserved_mb = None
        except Exception as exc:
            raise InferenceError(f"YOLO inference failed for '{source}': {exc}") from exc

        try:
            detections = extract_detections(last_result)
        except Exception as exc:
            raise InferenceError(f"Could not format YOLO detections for '{source}': {exc}") from exc
        visualization_started = perf_counter()
        try:
            annotated = annotate_image(image, detections)
            destination = output_image_path(output_dir, source)
            save_annotated_image(annotated, destination)
        except Exception as exc:
            raise InferenceError(f"Could not create annotated output: {exc}") from exc
        annotation_save_ms = (perf_counter() - visualization_started) * 1000.0
        total_ms = (perf_counter() - total_started) * 1000.0 + self.model_load_ms

        metrics = PerformanceMetrics(
            model_load_ms=self.model_load_ms,
            image_load_ms=image_load_ms,
            warmup_ms=warmup_ms,
            inference_mean_ms=mean(samples),
            inference_median_ms=median(samples),
            inference_samples_ms=tuple(samples),
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            annotation_save_ms=annotation_save_ms,
            total_ms=total_ms,
            gpu_memory_baseline_mb=gpu_memory_baseline_mb,
            gpu_memory_peak_mb=gpu_memory_peak_mb,
            gpu_memory_peak_reserved_mb=gpu_memory_peak_reserved_mb,
            runtime_name=self.runtime_name,
            runtime_version=self.runtime_version,
            execution_provider=self.execution_provider,
            input_size=image_size,
            iou_threshold=iou_threshold,
            preprocessing_mean_ms=float(np.mean(preprocess_samples)) if image_size else None,
            runtime_execution_mean_ms=float(np.mean(runtime_samples)) if image_size else None,
            postprocessing_mean_ms=float(np.mean(postprocess_samples)) if image_size else None,
            p95_inference_ms=float(np.percentile(samples, 95)),
            throughput_per_second=1000.0 / mean(samples),
            model_size_bytes=self.model_size_bytes,
        )
        return DetectionRun(
            image_path=str(source),
            annotated_image_path=str(destination.resolve()),
            model=self.model_name,
            device=self.device,
            confidence_threshold=confidence,
            detections=tuple(detections),
            performance=metrics,
        )
