"""Fixed-shape YOLO ONNX Runtime detector with Ultralytics-compatible I/O."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from statistics import mean, median
from threading import RLock
from time import perf_counter
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import onnxruntime as ort

from edgeai.config import DEFAULT_CONFIDENCE, DEFAULT_MEASURED_RUNS, DEFAULT_WARMUP_RUNS
from edgeai.detector_types import Detection, DetectionRun, PerformanceMetrics
from edgeai.errors import (
    DeviceUnavailableError,
    ImageInputError,
    InferenceError,
    ModelLoadError,
)
from edgeai.image_io import load_image
from edgeai.onnx_export import DEFAULT_ONNX_PATH, INPUT_SIZE
from edgeai.postprocess import annotate_image, output_image_path, save_annotated_image

DEFAULT_IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 300


def _load_names(metadata: dict[str, str]) -> dict[int, str]:
    raw = metadata.get("names")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
    if isinstance(parsed, dict):
        return {int(key): str(value) for key, value in parsed.items()}
    if isinstance(parsed, list):
        return {index: str(value) for index, value in enumerate(parsed)}
    return {}


def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    pad_width = size - resized_width
    pad_height = size - resized_height
    left = int(round(pad_width / 2 - 0.1))
    top = int(round(pad_height / 2 - 0.1))
    right = pad_width - left
    bottom = pad_height - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, left, top


def prepare_onnx_input(
    image: np.ndarray, size: int = INPUT_SIZE
) -> tuple[np.ndarray, float, int, int]:
    """Use the identical letterbox, RGB conversion, layout, and normalization for all runtimes."""
    if size not in {320, 480, 640}:
        raise ValueError("ONNX input size must be one of 320, 480, or 640")
    padded, scale, left, top = _letterbox(image, size)
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)[None]
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    np.divide(tensor, np.float32(255.0), out=tensor)
    return tensor, scale, left, top


def _box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    left_top = np.maximum(one[:2], many[:, :2])
    right_bottom = np.minimum(one[2:], many[:, 2:])
    intersection = np.prod(np.maximum(right_bottom - left_top, 0), axis=1)
    one_area = np.prod(np.maximum(one[2:] - one[:2], 0))
    many_area = np.prod(np.maximum(many[:, 2:] - many[:, :2], 0), axis=1)
    return intersection / np.maximum(one_area + many_area - intersection, 1e-9)


def _decode_yolo_output(
    output: np.ndarray,
    *,
    scale: float,
    pad_left: int,
    pad_top: int,
    image_width: int,
    image_height: int,
    names: dict[int, str],
    confidence: float,
    iou_threshold: float,
) -> list[Detection]:
    prediction = np.asarray(output, dtype=np.float32)
    if prediction.ndim == 3:
        if prediction.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1, got output shape {prediction.shape}"
            )
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError(
            f"Expected a rank-2 YOLO output after removing batch, got {prediction.shape}"
        )
    feature_count = 4 + len(names) if names else None
    if feature_count is not None and prediction.shape[0] == feature_count:
        prediction = prediction.T
    elif feature_count is None and prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.T
    if prediction.shape[1] < 5:
        raise ValueError(f"YOLO output has too few channels: {prediction.shape}")

    class_scores = prediction[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]
    keep = np.flatnonzero(scores >= confidence)
    if keep.size == 0:
        return []
    prediction = prediction[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    centers = prediction[:, :2]
    sizes = prediction[:, 2:4]
    boxes = np.column_stack((centers - sizes / 2, centers + sizes / 2))

    selected = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        order = indices[np.argsort(scores[indices])[::-1]]
        while order.size:
            current = int(order[0])
            selected.append(current)
            rest = order[1:]
            if not rest.size:
                break
            order = rest[_box_iou(boxes[current], boxes[rest]) <= iou_threshold]
    if len(selected) > MAX_DETECTIONS:
        selected = sorted(
            selected, key=lambda index: float(scores[index]), reverse=True
        )[:MAX_DETECTIONS]

    detections = []
    for index in selected:
        x1, y1, x2, y2 = boxes[index]
        x1 = np.clip((x1 - pad_left) / scale, 0, image_width)
        y1 = np.clip((y1 - pad_top) / scale, 0, image_height)
        x2 = np.clip((x2 - pad_left) / scale, 0, image_width)
        y2 = np.clip((y2 - pad_top) / scale, 0, image_height)
        class_id = int(class_ids[index])
        detections.append(
            Detection(
                class_id=class_id,
                class_name=names.get(class_id, str(class_id)),
                confidence=float(scores[index]),
                bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
            )
        )
    return detections


def _as_ultralytics_result(
    detections: list[Detection], names: dict[int, str], runtime_ms: float
):
    boxes = _DetectionBoxes(detections)
    return SimpleNamespace(boxes=boxes, names=names, speed={"inference": runtime_ms})


class _DetectionBoxes:
    def __init__(self, detections: list[Detection]):
        self.xyxy = np.asarray(
            [item.bbox_xyxy for item in detections], dtype=np.float32
        ).reshape(-1, 4)
        self.conf = np.asarray(
            [item.confidence for item in detections], dtype=np.float32
        )
        self.cls = np.asarray([item.class_id for item in detections], dtype=np.float32)
        self.id = None

    def __len__(self):
        return len(self.xyxy)


class OnnxDetector:
    """Load one static ONNX model and reuse its selected ORT session."""

    gpu_memory_supported = False

    def __init__(
        self,
        model_path: str | Path = DEFAULT_ONNX_PATH,
        device: str = "auto",
        *,
        cpu_threads: int = 4,
        precision: str = "fp32",
        require_cuda: bool = False,
        profile: bool = False,
        profile_file_prefix: str | Path | None = None,
    ):
        self.require_cuda = require_cuda
        self.profile = profile
        self.profile_file_prefix = (
            str(profile_file_prefix) if profile_file_prefix is not None else None
        )
        self._profiling_ended = False
        self.model_path = Path(model_path).expanduser().resolve()
        self.inference_lock = RLock()
        if not self.model_path.is_file():
            raise ModelLoadError(f"ONNX model file does not exist: {self.model_path}")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if cpu_threads < 1:
            raise ValueError("cpu_threads must be at least 1")
        if precision not in {"fp32", "fp16", "int8"}:
            raise ValueError("precision must be 'fp32', 'fp16', or 'int8'")
        if precision == "fp16" and not require_cuda:
            raise ValueError("FP16 ONNX inference requires strict CUDA execution.")
        if precision == "int8" and device != "cpu":
            raise ValueError(
                "INT8 benchmark inference is CPU-only in this phase; select device='cpu'."
            )
        if require_cuda and device != "cuda":
            raise DeviceUnavailableError(
                "Strict CUDA execution requires device='cuda'."
            )
        self.requested_device = device
        self.precision = precision.upper()
        self.require_cuda = require_cuda
        self.model_name = self.model_path.name
        self.model_size_bytes = self.model_path.stat().st_size
        self.last_timings = {
            "preprocess_ms": 0.0,
            "runtime_ms": 0.0,
            "postprocess_ms": 0.0,
        }
        self.fallback_error: str | None = None
        self._cpu_threads = cpu_threads

        started = perf_counter()
        self.session = self._create_session(device)
        self.model_load_ms = (perf_counter() - started) * 1000.0
        self.device = (
            "cuda:0"
            if self.session.get_providers()[0] == "CUDAExecutionProvider"
            else "cpu"
        )
        self.execution_provider = self.session.get_providers()[0]
        self.runtime_name = (
            "onnxruntime-gpu" if self.device == "cuda:0" else "onnxruntime"
        )
        self.runtime_version = ort.__version__
        self.input = self.session.get_inputs()[0]
        self.output_names = [item.name for item in self.session.get_outputs()]
        if (
            len(self.input.shape) != 4
            or self.input.shape[0] != 1
            or self.input.shape[1] != 3
            or self.input.shape[2] != self.input.shape[3]
            or self.input.shape[2] not in {320, 480, 640}
        ):
            raise ModelLoadError(
                f"ONNX input must be fixed [1, 3, S, S] for S in 320, 480, or 640; got {self.input.shape}"
            )
        self.input_size = int(self.input.shape[2])
        self.names = _load_names(self.session.get_modelmeta().custom_metadata_map)
        self.last_detections: list[Detection] = []

    def _session_options(self, *, gpu: bool):
        options = ort.SessionOptions()
        options.enable_profiling = self.profile
        if self.profile_file_prefix is not None:
            options.profile_file_prefix = self.profile_file_prefix
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if self.require_cuda:
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

        if not gpu:
            options.intra_op_num_threads = self._cpu_threads
            options.inter_op_num_threads = 1

        return options

    def end_profiling(self) -> str | None:
        """Stop ONNX Runtime profiling and return the generated trace path."""
        if not self.profile or self._profiling_ended:
            return None

        with self.inference_lock:
            if self._profiling_ended:
                return None

            profile_path = self.session.end_profiling()
            self._profiling_ended = True

        return profile_path

    def _create_session(self, requested: str):
        providers = ort.get_available_providers()
        if self.require_cuda:
            if not torch.cuda.is_available():
                raise DeviceUnavailableError(
                    "Strict CUDA execution was requested, but torch.cuda.is_available() is False."
                )
            if "CUDAExecutionProvider" not in providers:
                raise DeviceUnavailableError(
                    f"CUDAExecutionProvider unavailable; installed providers: {providers}"
                )
            try:
                session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=self._session_options(gpu=True),
                    providers=["CUDAExecutionProvider"],
                )
                if (
                    not session.get_providers()
                    or session.get_providers()[0] != "CUDAExecutionProvider"
                ):
                    raise RuntimeError(
                        f"CUDA was not selected: {session.get_providers()}"
                    )
                return session
            except Exception as exc:
                raise DeviceUnavailableError(
                    f"Strict CUDA session initialization failed (CPU node fallback disabled): {type(exc).__name__}: {exc}"
                ) from exc
        wants_cuda = requested != "cpu" and torch.cuda.is_available()
        cuda_error = None
        if requested != "cpu" and not torch.cuda.is_available():
            cuda_error = "PyTorch reports CUDA unavailable on this machine."
        elif wants_cuda and "CUDAExecutionProvider" not in providers:
            cuda_error = (
                f"CUDAExecutionProvider unavailable; installed providers: {providers}"
            )
        elif wants_cuda:
            try:
                # Importing torch above loads the compatible CUDA/cuDNN DLLs
                # from the existing PyTorch cu130 installation.
                session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=self._session_options(gpu=True),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                if (
                    not session.get_providers()
                    or session.get_providers()[0] != "CUDAExecutionProvider"
                ):
                    raise RuntimeError(
                        f"CUDAExecutionProvider was not selected; session providers: {session.get_providers()}"
                    )
                return session
            except Exception as exc:
                cuda_error = f"CUDAExecutionProvider initialization failed: {type(exc).__name__}: {exc}"
        if requested == "cuda" or requested == "auto":
            self.fallback_error = cuda_error
        return ort.InferenceSession(
            str(self.model_path),
            sess_options=self._session_options(gpu=False),
            providers=["CPUExecutionProvider"],
        )

    def _prepare(self, image: np.ndarray):
        return prepare_onnx_input(image, self.input_size)

    def predict_frame(
        self,
        frame: np.ndarray,
        confidence: float = DEFAULT_CONFIDENCE,
        *,
        image_size: int | None = None,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ):
        if image_size not in (None, self.input_size):
            raise ValueError(
                f"This ONNX export has a fixed input size of {self.input_size}x{self.input_size}"
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")
        if frame is None or frame.ndim != 3 or frame.size == 0:
            raise ImageInputError("Inference input must be a non-empty BGR image.")
        try:
            with self.inference_lock:
                started = perf_counter()
                tensor, scale, left, top = self._prepare(frame)
                preprocess_ms = (perf_counter() - started) * 1000.0
                started = perf_counter()
                outputs = self.session.run(self.output_names, {self.input.name: tensor})
                # ONNX Runtime's default session.run API is synchronous: outputs
                # are ready when it returns. Do not use PyTorch CUDA synchronization
                # to time work scheduled by a separate runtime.
                runtime_ms = (perf_counter() - started) * 1000.0
                started = perf_counter()
                height, width = frame.shape[:2]
                detections = _decode_yolo_output(
                    outputs[0],
                    scale=scale,
                    pad_left=left,
                    pad_top=top,
                    image_width=width,
                    image_height=height,
                    names=self.names,
                    confidence=confidence,
                    iou_threshold=iou_threshold,
                )
                postprocess_ms = (perf_counter() - started) * 1000.0
                self.last_timings = {
                    "preprocess_ms": preprocess_ms,
                    "runtime_ms": runtime_ms,
                    "postprocess_ms": postprocess_ms,
                }
                self.last_detections = detections
                return _as_ultralytics_result(detections, self.names, runtime_ms), (
                    preprocess_ms + runtime_ms + postprocess_ms
                )
        except Exception as exc:
            if isinstance(exc, (ImageInputError, ValueError)):
                raise
            raise InferenceError(f"ONNX Runtime inference failed: {exc}") from exc

    def detect_image(
        self,
        image_path: str | Path,
        output_dir: str | Path = "outputs",
        confidence: float = DEFAULT_CONFIDENCE,
        warmup_runs: int = DEFAULT_WARMUP_RUNS,
        measured_runs: int = DEFAULT_MEASURED_RUNS,
        *,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> DetectionRun:
        if warmup_runs < 0 or measured_runs < 1:
            raise ValueError(
                "warmup_runs must be non-negative and measured_runs must be at least 1"
            )
        image, load_ms = load_image(image_path)
        warmup_started = perf_counter()
        for _ in range(warmup_runs):
            self.predict_frame(image, confidence, iou_threshold=iou_threshold)
        warmup_ms = (perf_counter() - warmup_started) * 1000.0
        samples = []
        preprocess_samples = []
        runtime_samples = []
        postprocess_samples = []
        for _ in range(measured_runs):
            _, total_ms = self.predict_frame(
                image, confidence, iou_threshold=iou_threshold
            )
            samples.append(total_ms)
            preprocess_samples.append(self.last_timings["preprocess_ms"])
            runtime_samples.append(self.last_timings["runtime_ms"])
            postprocess_samples.append(self.last_timings["postprocess_ms"])
        detections = self.last_detections
        draw_started = perf_counter()
        destination = output_image_path(output_dir, image_path)
        save_annotated_image(annotate_image(image, detections), destination)
        annotate_save_ms = (perf_counter() - draw_started) * 1000.0
        metrics = PerformanceMetrics(
            model_load_ms=self.model_load_ms,
            image_load_ms=load_ms,
            warmup_ms=warmup_ms,
            inference_mean_ms=mean(samples),
            inference_median_ms=median(samples),
            inference_samples_ms=tuple(samples),
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            annotation_save_ms=annotate_save_ms,
            total_ms=load_ms
            + warmup_ms
            + sum(samples)
            + annotate_save_ms
            + self.model_load_ms,
            gpu_memory_baseline_mb=None,
            gpu_memory_peak_mb=None,
            gpu_memory_peak_reserved_mb=None,
            runtime_name=self.runtime_name,
            runtime_version=self.runtime_version,
            execution_provider=self.execution_provider,
            input_size=self.input_size,
            iou_threshold=iou_threshold,
            preprocessing_mean_ms=mean(preprocess_samples),
            runtime_execution_mean_ms=mean(runtime_samples),
            postprocessing_mean_ms=mean(postprocess_samples),
            p95_inference_ms=float(np.percentile(samples, 95)),
            throughput_per_second=1000.0 / mean(samples),
            model_size_bytes=self.model_size_bytes,
            fallback_error=self.fallback_error,
            precision=self.precision,
        )
        return DetectionRun(
            image_path=str(Path(image_path).resolve()),
            annotated_image_path=str(destination.resolve()),
            model=self.model_name,
            device=self.device,
            confidence_threshold=confidence,
            detections=tuple(detections),
            performance=metrics,
        )
