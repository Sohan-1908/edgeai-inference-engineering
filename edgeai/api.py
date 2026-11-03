"""FastAPI image-inference service backed by EdgeAI's existing detectors."""

from __future__ import annotations

import base64
import gc
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, StreamingResponse

from edgeai.config import DEFAULT_CONFIDENCE, DEFAULT_MODEL, MAX_IMAGE_BYTES
from edgeai.detector import ImageDetector
from edgeai.detector_types import Detection
from edgeai.errors import DeviceUnavailableError, InferenceError, ModelLoadError
from edgeai.onnx_detector import DEFAULT_IOU_THRESHOLD, DEFAULT_ONNX_PATH, OnnxDetector
from edgeai.postprocess import annotate_image, extract_detections
from edgeai.api_video import (
    cleanup_temp_directory,
    copy_upload_to_path,
    probe_video,
    stream_file_and_cleanup,
    validate_video_filename_and_type,
)
from edgeai.errors import VideoInputError, VideoProcessingError
from edgeai.video import VideoProcessor
from edgeai.webcam import SUPPORTED_WEBCAM_SIZES

logger = logging.getLogger("edgeai.api")
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
JPEG_SIGNATURE = b"\xff\xd8\xff"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class ApiSettings:
    runtime: str = "pytorch"
    device: str = "cuda"
    input_size: int = 640
    confidence: float = DEFAULT_CONFIDENCE
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    model_name: str = DEFAULT_MODEL
    onnx_model_path: str = str(DEFAULT_ONNX_PATH)
    max_upload_bytes: int = MAX_IMAGE_BYTES
    max_video_bytes: int = 100 * 1024 * 1024
    max_video_seconds: float = 120.0
    max_video_frames: int = 3600

    def __post_init__(self):
        if self.runtime not in {"pytorch", "onnx"}:
            raise ValueError("EDGEAI_API_RUNTIME must be 'pytorch' or 'onnx'.")
        if self.device != "cuda":
            raise ValueError("The inference API requires CUDA; EDGEAI_API_DEVICE must be 'cuda'.")
        if self.input_size not in SUPPORTED_WEBCAM_SIZES:
            raise ValueError(f"EDGEAI_API_IMGSZ must be one of {SUPPORTED_WEBCAM_SIZES}.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("EDGEAI_API_CONF must be between 0 and 1.")
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("EDGEAI_API_IOU must be greater than 0 and at most 1.")
        if self.max_upload_bytes < 1:
            raise ValueError("EDGEAI_API_MAX_UPLOAD_BYTES must be positive.")
        if self.max_video_bytes < 1:
            raise ValueError("EDGEAI_API_MAX_VIDEO_BYTES must be positive.")
        if self.max_video_seconds <= 0:
            raise ValueError("EDGEAI_API_MAX_VIDEO_SECONDS must be positive.")
        if self.max_video_frames < 1:
            raise ValueError("EDGEAI_API_MAX_VIDEO_FRAMES must be positive.")

    @classmethod
    def from_environment(cls) -> "ApiSettings":
        return cls(
            runtime=os.getenv("EDGEAI_API_RUNTIME", "pytorch").strip().lower(),
            device=os.getenv("EDGEAI_API_DEVICE", "cuda").strip().lower(),
            input_size=int(os.getenv("EDGEAI_API_IMGSZ", "640")),
            confidence=float(os.getenv("EDGEAI_API_CONF", str(DEFAULT_CONFIDENCE))),
            iou_threshold=float(os.getenv("EDGEAI_API_IOU", str(DEFAULT_IOU_THRESHOLD))),
            model_name=os.getenv("EDGEAI_API_MODEL", DEFAULT_MODEL),
            onnx_model_path=os.getenv("EDGEAI_API_ONNX_MODEL", str(DEFAULT_ONNX_PATH)),
            max_upload_bytes=int(os.getenv("EDGEAI_API_MAX_UPLOAD_BYTES", str(MAX_IMAGE_BYTES))),
            max_video_bytes=int(os.getenv("EDGEAI_API_MAX_VIDEO_BYTES", str(100 * 1024 * 1024))),
            max_video_seconds=float(os.getenv("EDGEAI_API_MAX_VIDEO_SECONDS", "120")),
            max_video_frames=int(os.getenv("EDGEAI_API_MAX_VIDEO_FRAMES", "3600")),
        )


class HealthResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    detector_initialized: bool
    active_runtime: str | None
    active_device: str | None
    model_input_resolution: int | None


class SupportedConfiguration(BaseModel):
    runtime: Literal["pytorch", "onnx"]
    device: Literal["cuda"]
    input_resolutions: list[int]


class ConfigResponse(BaseModel):
    model_name: str
    runtime: Literal["pytorch", "onnx"]
    device: Literal["cuda"]
    execution_provider: str | None
    input_resolution: int
    confidence_threshold: float
    nms_iou_threshold: float
    max_upload_bytes: int
    max_video_bytes: int
    max_video_seconds: float
    max_video_frames: int
    supported_configurations: list[SupportedConfiguration]


class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float] = Field(
        description=(
            "Absolute pixel coordinates [x_min, y_min, x_max, y_max] in the original image; "
            "origin is the top-left corner."
        )
    )


class DetectTimings(BaseModel):
    image_decode_ms: float
    inference_ms: float = Field(description="Detector call time, including runtime preprocessing and post-processing.")
    preprocessing_ms: float | None
    model_execution_ms: float | None
    postprocessing_ms: float | None
    annotation_encoding_ms: float | None
    total_request_processing_ms: float


class DetectResponse(BaseModel):
    image_width: int
    image_height: int
    model_input_resolution: int
    bbox_coordinate_system: str
    number_of_detections: int
    detections: list[DetectionItem]
    timings: DetectTimings
    annotated_image_base64: str | None = Field(
        default=None,
        description="Base64-encoded JPEG, returned only when include_annotated_image=true.",
    )


class VideoProcessingMetadata(BaseModel):
    source_frame_count: int
    processed_frame_count: int
    warmup_frames: int
    source_fps: float | None
    output_fps: float
    video_width: int
    video_height: int
    model_input_resolution: int
    warmup_inference_ms: float
    mean_inference_latency_ms: float
    median_inference_latency_ms: float
    model_inference_mean_ms: float
    model_inference_median_ms: float
    tracking_processing_mean_ms: float | None
    tracking_processing_median_ms: float | None
    p95_inference_latency_ms: float
    end_to_end_processing_fps: float
    whole_video_processing_fps: float
    total_processing_time_ms: float
    annotation_ms: float
    output_writing_ms: float
    video_decoding_ms: float
    measured_pipeline_ms: float
    runtime: Literal["pytorch", "onnx"]
    device: str
    tracking_enabled: bool
    tracking_inference_confidence: float | None
    total_detections: int
    unique_tracking_ids: int


def _onnx_path_for_size(settings: ApiSettings) -> Path:
    path = Path(settings.onnx_model_path).expanduser()
    if settings.input_size != 640:
        path = path.with_name(f"{path.stem}_{settings.input_size}{path.suffix}")
    return path


def build_detector(settings: ApiSettings):
    """Build one strict CUDA detector; a CPU fallback is never accepted here."""
    if not torch.cuda.is_available():
        raise DeviceUnavailableError("The API requires CUDA, but torch.cuda.is_available() is False.")
    if settings.runtime == "pytorch":
        detector = ImageDetector(model_name=settings.model_name, device="cuda")
        if detector.device != "cuda:0":
            raise DeviceUnavailableError("PyTorch did not initialize on CUDA device 0.")
        return detector

    model_path = _onnx_path_for_size(settings)
    detector = OnnxDetector(
        model_path=model_path,
        device="cuda",
        precision="fp32",
        require_cuda=True,
    )
    if detector.device != "cuda:0" or detector.execution_provider != "CUDAExecutionProvider":
        raise DeviceUnavailableError("ONNX Runtime did not initialize CUDAExecutionProvider.")
    if detector.input_size != settings.input_size:
        raise ModelLoadError(
            f"Selected ONNX graph is {detector.input_size}x{detector.input_size}; "
            f"expected {settings.input_size}x{settings.input_size}."
        )
    return detector


def _read_upload(upload: UploadFile, max_bytes: int) -> bytes:
    """Read a spooled upload in bounded chunks; never persist it as a project file."""
    buffer = bytearray()
    while True:
        chunk = upload.file.read(min(64 * 1024, max_bytes + 1 - len(buffer)))
        if not chunk:
            return bytes(buffer)
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise HTTPException(status_code=413, detail="Uploaded image exceeds the configured size limit.")


def _decode_uploaded_image(contents: bytes, filename: str) -> tuple[np.ndarray, float]:
    started = perf_counter()
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Only JPG, JPEG, and PNG images are supported.")
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    is_jpeg = contents.startswith(JPEG_SIGNATURE)
    is_png = contents.startswith(PNG_SIGNATURE)
    if not ((suffix in {".jpg", ".jpeg"} and is_jpeg) or (suffix == ".png" and is_png)):
        raise HTTPException(status_code=415, detail="Image contents must match a supported JPEG or PNG format.")
    try:
        image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        image = None
    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="Uploaded image is corrupt or cannot be decoded.")
    return image, (perf_counter() - started) * 1000.0


def _infer_image(
    detector,
    image: np.ndarray,
    image_decode_ms: float,
    settings: ApiSettings,
    include_annotated: bool,
    inference_lock,
) -> dict:
    """Inference runs off the event loop and snapshots shared detector timing under a lock."""
    height, width = image.shape[:2]
    annotation_encoding_ms = None
    annotated_base64 = None

    with inference_lock:
        result, inference_ms = detector.predict_frame(
            image,
            settings.confidence,
            image_size=settings.input_size,
            iou_threshold=settings.iou_threshold,
        )
        detector_timings = dict(getattr(detector, "last_timings", {}) or {})
        detections = extract_detections(result)

    detection_payload = [item.to_dict() for item in detections]
    if include_annotated:
        annotation_started = perf_counter()
        annotated = annotate_image(image, detections)
        encoded, data = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not encoded:
            raise InferenceError("OpenCV could not encode the annotated image.")
        annotated_base64 = base64.b64encode(data.tobytes()).decode("ascii")
        annotation_encoding_ms = (perf_counter() - annotation_started) * 1000.0

    return {
        "image_width": width,
        "image_height": height,
        "model_input_resolution": settings.input_size,
        "bbox_coordinate_system": "absolute_xyxy_pixels_in_original_image_origin_top_left",
        "number_of_detections": len(detection_payload),
        "detections": detection_payload,
        "inference_ms": float(inference_ms),
        "image_decode_ms": image_decode_ms,
        "preprocessing_ms": detector_timings.get("preprocess_ms"),
        "model_execution_ms": detector_timings.get("runtime_ms"),
        "postprocessing_ms": detector_timings.get("postprocess_ms"),
        "annotation_encoding_ms": annotation_encoding_ms,
        "annotated_image_base64": annotated_base64,
    }


def _run_video_inference(detector, source: Path, settings: ApiSettings, track: bool, inference_lock):
    """Run the established CLI video pipeline without loading another model."""
    processor = VideoProcessor(detector)
    with inference_lock:
        return processor.process_video(
            source,
            output_dir=source.parent,
            confidence=settings.confidence,
            warmup_frames=3,
            track=track,
            image_size=settings.input_size,
            iou_threshold=settings.iou_threshold,
            max_frames=settings.max_video_frames,
            max_duration_seconds=settings.max_video_seconds,
        )


def _video_metadata(run, settings: ApiSettings) -> VideoProcessingMetadata:
    metrics = run.performance
    p95 = float(np.percentile(metrics.inference_samples_ms, 95))
    processed_count = metrics.processed_frames
    return VideoProcessingMetadata(
        # The processor reads the entire source to EOF on successful requests.
        source_frame_count=processed_count,
        processed_frame_count=processed_count,
        warmup_frames=metrics.warmup_frames,
        source_fps=metrics.input_video_fps,
        output_fps=metrics.output_video_fps,
        video_width=run.frame_width,
        video_height=run.frame_height,
        model_input_resolution=settings.input_size,
        warmup_inference_ms=metrics.warmup_inference_ms,
        mean_inference_latency_ms=metrics.inference_mean_ms,
        median_inference_latency_ms=metrics.inference_median_ms,
        model_inference_mean_ms=metrics.model_inference_mean_ms,
        model_inference_median_ms=metrics.model_inference_median_ms,
        tracking_processing_mean_ms=metrics.tracking_processing_mean_ms,
        tracking_processing_median_ms=metrics.tracking_processing_median_ms,
        p95_inference_latency_ms=p95,
        end_to_end_processing_fps=metrics.end_to_end_fps,
        whole_video_processing_fps=metrics.whole_video_processing_fps,
        total_processing_time_ms=metrics.total_processing_ms,
        annotation_ms=metrics.annotation_ms,
        output_writing_ms=metrics.output_writing_ms,
        video_decoding_ms=metrics.video_decoding_ms,
        measured_pipeline_ms=metrics.measured_pipeline_ms,
        runtime=settings.runtime,
        device=run.device,
        tracking_enabled=run.tracking_enabled,
        tracking_inference_confidence=metrics.tracking_inference_confidence,
        total_detections=run.total_detections,
        unique_tracking_ids=run.unique_tracking_ids,
    )


def create_app(
    settings: ApiSettings | None = None,
    *,
    detector_factory: Callable[[ApiSettings], object] | None = None,
) -> FastAPI:
    config = settings or ApiSettings.from_environment()
    make_detector = detector_factory or build_detector

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        detector = await run_in_threadpool(make_detector, config)
        if getattr(detector, "device", None) != "cuda:0":
            raise DeviceUnavailableError("The API detector must be initialized on CUDA device 0.")
        if config.runtime == "onnx" and getattr(detector, "execution_provider", None) != "CUDAExecutionProvider":
            raise DeviceUnavailableError("The API requires ONNX Runtime CUDAExecutionProvider.")
        active_size = getattr(detector, "input_size", config.input_size)
        if active_size != config.input_size:
            raise ModelLoadError("Loaded detector input resolution does not match API configuration.")
        application.state.detector = detector
        try:
            yield
        finally:
            application.state.detector = None
            is_real_cuda_detector = isinstance(detector, (ImageDetector, OnnxDetector))
            if isinstance(detector, OnnxDetector):
                detector.session = None
            elif isinstance(detector, ImageDetector):
                detector.model = None
            del detector
            gc.collect()
            if is_real_cuda_detector and torch.cuda.is_available():
                torch.cuda.empty_cache()

    app = FastAPI(
        title="EdgeAI Inference API",
        description=(
            "CUDA-only image object detection using the existing YOLO11n PyTorch or ONNX FP32 detector. "
            "Bounding boxes use absolute xyxy pixel coordinates in the original image."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.api_settings = config
    app.state.detector = None
    app.state.detector_factory = make_detector
    from threading import RLock

    app.state.inference_lock = RLock()

    @app.middleware("http")
    async def record_start_and_reject_oversized_body(request: Request, call_next):
        request.state.started_at = perf_counter()
        if request.url.path in {"/api/v1/detect", "/api/v1/detect/video"}:
            content_length = request.headers.get("content-length")
            if content_length and content_length.isdigit():
                limit = config.max_video_bytes if request.url.path.endswith("/video") else config.max_upload_bytes
                if int(content_length) > limit + 64 * 1024:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Uploaded file exceeds the configured size limit."},
                    )
        return await call_next(request)

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    async def health(request: Request):
        detector = request.app.state.detector
        ready = detector is not None
        return HealthResponse(
            status="ready" if ready else "not_ready",
            detector_initialized=ready,
            active_runtime=(config.runtime if ready else None),
            active_device=(getattr(detector, "device", None) if ready else None),
            model_input_resolution=(getattr(detector, "input_size", config.input_size) if ready else None),
        )

    @app.get("/api/v1/config", response_model=ConfigResponse, tags=["service"])
    async def api_config():
        detector = app.state.detector
        model_name = (
            getattr(detector, "model_name", config.model_name)
            if config.runtime == "pytorch"
            else Path(config.onnx_model_path).name
        )
        if config.runtime == "onnx" and config.input_size != 640:
            model_name = _onnx_path_for_size(config).name
        return ConfigResponse(
            model_name=model_name,
            runtime=config.runtime,
            device="cuda",
            execution_provider=(
                getattr(detector, "execution_provider", None)
                if config.runtime == "onnx"
                else None
            ),
            input_resolution=config.input_size,
            confidence_threshold=config.confidence,
            nms_iou_threshold=config.iou_threshold,
            max_upload_bytes=config.max_upload_bytes,
            max_video_bytes=config.max_video_bytes,
            max_video_seconds=config.max_video_seconds,
            max_video_frames=config.max_video_frames,
            supported_configurations=[
                SupportedConfiguration(runtime="pytorch", device="cuda", input_resolutions=list(SUPPORTED_WEBCAM_SIZES)),
                SupportedConfiguration(runtime="onnx", device="cuda", input_resolutions=list(SUPPORTED_WEBCAM_SIZES)),
            ],
        )

    @app.post(
        "/api/v1/detect",
        response_model=DetectResponse,
        tags=["inference"],
        summary="Detect objects in an uploaded image",
        responses={
            400: {"description": "Empty, corrupt, or undecodable image."},
            413: {"description": "Uploaded file exceeds the configured limit."},
            415: {"description": "Unsupported image extension or file format."},
            503: {"description": "Detector has not initialized."},
        },
    )
    async def detect_image(
        request: Request,
        file: UploadFile = File(..., description="JPG, JPEG, or PNG image."),
        include_annotated_image: bool = Form(
            False, description="Return the annotated image as a Base64 JPEG when true."
        ),
    ):
        total_started = getattr(request.state, "started_at", perf_counter())
        detector = request.app.state.detector
        if detector is None:
            raise HTTPException(status_code=503, detail="Detector is not initialized.")
        filename = file.filename or ""
        suffix = Path(filename).suffix.lower()
        try:
            if suffix not in SUPPORTED_SUFFIXES:
                raise HTTPException(status_code=415, detail="Only JPG, JPEG, and PNG images are supported.")
            contents = await run_in_threadpool(_read_upload, file, config.max_upload_bytes)
            image, image_decode_ms = await run_in_threadpool(_decode_uploaded_image, contents, filename)
            result = await run_in_threadpool(
                _infer_image,
                detector,
                image,
                image_decode_ms,
                config,
                include_annotated_image,
                request.app.state.inference_lock,
            )
            total_ms = (perf_counter() - total_started) * 1000.0
            return DetectResponse(
                image_width=result["image_width"],
                image_height=result["image_height"],
                model_input_resolution=result["model_input_resolution"],
                bbox_coordinate_system=result["bbox_coordinate_system"],
                number_of_detections=result["number_of_detections"],
                detections=[DetectionItem(**item) for item in result["detections"]],
                timings=DetectTimings(
                    image_decode_ms=result["image_decode_ms"],
                    inference_ms=result["inference_ms"],
                    preprocessing_ms=result["preprocessing_ms"],
                    model_execution_ms=result["model_execution_ms"],
                    postprocessing_ms=result["postprocessing_ms"],
                    annotation_encoding_ms=result["annotation_encoding_ms"],
                    total_request_processing_ms=total_ms,
                ),
                annotated_image_base64=result["annotated_image_base64"],
            )
        except HTTPException:
            raise
        except (InferenceError, cv2.error, ValueError) as exc:
            logger.warning("Image inference request failed (%s).", type(exc).__name__)
            raise HTTPException(status_code=500, detail="Image inference failed.") from None
        except Exception as exc:
            logger.warning("Image inference request failed (%s).", type(exc).__name__)
            raise HTTPException(status_code=500, detail="Image inference failed.") from None
        finally:
            await file.close()

    @app.post(
        "/api/v1/detect/video",
        response_class=StreamingResponse,
        tags=["inference"],
        summary="Detect objects in a video and download the annotated result",
        description=(
            "Upload MP4 or AVI video. The annotated file is streamed as the response body. "
            "Compact JSON processing metadata is returned in the X-EdgeAI-Metadata response header. "
            "Tracking is supported only with the PyTorch runtime."
        ),
        responses={
            200: {
                "description": "Annotated video stream. Read X-EdgeAI-Metadata for processing metrics.",
                "content": {"video/mp4": {}, "video/x-msvideo": {}},
                "headers": {
                    "X-EdgeAI-Metadata": {
                        "description": "JSON object containing source/output frame and timing metadata.",
                        "content": {
                            "application/json": {
                                "schema": VideoProcessingMetadata.model_json_schema(),
                            }
                        },
                    },
                    "Content-Disposition": {
                        "description": "Download filename for the annotated video.",
                        "schema": {"type": "string"},
                    },
                },
            },
            400: {"description": "Empty, corrupt, or unreadable video."},
            413: {"description": "Upload, frame count, or duration exceeds configured limits."},
            415: {"description": "Unsupported video extension or MIME type."},
            422: {"description": "Tracking requested with a runtime that does not support it."},
            500: {"description": "Video processing or output encoding failed."},
            503: {"description": "CUDA detector has not initialized."},
        },
    )
    async def detect_video(
        request: Request,
        file: UploadFile = File(..., description="MP4 or AVI video; supported by installed OpenCV codecs."),
        track: bool = Form(False, description="Enable ByteTrack (PyTorch runtime only)."),
    ):
        detector = request.app.state.detector
        if detector is None:
            await file.close()
            raise HTTPException(status_code=503, detail="Detector is not initialized.")
        if track and config.runtime != "pytorch":
            await file.close()
            raise HTTPException(status_code=422, detail="ByteTrack is supported only with the PyTorch runtime.")

        temp_directory: Path | None = None
        response_created = False
        try:
            temp_directory = Path(tempfile.mkdtemp(prefix="edgeai-api-video-"))
            suffix = validate_video_filename_and_type(file.filename or "", file.content_type)
            input_path = temp_directory / f"input{suffix}"
            await run_in_threadpool(copy_upload_to_path, file, input_path, config.max_video_bytes)
            await run_in_threadpool(
                probe_video,
                input_path,
                max_frames=config.max_video_frames,
                max_duration_seconds=config.max_video_seconds,
            )
            run = await run_in_threadpool(
                _run_video_inference,
                detector,
                input_path,
                config,
                track,
                request.app.state.inference_lock,
            )
            reported_frames = run.performance.reported_input_frames
            if reported_frames is not None and run.performance.processed_frames < reported_frames:
                raise HTTPException(
                    status_code=400,
                    detail="Video ended before all frames reported by its container could be decoded.",
                )
            output_path = Path(run.annotated_video_path)
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise VideoProcessingError("OpenCV did not create a readable output file.")
            metadata = _video_metadata(run, config).model_dump()
            media_type = "video/mp4" if suffix == ".mp4" else "video/x-msvideo"
            download_name = f"edgeai_annotated{suffix}"
            response = StreamingResponse(
                stream_file_and_cleanup(output_path, temp_directory),
                media_type=media_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{download_name}"',
                    "X-EdgeAI-Metadata": json.dumps(metadata, separators=(",", ":"), ensure_ascii=True),
                },
                background=BackgroundTask(cleanup_temp_directory, temp_directory),
            )
            response_created = True
            return response
        except HTTPException:
            raise
        except VideoInputError as exc:
            detail = str(exc)
            if "limit" in detail.lower() or "exceeds" in detail.lower():
                raise HTTPException(status_code=413, detail="Video exceeds the configured duration or frame limit.") from None
            raise HTTPException(status_code=400, detail="Video is corrupt or cannot be decoded.") from None
        except VideoProcessingError as exc:
            # VideoProcessor can report input/decode failures with this base error;
            # classify those as bad input while keeping writer/inference errors generic.
            detail = str(exc).lower()
            if "exceeds the configured" in detail:
                raise HTTPException(status_code=413, detail="Video exceeds the configured duration or frame limit.") from None
            if "could not open video" in detail or "no decodable frames" in detail or "invalid frame" in detail:
                raise HTTPException(status_code=400, detail="Video is corrupt or cannot be decoded.") from None
            logger.exception("Video inference request failed (%s).", type(exc).__name__)
            raise HTTPException(status_code=500, detail="Video processing failed.") from None
        except Exception as exc:
            logger.exception("Video inference request failed (%s).", type(exc).__name__)
            raise HTTPException(status_code=500, detail="Video processing failed.") from None
        finally:
            try:
                await file.close()
            except Exception as exc:
                logger.warning("Video upload close failed (%s).", type(exc).__name__)
            finally:
                if temp_directory is not None and not response_created:
                    await run_in_threadpool(cleanup_temp_directory, temp_directory)

    return app


app = create_app()
