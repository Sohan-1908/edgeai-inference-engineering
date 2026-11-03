"""Frame-by-frame YOLO video inference using the shared image detector."""

from pathlib import Path
from statistics import mean, median
from time import perf_counter

import cv2
import torch

from edgeai.config import DEFAULT_CONFIDENCE, DEFAULT_OUTPUT_DIR
from edgeai.detector import ImageDetector
from edgeai.errors import VideoInputError, VideoProcessingError
from edgeai.postprocess import annotate_image, extract_detections
from edgeai.tracking import BYTE_TRACKER_CONFIG, BYTE_TRACKER_MIN_CONFIDENCE, ByteTrackSession
from edgeai.video_io import output_video_path, validate_video_path
from edgeai.video_types import VideoDetectionRun, VideoPerformanceMetrics

FALLBACK_OUTPUT_FPS = 30.0


class VideoProcessor:
    """Process a video with an already-loaded :class:`ImageDetector`.

    ``detector`` owns the single YOLO model instance. Each frame reuses its
    timed ``predict_frame`` interface; this class never loads model weights.
    """

    def __init__(self, detector: ImageDetector, *, capture_factory=None, writer_factory=None):
        self.detector = detector
        self.capture_factory = capture_factory
        self.writer_factory = writer_factory

    @staticmethod
    def _validate_settings(confidence: float, warmup_frames: int):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if warmup_frames < 0:
            raise ValueError("warmup_frames cannot be negative")

    def process_video(
        self,
        video_path: str | Path,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        confidence: float = DEFAULT_CONFIDENCE,
        warmup_frames: int = 3,
        track: bool = False,
        image_size: int | None = None,
        iou_threshold: float | None = None,
        frame_callback=None,
        max_frames: int | None = None,
        max_duration_seconds: float | None = None,
    ) -> VideoDetectionRun:
        self._validate_settings(confidence, warmup_frames)
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive when provided")
        if max_duration_seconds is not None and max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive when provided")
        source = validate_video_path(video_path).resolve()
        tracker_session = ByteTrackSession(self.detector) if track else None
        if tracker_session is not None:
            tracker_session.reset()
        started_total = perf_counter()
        capture = None
        writer = None
        destination = output_video_path(output_dir, source)
        decode_ms = 0.0
        warmup_inference_ms = 0.0
        annotation_ms = 0.0
        writing_ms = 0.0
        measured_pipeline_started = None
        measured_pipeline_ms = 0.0
        inference_samples = []
        model_inference_samples = []
        tracking_processing_samples = []
        runtime_preprocess_samples = []
        runtime_execution_samples = []
        runtime_postprocess_samples = []
        tracking_ids = set()
        current_visible_tracked_objects = 0
        processed_frames = 0
        total_detections = 0
        output_fps = FALLBACK_OUTPUT_FPS
        input_fps = None
        reported_frames = None
        width = height = 0
        effective_warmup_frames = warmup_frames
        memory_reset = False
        gpu_memory_baseline_mb = None
        gpu_memory_peak_allocated_mb = None
        gpu_memory_peak_reserved_mb = None

        capture_maker = self.capture_factory or cv2.VideoCapture
        writer_maker = self.writer_factory or cv2.VideoWriter
        try:
            capture = capture_maker(str(source))
            if not capture.isOpened():
                raise VideoInputError(f"OpenCV could not open video: {source}")

            raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if raw_fps > 0 and raw_fps < float("inf"):
                input_fps = raw_fps
                output_fps = raw_fps
            raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if raw_count > 0 and raw_count < float("inf"):
                reported_frames = int(round(raw_count))
                # Keep at least one frame for measured latency on short clips.
                effective_warmup_frames = min(warmup_frames, max(reported_frames - 1, 0))

            while True:
                frame_started = perf_counter()
                decode_started = perf_counter()
                ok, frame = capture.read()
                decode_ms += (perf_counter() - decode_started) * 1000.0
                if not ok:
                    break
                if frame is None or frame.size == 0:
                    raise VideoInputError(f"Video contains an invalid frame near index {processed_frames}.")
                if max_frames is not None and processed_frames >= max_frames:
                    raise VideoInputError("Video exceeds the configured frame limit.")
                duration_fps = input_fps or FALLBACK_OUTPUT_FPS
                if (
                    max_duration_seconds is not None
                    and (processed_frames + 1) / duration_fps > max_duration_seconds
                ):
                    raise VideoInputError("Video exceeds the configured duration limit.")

                if width == 0 or height == 0:
                    height, width = frame.shape[:2]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    codec = "mp4v" if source.suffix.lower() == ".mp4" else "MJPG"
                    writer = writer_maker(
                        str(destination),
                        cv2.VideoWriter_fourcc(*codec),
                        output_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise VideoProcessingError(
                            f"OpenCV could not create output video '{destination}' using {codec}."
                        )
                elif frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

                is_warmup = processed_frames < effective_warmup_frames
                if not is_warmup and measured_pipeline_started is None:
                    measured_pipeline_started = frame_started
                try:
                    if is_warmup:
                        if tracker_session is not None:
                            result, elapsed_ms, _ = tracker_session.predict_frame(
                                frame,
                                confidence,
                                image_size=image_size,
                                iou_threshold=iou_threshold,
                            )
                        elif image_size is not None or iou_threshold is not None:
                            result, elapsed_ms = self.detector.predict_frame(
                                frame,
                                confidence,
                                image_size=image_size,
                                iou_threshold=iou_threshold,
                            )
                        else:
                            result, elapsed_ms = self.detector.predict_frame(frame, confidence)
                        warmup_inference_ms += elapsed_ms
                    else:
                        if (
                            self.detector.device == "cuda:0"
                            and getattr(self.detector, "gpu_memory_supported", True)
                            and not memory_reset
                        ):
                            torch.cuda.synchronize(0)
                            torch.cuda.reset_peak_memory_stats(0)
                            gpu_memory_baseline_mb = torch.cuda.memory_allocated(0) / (1024**2)
                            memory_reset = True
                        if tracker_session is not None:
                            result, elapsed_ms, tracker_ms = tracker_session.predict_frame(
                                frame,
                                confidence,
                                image_size=image_size,
                                iou_threshold=iou_threshold,
                            )
                            tracking_processing_samples.append(tracker_ms)
                        elif image_size is not None or iou_threshold is not None:
                            result, elapsed_ms = self.detector.predict_frame(
                                frame,
                                confidence,
                                image_size=image_size,
                                iou_threshold=iou_threshold,
                            )
                        else:
                            result, elapsed_ms = self.detector.predict_frame(frame, confidence)
                        inference_samples.append(elapsed_ms)
                        timing = getattr(self.detector, "last_timings", {}) or {}
                        if timing:
                            runtime_preprocess_samples.append(float(timing.get("preprocess_ms", 0.0)))
                            runtime_execution_samples.append(float(timing.get("runtime_ms", elapsed_ms)))
                            runtime_postprocess_samples.append(float(timing.get("postprocess_ms", 0.0)))
                    detections = extract_detections(result)
                    if frame_callback is not None:
                        frame_callback(processed_frames, detections)
                    if tracker_session is not None:
                        frame_ids = [detection.track_id for detection in detections if detection.track_id is not None]
                        tracking_ids.update(frame_ids)
                        current_visible_tracked_objects = len(frame_ids)
                except Exception as exc:
                    raise VideoProcessingError(
                        f"Video inference failed at frame {processed_frames}: {exc}"
                    ) from exc

                annotation_started = perf_counter()
                try:
                    annotated = annotate_image(frame, detections)
                except Exception as exc:
                    raise VideoProcessingError(
                        f"Could not annotate video frame {processed_frames}: {exc}"
                    ) from exc
                annotation_ms += (perf_counter() - annotation_started) * 1000.0

                write_started = perf_counter()
                try:
                    writer.write(annotated)
                except Exception as exc:
                    raise VideoProcessingError(
                        f"Could not write video frame {processed_frames}: {exc}"
                    ) from exc
                writing_ms += (perf_counter() - write_started) * 1000.0
                total_detections += len(detections)
                speed = getattr(result, "speed", {}) or {}
                model_inference_ms = float(speed.get("inference", elapsed_ms))
                if not is_warmup:
                    model_inference_samples.append(model_inference_ms)
                processed_frames += 1

            if processed_frames == 0:
                raise VideoInputError(f"Video contains no decodable frames: {source}")
            if not inference_samples:
                raise VideoInputError(
                    "Video is too short to contain measured frames after warm-up; use --warmup-frames 0."
                )
            if (
                self.detector.device == "cuda:0"
                and getattr(self.detector, "gpu_memory_supported", True)
            ):
                torch.cuda.synchronize(0)
                gpu_memory_peak_allocated_mb = torch.cuda.max_memory_allocated(0) / (1024**2)
                gpu_memory_peak_reserved_mb = torch.cuda.max_memory_reserved(0) / (1024**2)
        except (VideoInputError, VideoProcessingError):
            raise
        except Exception as exc:
            raise VideoProcessingError(f"Video processing failed for '{source}': {exc}") from exc
        finally:
            try:
                if writer is not None:
                    writer.release()
            finally:
                if capture is not None:
                    capture.release()

        total_processing_ms = (perf_counter() - started_total) * 1000.0
        if measured_pipeline_started is not None:
            measured_pipeline_ms = (perf_counter() - measured_pipeline_started) * 1000.0
        model_inference_total_ms = sum(model_inference_samples)
        model_only_fps = len(model_inference_samples) * 1000.0 / model_inference_total_ms
        end_to_end_fps = len(inference_samples) * 1000.0 / measured_pipeline_ms
        whole_video_processing_fps = processed_frames * 1000.0 / total_processing_ms
        metrics = VideoPerformanceMetrics(
            model_load_ms=self.detector.model_load_ms,
            video_decoding_ms=decode_ms,
            warmup_inference_ms=warmup_inference_ms,
            inference_mean_ms=mean(inference_samples),
            inference_median_ms=median(inference_samples),
            inference_samples_ms=tuple(inference_samples),
            model_inference_mean_ms=mean(model_inference_samples),
            model_inference_median_ms=median(model_inference_samples),
            model_inference_samples_ms=tuple(model_inference_samples),
            tracking_processing_mean_ms=(
                mean(tracking_processing_samples) if tracker_session is not None else None
            ),
            tracking_processing_median_ms=(
                median(tracking_processing_samples) if tracker_session is not None else None
            ),
            tracking_processing_samples_ms=tuple(tracking_processing_samples),
            annotation_ms=annotation_ms,
            output_writing_ms=writing_ms,
            total_processing_ms=total_processing_ms,
            measured_pipeline_ms=measured_pipeline_ms,
            input_video_fps=input_fps,
            output_video_fps=output_fps,
            model_only_fps=model_only_fps,
            end_to_end_fps=end_to_end_fps,
            whole_video_processing_fps=whole_video_processing_fps,
            warmup_frames=effective_warmup_frames,
            measured_frames=len(inference_samples),
            processed_frames=processed_frames,
            reported_input_frames=reported_frames,
            gpu_memory_baseline_mb=gpu_memory_baseline_mb,
            gpu_memory_peak_allocated_mb=gpu_memory_peak_allocated_mb,
            gpu_memory_peak_reserved_mb=gpu_memory_peak_reserved_mb,
            tracking_enabled=tracker_session is not None,
            tracker_name=BYTE_TRACKER_CONFIG if tracker_session is not None else None,
            tracking_inference_confidence=(
                min(confidence, BYTE_TRACKER_MIN_CONFIDENCE)
                if tracker_session is not None
                else None
            ),
            runtime_name=getattr(self.detector, "runtime_name", "pytorch"),
            runtime_version=getattr(self.detector, "runtime_version", None),
            execution_provider=getattr(self.detector, "execution_provider", None),
            model_size_bytes=getattr(self.detector, "model_size_bytes", None),
            fallback_error=getattr(self.detector, "fallback_error", None),
            runtime_preprocessing_mean_ms=(
                mean(runtime_preprocess_samples) if runtime_preprocess_samples else None
            ),
            runtime_execution_mean_ms=(
                mean(runtime_execution_samples) if runtime_execution_samples else None
            ),
            runtime_postprocessing_mean_ms=(
                mean(runtime_postprocess_samples) if runtime_postprocess_samples else None
            ),
            precision=getattr(self.detector, "precision", "FP32"),
        )
        return VideoDetectionRun(
            video_path=str(source),
            annotated_video_path=str(destination.resolve()),
            model=self.detector.model_name,
            device=self.detector.device,
            confidence_threshold=confidence,
            frame_width=width,
            frame_height=height,
            total_detections=total_detections,
            unique_tracking_ids=len(tracking_ids),
            current_visible_tracked_objects=current_visible_tracked_objects,
            tracking_enabled=tracker_session is not None,
            performance=metrics,
        )
