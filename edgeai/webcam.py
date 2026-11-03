"""Interactive webcam inference built on EdgeAI's existing detectors."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from statistics import mean, median
from threading import Condition, Event, Lock, Thread
from time import perf_counter

import cv2
import numpy as np

from edgeai.config import DEFAULT_CONFIDENCE
from edgeai.detector_types import Detection
from edgeai.errors import InferenceError, WebcamInputError
from edgeai.postprocess import annotate_image, extract_detections
from edgeai.tracking import ByteTrackSession


SUPPORTED_WEBCAM_SIZES = (640, 480, 320)


@dataclass(frozen=True)
class WebcamRun:
    camera_index: int
    requested_capture_width: int
    requested_capture_height: int
    capture_width: int
    capture_height: int
    inference_size: int
    runtime: str
    device: str
    execution_provider: str
    model_load_ms: float
    warmup_ms: float
    warmup_runs: int
    processed_frames: int
    captured_frames: int
    captured_frames_total: int
    frames_replaced: int
    frames_replaced_total: int
    frame_replacement_rate: float | None
    capture_fps: float | None
    inference_fps: float | None
    elapsed_ms: float
    measured_elapsed_ms: float
    inference_mean_ms: float | None
    inference_median_ms: float | None
    inference_p95_ms: float | None
    model_runtime_mean_ms: float | None
    end_to_end_fps: float | None
    frame_age_mean_ms: float | None
    frame_age_median_ms: float | None
    frame_age_p95_ms: float | None
    approximate_frame_to_display_mean_ms: float | None
    capture_mode: str
    tracking_enabled: bool
    unique_tracking_ids: int
    termination_reason: str
    last_frame_detections: tuple[Detection, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["last_frame_detections"] = [item.to_dict() for item in self.last_frame_detections]
        return data


def validate_webcam_settings(
    camera_index: int,
    capture_width: int,
    capture_height: int,
    inference_size: int,
    warmup_runs: int,
) -> None:
    if camera_index < 0:
        raise ValueError("camera_index must be zero or greater")
    if capture_width < 1 or capture_height < 1:
        raise ValueError("camera capture width and height must be positive")
    if inference_size not in SUPPORTED_WEBCAM_SIZES:
        raise ValueError(f"inference_size must be one of {SUPPORTED_WEBCAM_SIZES}")
    if warmup_runs < 0:
        raise ValueError("warmup_runs cannot be negative")


@dataclass(frozen=True)
class CapturedFrame:
    frame: np.ndarray
    sequence: int
    captured_at: float


@dataclass(frozen=True)
class CaptureMarker:
    captured_frames: int
    replaced_frames: int
    timestamp: float


_ACTIVE_CAPTURE_LOCK = Lock()
_ACTIVE_CAPTURE_CAMERAS: set[int] = set()
_ACTIVE_DETECTOR_LOCK = Lock()
_ACTIVE_DETECTORS: set[int] = set()


class LatestFrameCapture:
    """Single-camera capture worker with one replaceable, thread-safe slot."""

    def __init__(self, camera_index: int, width: int, height: int, *, capture_factory=None):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.capture_factory = capture_factory or cv2.VideoCapture
        self._condition = Condition()
        self._started = Event()
        self._stop = Event()
        self._thread: Thread | None = None
        self._capture = None
        self._latest: CapturedFrame | None = None
        self._slot_consumed = True
        self._captured = 0
        self._replaced = 0
        self._sequence = 0
        self._startup_error: Exception | None = None
        self._thread_error: Exception | None = None
        self._disconnected = False
        self._capture_started_at: float | None = None
        self._capture_stopped_at: float | None = None
        self._reserved = False
        self._closed = False

    def start(self, timeout: float = 10.0) -> "LatestFrameCapture":
        with _ACTIVE_CAPTURE_LOCK:
            if self.camera_index in _ACTIVE_CAPTURE_CAMERAS:
                raise WebcamInputError(
                    f"A webcam capture thread is already active for camera index {self.camera_index}."
                )
            _ACTIVE_CAPTURE_CAMERAS.add(self.camera_index)
            self._reserved = True
        self._thread = Thread(
            target=self._capture_loop,
            name=f"edgeai-webcam-capture-{self.camera_index}",
            daemon=False,
        )
        self._thread.start()
        if not self._started.wait(timeout):
            self.stop()
            raise WebcamInputError(f"Timed out opening webcam camera index {self.camera_index}.")
        if self._startup_error is not None:
            error = self._startup_error
            self.stop()
            if isinstance(error, WebcamInputError):
                raise error
            raise WebcamInputError(f"Could not initialize webcam camera index {self.camera_index}: {error}") from error
        return self

    def _release_camera_reservation(self):
        if self._reserved:
            with _ACTIVE_CAPTURE_LOCK:
                _ACTIVE_CAPTURE_CAMERAS.discard(self.camera_index)
            self._reserved = False

    def _capture_loop(self):
        capture = None
        try:
            capture = self.capture_factory(self.camera_index)
            self._capture = capture
            if capture is None or not capture.isOpened():
                raise WebcamInputError(f"Could not open webcam camera index {self.camera_index}.")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._capture_started_at = perf_counter()
            self._started.set()
            while not self._stop.is_set():
                ok, frame = capture.read()
                captured_at = perf_counter()
                if self._stop.is_set():
                    break
                if not ok or frame is None or frame.size == 0:
                    self._disconnected = True
                    break
                with self._condition:
                    self._sequence += 1
                    self._captured += 1
                    if self._latest is not None and not self._slot_consumed:
                        self._replaced += 1
                    self._latest = CapturedFrame(frame, self._sequence, captured_at)
                    self._slot_consumed = False
                    self._condition.notify_all()
        except Exception as exc:
            if not self._started.is_set():
                self._startup_error = exc
            else:
                self._thread_error = exc
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception as exc:
                    if self._thread_error is None:
                        self._thread_error = exc
            self._capture_stopped_at = perf_counter()
            self._started.set()
            with self._condition:
                self._condition.notify_all()
            self._release_camera_reservation()

    def get_latest(self, after_sequence: int = 0, timeout: float | None = 1.0) -> CapturedFrame | None:
        deadline = None if timeout is None else perf_counter() + timeout
        with self._condition:
            while True:
                if self._thread_error is not None:
                    raise WebcamInputError(f"Webcam capture thread failed: {self._thread_error}") from self._thread_error
                if self._latest is not None and self._latest.sequence > after_sequence:
                    self._slot_consumed = True
                    packet = self._latest
                    return CapturedFrame(packet.frame.copy(), packet.sequence, packet.captured_at)
                if self._capture_stopped_at is not None or self._stop.is_set():
                    return None
                remaining = None if deadline is None else deadline - perf_counter()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def measurement_marker(self) -> CaptureMarker:
        with self._condition:
            # A pending warm-up frame is intentionally discarded at the
            # measurement boundary and is not counted as a measured drop.
            self._slot_consumed = True
            return CaptureMarker(self._captured, self._replaced, perf_counter())

    def measurement_stats(self, marker: CaptureMarker) -> tuple[int, int, float]:
        with self._condition:
            captured = max(0, self._captured - marker.captured_frames)
            replaced = max(0, self._replaced - marker.replaced_frames)
            stopped = self._capture_stopped_at or perf_counter()
            duration = max(0.0, stopped - marker.timestamp)
            return captured, replaced, duration

    def total_stats(self) -> tuple[int, int]:
        with self._condition:
            return self._captured, self._replaced

    @property
    def capture_size(self) -> tuple[int, int]:
        if self._latest is not None:
            return int(self._latest.frame.shape[1]), int(self._latest.frame.shape[0])
        return 0, 0

    @property
    def disconnected(self) -> bool:
        return self._disconnected

    def stop(self, timeout: float = 3.0):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
            if thread.is_alive():
                # Some camera backends block in read(); releasing the handle
                # is the unblock mechanism before one final bounded join.
                if self._capture is not None:
                    self._capture.release()
                thread.join(timeout)
            if thread.is_alive():
                self._release_camera_reservation()
                raise WebcamInputError(
                    f"Webcam capture thread for camera index {self.camera_index} did not stop cleanly."
                )
        with self._condition:
            if self._latest is not None and not self._slot_consumed:
                self._replaced += 1
                self._slot_consumed = True
        self._release_camera_reservation()


class SequentialWebcamCapture:
    """Synchronous capture source retained for controlled comparisons."""

    def __init__(self, camera_index: int, width: int, height: int, *, capture_factory=None):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.capture_factory = capture_factory or cv2.VideoCapture
        self.capture = None
        self.sequence = 0
        self.captured = 0
        self.replaced = 0
        self.started_at = None
        self.stopped_at = None
        self._capture_size = (0, 0)
        self._disconnected = False
        self._closed = False

    def start(self):
        self.capture = self.capture_factory(self.camera_index)
        if self.capture is None or not self.capture.isOpened():
            if self.capture is not None:
                self.capture.release()
            raise WebcamInputError(f"Could not open webcam camera index {self.camera_index}.")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.started_at = perf_counter()
        return self

    def get_latest(self, after_sequence: int = 0, timeout: float | None = None) -> CapturedFrame | None:
        if self.capture is None or self._disconnected:
            return None
        ok, frame = self.capture.read()
        captured_at = perf_counter()
        if not ok or frame is None or frame.size == 0:
            self._disconnected = True
            return None
        self.sequence += 1
        self.captured += 1
        self._capture_size = (int(frame.shape[1]), int(frame.shape[0]))
        return CapturedFrame(frame, self.sequence, captured_at)

    def measurement_marker(self) -> CaptureMarker:
        return CaptureMarker(self.captured, self.replaced, perf_counter())

    def measurement_stats(self, marker: CaptureMarker) -> tuple[int, int, float]:
        end = self.stopped_at or perf_counter()
        return max(0, self.captured - marker.captured_frames), 0, max(0.0, end - marker.timestamp)

    def total_stats(self) -> tuple[int, int]:
        return self.captured, self.replaced

    @property
    def capture_size(self) -> tuple[int, int]:
        return self._capture_size

    @property
    def disconnected(self) -> bool:
        return self._disconnected

    def stop(self):
        if self._closed:
            return
        self._closed = True
        self.stopped_at = perf_counter()
        if self.capture is not None:
            self.capture.release()


def _reserve_detector(detector) -> int:
    identity = id(detector)
    with _ACTIVE_DETECTOR_LOCK:
        if identity in _ACTIVE_DETECTORS:
            raise InferenceError("This detector already has an active webcam inference session.")
        _ACTIVE_DETECTORS.add(identity)
    return identity


def _release_detector(identity: int):
    with _ACTIVE_DETECTOR_LOCK:
        _ACTIVE_DETECTORS.discard(identity)


def _overlay_status(
    frame: np.ndarray,
    *,
    runtime: str,
    input_size: int,
    rolling_fps: float | None,
    rolling_latency_ms: float | None,
    tracking: bool,
) -> np.ndarray:
    result = frame.copy()
    height, width = result.shape[:2]
    panel_width = min(width, 440)
    panel_height = min(height, 84)
    if panel_width <= 0 or panel_height <= 0:
        return result
    region = result[:panel_height, :panel_width]
    shade = np.zeros_like(region)
    result[:panel_height, :panel_width] = cv2.addWeighted(region, 0.35, shade, 0.65, 0)
    fps_label = "warming up" if rolling_fps is None else f"{rolling_fps:.1f} FPS"
    latency_label = "--" if rolling_latency_ms is None else f"{rolling_latency_ms:.1f} ms"
    suffix = " | ByteTrack" if tracking else ""
    cv2.putText(result, f"{runtime.upper()} CUDA | {input_size}x{input_size}{suffix}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"Processing: {fps_label} | Inference: {latency_label}",
                (10, min(panel_height - 10, 55)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (90, 240, 120), 2, cv2.LINE_AA)
    return result


def run_webcam(
    detector,
    *,
    camera_index: int = 0,
    capture_width: int = 1280,
    capture_height: int = 720,
    inference_size: int = 640,
    confidence: float = DEFAULT_CONFIDENCE,
    iou_threshold: float = 0.7,
    tracking: bool = False,
    warmup_runs: int = 10,
    display: bool = True,
    max_frames: int | None = None,
    duration_seconds: float | None = None,
    capture_mode: str = "threaded",
    capture_factory=None,
    key_reader=None,
) -> WebcamRun:
    """Capture and display webcam detections until Q/ESC or camera disconnect.

    Threaded capture is the default. Its single frame slot always contains the
    newest frame, replacing an unread stale frame when the camera outpaces
    inference. ``capture_mode='sequential'`` retains the old behavior for
    controlled comparisons. ``max_frames``, ``duration_seconds``, factories,
    and ``display=False`` support bounded tests and benchmark runs.
    """
    validate_webcam_settings(camera_index, capture_width, capture_height, inference_size, warmup_runs)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be greater than zero and at most one")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least one")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    if max_frames is not None and duration_seconds is not None:
        raise ValueError("Choose either max_frames or duration_seconds, not both")
    if capture_mode not in {"threaded", "sequential"}:
        raise ValueError("capture_mode must be 'threaded' or 'sequential'")
    if tracking and getattr(detector, "runtime_name", "pytorch") != "pytorch":
        raise ValueError("ByteTrack webcam mode is supported with the PyTorch runtime only.")

    detector_identity = _reserve_detector(detector)
    source = (
        LatestFrameCapture(camera_index, capture_width, capture_height, capture_factory=capture_factory)
        if capture_mode == "threaded"
        else SequentialWebcamCapture(camera_index, capture_width, capture_height,
                                     capture_factory=capture_factory)
    )
    read_key = key_reader or cv2.waitKey
    tracker = None
    inference_samples: list[float] = []
    model_runtime_samples: list[float] = []
    frame_age_samples: list[float] = []
    frame_to_display_samples: list[float] = []
    latencies_window: deque[float] = deque(maxlen=30)
    frame_times_window: deque[float] = deque(maxlen=30)
    seen_ids: set[int] = set()
    latest_detections: tuple[Detection, ...] = ()
    capture_width_actual = 0
    capture_height_actual = 0
    warmup_ms = 0.0
    measured_started: float | None = None
    end_time: float | None = None
    capture_marker: CaptureMarker | None = None
    frame_count = 0
    termination_reason = "user_quit"
    started_total = perf_counter()
    window_name = f"EdgeAI Webcam — {detector.runtime_name} CUDA"

    try:
        tracker = ByteTrackSession(detector) if tracking else None
        source.start()
        first_packet = source.get_latest(timeout=5.0)
        if first_packet is None:
            raise WebcamInputError(f"Webcam camera index {camera_index} opened but did not provide a frame.")
        warmup_frame = first_packet.frame
        capture_width_actual, capture_height_actual = warmup_frame.shape[1], warmup_frame.shape[0]

        if tracker is not None:
            tracker.reset()
        warm_started = perf_counter()
        for _ in range(warmup_runs):
            if tracker is not None:
                tracker.predict_frame(warmup_frame, confidence, image_size=inference_size,
                                      iou_threshold=iou_threshold)
            else:
                detector.predict_frame(warmup_frame, confidence, image_size=inference_size,
                                       iou_threshold=iou_threshold)
        if tracker is not None:
            tracker.reset()
        warmup_ms = (perf_counter() - warm_started) * 1000.0

        if display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        capture_marker = source.measurement_marker()
        measured_started = perf_counter()
        last_sequence = first_packet.sequence
        while True:
            if duration_seconds is not None and perf_counter() - measured_started >= duration_seconds:
                termination_reason = "duration_limit"
                break
            packet = source.get_latest(after_sequence=last_sequence, timeout=0.2)
            if packet is None:
                if source.disconnected:
                    termination_reason = "camera_disconnected_or_frame_read_failed"
                    break
                continue
            last_sequence = packet.sequence
            frame = packet.frame
            frame_age_samples.append(max(0.0, (perf_counter() - packet.captured_at) * 1000.0))
            if tracker is not None:
                result, inference_ms, _ = tracker.predict_frame(
                    frame, confidence, image_size=inference_size, iou_threshold=iou_threshold
                )
            else:
                result, inference_ms = detector.predict_frame(
                    frame, confidence, image_size=inference_size, iou_threshold=iou_threshold
                )
            latest_detections = tuple(extract_detections(result))
            if tracker is not None:
                seen_ids.update(item.track_id for item in latest_detections if item.track_id is not None)
            annotated = annotate_image(frame, list(latest_detections))
            now = perf_counter()
            inference_samples.append(float(inference_ms))
            speed = getattr(result, "speed", {}) or {}
            runtime_ms = speed.get("inference")
            if runtime_ms is None:
                runtime_ms = getattr(detector, "last_timings", {}).get("runtime_ms")
            if runtime_ms is not None:
                model_runtime_samples.append(float(runtime_ms))
            latencies_window.append(float(inference_ms))
            frame_times_window.append(now)
            rolling_fps = None
            if len(frame_times_window) >= 2:
                rolling_fps = (len(frame_times_window) - 1) / max(
                    frame_times_window[-1] - frame_times_window[0], 1e-9
                )
            rolling_latency = mean(latencies_window) if latencies_window else None
            shown = _overlay_status(
                annotated,
                runtime=getattr(detector, "runtime_name", "pytorch"),
                input_size=inference_size,
                rolling_fps=rolling_fps,
                rolling_latency_ms=rolling_latency,
                tracking=tracking,
            )
            frame_count += 1
            end_time = perf_counter()
            if display:
                cv2.imshow(window_name, shown)
                key = read_key(1) & 0xFF
                frame_to_display_samples.append(max(0.0, (perf_counter() - packet.captured_at) * 1000.0))
                if key in (ord("q"), 27):
                    end_time = perf_counter()
                    termination_reason = "user_quit"
                    break
            end_time = perf_counter()
            if max_frames is not None and frame_count >= max_frames:
                termination_reason = "frame_limit"
                break

        if frame_count == 0:
            raise WebcamInputError(f"Webcam camera index {camera_index} ended before a frame was processed.")
    except WebcamInputError:
        raise
    except Exception as exc:
        raise InferenceError(f"Webcam processing failed: {type(exc).__name__}: {exc}") from exc
    finally:
        try:
            source.stop()
        finally:
            _release_detector(detector_identity)
            if display:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

    elapsed_ms = perf_counter() - started_total
    measured_elapsed_ms = (
        max(0.0, (end_time or perf_counter()) - measured_started) * 1000.0
        if measured_started is not None else 0.0
    )
    captured_frames, replaced_frames, capture_duration = (
        source.measurement_stats(capture_marker) if capture_marker is not None else (0, 0, 0.0)
    )
    captured_frames_total, replaced_frames_total = source.total_stats()
    inference_mean = mean(inference_samples) if inference_samples else None
    return WebcamRun(
        camera_index=camera_index,
        requested_capture_width=capture_width,
        requested_capture_height=capture_height,
        capture_width=capture_width_actual,
        capture_height=capture_height_actual,
        inference_size=inference_size,
        runtime=getattr(detector, "runtime_name", "pytorch"),
        device=detector.device,
        execution_provider=getattr(detector, "execution_provider", "CUDAExecutionProvider"),
        model_load_ms=float(getattr(detector, "model_load_ms", 0.0)),
        warmup_ms=warmup_ms,
        warmup_runs=warmup_runs,
        processed_frames=frame_count,
        captured_frames=captured_frames,
        captured_frames_total=captured_frames_total,
        frames_replaced=replaced_frames,
        frames_replaced_total=replaced_frames_total,
        frame_replacement_rate=(100.0 * replaced_frames / captured_frames) if captured_frames else None,
        capture_fps=(captured_frames / capture_duration) if capture_duration > 0 else None,
        inference_fps=(frame_count * 1000.0 / sum(inference_samples)) if inference_samples else None,
        elapsed_ms=elapsed_ms * 1000.0,
        measured_elapsed_ms=measured_elapsed_ms,
        inference_mean_ms=inference_mean,
        inference_median_ms=median(inference_samples) if inference_samples else None,
        inference_p95_ms=float(np.percentile(inference_samples, 95)) if inference_samples else None,
        model_runtime_mean_ms=mean(model_runtime_samples) if model_runtime_samples else None,
        end_to_end_fps=(frame_count * 1000.0 / measured_elapsed_ms) if measured_elapsed_ms > 0 else None,
        frame_age_mean_ms=mean(frame_age_samples) if frame_age_samples else None,
        frame_age_median_ms=median(frame_age_samples) if frame_age_samples else None,
        frame_age_p95_ms=float(np.percentile(frame_age_samples, 95)) if frame_age_samples else None,
        approximate_frame_to_display_mean_ms=(
            mean(frame_to_display_samples) if frame_to_display_samples else None
        ),
        capture_mode=capture_mode,
        tracking_enabled=tracking,
        unique_tracking_ids=len(seen_ids),
        termination_reason=termination_reason,
        last_frame_detections=latest_detections,
    )
