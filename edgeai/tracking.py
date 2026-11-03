"""Ultralytics ByteTrack session and per-update timing support."""

from time import perf_counter
from contextlib import nullcontext

import torch

from edgeai.device import ultralytics_device
from edgeai.errors import InferenceError

BYTE_TRACKER_CONFIG = "bytetrack.yaml"
BYTE_TRACKER_MIN_CONFIDENCE = 0.1


class _TrackerUpdateTimer:
    """Time ByteTrack.update without folding it into model forward latency."""

    def __init__(self):
        self.last_update_ms: float | None = None

    def clear(self):
        self.last_update_ms = None

    def attach(self, predictor):
        for tracker in getattr(predictor, "trackers", ()):
            if getattr(tracker, "_edgeai_update_timer", None) is self:
                continue
            original_update = tracker.update
            timer = self

            def timed_update(*args, _original_update=original_update, **kwargs):
                started = perf_counter()
                try:
                    return _original_update(*args, **kwargs)
                finally:
                    timer.last_update_ms = (perf_counter() - started) * 1000.0

            tracker.update = timed_update
            tracker._edgeai_update_timer = self


def _get_update_timer(model) -> _TrackerUpdateTimer:
    timer = getattr(model, "_edgeai_bytetrack_update_timer", None)
    if timer is None:
        from ultralytics.trackers import register_tracker

        # Register Ultralytics' callbacks first, then our timer callback. The
        # model.track() API refreshes the official callbacks in place, leaving
        # this callback after tracker creation and before each update.
        register_tracker(model, persist=True)
        timer = _TrackerUpdateTimer()
        model.add_callback("on_predict_start", timer.attach)
        model._edgeai_bytetrack_update_timer = timer
    return timer


class ByteTrackSession:
    """Use Ultralytics' supported model.track API for one video stream.

    The model weights stay on the shared detector. Tracker state is reset at
    the start of each session so IDs are not carried between unrelated videos.
    """

    def __init__(self, detector, tracker_config: str = BYTE_TRACKER_CONFIG):
        self.detector = detector
        self.tracker_config = tracker_config
        self.timer = _get_update_timer(detector.model)

    def reset(self):
        predictor = getattr(self.detector.model, "predictor", None)
        if predictor is not None:
            if hasattr(predictor, "trackers"):
                # Ultralytics will create fresh ByteTrack instances on the
                # next persist=True call when this list is empty.
                predictor.trackers = []
            if hasattr(predictor, "vid_path"):
                predictor.vid_path = [None]
        self.timer.clear()

    def predict_frame(
        self,
        frame,
        confidence: float,
        *,
        image_size: int | None = None,
        iou_threshold: float | None = None,
    ):
        self.timer.clear()
        try:
            guard = getattr(self.detector, "inference_lock", None) or nullcontext()
            with guard:
                if self.detector.device == "cuda:0":
                    torch.cuda.synchronize(0)
                started = perf_counter()
                options = {}
                if image_size is not None:
                    options.update(imgsz=int(image_size), rect=False)
                if iou_threshold is not None:
                    options["iou"] = float(iou_threshold)
                results = self.detector.model.track(
                    source=frame,
                    # Keep ByteTrack's low-confidence association candidates. The
                    # tracker applies its own high/low thresholds after inference.
                    conf=min(confidence, BYTE_TRACKER_MIN_CONFIDENCE),
                    device=ultralytics_device(self.detector.device),
                    tracker=self.tracker_config,
                    persist=True,
                    verbose=False,
                    **options,
                )
                if self.detector.device == "cuda:0":
                    torch.cuda.synchronize(0)
                elapsed_ms = (perf_counter() - started) * 1000.0
                return results[0], elapsed_ms, self.timer.last_update_ms or 0.0
        except Exception as exc:
            raise InferenceError(f"YOLO ByteTrack inference failed: {exc}") from exc
