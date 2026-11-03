import unittest
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from edgeai.errors import InferenceError, WebcamInputError
from edgeai.detector import ImageDetector
from edgeai.webcam import LatestFrameCapture, _reserve_detector, _release_detector, run_webcam, validate_webcam_settings
from main import build_parser


class _Boxes:
    xyxy = np.asarray([[8, 9, 35, 38]], dtype=np.float32)
    conf = np.asarray([0.9], dtype=np.float32)
    cls = np.asarray([0], dtype=np.float32)
    id = None

    def __len__(self):
        return 1


class _FakeDetector:
    runtime_name = "pytorch"
    runtime_version = "test"
    device = "cuda:0"
    execution_provider = "CUDAExecutionProvider"
    model_load_ms = 4.0
    last_timings = {"runtime_ms": 2.0}

    def __init__(self, broken=False):
        self.calls = []
        self.broken = broken

    def predict_frame(self, frame, confidence, *, image_size, iou_threshold):
        self.calls.append((confidence, image_size, iou_threshold))
        if self.broken:
            raise RuntimeError("synthetic inference failure")
        result = SimpleNamespace(
            boxes=_Boxes(), names={0: "person"}, speed={"inference": 2.0}
        )
        return result, 2.5


class _Capture:
    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False
        self.settings = {}

    def isOpened(self):
        return self.opened

    def set(self, key, value):
        self.settings[key] = value
        return True

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


def _frame(value=0):
    return np.full((48, 64, 3), value, dtype=np.uint8)


class WebcamValidationTests(unittest.TestCase):
    def test_camera_and_inference_settings_are_validated(self):
        validate_webcam_settings(0, 1280, 720, 480, 10)
        for values in ((-1, 1280, 720, 640, 10), (0, 0, 720, 640, 10),
                       (0, 1280, 720, 400, 10), (0, 1280, 720, 640, -1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_webcam_settings(*values)

    def test_cli_keeps_existing_modes_and_adds_webcam_options(self):
        parser = build_parser()
        image_args = parser.parse_args(["--image", "x.jpg"])
        webcam_args = parser.parse_args([
            "--webcam", "--camera-index", "1", "--runtime", "onnx",
            "--device", "cuda", "--imgsz", "320", "--track",
            "--webcam-capture-mode", "sequential", "--webcam-duration-seconds", "2",
        ])
        self.assertEqual(image_args.image, "x.jpg")
        self.assertEqual(webcam_args.camera_index, 1)
        self.assertEqual(webcam_args.imgsz, 320)
        self.assertTrue(webcam_args.track)
        self.assertEqual(webcam_args.webcam_capture_mode, "sequential")
        self.assertEqual(webcam_args.webcam_duration_seconds, 2.0)


class WebcamCaptureTests(unittest.TestCase):
    def test_camera_open_failure_is_clear_and_releases_capture(self):
        capture = _Capture([], opened=False)
        with self.assertRaisesRegex(WebcamInputError, "camera index 3"):
            run_webcam(_FakeDetector(), camera_index=3, display=False,
                       capture_factory=lambda _: capture)
        self.assertTrue(capture.released)

    def test_first_frame_failure_is_reported(self):
        capture = _Capture([])
        with self.assertRaisesRegex(WebcamInputError, "did not provide a frame"):
            run_webcam(_FakeDetector(), display=False, capture_factory=lambda _: capture)
        self.assertTrue(capture.released)

    def test_read_failure_after_processing_exits_cleanly(self):
        capture = _Capture([_frame(), _frame(10)])
        result = run_webcam(_FakeDetector(), warmup_runs=0, display=False,
                            capture_mode="sequential", capture_factory=lambda _: capture)
        self.assertEqual(result.processed_frames, 1)
        self.assertEqual(result.termination_reason, "camera_disconnected_or_frame_read_failed")
        self.assertTrue(capture.released)

    def test_q_key_stops_display_and_closes_windows(self):
        capture = _Capture([_frame(), _frame(10)])
        with patch("edgeai.webcam.cv2.namedWindow") as named, \
             patch("edgeai.webcam.cv2.imshow") as shown, \
             patch("edgeai.webcam.cv2.waitKey", return_value=ord("q")), \
             patch("edgeai.webcam.cv2.destroyAllWindows") as destroyed:
            result = run_webcam(_FakeDetector(), warmup_runs=0,
                                capture_mode="sequential", capture_factory=lambda _: capture)
        self.assertEqual(result.termination_reason, "user_quit")
        self.assertEqual(result.processed_frames, 1)
        named.assert_called_once()
        shown.assert_called_once()
        destroyed.assert_called_once()

    def test_frames_metrics_and_resolution_selection(self):
        capture = _Capture([_frame(), _frame(20), _frame(40), _frame(60)])
        detector = _FakeDetector()
        result = run_webcam(
            detector, capture_width=800, capture_height=600, inference_size=480,
            warmup_runs=2, display=False, max_frames=3,
            capture_mode="sequential", capture_factory=lambda _: capture,
        )
        self.assertEqual(result.processed_frames, 3)
        self.assertEqual((result.capture_width, result.capture_height), (64, 48))
        self.assertEqual((result.requested_capture_width, result.requested_capture_height), (800, 600))
        self.assertEqual(result.inference_size, 480)
        self.assertEqual(result.model_load_ms, 4.0)
        self.assertEqual(result.inference_mean_ms, 2.5)
        self.assertEqual(result.model_runtime_mean_ms, 2.0)
        self.assertGreater(result.end_to_end_fps, 0)
        self.assertEqual(result.captured_frames, 3)
        self.assertEqual(result.frames_replaced, 0)
        self.assertGreater(result.capture_fps, 0)
        self.assertAlmostEqual(result.inference_fps, 400.0)
        self.assertIsNotNone(result.frame_age_mean_ms)
        self.assertIsNone(result.approximate_frame_to_display_mean_ms)
        self.assertEqual(len(result.last_frame_detections), 1)
        self.assertEqual(len(detector.calls), 5)  # two warmups plus three displayed frames
        self.assertEqual(detector.calls[-1][1], 480)
        self.assertEqual(capture.settings, {3: 800, 4: 600})
        self.assertTrue(capture.released)

    def test_tracking_sessions_reset_and_are_recreated(self):
        made = []

        class _Tracker:
            def __init__(self, detector):
                self.detector = detector
                self.reset_calls = 0
                made.append(self)

            def reset(self):
                self.reset_calls += 1

            def predict_frame(self, frame, confidence, *, image_size, iou_threshold):
                result = SimpleNamespace(boxes=_Boxes(), names={0: "person"}, speed={"inference": 1.5})
                return result, 2.0, 0.1

        for _ in range(2):
            capture = _Capture([_frame(), _frame(10)])
            with patch("edgeai.webcam.ByteTrackSession", _Tracker):
                result = run_webcam(
                    _FakeDetector(), tracking=True, warmup_runs=1,
                    display=False, max_frames=1, capture_mode="sequential",
                    capture_factory=lambda _, cap=capture: cap,
                )
            self.assertTrue(result.tracking_enabled)
        self.assertEqual(len(made), 2)
        self.assertEqual([tracker.reset_calls for tracker in made], [2, 2])

    def test_inference_exception_releases_capture(self):
        capture = _Capture([_frame(), _frame(10)])
        with self.assertRaisesRegex(InferenceError, "Webcam processing failed"):
            run_webcam(
                _FakeDetector(broken=True), warmup_runs=0, display=False,
                capture_mode="sequential", capture_factory=lambda _: capture,
            )
        self.assertTrue(capture.released)

    def test_runtime_is_reported_without_cross_runtime_sync_assumptions(self):
        capture = _Capture([_frame(), _frame(10)])
        detector = _FakeDetector()
        detector.runtime_name = "onnxruntime-gpu"
        result = run_webcam(detector, warmup_runs=0, display=False, max_frames=1,
                            capture_mode="sequential", capture_factory=lambda _: capture)
        self.assertEqual(result.runtime, "onnxruntime-gpu")
        self.assertEqual(result.execution_provider, "CUDAExecutionProvider")


class LatestFrameCaptureTests(unittest.TestCase):
    def test_camera_open_failure_does_not_leave_capture_thread_or_reservation(self):
        camera = _Capture([], opened=False)
        worker = LatestFrameCapture(935, 320, 240, capture_factory=lambda _: camera)
        with self.assertRaisesRegex(WebcamInputError, "Could not open webcam"):
            worker.start()
        self.assertTrue(camera.released)
        self.assertFalse(worker._thread.is_alive())
        # A later worker can claim the same index after the open failure.
        retry_camera = _Capture([], opened=False)
        with self.assertRaises(WebcamInputError):
            LatestFrameCapture(935, 320, 240, capture_factory=lambda _: retry_camera).start()

    def test_worker_starts_replaces_stale_slot_and_shuts_down(self):
        class ContinuousCapture(_Capture):
            def __init__(self):
                super().__init__([])
                self.value = 0

            def read(self):
                import time
                time.sleep(0.002)
                if self.released:
                    return False, None
                self.value += 1
                return True, _frame(self.value % 255)

        camera = ContinuousCapture()
        worker = LatestFrameCapture(936, 320, 240, capture_factory=lambda _: camera)
        worker.start()
        try:
            import time
            time.sleep(0.04)
            packet = worker.get_latest(timeout=1)
            self.assertIsNotNone(packet)
            self.assertGreater(packet.sequence, 1)
            self.assertGreater(worker._captured, 1)
            self.assertLessEqual(worker._sequence - worker._captured, 0)
            self.assertGreater(worker._replaced, 0)
        finally:
            worker.stop()
        self.assertFalse(worker._thread.is_alive())
        self.assertTrue(camera.released)

    def test_concurrent_retrieval_returns_safe_frame_snapshots(self):
        class ContinuousCapture(_Capture):
            def __init__(self):
                super().__init__([])
                self.value = 0

            def read(self):
                import time
                time.sleep(0.003)
                if self.released:
                    return False, None
                self.value += 1
                return True, _frame(self.value % 255)

        camera = ContinuousCapture()
        worker = LatestFrameCapture(940, 320, 240, capture_factory=lambda _: camera).start()
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                snapshots = list(pool.map(lambda _: worker.get_latest(timeout=1), range(4)))
            self.assertTrue(all(packet is not None for packet in snapshots))
            self.assertTrue(all(packet.frame.shape == (48, 64, 3) for packet in snapshots))
            self.assertEqual(len({id(packet.frame) for packet in snapshots}), 4)
            self.assertIsInstance(worker._latest.frame, np.ndarray)  # exactly one slot, not a queue
        finally:
            worker.stop()
        self.assertFalse(worker._thread.is_alive())

    def test_only_one_thread_can_claim_the_same_camera(self):
        class ContinuousCapture(_Capture):
            def __init__(self):
                super().__init__([])

            def read(self):
                import time
                time.sleep(0.01)
                return (False, None) if self.released else (True, _frame())

        first_camera = ContinuousCapture()
        second_camera = ContinuousCapture()
        first = LatestFrameCapture(937, 320, 240, capture_factory=lambda _: first_camera).start()
        try:
            with self.assertRaisesRegex(WebcamInputError, "already active"):
                LatestFrameCapture(937, 320, 240, capture_factory=lambda _: second_camera).start()
        finally:
            first.stop()
        self.assertFalse(first._thread.is_alive())
        self.assertFalse(second_camera.released)

    def test_capture_exception_is_reported_and_resource_is_released(self):
        class BrokenCapture(_Capture):
            def read(self):
                raise RuntimeError("synthetic capture failure")

        camera = BrokenCapture([])
        worker = LatestFrameCapture(938, 320, 240, capture_factory=lambda _: camera)
        worker.start()
        try:
            with self.assertRaisesRegex(WebcamInputError, "capture thread failed"):
                worker.get_latest(timeout=1)
        finally:
            worker.stop()
        self.assertTrue(camera.released)
        self.assertFalse(worker._thread.is_alive())

    def test_camera_disconnect_keeps_latest_frame_then_returns_none(self):
        camera = _Capture([_frame()])
        worker = LatestFrameCapture(939, 320, 240, capture_factory=lambda _: camera).start()
        try:
            packet = worker.get_latest(timeout=1)
            self.assertIsNotNone(packet)
            later = worker.get_latest(after_sequence=packet.sequence, timeout=1)
            self.assertIsNone(later)
            self.assertTrue(worker.disconnected)
        finally:
            worker.stop()
        self.assertFalse(worker._thread.is_alive())
        self.assertTrue(camera.released)

    def test_same_detector_cannot_run_two_webcam_inference_sessions(self):
        detector = _FakeDetector()
        identity = _reserve_detector(detector)
        try:
            with self.assertRaisesRegex(InferenceError, "already has an active"):
                run_webcam(detector, display=False, capture_mode="sequential")
        finally:
            _release_detector(identity)

    def test_detector_serializes_overlapping_inference_calls(self):
        class ConcurrentModel:
            names = {0: "person"}

            def __init__(self):
                self.guard = threading.Lock()
                self.active = 0
                self.peak_active = 0

            def predict(self, **kwargs):
                with self.guard:
                    self.active += 1
                    self.peak_active = max(self.peak_active, self.active)
                time.sleep(0.03)
                with self.guard:
                    self.active -= 1
                return [SimpleNamespace(boxes=_Boxes(), names=self.names, speed={"inference": 1.0})]

        model = ConcurrentModel()
        detector = ImageDetector(model=model, device="cpu")
        frame = _frame()
        workers = [threading.Thread(target=detector.predict_frame, args=(frame,)) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=2)
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(model.peak_active, 1)


if __name__ == "__main__":
    unittest.main()
