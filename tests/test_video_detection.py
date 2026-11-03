import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from edgeai.detector import ImageDetector
from edgeai.errors import VideoInputError, VideoProcessingError
from edgeai.video import VideoProcessor


class EmptyVideoDetector:
    model_name = "test-yolo"
    device = "cpu"
    model_load_ms = 0.0

    def __init__(self):
        self.calls = 0

    def predict_frame(self, frame, confidence):
        self.calls += 1
        return SimpleNamespace(boxes=None, names={}), 2.0


class BrokenVideoDetector(EmptyVideoDetector):
    def predict_frame(self, frame, confidence):
        raise RuntimeError("synthetic inference failure")


class FakeUltralyticsModel:
    def predict(self, **kwargs):
        return [SimpleNamespace(boxes=None, names={})]


def make_video(path: Path, frame_count=8, width=96, height=64, fps=12.0):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV MJPG video writer is unavailable")
    for index in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (index, 8), (min(width - 1, index + 20), 28), (40, 180, 240), -1)
        writer.write(frame)
    writer.release()


class VideoInputTests(unittest.TestCase):
    def test_invalid_video_path_is_reported(self):
        processor = VideoProcessor(EmptyVideoDetector())
        with self.assertRaisesRegex(VideoInputError, "does not exist"):
            processor.process_video("missing-video.avi")

    def test_unsupported_or_unreadable_video_is_reported(self):
        processor = VideoProcessor(EmptyVideoDetector())
        with tempfile.TemporaryDirectory() as directory:
            unsupported = Path(directory) / "clip.mkv"
            unsupported.write_bytes(b"not supported")
            with self.assertRaisesRegex(VideoInputError, "Unsupported video format"):
                processor.process_video(unsupported)

            corrupt = Path(directory) / "corrupt.avi"
            corrupt.write_bytes(b"not a video")
            with self.assertRaises(VideoInputError):
                processor.process_video(corrupt, warmup_frames=0)


class VideoProcessingTests(unittest.TestCase):
    def test_extracts_and_writes_all_frames_with_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.avi"
            make_video(source, frame_count=8)
            detector = EmptyVideoDetector()
            result = VideoProcessor(detector).process_video(
                source,
                output_dir=Path(directory) / "outputs",
                warmup_frames=2,
            )

            self.assertEqual(result.performance.processed_frames, 8)
            self.assertEqual(result.performance.reported_input_frames, 8)
            self.assertEqual(result.performance.measured_frames, 6)
            self.assertEqual(result.performance.warmup_frames, 2)
            self.assertEqual(detector.calls, 8)
            self.assertEqual((result.frame_width, result.frame_height), (96, 64))
            self.assertEqual(result.performance.input_video_fps, 12.0)
            self.assertEqual(result.performance.output_video_fps, 12.0)
            self.assertEqual(result.performance.inference_mean_ms, 2.0)
            self.assertEqual(result.performance.model_only_fps, 500.0)
            self.assertGreaterEqual(result.performance.video_decoding_ms, 0.0)
            self.assertGreaterEqual(result.performance.annotation_ms, 0.0)
            self.assertGreaterEqual(result.performance.output_writing_ms, 0.0)
            self.assertGreater(result.performance.end_to_end_fps, 0.0)
            self.assertTrue(Path(result.annotated_video_path).is_file())

            output = cv2.VideoCapture(result.annotated_video_path)
            self.assertTrue(output.isOpened())
            self.assertEqual(int(round(output.get(cv2.CAP_PROP_FRAME_COUNT))), 8)
            self.assertEqual(int(round(output.get(cv2.CAP_PROP_FRAME_WIDTH))), 96)
            self.assertEqual(int(round(output.get(cv2.CAP_PROP_FRAME_HEIGHT))), 64)
            output.release()

    def test_auto_device_uses_cpu_fallback_for_video(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.avi"
            make_video(source, frame_count=3)
            with patch("edgeai.device.torch.cuda.is_available", return_value=False):
                detector = ImageDetector(model=FakeUltralyticsModel(), device="auto")
                result = VideoProcessor(detector).process_video(source, warmup_frames=1)
            self.assertEqual(result.device, "cpu")
            self.assertEqual(result.performance.processed_frames, 3)

    def test_capture_and_writer_are_released_after_inference_error(self):
        class TrackedResource:
            def __init__(self, inner):
                self.inner = inner
                self.was_released = False

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def release(self):
                self.was_released = True
                self.inner.release()

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.avi"
            make_video(source, frame_count=4)
            captures = []
            writers = []

            def capture_factory(path):
                resource = TrackedResource(cv2.VideoCapture(path))
                captures.append(resource)
                return resource

            def writer_factory(*args):
                resource = TrackedResource(cv2.VideoWriter(*args))
                writers.append(resource)
                return resource

            processor = VideoProcessor(
                BrokenVideoDetector(),
                capture_factory=capture_factory,
                writer_factory=writer_factory,
            )
            with self.assertRaisesRegex(VideoProcessingError, "frame 0"):
                processor.process_video(source, warmup_frames=0)
            self.assertTrue(captures[0].was_released)
            self.assertTrue(writers[0].was_released)

    def test_video_too_short_for_requested_warmup_keeps_measured_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "one-frame.avi"
            make_video(source, frame_count=1)
            result = VideoProcessor(EmptyVideoDetector()).process_video(source, warmup_frames=3)
            self.assertEqual(result.performance.warmup_frames, 0)
            self.assertEqual(result.performance.measured_frames, 1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class VideoCudaInferenceTests(unittest.TestCase):
    def test_pretrained_model_runs_on_video_frames_with_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cuda-video.avi"
            make_video(source, frame_count=4)
            detector = ImageDetector(device="cuda")
            result = VideoProcessor(detector).process_video(
                source,
                output_dir=Path(directory) / "outputs",
                warmup_frames=1,
            )
            self.assertEqual(result.device, "cuda:0")
            self.assertEqual(result.performance.processed_frames, 4)
            self.assertEqual(result.performance.measured_frames, 3)
            self.assertIsNotNone(result.performance.gpu_memory_peak_allocated_mb)
            self.assertTrue(Path(result.annotated_video_path).is_file())

    def test_pretrained_model_runs_bytetrack_on_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cuda-tracking-video.avi"
            make_video(source, frame_count=4)
            detector = ImageDetector(device="cuda")
            result = VideoProcessor(detector).process_video(
                source,
                output_dir=Path(directory) / "outputs",
                warmup_frames=1,
                track=True,
            )
            self.assertEqual(result.device, "cuda:0")
            self.assertTrue(result.tracking_enabled)
            self.assertEqual(result.performance.processed_frames, 4)
            self.assertEqual(result.performance.tracker_name, "bytetrack.yaml")
            self.assertIsNotNone(result.performance.tracking_processing_mean_ms)
            self.assertIsNotNone(result.performance.gpu_memory_peak_allocated_mb)
            self.assertTrue(Path(result.annotated_video_path).is_file())


if __name__ == "__main__":
    unittest.main()
