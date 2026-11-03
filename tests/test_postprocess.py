"""Tests for shared image and video annotation rendering."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from edgeai.detector_types import Detection
from edgeai.postprocess import annotate_image
from edgeai.video import VideoProcessor


def _overlaps(first, second):
    return (
        first[0] < second[2]
        and second[0] < first[2]
        and first[1] < second[3]
        and second[1] < first[3]
    )


class AnnotationTests(unittest.TestCase):
    def test_overlapping_objects_receive_non_overlapping_labels_when_space_allows(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(index % 4, f"class-{index}", 0.99 - index * 0.01,
                      (220.0, 170.0, 420.0, 310.0))
            for index in range(6)
        ]
        label_rectangles = []
        original_rectangle = cv2.rectangle

        def capture_rectangle(target, start, end, color, thickness=1, *args, **kwargs):
            if thickness == -1:
                label_rectangles.append((start[0], start[1], end[0] + 1, end[1] + 1))
            return original_rectangle(
                target, start, end, color, thickness, *args, **kwargs
            )

        with patch("edgeai.postprocess.cv2.rectangle", side_effect=capture_rectangle):
            annotated = annotate_image(image, detections)

        self.assertEqual(len(label_rectangles), len(detections))
        for index, first in enumerate(label_rectangles):
            self.assertTrue(
                all(not _overlaps(first, second) for second in label_rectangles[index + 1:])
            )
        self.assertFalse(np.array_equal(annotated, image))

    def test_boxes_and_labels_at_all_four_image_boundaries_stay_in_frame(self):
        width, height = 160, 120
        image = np.zeros((height, width, 3), dtype=np.uint8)
        detections = [
            Detection(0, "top-left", 0.9, (0.0, 0.0, 15.0, 12.0)),
            Detection(1, "top-right", 0.9, (145.0, 0.0, 159.0, 12.0)),
            Detection(2, "bottom-left", 0.9, (0.0, 108.0, 15.0, 119.0)),
            Detection(3, "bottom-right", 0.9, (145.0, 108.0, 159.0, 119.0)),
        ]
        rectangles = []
        original_rectangle = cv2.rectangle

        def capture_rectangle(target, start, end, color, thickness=1, *args, **kwargs):
            rectangles.append((start, end))
            return original_rectangle(
                target, start, end, color, thickness, *args, **kwargs
            )

        with patch("edgeai.postprocess.cv2.rectangle", side_effect=capture_rectangle):
            annotated = annotate_image(image, detections)

        self.assertEqual(annotated.shape, image.shape)
        self.assertTrue(rectangles)
        for start, end in rectangles:
            self.assertGreaterEqual(start[0], 0)
            self.assertGreaterEqual(start[1], 0)
            self.assertLess(end[0], width)
            self.assertLess(end[1], height)

    def test_very_small_boxes_are_drawn_and_tracking_labels_keep_supplied_ids(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        detections = [
            Detection(0, "person", 0.93, (0.0, 0.0, 2.0, 2.0), track_id=12),
            Detection(2, "car", 0.81, (318.0, 238.0, 319.0, 239.0), track_id=44),
        ]
        labels = []
        original_put_text = cv2.putText

        def capture_put_text(target, text, *args, **kwargs):
            labels.append(text)
            return original_put_text(target, text, *args, **kwargs)

        with patch("edgeai.postprocess.cv2.putText", side_effect=capture_put_text):
            annotated = annotate_image(image, detections)

        self.assertIn("person #12 | 0.93", labels)
        self.assertIn("car #44 | 0.81", labels)
        self.assertFalse(np.array_equal(annotated, image))

    def test_no_detections_return_an_unchanged_copy(self):
        image = np.full((32, 48, 3), 127, dtype=np.uint8)
        annotated = annotate_image(image, [])
        self.assertIsNot(annotated, image)
        np.testing.assert_array_equal(annotated, image)

    def test_annotation_cost_is_bounded_for_a_crowded_720p_frame(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = [
            Detection(
                index % 5,
                f"class-{index % 5}",
                0.95 - index * 0.001,
                (
                    float(360 + index % 10 * 28),
                    float(180 + index // 10 * 24),
                    float(470 + index % 10 * 28),
                    float(285 + index // 10 * 24),
                ),
                track_id=index,
            )
            for index in range(50)
        ]
        annotate_image(image, detections)
        started = time.perf_counter()
        for _ in range(3):
            annotate_image(image, detections)
        mean_ms = (time.perf_counter() - started) * 1000 / 3
        self.assertLess(mean_ms, 100.0)

    def test_video_frames_use_the_same_annotation_renderer(self):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)

        class FakeBoxes:
            xyxy = [[12.0, 14.0, 58.0, 61.0]]
            conf = [0.93]
            cls = [0]
            id = [12]

            def __len__(self):
                return 1

        class FakeDetector:
            device = "cpu"
            model_name = "test-model"
            model_load_ms = 0.0
            last_timings = {}

            def predict_frame(self, _frame, _confidence):
                return (
                    SimpleNamespace(
                        boxes=FakeBoxes(),
                        names={0: "person"},
                        speed={"inference": 1.0},
                    ),
                    1.0,
                )

        class FakeCapture:
            def __init__(self):
                self.frames = [frame.copy()]

            def isOpened(self):
                return True

            def get(self, prop):
                if prop == cv2.CAP_PROP_FPS:
                    return 10.0
                if prop == cv2.CAP_PROP_FRAME_COUNT:
                    return 1.0
                return 0.0

            def read(self):
                return (True, self.frames.pop()) if self.frames else (False, None)

            def release(self):
                pass

        class FakeWriter:
            def __init__(self):
                self.frames = []

            def isOpened(self):
                return True

            def write(self, value):
                self.frames.append(value.copy())

            def release(self):
                pass

        writer = FakeWriter()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "one-frame.avi"
            source.write_bytes(b"synthetic video")
            processor = VideoProcessor(
                FakeDetector(),
                capture_factory=lambda _path: FakeCapture(),
                writer_factory=lambda *_args: writer,
            )
            with patch("edgeai.video.annotate_image", wraps=annotate_image) as renderer:
                processor.process_video(source, output_dir=Path(directory) / "outputs", warmup_frames=0)

        self.assertEqual(renderer.call_count, 1)
        expected = annotate_image(
            frame,
            [Detection(0, "person", 0.93, (12.0, 14.0, 58.0, 61.0), track_id=12)],
        )
        np.testing.assert_array_equal(writer.frames[0], expected)


if __name__ == "__main__":
    unittest.main()
