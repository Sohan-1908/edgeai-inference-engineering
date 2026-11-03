import unittest
from types import SimpleNamespace
from unittest.mock import patch

from edgeai.detector import ImageDetector
from edgeai.postprocess import extract_detections
from edgeai.tracking import ByteTrackSession


class FakeTracker:
    def __init__(self):
        self.calls = 0

    def update(self, detections, image=None):
        self.calls += 1
        return [17]


class FakeBoxes:
    def __init__(self, track_ids):
        self.xyxy = [[1, 2, 20, 24] for _ in track_ids]
        self.conf = [0.91 for _ in track_ids]
        self.cls = [0 for _ in track_ids]
        self.id = track_ids or None

    def __len__(self):
        return len(self.xyxy)


class FakeTrackingModel:
    def __init__(self):
        self.callbacks = {"on_predict_start": []}
        self.predictor = SimpleNamespace(trackers=[], vid_path=[None])
        self.track_calls = []

    def add_callback(self, event, callback):
        self.callbacks.setdefault(event, []).append(callback)

    def track(self, **kwargs):
        self.track_calls.append(kwargs)
        if not self.predictor.trackers:
            self.predictor.trackers = [FakeTracker()]
        for callback in self.callbacks.get("on_predict_start", []):
            callback(self.predictor)
        self.predictor.trackers[0].update([], None)
        return [SimpleNamespace(boxes=FakeBoxes([17]), names={0: "person"}, speed={"inference": 1.0})]


class ByteTrackSessionTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeTrackingModel()
        self.detector = ImageDetector(model=self.model, device="cpu")
        self.register_patch = patch("ultralytics.trackers.register_tracker")
        self.register_patch.start()
        self.addCleanup(self.register_patch.stop)

    def test_tracking_ids_are_present_and_persist_across_frames(self):
        session = ByteTrackSession(self.detector)
        session.reset()
        first, _, first_tracker_ms = session.predict_frame(object(), 0.25)
        tracker = self.model.predictor.trackers[0]
        second, _, second_tracker_ms = session.predict_frame(object(), 0.25)

        self.assertEqual([item.track_id for item in extract_detections(first)], [17])
        self.assertEqual([item.track_id for item in extract_detections(second)], [17])
        self.assertIs(self.model.predictor.trackers[0], tracker)
        self.assertEqual(tracker.calls, 2)
        self.assertGreaterEqual(first_tracker_ms, 0)
        self.assertGreaterEqual(second_tracker_ms, 0)
        self.assertTrue(all(call["persist"] for call in self.model.track_calls))
        self.assertTrue(all(call["tracker"] == "bytetrack.yaml" for call in self.model.track_calls))
        self.assertTrue(all(call["device"] == "cpu" for call in self.model.track_calls))
        self.assertTrue(all(call["conf"] == 0.1 for call in self.model.track_calls))

    def test_reset_discards_state_before_a_new_video(self):
        session = ByteTrackSession(self.detector)
        session.reset()
        session.predict_frame(object(), 0.25)
        previous_tracker = self.model.predictor.trackers[0]
        session.reset()
        self.assertEqual(self.model.predictor.trackers, [])
        self.assertEqual(self.model.predictor.vid_path, [None])
        session.predict_frame(object(), 0.25)
        self.assertIsNot(self.model.predictor.trackers[0], previous_tracker)

    def test_empty_frame_has_no_detections_but_tracking_call_succeeds(self):
        result = SimpleNamespace(boxes=None, names={}, speed={"inference": 0.5})

        def no_detection_track(**kwargs):
            return [result]

        self.model.track = no_detection_track
        session = ByteTrackSession(self.detector)
        session.reset()
        tracked_result, _, _ = session.predict_frame(object(), 0.25)
        self.assertEqual(extract_detections(tracked_result), [])


if __name__ == "__main__":
    unittest.main()
