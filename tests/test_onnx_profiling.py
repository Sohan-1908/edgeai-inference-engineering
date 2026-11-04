import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from edgeai.onnx_detector import OnnxDetector


class _FakeSession:
    def __init__(self):
        self.end_profiling_calls = 0

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return [SimpleNamespace(name="images", shape=[1, 3, 640, 640])]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map={})

    def end_profiling(self):
        self.end_profiling_calls += 1
        return "trace.json"


class OnnxDetectorProfilingTests(unittest.TestCase):
    def _create_detector(self, **kwargs):
        session = _FakeSession()
        options = []

        def create_session(model_path, *, sess_options, providers):
            options.append(sess_options)
            return session

        with patch("edgeai.onnx_detector.ort.InferenceSession", side_effect=create_session):
            detector = OnnxDetector(
                Path("models/yolo11n.onnx"),
                device="cpu",
                **kwargs,
            )
        return detector, session, options[0]

    def test_profiling_is_disabled_by_default(self):
        detector, session, options = self._create_detector()

        self.assertFalse(detector.profile)
        self.assertFalse(options.enable_profiling)
        self.assertIsNone(detector.end_profiling())
        self.assertEqual(session.end_profiling_calls, 0)

    def test_profiling_is_enabled_when_requested(self):
        detector, _, options = self._create_detector(profile=True)

        self.assertTrue(detector.profile)
        self.assertTrue(options.enable_profiling)

    def test_profile_file_prefix_is_configured_when_provided(self):
        prefix = Path("outputs/ort_profiles/test_trace")
        _, _, options = self._create_detector(
            profile=True,
            profile_file_prefix=prefix,
        )

        self.assertEqual(options.profile_file_prefix, str(prefix))

    def test_end_profiling_returns_trace_path(self):
        detector, session, _ = self._create_detector(profile=True)

        self.assertEqual(detector.end_profiling(), "trace.json")
        self.assertEqual(session.end_profiling_calls, 1)

    def test_repeated_end_profiling_does_not_end_session_twice(self):
        detector, session, _ = self._create_detector(profile=True)

        self.assertEqual(detector.end_profiling(), "trace.json")
        self.assertIsNone(detector.end_profiling())
        self.assertEqual(session.end_profiling_calls, 1)


if __name__ == "__main__":
    unittest.main()
