import base64
import json
import shutil
import threading
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from pydantic import ValidationError
from fastapi.testclient import TestClient

from edgeai.api import (
    ApiSettings,
    VideoProcessingMetadata,
    _video_metadata,
    build_detector,
    create_app,
)
from edgeai.api_video import probe_video, stream_file_and_cleanup
from edgeai.errors import DeviceUnavailableError, VideoProcessingError
from edgeai.video import VideoProcessor


class _Boxes:
    def __init__(self):
        self.xyxy = np.asarray([[5.0, 6.0, 31.0, 39.0]], dtype=np.float32)
        self.conf = np.asarray([0.875], dtype=np.float32)
        self.cls = np.asarray([0.0], dtype=np.float32)
        self.id = None

    def __len__(self):
        return 1


class _FakeDetector:
    runtime_name = "pytorch"
    runtime_version = "test"
    device = "cuda:0"
    execution_provider = "CUDAExecutionProvider"
    model_name = "yolo11n.pt"
    input_size = 640
    gpu_memory_supported = False
    model_load_ms = 1.0

    def __init__(self, *, broken=False):
        self.inference_lock = RLock()
        self.last_timings = {}
        self.broken = broken
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.guard = threading.Lock()
        self.inference_started = threading.Event()

    def predict_frame(self, frame, confidence, *, image_size, iou_threshold):
        if self.broken:
            raise RuntimeError("private file path should not be returned")
        self.inference_started.set()
        with self.guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self.guard:
            self.active -= 1
            self.calls += 1
        self.last_timings = {
            "preprocess_ms": 1.0,
            "runtime_ms": 12.0,
            "postprocess_ms": 0.5,
        }
        result = SimpleNamespace(
            boxes=_Boxes(), names={0: "person"}, speed={"inference": 12.0}
        )
        return result, 13.8


def _jpeg(width=80, height=60):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(image, (5, 6), (31, 39), (0, 255, 0), -1)
    ok, data = cv2.imencode(".jpg", image)
    assert ok
    return data.tobytes()


def _synthetic_video(directory: Path, *, frames=6, fps=10):
    path = directory / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (96, 64))
    if not writer.isOpened():
        raise RuntimeError("MJPG writer unavailable for synthetic API test")
    for index in range(frames):
        frame = np.full((64, 96, 3), index * 15, dtype=np.uint8)
        cv2.rectangle(frame, (8 + index, 10), (40 + index, 48), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return path


class _FakeByteTrackSession:
    resets = 0
    predictions = 0

    def __init__(self, detector):
        self.detector = detector

    def reset(self):
        type(self).resets += 1

    def predict_frame(self, frame, confidence, *, image_size=None, iou_threshold=None):
        result, inference_ms = self.detector.predict_frame(
            frame, confidence, image_size=image_size, iou_threshold=iou_threshold
        )
        result.boxes.id = np.asarray([73], dtype=np.float32)
        type(self).predictions += 1
        return result, inference_ms, 0.2


class ApiSettingsTests(unittest.TestCase):
    def test_cuda_only_and_valid_runtime_resolution_thresholds(self):
        settings = ApiSettings(runtime="onnx", input_size=480, confidence=0.4, iou_threshold=0.6)
        self.assertEqual(settings.runtime, "onnx")
        for kwargs in (
            {"runtime": "fp16"},
            {"device": "cpu"},
            {"input_size": 400},
            {"confidence": 1.1},
            {"iou_threshold": 0.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ApiSettings(**kwargs)

    def test_cuda_initialization_failure_is_explicit_and_never_builds_cpu_model(self):
        with patch("edgeai.api.torch.cuda.is_available", return_value=False), \
             patch("edgeai.api.ImageDetector") as detector_type:
            with self.assertRaisesRegex(DeviceUnavailableError, "requires CUDA"):
                build_detector(ApiSettings(runtime="pytorch"))
        detector_type.assert_not_called()

    def test_onnx_initialization_failure_is_explicit_and_never_builds_cpu_model(self):
        with patch("edgeai.api.torch.cuda.is_available", return_value=False), \
             patch("edgeai.api.OnnxDetector") as detector_type:
            with self.assertRaisesRegex(DeviceUnavailableError, "requires CUDA"):
                build_detector(ApiSettings(runtime="onnx"))
        detector_type.assert_not_called()

    def test_onnx_detector_is_constructed_with_strict_cuda_fp32(self):
        detector = SimpleNamespace(
            device="cuda:0",
            execution_provider="CUDAExecutionProvider",
            input_size=640,
        )
        with patch("edgeai.api.torch.cuda.is_available", return_value=True), \
             patch("edgeai.api.OnnxDetector", return_value=detector) as detector_type:
            built = build_detector(ApiSettings(runtime="onnx"))
        self.assertIs(built, detector)
        kwargs = detector_type.call_args.kwargs
        self.assertEqual(kwargs["device"], "cuda")
        self.assertEqual(kwargs["precision"], "fp32")
        self.assertTrue(kwargs["require_cuda"])

    def test_onnx_resolution_selects_matching_static_export(self):
        from edgeai.api import _onnx_path_for_size

        path = _onnx_path_for_size(ApiSettings(runtime="onnx", input_size=480))
        self.assertEqual(path.name, "yolo11n_480.onnx")


class ApiVideoHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_cancellation_removes_request_temporary_directory(self):
        with tempfile.TemporaryDirectory() as parent:
            directory = Path(parent) / "request"
            directory.mkdir()
            output = directory / "annotated.avi"
            output.write_bytes(b"video bytes")
            stream = stream_file_and_cleanup(output, directory, chunk_size=4)
            first_chunk = await anext(stream)
            self.assertEqual(first_chunk, b"vide")
            await stream.aclose()
            self.assertFalse(directory.exists())


class ApiVideoProbeTests(unittest.TestCase):
    def test_video_probe_uses_frame_dimensions_and_tolerates_missing_fps_metadata(self):
        class CaptureWithoutMetadata:
            def __init__(self, path):
                self.released = False

            def isOpened(self):
                return True

            def read(self):
                return True, np.zeros((48, 80, 3), dtype=np.uint8)

            def get(self, prop):
                return 0.0

            def release(self):
                self.released = True

        with patch("edgeai.api_video.cv2.VideoCapture", CaptureWithoutMetadata):
            metadata = probe_video(Path("synthetic.avi"), max_frames=100, max_duration_seconds=5)
        self.assertEqual(metadata["width"], 80)
        self.assertEqual(metadata["height"], 48)
        self.assertIsNone(metadata["source_fps"])
        self.assertIsNone(metadata["source_frame_count"])


class ApiConfigProviderTests(unittest.TestCase):
    def _get_config(self, runtime):
        detector = _FakeDetector()
        app = create_app(ApiSettings(runtime=runtime), detector_factory=lambda _: detector)
        app.state.detector = detector
        endpoint = next(
            route.endpoint
            for route in app.routes
            if getattr(route, "path", None) == "/api/v1/config"
        )
        response = endpoint()
        try:
            response.send(None)
        except StopIteration as completed:
            return completed.value
        self.fail("The config endpoint unexpectedly suspended.")

    def test_config_reports_actual_onnx_execution_provider(self):
        payload = self._get_config("onnx")
        self.assertEqual(payload.runtime, "onnx")
        self.assertEqual(payload.execution_provider, "CUDAExecutionProvider")

    def test_config_does_not_report_onnx_provider_for_pytorch(self):
        payload = self._get_config("pytorch")
        self.assertEqual(payload.runtime, "pytorch")
        self.assertIsNone(payload.execution_provider)


class ApiEndpointTests(unittest.TestCase):
    def setUp(self):
        self.detector = _FakeDetector()
        self.factory_calls = []

        def factory(settings):
            self.factory_calls.append(settings)
            return self.detector

        self.app = create_app(ApiSettings(), detector_factory=factory)
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)

    def test_health_reports_initialized_cuda_detector(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "ready",
            "detector_initialized": True,
            "active_runtime": "pytorch",
            "active_device": "cuda:0",
            "model_input_resolution": 640,
        })

    def test_config_reports_static_settings_and_supported_configurations(self):
        response = self.client.get("/api/v1/config")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["runtime"], "pytorch")
        self.assertEqual(payload["device"], "cuda")
        self.assertIsNone(payload["execution_provider"])
        self.assertEqual(payload["input_resolution"], 640)
        self.assertEqual(payload["confidence_threshold"], 0.25)
        self.assertEqual(payload["nms_iou_threshold"], 0.7)
        self.assertEqual(len(payload["supported_configurations"]), 2)

    def test_valid_image_returns_structured_detections_and_timings(self):
        response = self.client.post("/api/v1/detect", files={"file": ("person.jpg", _jpeg(), "image/jpeg")})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual((payload["image_width"], payload["image_height"]), (80, 60))
        self.assertEqual(payload["model_input_resolution"], 640)
        self.assertEqual(payload["number_of_detections"], 1)
        self.assertEqual(payload["detections"][0]["class_name"], "person")
        self.assertEqual(payload["detections"][0]["bbox_xyxy"], [5.0, 6.0, 31.0, 39.0])
        self.assertEqual(payload["bbox_coordinate_system"], "absolute_xyxy_pixels_in_original_image_origin_top_left")
        self.assertEqual(payload["timings"]["inference_ms"], 13.8)
        self.assertEqual(payload["timings"]["preprocessing_ms"], 1.0)
        self.assertEqual(payload["timings"]["model_execution_ms"], 12.0)
        self.assertGreater(payload["timings"]["total_request_processing_ms"], 0)
        self.assertIsNone(payload["annotated_image_base64"])

    def test_annotated_image_is_returned_as_base64_jpeg(self):
        response = self.client.post(
            "/api/v1/detect",
            files={"file": ("person.jpg", _jpeg(), "image/jpeg")},
            data={"include_annotated_image": "true"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        encoded = response.json()["annotated_image_base64"]
        decoded = base64.b64decode(encoded, validate=True)
        image = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[:2], (60, 80))

    def test_unsupported_extension_and_format_are_rejected(self):
        extension = self.client.post("/api/v1/detect", files={"file": ("image.gif", b"GIF89a", "image/gif")})
        content = self.client.post(
            "/api/v1/detect", files={"file": ("image.jpg", _jpeg().replace(b"\xff\xd8\xff", b"BAD"), "image/jpeg")}
        )
        self.assertEqual(extension.status_code, 415)
        self.assertEqual(content.status_code, 415)

    def test_empty_and_corrupt_uploads_are_rejected(self):
        empty = self.client.post("/api/v1/detect", files={"file": ("empty.jpg", b"", "image/jpeg")})
        corrupt = self.client.post("/api/v1/detect", files={"file": ("bad.jpg", b"\xff\xd8\xffbroken", "image/jpeg")})
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(corrupt.status_code, 400)

    def test_upload_size_limit_and_missing_file_errors(self):
        app = create_app(
            ApiSettings(max_upload_bytes=100), detector_factory=lambda _: _FakeDetector()
        )
        with TestClient(app) as client:
            too_large = client.post("/api/v1/detect", files={"file": ("large.jpg", _jpeg(), "image/jpeg")})
            missing = client.post("/api/v1/detect", data={"include_annotated_image": "false"})
        self.assertEqual(too_large.status_code, 413)
        self.assertEqual(missing.status_code, 422)

    def test_multiple_requests_reuse_one_detector_initialization(self):
        for _ in range(2):
            response = self.client.post("/api/v1/detect", files={"file": ("x.jpg", _jpeg(), "image/jpeg")})
            self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.factory_calls), 1)
        self.assertEqual(self.detector.calls, 2)

    def test_concurrent_requests_are_serialized_on_shared_detector(self):
        def request_once(_):
            return self.client.post("/api/v1/detect", files={"file": ("x.jpg", _jpeg(), "image/jpeg")})

        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(request_once, range(4)))
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(self.detector.max_active, 1)

    def test_inference_errors_are_generic_and_do_not_expose_internals(self):
        app = create_app(ApiSettings(), detector_factory=lambda _: _FakeDetector(broken=True))
        with TestClient(app) as client:
            response = client.post("/api/v1/detect", files={"file": ("x.jpg", _jpeg(), "image/jpeg")})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Image inference failed.")
        self.assertNotIn("private file path", response.text)

    def test_valid_video_upload_streams_output_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _synthetic_video(Path(directory))
            response = self.client.post(
                "/api/v1/detect/video",
                files={"file": ("client-name.avi", source.read_bytes(), "video/x-msvideo")},
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers["content-type"], "video/x-msvideo")
            self.assertEqual(response.headers["content-disposition"], 'attachment; filename="edgeai_annotated.avi"')
            metadata = json.loads(response.headers["x-edgeai-metadata"])
            self.assertEqual(metadata["source_frame_count"], 6)
            self.assertEqual(metadata["processed_frame_count"], 6)
            self.assertEqual(metadata["source_fps"], 10.0)
            self.assertEqual((metadata["video_width"], metadata["video_height"]), (96, 64))
            self.assertEqual(metadata["model_input_resolution"], 640)
            self.assertGreaterEqual(metadata["mean_inference_latency_ms"], 0)
            self.assertGreaterEqual(metadata["median_inference_latency_ms"], 0)
            self.assertGreaterEqual(metadata["p95_inference_latency_ms"], 0)
            self.assertGreater(metadata["end_to_end_processing_fps"], 0)
            self.assertGreater(metadata["total_processing_time_ms"], 0)
            self.assertEqual(metadata["runtime"], "pytorch")
            self.assertEqual(metadata["device"], "cuda:0")
            output_path = Path(directory) / "response.avi"
            output_path.write_bytes(response.content)
            capture = cv2.VideoCapture(str(output_path))
            self.assertTrue(capture.isOpened())
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            self.assertEqual(count, 6)

    def test_video_empty_corrupt_extension_and_mime_are_rejected(self):
        empty = self.client.post(
            "/api/v1/detect/video", files={"file": ("empty.avi", b"", "video/x-msvideo")}
        )
        corrupt = self.client.post(
            "/api/v1/detect/video", files={"file": ("bad.avi", b"not a video", "video/x-msvideo")}
        )
        extension = self.client.post(
            "/api/v1/detect/video", files={"file": ("movie.mkv", b"x", "video/x-matroska")}
        )
        mime = self.client.post(
            "/api/v1/detect/video", files={"file": ("movie.avi", b"x", "video/mp4")}
        )
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(corrupt.status_code, 400)
        self.assertEqual(extension.status_code, 415)
        self.assertEqual(mime.status_code, 415)

    def test_video_byte_duration_and_frame_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _synthetic_video(Path(directory))
            payload = source.read_bytes()
            size_app = create_app(
                ApiSettings(max_video_bytes=100), detector_factory=lambda _: _FakeDetector()
            )
            duration_app = create_app(
                ApiSettings(max_video_seconds=0.2), detector_factory=lambda _: _FakeDetector()
            )
            frame_app = create_app(
                ApiSettings(max_video_frames=4), detector_factory=lambda _: _FakeDetector()
            )
            with TestClient(size_app) as client:
                too_large = client.post(
                    "/api/v1/detect/video", files={"file": ("clip.avi", payload, "video/x-msvideo")}
                )
            with TestClient(duration_app) as client:
                too_long = client.post(
                    "/api/v1/detect/video", files={"file": ("clip.avi", payload, "video/x-msvideo")}
                )
            with TestClient(frame_app) as client:
                too_many = client.post(
                    "/api/v1/detect/video", files={"file": ("clip.avi", payload, "video/x-msvideo")}
                )
        self.assertEqual(too_large.status_code, 413)
        self.assertEqual(too_long.status_code, 413)
        self.assertEqual(too_many.status_code, 413)

    def test_video_early_decode_stop_is_rejected_when_container_declares_more_frames(self):
        partial_run = SimpleNamespace(
            performance=SimpleNamespace(reported_input_frames=6, processed_frames=3)
        )
        with tempfile.TemporaryDirectory() as directory:
            source = _synthetic_video(Path(directory))
            with patch("edgeai.api._run_video_inference", return_value=partial_run):
                response = self.client.post(
                    "/api/v1/detect/video",
                    files={"file": ("clip.avi", source.read_bytes(), "video/x-msvideo")},
                )
        self.assertEqual(response.status_code, 400)
        self.assertIn("could be decoded", response.json()["detail"])

    def test_tracking_resets_for_each_video_and_returns_track_statistics(self):
        _FakeByteTrackSession.resets = 0
        _FakeByteTrackSession.predictions = 0
        with tempfile.TemporaryDirectory() as directory:
            source = _synthetic_video(Path(directory))
            payload = source.read_bytes()
            with patch("edgeai.video.ByteTrackSession", _FakeByteTrackSession):
                for _ in range(2):
                    response = self.client.post(
                        "/api/v1/detect/video",
                        files={"file": ("clip.avi", payload, "video/x-msvideo")},
                        data={"track": "true"},
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    metadata = json.loads(response.headers["x-edgeai-metadata"])
                    self.assertTrue(metadata["tracking_enabled"])
                    self.assertEqual(metadata["unique_tracking_ids"], 1)
        self.assertEqual(_FakeByteTrackSession.resets, 2)
        self.assertGreater(_FakeByteTrackSession.predictions, 2)

    def test_onnx_tracking_is_rejected_instead_of_silently_disabled(self):
        app = create_app(ApiSettings(runtime="onnx"), detector_factory=lambda _: _FakeDetector())
        with tempfile.TemporaryDirectory() as directory:
            source = _synthetic_video(Path(directory))
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/detect/video",
                    files={"file": ("clip.avi", source.read_bytes(), "video/x-msvideo")},
                    data={"track": "true"},
                )
        self.assertEqual(response.status_code, 422)

    def test_video_requests_serialize_detector_and_health_stays_responsive(self):
        self.detector.inference_started.clear()
        with tempfile.TemporaryDirectory() as directory:
            source = _synthetic_video(Path(directory), frames=12)
            payload = source.read_bytes()

            def request_video():
                return self.client.post(
                    "/api/v1/detect/video", files={"file": ("clip.avi", payload, "video/x-msvideo")}
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(request_video)
                self.assertTrue(self.detector.inference_started.wait(5))
                health_started = time.perf_counter()
                health = self.client.get("/health")
                health_elapsed = time.perf_counter() - health_started
                second = pool.submit(request_video)
                responses = [first.result(timeout=15), second.result(timeout=15)]
        self.assertEqual(health.status_code, 200)
        self.assertLess(health_elapsed, 1.0)
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(self.detector.max_active, 1)

    def test_video_temporary_files_are_removed_on_success_and_processing_error(self):
        with tempfile.TemporaryDirectory() as fixtures, tempfile.TemporaryDirectory() as temp_root:
            source = _synthetic_video(Path(fixtures))
            def fake_mkdtemp(prefix):
                directory = Path(temp_root) / f"{prefix}unique"
                directory.mkdir()
                return str(directory)

            with patch("edgeai.api.tempfile.mkdtemp", side_effect=fake_mkdtemp):
                success = self.client.post(
                    "/api/v1/detect/video",
                    files={"file": ("clip.avi", source.read_bytes(), "video/x-msvideo")},
                )
            self.assertEqual(success.status_code, 200)
            self.assertEqual(list(Path(temp_root).iterdir()), [])

            class BrokenProcessor:
                def __init__(self, detector):
                    pass

                def process_video(self, *args, **kwargs):
                    raise VideoProcessingError("sensitive internal path")

            with patch("edgeai.api.tempfile.mkdtemp", side_effect=fake_mkdtemp), \
                 patch("edgeai.api.VideoProcessor", BrokenProcessor):
                failure = self.client.post(
                    "/api/v1/detect/video",
                    files={"file": ("clip.avi", source.read_bytes(), "video/x-msvideo")},
                )
            self.assertEqual(failure.status_code, 500)
            self.assertNotIn("sensitive internal path", failure.text)
            self.assertEqual(list(Path(temp_root).iterdir()), [])

class ApiVideoProcessingTests(unittest.TestCase):
    def assert_metadata_matches_performance(self, metadata, run, runtime):
        metrics = run.performance
        metric_fields = (
            "warmup_frames",
            "warmup_inference_ms",
            "mean_inference_latency_ms",
            "median_inference_latency_ms",
            "model_inference_mean_ms",
            "model_inference_median_ms",
            "tracking_processing_mean_ms",
            "tracking_processing_median_ms",
            "p95_inference_latency_ms",
            "end_to_end_processing_fps",
            "whole_video_processing_fps",
            "total_processing_time_ms",
            "annotation_ms",
            "output_writing_ms",
            "video_decoding_ms",
            "measured_pipeline_ms",
            "tracking_inference_confidence",
        )
        for field in metric_fields:
            with self.subTest(field=field):
                expected_field = {
                    "mean_inference_latency_ms": "inference_mean_ms",
                    "median_inference_latency_ms": "inference_median_ms",
                    "p95_inference_latency_ms": None,
                    "end_to_end_processing_fps": "end_to_end_fps",
                    "total_processing_time_ms": "total_processing_ms",
                }.get(field, field)
                expected = (
                    float(np.percentile(metrics.inference_samples_ms, 95))
                    if expected_field is None
                    else getattr(metrics, expected_field)
                )
                self.assertEqual(getattr(metadata, field), expected)

        self.assertEqual(metadata.source_frame_count, metrics.processed_frames)
        self.assertEqual(metadata.processed_frame_count, metrics.processed_frames)
        self.assertEqual(metadata.source_fps, metrics.input_video_fps)
        self.assertEqual(metadata.output_fps, metrics.output_video_fps)
        self.assertEqual((metadata.video_width, metadata.video_height), (run.frame_width, run.frame_height))
        self.assertEqual(metadata.runtime, runtime)
        self.assertEqual(metadata.device, run.device)
        self.assertEqual(metadata.tracking_enabled, run.tracking_enabled)
        self.assertEqual(metadata.total_detections, run.total_detections)
        self.assertEqual(metadata.unique_tracking_ids, run.unique_tracking_ids)

    def test_pytorch_video_metadata_remains_compatible(self):
        detector = _FakeDetector()
        settings = ApiSettings(runtime="pytorch")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _synthetic_video(root)
            run = VideoProcessor(detector).process_video(
                source,
                output_dir=root,
                warmup_frames=3,
                image_size=settings.input_size,
                iou_threshold=settings.iou_threshold,
                max_frames=settings.max_video_frames,
                max_duration_seconds=settings.max_video_seconds,
            )
            metadata = _video_metadata(run, settings)
            self.assertEqual(metadata.runtime, "pytorch")
            self.assertFalse(metadata.tracking_enabled)
            self.assert_metadata_matches_performance(metadata, run, settings.runtime)
            self.assertIsNone(metadata.tracking_processing_mean_ms)
            self.assertIsNone(metadata.tracking_processing_median_ms)
            self.assertIsNone(metadata.tracking_inference_confidence)

    def test_onnx_video_returns_validated_metadata(self):
        detector = _FakeDetector()
        detector.runtime_name = "onnxruntime-gpu"
        detector.runtime_version = "test"
        detector.model_name = "yolo11n.onnx"
        detector.execution_provider = "CUDAExecutionProvider"
        settings = ApiSettings(runtime="onnx")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _synthetic_video(root)
            run = VideoProcessor(detector).process_video(
                source,
                output_dir=root,
                warmup_frames=3,
                image_size=settings.input_size,
                iou_threshold=settings.iou_threshold,
                max_frames=settings.max_video_frames,
                max_duration_seconds=settings.max_video_seconds,
            )
            self.assertTrue(Path(run.annotated_video_path).is_file())
            metadata = _video_metadata(run, settings)
            validated = VideoProcessingMetadata.model_validate(metadata.model_dump())
            self.assertEqual(validated.runtime, "onnx")
            self.assertEqual(validated.device, "cuda:0")
            self.assertEqual(validated.processed_frame_count, 6)
            self.assertEqual(validated.total_detections, 6)
            self.assert_metadata_matches_performance(validated, run, settings.runtime)
            self.assertIsNone(validated.tracking_processing_mean_ms)
            self.assertIsNone(validated.tracking_processing_median_ms)
            self.assertIsNone(validated.tracking_inference_confidence)
            invalid_payload = metadata.model_dump()
            invalid_payload["runtime"] = run.performance.runtime_name
            with self.assertRaises(ValidationError) as validation_error:
                VideoProcessingMetadata.model_validate(invalid_payload)
            self.assertEqual(validation_error.exception.errors()[0]["loc"], ("runtime",))

    def test_tracking_video_metadata_includes_tracker_metrics_and_confidence(self):
        detector = _FakeDetector()
        settings = ApiSettings(runtime="pytorch")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _synthetic_video(root)
            with patch("edgeai.video.ByteTrackSession", _FakeByteTrackSession):
                run = VideoProcessor(detector).process_video(
                    source,
                    output_dir=root,
                    warmup_frames=3,
                    track=True,
                    image_size=settings.input_size,
                    iou_threshold=settings.iou_threshold,
                    max_frames=settings.max_video_frames,
                    max_duration_seconds=settings.max_video_seconds,
                )
            metadata = _video_metadata(run, settings)
            self.assertTrue(metadata.tracking_enabled)
            self.assert_metadata_matches_performance(metadata, run, settings.runtime)
            self.assertIsNotNone(metadata.tracking_processing_mean_ms)
            self.assertIsNotNone(metadata.tracking_processing_median_ms)
            self.assertIsNotNone(metadata.tracking_inference_confidence)

    def test_openapi_video_metadata_schema_contains_observability_fields(self):
        operation = create_app(
            ApiSettings(), detector_factory=lambda _: _FakeDetector()
        ).openapi()["paths"]["/api/v1/detect/video"]["post"]
        schema = operation["responses"]["200"]["headers"]["X-EdgeAI-Metadata"]["content"][
            "application/json"
        ]["schema"]
        expected_fields = {
            "warmup_frames",
            "warmup_inference_ms",
            "model_inference_mean_ms",
            "model_inference_median_ms",
            "tracking_processing_mean_ms",
            "tracking_processing_median_ms",
            "annotation_ms",
            "output_writing_ms",
            "video_decoding_ms",
            "measured_pipeline_ms",
            "whole_video_processing_fps",
            "tracking_inference_confidence",
        }
        self.assertTrue(expected_fields.issubset(schema["properties"]))
        for field in (
            "tracking_processing_mean_ms",
            "tracking_processing_median_ms",
            "tracking_inference_confidence",
        ):
            self.assertIn(
                {"type": "null"},
                schema["properties"][field]["anyOf"],
            )

if __name__ == "__main__":
    unittest.main()
