from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from benchmark_runtime_comparison import (
    BenchmarkError,
    CSV_FIELDS,
    RuntimeBenchmarkClient,
    aggregate_runs,
    make_report,
    parse_video_metadata,
    regenerate_report_from_csv,
    run_benchmark,
    write_results,
)


def _metadata(**overrides):
    payload = {
        "runtime": "onnx",
        "device": "cuda:0",
        "model_input_resolution": 640,
        "source_frame_count": 62,
        "processed_frame_count": 62,
        "source_fps": 30.0,
        "output_fps": 30.0,
        "mean_inference_latency_ms": 20.0,
        "median_inference_latency_ms": 19.0,
        "p95_inference_latency_ms": 28.0,
        "end_to_end_processing_fps": 35.0,
        "total_processing_time_ms": 1800.0,
        "total_detections": 124,
        "tracking_enabled": False,
    }
    payload.update(overrides)
    return payload


class FakeResponse:
    def __init__(self, *, status_code=200, headers=None, chunks=(), body=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.chunks = chunks
        self.body = body
        self.closed = False

    def iter_content(self, chunk_size):
        yield from self.chunks

    def json(self):
        if self.body is None:
            raise ValueError("not JSON")
        return self.body

    def close(self):
        self.closed = True


class RuntimeBenchmarkMetadataTests(unittest.TestCase):
    def test_parses_valid_video_metadata(self):
        metadata = parse_video_metadata(json.dumps(_metadata()))
        self.assertEqual(metadata["runtime"], "onnx")
        self.assertEqual(metadata["processed_frame_count"], 62)

    def test_rejects_missing_or_malformed_metrics(self):
        with self.assertRaisesRegex(BenchmarkError, "missing"):
            parse_video_metadata(None)
        with self.assertRaisesRegex(BenchmarkError, "valid JSON"):
            parse_video_metadata("{broken")
        with self.assertRaisesRegex(BenchmarkError, "mean_inference_latency_ms"):
            parse_video_metadata(json.dumps(_metadata(mean_inference_latency_ms="20")))
        with self.assertRaisesRegex(BenchmarkError, "tracking"):
            parse_video_metadata(json.dumps(_metadata(tracking_enabled=True)))
        with self.assertRaisesRegex(BenchmarkError, "incomplete frame count"):
            parse_video_metadata(json.dumps(_metadata(processed_frame_count=61)))


class RuntimeBenchmarkAggregationTests(unittest.TestCase):
    def test_aggregates_latency_fps_counts_and_gpu_memory(self):
        rows = [
            {
                "record_type": "measured",
                "status": "ok",
                "request_latency_ms": 100.0,
                "api_mean_inference_latency_ms": 10.0,
                "api_p95_inference_latency_ms": 15.0,
                "api_processing_time_ms": 200.0,
                "api_end_to_end_fps": 30.0,
                "client_end_to_end_fps": 100.0,
                "processed_frame_count": 10,
                "total_detections": 22,
                "gpu_memory_before_mb": 1000,
                "gpu_memory_after_mb": 1200,
            },
            {
                "record_type": "measured",
                "status": "ok",
                "request_latency_ms": 400.0,
                "api_mean_inference_latency_ms": 20.0,
                "api_p95_inference_latency_ms": 25.0,
                "api_processing_time_ms": 400.0,
                "api_end_to_end_fps": 20.0,
                "client_end_to_end_fps": 50.0,
                "processed_frame_count": 20,
                "total_detections": 45,
                "gpu_memory_before_mb": 1100,
                "gpu_memory_after_mb": 1300,
            },
            {"record_type": "measured", "status": "failed"},
            {"record_type": "warmup", "status": "ok"},
        ]
        result = aggregate_runs(rows)
        self.assertEqual(result["successful_runs"], 2)
        self.assertEqual(result["failed_runs"], 1)
        self.assertEqual(result["mean_request_latency_ms"], 250)
        self.assertEqual(result["median_request_latency_ms"], 250)
        self.assertEqual(result["p95_request_latency_ms"], 385)
        self.assertEqual(result["mean_per_run_api_p95_inference_ms"], 20)
        self.assertEqual(result["median_per_run_api_p95_inference_ms"], 20)
        self.assertEqual(result["p95_of_per_run_api_p95_inference_ms"], 24.5)
        self.assertEqual(result["mean_client_end_to_end_fps"], 75)
        self.assertEqual(result["pooled_client_fps"], 60)
        self.assertEqual(result["mean_api_end_to_end_fps"], 25)
        self.assertEqual(result["whole_video_api_fps"], 50)
        self.assertEqual(result["total_processed_frames"], 30)
        self.assertEqual(result["detection_counts"], "22,45")
        self.assertEqual(result["mean_gpu_memory_observed_mb"], 1150)

    def test_api_preflight_requires_active_onnx_cuda_provider(self):
        healthy = FakeResponse(
            body={
                "status": "ready",
                "detector_initialized": True,
                "active_runtime": "onnx",
                "active_device": "cuda:0",
            }
        )
        config = FakeResponse(
            body={
                "runtime": "onnx",
                "device": "cuda",
                "execution_provider": "CPUExecutionProvider",
                "input_resolution": 640,
                "confidence_threshold": 0.25,
                "nms_iou_threshold": 0.7,
            }
        )
        session = Mock()
        session.get.side_effect = [healthy, config]
        client = RuntimeBenchmarkClient("http://api", session=session)
        with self.assertRaisesRegex(BenchmarkError, "CUDAExecutionProvider"):
            client.get_api_settings()

    def test_reports_failed_request_and_missing_metrics_as_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "clip.avi"
            video.write_bytes(b"sample")
            for response in (
                FakeResponse(
                    status_code=500,
                    body={"detail": "Video processing failed."},
                ),
                FakeResponse(
                    headers={"Content-Type": "video/x-msvideo"},
                    chunks=[b"annotated"],
                ),
            ):
                session = Mock()
                session.post.return_value = response
                client = RuntimeBenchmarkClient(
                    "http://api",
                    session=session,
                    gpu_probe=lambda: (None, "nvidia-smi unavailable"),
                )
                row = client.run_video(
                    video_path=video,
                    runtime="onnx",
                    config={"device": "cuda", "input_resolution": 640},
                    comparison_id="test",
                    video_sha256="hash",
                    run_type="measured",
                    run_index=1,
                    timestamp_utc="now",
                )
                self.assertEqual(row["status"], "failed")
                self.assertTrue(row["error"])
                self.assertTrue(response.closed)
                self.assertEqual(session.post.call_args.kwargs["data"], {"track": "false"})

    def test_successful_request_records_metadata_and_download_latency(self):
        response = FakeResponse(
            headers={
                "Content-Type": "video/x-msvideo",
                "X-EdgeAI-Metadata": json.dumps(_metadata()),
            },
            chunks=[b"annotated", b"video"],
        )
        session = Mock()
        session.post.return_value = response
        times = iter((1_000_000_000, 2_500_000_000))
        client = RuntimeBenchmarkClient(
            "http://api",
            session=session,
            gpu_probe=lambda: ({"memory_mb": 1000, "utilization_percent": 20}, "ok"),
            clock=lambda: next(times),
        )
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "clip.avi"
            video.write_bytes(b"input")
            row = client.run_video(
                video_path=video,
                runtime="onnx",
                config={"device": "cuda", "input_resolution": 640},
                comparison_id="test",
                video_sha256="hash",
                run_type="measured",
                run_index=1,
                timestamp_utc="now",
            )
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["request_latency_ms"], 1500)
        self.assertEqual(row["api_mean_inference_latency_ms"], 20.0)
        self.assertEqual(row["api_median_inference_latency_ms"], 19.0)
        self.assertEqual(row["api_p95_inference_latency_ms"], 28.0)
        self.assertEqual(row["processed_frame_count"], 62)
        self.assertEqual(row["total_detections"], 124)
        self.assertEqual(row["output_bytes"], len(b"annotatedvideo"))
        self.assertAlmostEqual(
            row["client_end_to_end_fps"],
            row["processed_frame_count"] * 1000 / row["request_latency_ms"],
        )
        self.assertEqual(session.post.call_args.kwargs["data"], {"track": "false"})


class RuntimeBenchmarkReportTests(unittest.TestCase):
    def test_writes_csv_and_comparison_markdown(self):
        rows = []
        for runtime, latency, detections in (("pytorch", 120.0, 100), ("onnx", 90.0, 100)):
            row = {field: "" for field in CSV_FIELDS}
            row.update(
                comparison_id="same-video",
                video_sha256="abc123",
                input_resolution=640,
                confidence_threshold=0.25,
                nms_iou_threshold=0.7,
                runtime=runtime,
                record_type="measured",
                run_index=1,
                status="ok",
                request_latency_ms=latency,
                api_mean_inference_latency_ms=22,
                api_median_inference_latency_ms=20,
                api_p95_inference_latency_ms=30,
                api_processing_time_ms=1100,
                api_end_to_end_fps=30,
                client_end_to_end_fps=25,
                processed_frame_count=62,
                total_detections=detections,
                gpu_memory_before_mb=1000,
                gpu_memory_after_mb=1200,
            )
            rows.append(row)
        report = make_report(rows)
        self.assertIn("**25.00% improvement** relative to PyTorch", report)
        self.assertIn("**33.33% higher** than PyTorch", report)
        self.assertIn("Detection-count consistency by measured run index: **consistent**", report)
        self.assertIn("frame-latency samples cannot be reconstructed", report)
        self.assertIn("1000 / 1200", report)
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "runtime_benchmark.csv"
            report_path = Path(directory) / "runtime_benchmark.md"
            write_results(csv_path, report_path, rows)
            with csv_path.open(newline="", encoding="utf-8") as stream:
                written = list(csv.DictReader(stream))
            self.assertEqual(len(written), 2)
            self.assertTrue(report_path.is_file())
            self.assertIn("PyTorch CUDA vs ONNX CUDA", report_path.read_text(encoding="utf-8"))

    def test_detection_counts_compare_ordered_runs_not_only_sets(self):
        rows = []
        for runtime, counts in (("pytorch", (1, 2)), ("onnx", (2, 1))):
            for index, detection_count in enumerate(counts, start=1):
                row = {field: "" for field in CSV_FIELDS}
                row.update(
                    comparison_id="ordered-counts",
                    video_sha256="hash",
                    input_resolution=640,
                    confidence_threshold=0.25,
                    nms_iou_threshold=0.7,
                    runtime=runtime,
                    record_type="measured",
                    run_index=index,
                    status="ok",
                    request_latency_ms=100,
                    api_processing_time_ms=100,
                    api_mean_inference_latency_ms=10,
                    api_p95_inference_latency_ms=15,
                    api_end_to_end_fps=10,
                    processed_frame_count=1,
                    total_detections=detection_count,
                )
                rows.append(row)
        report = make_report(rows)
        self.assertIn("Detection-count consistency by measured run index: **not consistent**", report)

    def test_regenerates_report_from_existing_csv_without_changing_raw_rows(self):
        project_csv = Path(__file__).resolve().parents[1] / "runtime_benchmark.csv"
        original_bytes = project_csv.read_bytes()
        with project_csv.open(newline="", encoding="utf-8") as stream:
            original_fields = next(csv.reader(stream))
        with project_csv.open(newline="", encoding="utf-8") as stream:
            historical_rows = list(csv.DictReader(stream))
        for row in historical_rows:
            if row["record_type"] == "measured" and row["status"] == "ok":
                expected_fps = int(row["processed_frame_count"]) * 1000 / float(row["request_latency_ms"])
                self.assertTrue(
                    math.isclose(float(row["client_end_to_end_fps"]), expected_fps, rel_tol=1e-12)
                )
        for runtime in ("pytorch", "onnx"):
            raw_runs = [
                row for row in historical_rows
                if row["runtime"] == runtime and row["record_type"] == "measured"
            ]
            computed = aggregate_runs(raw_runs)
            stored = next(
                row for row in historical_rows
                if row["runtime"] == runtime and row["record_type"] == "aggregate"
            )
            for key in (
                "successful_runs",
                "failed_runs",
                "mean_request_latency_ms",
                "median_request_latency_ms",
                "p95_request_latency_ms",
                "mean_run_api_inference_ms",
                "median_run_api_inference_ms",
                "p95_run_api_inference_ms",
                "mean_api_end_to_end_fps",
                "mean_client_end_to_end_fps",
                "total_processed_frames",
                "mean_total_detections",
                "detection_counts",
                "mean_gpu_memory_observed_mb",
            ):
                if isinstance(computed[key], (int, float)):
                    self.assertTrue(math.isclose(float(stored[key]), computed[key], rel_tol=1e-12), key)
                else:
                    self.assertEqual(stored[key], computed[key], key)
        with tempfile.TemporaryDirectory() as directory:
            copy_csv = Path(directory) / "runtime_benchmark.csv"
            copy_report = Path(directory) / "runtime_benchmark.md"
            copy_csv.write_bytes(original_bytes)
            regenerate_report_from_csv(copy_csv, copy_report)
            with copy_csv.open(newline="", encoding="utf-8") as stream:
                refreshed_rows = list(csv.DictReader(stream))
            original_raw_rows = [
                {key: row.get(key, "") for key in original_fields}
                for row in historical_rows
                if row["record_type"] in {"warmup", "measured"}
            ]
            refreshed_raw_rows = [
                {key: row.get(key, "") for key in original_fields}
                for row in refreshed_rows
                if row["record_type"] in {"warmup", "measured"}
            ]
            self.assertEqual(refreshed_raw_rows, original_raw_rows)
            for runtime in ("pytorch", "onnx"):
                expected = aggregate_runs([
                    row for row in refreshed_rows
                    if row["runtime"] == runtime and row["record_type"] == "measured"
                ])
                stored = next(
                    row for row in refreshed_rows
                    if row["runtime"] == runtime and row["record_type"] == "aggregate"
                )
                for key in (
                    "mean_per_run_api_p95_inference_ms",
                    "median_per_run_api_p95_inference_ms",
                    "p95_of_per_run_api_p95_inference_ms",
                    "total_client_elapsed_ms",
                    "total_api_processing_time_ms",
                    "pooled_client_fps",
                    "whole_video_api_fps",
                ):
                    self.assertTrue(math.isclose(float(stored[key]), expected[key], rel_tol=1e-12), key)
            report = copy_report.read_text(encoding="utf-8")
            self.assertIn("36.58 / 33.20 / 46.86", report)
            self.assertIn("33.30 / 31.78 / 45.15", report)
            self.assertIn("38.49", report)
            self.assertIn("41.62", report)
            self.assertIn("consistent", report)

    def test_runner_persists_failed_measured_attempts_and_aggregates(self):
        class FakeClient:
            def get_api_settings(self):
                return {
                    "runtime": "pytorch",
                    "device": "cuda",
                    "input_resolution": 640,
                    "confidence_threshold": 0.25,
                    "nms_iou_threshold": 0.7,
                }

            def run_video(self, *, run_type, run_index, **kwargs):
                status = "failed" if run_type == "measured" and run_index == 1 else "ok"
                return {
                    "status": status,
                    "error": "mock inference failure" if status == "failed" else "",
                    "request_latency_ms": 100 if status == "ok" else 0,
                    "api_processing_time_ms": 2000 if status == "ok" else "",
                    "api_mean_inference_latency_ms": 20 if status == "ok" else "",
                    "api_median_inference_latency_ms": 19 if status == "ok" else "",
                    "api_p95_inference_latency_ms": 25 if status == "ok" else "",
                    "api_end_to_end_fps": 30 if status == "ok" else "",
                    "client_end_to_end_fps": 25 if status == "ok" else "",
                    "processed_frame_count": 62 if status == "ok" else "",
                    "total_detections": 100 if status == "ok" else "",
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(b"same input")
            csv_path = root / "runtime_benchmark.csv"
            report_path = root / "runtime_benchmark.md"
            rows = run_benchmark(
                video_path=video,
                runtime="pytorch",
                runs=2,
                comparison_id="persist-failures",
                csv_path=csv_path,
                report_path=report_path,
                client=FakeClient(),
            )
            measured = [row for row in rows if row["record_type"] == "measured"]
            aggregate = rows[-1]
            self.assertEqual([row["status"] for row in measured], ["failed", "ok"])
            self.assertEqual(aggregate["failed_runs"], 1)
            with csv_path.open(newline="", encoding="utf-8") as stream:
                written = list(csv.DictReader(stream))
            self.assertEqual(len(written), 4)
            self.assertIn("mock inference failure", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
