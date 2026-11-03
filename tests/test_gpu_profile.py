from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import benchmark_gpu_profile as profile


class FakeNvml:
    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_GRAPHICS = 1
    NVML_CLOCK_MEM = 2

    def __init__(self, *, unsupported=()):
        self.unsupported = set(unsupported)
        self.initialized = False
        self.shutdown_called = False

    def nvmlInit(self):
        self.initialized = True

    def nvmlShutdown(self):
        self.shutdown_called = True

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def _value(self, name, value):
        if name in self.unsupported:
            raise RuntimeError(f"{name} unsupported")
        return value

    def nvmlDeviceGetUtilizationRates(self, handle):
        value = self._value("utilization", None)
        return SimpleNamespace(gpu=48, memory=32) if value is None else value

    def nvmlDeviceGetMemoryInfo(self, handle):
        self._value("memory", None)
        return SimpleNamespace(used=512 * 1024**2, total=4096 * 1024**2)

    def nvmlDeviceGetTemperature(self, handle, sensor):
        return self._value("temperature", 52)

    def nvmlDeviceGetPowerUsage(self, handle):
        return self._value("power", 12_500)

    def nvmlDeviceGetClockInfo(self, handle, clock):
        return self._value("clock", 1200 if clock == self.NVML_CLOCK_GRAPHICS else 5000)

    def nvmlDeviceGetPerformanceState(self, handle):
        return self._value("pstate", 2)


class FakeProcess:
    def __init__(self, *, fail_rss=False, fail_cpu=False):
        self.fail_rss = fail_rss
        self.fail_cpu = fail_cpu
        self.cpu_calls = 0

    def memory_info(self):
        if self.fail_rss:
            raise OSError("rss denied")
        return SimpleNamespace(rss=256 * 1024**2)

    def cpu_percent(self, interval=None):
        if self.fail_cpu:
            raise OSError("cpu denied")
        self.cpu_calls += 1
        return 21.0


class FakeTorch:
    class cuda:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def memory_allocated(device):
            return 10 * 1024**2

        @staticmethod
        def memory_reserved(device):
            return 24 * 1024**2


def _sample_once(runtime="pytorch", *, nvml=None, process=None):
    phases = profile._PhaseTracker()
    phases.set_phase("measured", 2)
    sampler = profile._TelemetrySampler(
        runtime,
        0.5,
        phases,
        nvml=nvml or FakeNvml(),
        process=process or FakeProcess(),
        torch_module=FakeTorch,
    )
    sampler.nvml.nvmlInit()
    sampler._handle = sampler.nvml.nvmlDeviceGetHandleByIndex(0)
    sampler._collect()
    return sampler


class TelemetryCollectionTests(unittest.TestCase):
    def test_collects_gpu_metrics_timestamps_and_phase_labels(self):
        sampler = _sample_once()
        sample = sampler.samples[0]

        self.assertEqual(sample["gpu_utilization_percent"], 48.0)
        self.assertEqual(sample["gpu_memory_utilization_percent"], 32.0)
        self.assertEqual(sample["gpu_memory_used_mib"], 512.0)
        self.assertEqual(sample["gpu_memory_total_mib"], 4096.0)
        self.assertEqual(sample["gpu_temperature_c"], 52.0)
        self.assertEqual(sample["gpu_power_w"], 12.5)
        self.assertEqual(sample["gpu_graphics_clock_mhz"], 1200.0)
        self.assertEqual(sample["gpu_memory_clock_mhz"], 5000.0)
        self.assertEqual(sample["gpu_performance_state"], 2)
        self.assertEqual(sample["phase"], "measured")
        self.assertEqual(sample["trial"], 2)
        self.assertIsNotNone(sample["timestamp_utc"])
        self.assertIsInstance(sample["monotonic_seconds"], float)

    def test_missing_nvml_metrics_are_null_and_recorded(self):
        sampler = _sample_once(nvml=FakeNvml(unsupported={"memory", "power"}))
        sample = sampler.samples[0]

        self.assertIsNone(sample["gpu_memory_used_mib"])
        self.assertIsNone(sample["gpu_memory_total_mib"])
        self.assertIsNone(sample["gpu_power_w"])
        self.assertTrue(any(error["metric"] == "gpu_power_w" for error in sampler.errors))
        self.assertEqual(sampler.report()["availability"]["gpu_power_w"]["status"], "unavailable")

    def test_wddm_process_gpu_memory_is_unavailable_not_zero(self):
        sampler = _sample_once(runtime="onnx")
        sample = sampler.samples[0]
        report = sampler.report()

        self.assertIsNone(sample["process_gpu_memory_mib"])
        self.assertIn("WDDM", sample["process_gpu_memory_status"])
        self.assertEqual(report["scope"]["gpu_metrics"], "DEVICE-WIDE NVML measurements")
        self.assertEqual(
            report["availability"]["process_gpu_memory_mib"]["status"],
            "unavailable",
        )

    def test_psutil_failures_are_reported_without_fabricated_values(self):
        sampler = _sample_once(process=FakeProcess(fail_rss=True, fail_cpu=True))

        self.assertIsNone(sampler.samples[0]["worker_rss_mib"])
        self.assertIsNone(sampler.samples[0]["worker_cpu_percent"])
        self.assertTrue(any(error["metric"] == "worker_rss_mib" for error in sampler.errors))
        self.assertTrue(any(error["metric"] == "worker_cpu_percent" for error in sampler.errors))

    def test_first_nonblocking_cpu_sample_is_marked_unavailable(self):
        sampler = _sample_once(process=FakeProcess())

        self.assertIsNone(sampler.samples[0]["worker_cpu_percent"])
        self.assertIn("baseline", sampler.samples[0]["worker_cpu_percent_status"])

    def test_short_cpu_sampling_interval_is_not_reported_as_utilization(self):
        sampler = _sample_once(process=FakeProcess())
        sampler._cpu_percent_primed = True
        sampler._last_cpu_monotonic = time.perf_counter() - 0.01
        sampler._collect()

        self.assertIsNone(sampler.samples[-1]["worker_cpu_percent"])
        self.assertIn("shorter than half", sampler.samples[-1]["worker_cpu_percent_status"])

    def test_pytorch_allocator_metrics_are_clearly_scoped(self):
        sampler = _sample_once(runtime="pytorch")
        report = sampler.report()

        self.assertEqual(sampler.samples[0]["pytorch_allocator_allocated_mib"], 10.0)
        self.assertEqual(sampler.samples[0]["pytorch_allocator_reserved_mib"], 24.0)
        self.assertIn("PYTORCH-ALLOCATOR", report["scope"]["pytorch_allocator"])
        self.assertIn("not total process GPU memory", report["scope"]["pytorch_allocator"])

    def test_onnx_does_not_report_pytorch_allocator_values(self):
        sampler = _sample_once(runtime="onnx")

        self.assertIsNone(sampler.samples[0]["pytorch_allocator_allocated_mib"])
        self.assertIn("not_applicable", sampler.samples[0]["pytorch_allocator_status"])


class TelemetryStatisticsTests(unittest.TestCase):
    def test_statistics_include_required_distribution_and_count(self):
        result = profile._statistics([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        self.assertEqual(result["minimum"], 1)
        self.assertEqual(result["mean"], 5.5)
        self.assertEqual(result["median"], 5.5)
        self.assertAlmostEqual(result["p90"], 9.1)
        self.assertAlmostEqual(result["p95"], 9.55)
        self.assertEqual(result["maximum"], 10)
        self.assertEqual(result["sample_count"], 10)
        self.assertIsNone(profile._statistics([]))

    def test_sampling_gap_is_reported(self):
        sampler = _sample_once()
        sampler._last_sample_monotonic = time.perf_counter() - 3.1
        sampler._collect()

        self.assertGreaterEqual(
            sampler.samples[-1]["missed_sampling_intervals"],
            5,
        )
        self.assertGreater(sampler.samples[-1]["gap_since_previous_sample_ms"], 3000)

    def test_phase_intervals_record_start_end_and_duration(self):
        tracker = profile._PhaseTracker()
        with tracker.interval("warmup", 1):
            time.sleep(0.02)
        item = tracker.intervals[0]

        self.assertEqual(item["phase"], "warmup")
        self.assertEqual(item["trial"], 1)
        self.assertLess(item["started_monotonic_seconds"], item["ended_monotonic_seconds"])
        self.assertGreater(item["duration_ms"], 0)
        self.assertIsNotNone(item["started_utc"])
        self.assertIsNotNone(item["ended_utc"])


class WorkerReliabilityTests(unittest.TestCase):
    class FakeConnection:
        def __init__(self):
            self.value = None
            self.closed = False

        def send(self, value):
            self.value = value

        def close(self):
            self.closed = True

    def test_worker_exception_preserves_traceback_and_stops_sampler(self):
        sampler = Mock()
        sampler.report.return_value = {"samples": [], "phase_intervals": []}
        connection = self.FakeConnection()
        with (
            patch.object(profile, "_TelemetrySampler", return_value=sampler),
            patch.object(profile.baseline, "load_image", side_effect=ValueError("decode failure")),
        ):
            profile._profile_worker(
                "pytorch",
                "input.jpg",
                "weights.pt",
                "model.onnx",
                0.5,
                connection,
            )

        self.assertTrue(sampler.start.called)
        self.assertTrue(sampler.stop.called)
        self.assertEqual(connection.value["status"], "failed")
        self.assertEqual(connection.value["failure"]["type"], "ValueError")
        self.assertIn("decode failure", connection.value["failure"]["traceback"])
        self.assertTrue(connection.closed)

    def test_worker_runner_joins_worker_and_propagates_exit_status(self):
        class FakeReceive:
            def poll(self, timeout=0):
                return True

            def recv(self):
                return {"runtime": "onnx", "status": "failed", "failure": {"type": "Test"}}

            def close(self):
                self.closed = True

        class FakeSend:
            def close(self):
                self.closed = True

        class FakeWorker:
            pid = 1234
            exitcode = 0

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.joined = False

            def start(self):
                pass

            def is_alive(self):
                return False

            def join(self, timeout=None):
                self.joined = True

            def terminate(self):
                raise AssertionError("completed worker should not be terminated")

        worker = FakeWorker()

        class FakeContext:
            def Pipe(self, duplex=False):
                return FakeReceive(), FakeSend()

            def Process(self, **kwargs):
                return worker

        result = profile._run_worker(
            "onnx",
            image_path=Path("input.jpg"),
            weights_path=Path("weights.pt"),
            onnx_path=Path("model.onnx"),
            sample_interval_seconds=0.5,
            context=FakeContext(),
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["worker_exit_code"], 0)
        self.assertTrue(worker.joined)

    def test_worker_runner_terminates_process_that_does_not_exit(self):
        class FakeReceive:
            def poll(self, timeout=0):
                return True

            def recv(self):
                return {"runtime": "onnx", "status": "completed"}

            def close(self):
                pass

        class FakeSend:
            def close(self):
                pass

        class FakeWorker:
            pid = 1234
            exitcode = None

            def __init__(self, **kwargs):
                self.terminated = False
                self.join_count = 0

            def start(self):
                pass

            def is_alive(self):
                return not self.terminated

            def join(self, timeout=None):
                self.join_count += 1

            def terminate(self):
                self.terminated = True
                self.exitcode = -15

        worker = FakeWorker()

        class FakeContext:
            def Pipe(self, duplex=False):
                return FakeReceive(), FakeSend()

            def Process(self, **kwargs):
                return worker

        with self.assertRaisesRegex(RuntimeError, "did not exit"):
            profile._run_worker(
                "onnx",
                image_path=Path("input.jpg"),
                weights_path=Path("weights.pt"),
                onnx_path=Path("model.onnx"),
                sample_interval_seconds=0.5,
                context=FakeContext(),
            )

        self.assertTrue(worker.terminated)
        self.assertEqual(worker.join_count, 3)


class OutputSafetyTests(unittest.TestCase):
    def test_timestamped_paths_are_unique_and_existing_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first_json, first_markdown = profile._timestamped_outputs(directory)
            first_json.write_text("existing JSON", encoding="utf-8")
            first_markdown.write_text("existing Markdown", encoding="utf-8")

            second_json, second_markdown = profile._timestamped_outputs(directory)
            self.assertNotEqual(first_json, second_json)
            self.assertNotEqual(first_markdown, second_markdown)

            report = {
                "created_utc": "2026-01-01T00:00:00+00:00",
                "status": "incomplete",
                "hardware_software": {"gpu_name": "test GPU"},
                "input": {"path": "input.jpg", "width": 10, "height": 20},
                "model_lineage": {"source_weights_match": True},
                "configuration": {
                    "warmup_runs": 10,
                    "measured_runs": 30,
                    "trials": 3,
                    "sampling_interval_seconds": 0.5,
                },
                "runtimes": {
                    name: {
                        "status": "failed",
                        "worker_pid": None,
                        "failure": None,
                        "telemetry": {
                            "sample_count": 1,
                            "sampling_overhead_total_ms": 0.1,
                            "performance_state_distribution": {"P0": 1},
                            "availability": {
                                "process_gpu_memory_mib": {
                                    "status": "unavailable",
                                    "sample_count": 0,
                                }
                            },
                            "phase_intervals": [],
                            "phase_summaries": {
                                "shutdown": {
                                    "worker_rss_mib": {
                                        "minimum": 1.0,
                                        "mean": 1.0,
                                        "median": 1.0,
                                        "p90": 1.0,
                                        "p95": 1.0,
                                        "maximum": 1.0,
                                        "sample_count": 1,
                                    },
                                    "missed_sampling_intervals": 0,
                                }
                            },
                            "errors": [],
                        },
                    }
                    for name in ("pytorch", "onnx")
                },
            }
            with self.assertRaises(FileExistsError):
                profile._write_reports(report, first_json, directory / "new.md")
            self.assertEqual(first_json.read_text(encoding="utf-8"), "existing JSON")
            self.assertFalse((directory / "new.md").exists())

    def test_report_outputs_are_written_as_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            json_path = directory / "profile.json"
            markdown_path = directory / "profile.md"
            report = {
                "created_utc": "2026-01-01T00:00:00+00:00",
                "status": "incomplete",
                "hardware_software": {"gpu_name": "test GPU"},
                "input": {"path": "input.jpg", "width": 10, "height": 20},
                "model_lineage": {"source_weights_match": True},
                "configuration": {
                    "warmup_runs": 10,
                    "measured_runs": 30,
                    "trials": 3,
                    "sampling_interval_seconds": 0.5,
                },
                "runtimes": {
                    name: {
                        "status": "failed",
                        "worker_pid": None,
                        "failure": None,
                        "telemetry": {
                            "sample_count": 1,
                            "sampling_overhead_total_ms": 0.1,
                            "performance_state_distribution": {"P0": 1},
                            "availability": {
                                "process_gpu_memory_mib": {
                                    "status": "unavailable",
                                    "sample_count": 0,
                                }
                            },
                            "phase_intervals": [],
                            "phase_summaries": {
                                "shutdown": {
                                    "worker_rss_mib": {
                                        "minimum": 1.0,
                                        "mean": 1.0,
                                        "median": 1.0,
                                        "p90": 1.0,
                                        "p95": 1.0,
                                        "maximum": 1.0,
                                        "sample_count": 1,
                                    },
                                    "missed_sampling_intervals": 0,
                                }
                            },
                            "errors": [],
                        },
                    }
                    for name in ("pytorch", "onnx")
                },
            }
            profile._write_reports(report, json_path, markdown_path)

            saved = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["output_paths"]["json"], str(json_path.resolve()))
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Telemetry interpretation", markdown)
            self.assertIn("| Phase | Metric | Min | Mean | Median | P90 | P95 | Max | Samples |", markdown)
            self.assertIn("process_gpu_memory_mib | unavailable | 0", markdown)


if __name__ == "__main__":
    unittest.main()
