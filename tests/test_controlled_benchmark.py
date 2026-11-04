from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import controlled_benchmark as benchmark


class _FakeSession:
    def get_providers(self):
        return ["CPUExecutionProvider"]


class _FakeDetector:
    def __init__(self, model_path, *, device, cpu_threads, precision, profile):
        self.model_path = Path(model_path)
        self.device = device
        self.cpu_threads = cpu_threads
        self.precision = precision
        self.profile = profile
        self.input_size = benchmark.INPUT_SIZE
        self.session = _FakeSession()
        self.last_timings = {
            "preprocess_ms": 1.0,
            "runtime_ms": 2.0,
            "postprocess_ms": 3.0,
        }

    def predict_frame(self, image, confidence, *, iou_threshold):
        return None, 6.0


class ControlledBenchmarkTests(unittest.TestCase):
    def _fixtures(self, temporary_directory):
        root = Path(temporary_directory)
        images = []
        hashes = {}
        for index in range(4):
            path = root / f"image_{index}.jpg"
            payload = f"image-{index}".encode("ascii")
            path.write_bytes(payload)
            images.append(path)
            hashes[path.name] = hashlib.sha256(payload).hexdigest()
        models = {}
        for name in benchmark.MODEL_ORDER:
            path = root / f"{name}.onnx"
            path.write_bytes(f"model-{name}".encode("ascii"))
            models[name] = path
        return images, hashes, models

    def _run(self, root, images, hashes, models, **kwargs):
        snapshots = iter(
            [
                {"available_ram_bytes": 4_000_000_000, "total_ram_bytes": 8_000_000_000,
                 "cpu_utilization_average_percent": 20.0}
                for _ in range(7)
            ]
        )
        return benchmark.run_controlled_benchmark(
            output_root=root / "reports",
            image_paths=images,
            expected_image_hashes=hashes,
            model_paths=models,
            detector_factory=kwargs.pop("detector_factory", _FakeDetector),
            image_loader=lambda path: (np.zeros((360, 640, 3), dtype=np.uint8), 0.0),
            snapshotter=lambda: next(snapshots),
            **kwargs,
        )

    def test_interleaved_execution_order_is_exact(self):
        self.assertEqual(benchmark.execution_order(0), ("FP32", "QDQ_INT8"))
        self.assertEqual(benchmark.execution_order(1), ("QDQ_INT8", "FP32"))
        self.assertEqual(benchmark.execution_order(2), ("FP32", "QDQ_INT8"))
        with self.assertRaises(ValueError):
            benchmark.execution_order(3)

    def test_latency_aggregation_reports_distribution_and_throughput(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = benchmark.latency_statistics(values)
        self.assertEqual(result["sample_count"], 5)
        self.assertEqual(result["mean_ms"], 3.0)
        self.assertEqual(result["median_ms"], 3.0)
        self.assertEqual(result["p95_ms"], 4.8)
        self.assertAlmostEqual(result["standard_deviation_ms"], 1.5811388300841898)
        self.assertAlmostEqual(result["throughput_inferences_per_second"], 1000 / 3)

    def test_hashes_session_configuration_and_raw_samples_are_recorded(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images, hashes, models = self._fixtures(root)
            detector_calls = []

            def detector_factory(*args, **kwargs):
                detector_calls.append((args, kwargs))
                return _FakeDetector(*args, **kwargs)

            report = self._run(
                root, images, hashes, models, detector_factory=detector_factory
            )
            output = Path(report["output_directory"])
            raw = json.loads((output / "raw_measurements.json").read_text())
            config = report["configuration"]

            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(detector_calls), 6)
            self.assertEqual(len(raw["measurements"]), 300)
            self.assertEqual(config["execution_provider"], "CPUExecutionProvider")
            self.assertEqual(config["intra_op_threads"], 4)
            self.assertEqual(config["inter_op_threads"], 1)
            self.assertFalse(config["profiling_enabled_during_measurement"])
            self.assertEqual(
                [item["sha256"] for item in report["images"]],
                [hashes[path.name] for path in images],
            )
            for name, model_path in models.items():
                self.assertEqual(
                    report["models"][name]["sha256"],
                    hashlib.sha256(model_path.read_bytes()).hexdigest(),
                )
            self.assertTrue(
                all(
                    kwargs
                    == {
                        "device": "cpu",
                        "cpu_threads": 4,
                        "precision": "fp32" if args[0].stem == "FP32" else "int8",
                        "profile": False,
                    }
                    for args, kwargs in detector_calls
                )
            )
            self.assertTrue(
                all(item["hash_verified_against_previous_report"] for item in report["images"])
            )
            self.assertEqual(
                [item["model"] for item in raw["measurements"][:50]],
                ["FP32"] * 50,
            )
            self.assertEqual(
                [item["model"] for item in raw["measurements"][50:100]],
                ["QDQ_INT8"] * 50,
            )
            self.assertIn("session_creation_ms", report["trial_environment"][0]["session_runs"][0])
            self.assertEqual(report["aggregates"]["FP32"]["measurements"], 150)
            self.assertEqual(report["aggregates"]["QDQ_INT8"]["measurements"], 150)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "controlled_benchmark_report.json",
                    "controlled_benchmark_report.md",
                    "raw_measurements.json",
                    "benchmark_environment.json",
                },
            )

    def test_image_hash_mismatch_writes_failure_report_without_running_inference(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images, hashes, models = self._fixtures(root)
            hashes[images[0].name] = "0" * 64
            with self.assertRaises(benchmark.ControlledBenchmarkError):
                self._run(root, images, hashes, models)

            output = next((root / "reports").iterdir())
            report = json.loads(
                (output / "controlled_benchmark_report.json").read_text()
            )
            raw = json.loads((output / "raw_measurements.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("hash mismatch", report["errors"][0]["message"])
            self.assertEqual(raw["measurements"], [])
            self.assertEqual(
                report["images"][0]["sha256"],
                hashlib.sha256(images[0].read_bytes()).hexdigest(),
            )
            self.assertEqual(report["images"][0]["expected_sha256"], "0" * 64)
            self.assertFalse(
                report["images"][0]["hash_verified_against_previous_report"]
            )

    def test_wrong_provider_fails_and_preserves_partial_report(self):
        class WrongProviderDetector(_FakeDetector):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.session = type(
                    "WrongSession", (), {"get_providers": lambda self: ["CUDAExecutionProvider"]}
                )()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images, hashes, models = self._fixtures(root)
            with self.assertRaises(benchmark.ControlledBenchmarkError):
                self._run(
                    root,
                    images,
                    hashes,
                    models,
                    detector_factory=WrongProviderDetector,
                )
            output = next((root / "reports").iterdir())
            report = json.loads(
                (output / "controlled_benchmark_report.json").read_text()
            )
            self.assertEqual(report["status"], "failed")
            self.assertIn("CPUExecutionProvider only", report["errors"][0]["message"])

    def test_resource_thresholds_are_reported_as_warnings(self):
        snapshots = [
            {"cpu_utilization_average_percent": 95.0},
            {"cpu_utilization_average_percent": 20.0},
        ]
        records = [
            {"trial": 1, "system_before": snapshots[0], "system_after": snapshots[1]}
        ]
        warnings = benchmark._check_system_warnings(records)
        self.assertEqual(len(warnings), 1)
        self.assertIn("95.0%", warnings[0])

    def test_initial_resource_constraints_are_included_in_warnings(self):
        initial = {
            "available_ram_bytes": 500_000_000,
            "total_ram_bytes": 8_000_000_000,
            "cpu_utilization_average_percent": 10.0,
        }
        warnings = benchmark._check_system_warnings([], initial)
        self.assertEqual(len(warnings), 1)
        self.assertIn("Pre-run before", warnings[0])
        self.assertIn("available RAM", warnings[0])


if __name__ == "__main__":
    unittest.main()
