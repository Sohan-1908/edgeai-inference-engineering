from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np

import benchmark_gpu_baseline as benchmark
from edgeai.errors import DeviceUnavailableError, InferenceError


class FakeDetector:
    def __init__(self, runtime: str, calls: list[str]):
        self.runtime = runtime
        self.calls = calls
        self.runtime_version = "test-runtime"
        self.device = "cuda:0"
        self.execution_provider = "CUDAExecutionProvider"
        self.require_cuda = True
        self.input_size = benchmark.INPUT_SIZE
        self.session = self
        self.last_timings = {}

    def get_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def predict_frame(self, image, confidence, **kwargs):
        self.calls.append(self.runtime)
        self.last_timings = (
            {"preprocess_ms": 1.0, "runtime_ms": 2.0, "postprocess_ms": 0.5}
            if self.runtime == "pytorch"
            else {"preprocess_ms": 0.8, "runtime_ms": 3.0, "postprocess_ms": 0.4}
        )
        return object(), 5.0


def _create_model_pair(directory: Path) -> tuple[Path, Path]:
    weights = directory / "model.pt"
    onnx_model = directory / "model.onnx"
    weights.write_bytes(b"fake-pytorch-checkpoint")
    onnx_model.write_bytes(b"fake-onnx-model")
    manifest = {
        "source_weights_sha256": benchmark._sha256_file(weights),
        "sha256": benchmark._sha256_file(onnx_model),
        "precision": "FP32",
        "batch_size": 1,
        "input_size": [benchmark.INPUT_SIZE, benchmark.INPUT_SIZE],
    }
    onnx_model.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
    return weights, onnx_model


class GpuBaselineStatisticsTests(unittest.TestCase):
    def test_reports_required_latency_statistics(self):
        stats = benchmark._latency_statistics([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        self.assertEqual(stats["mean_ms"], 5.5)
        self.assertEqual(stats["median_ms"], 5.5)
        self.assertAlmostEqual(stats["p90_ms"], 9.1)
        self.assertAlmostEqual(stats["p95_ms"], 9.55)
        self.assertEqual(stats["minimum_ms"], 1.0)
        self.assertEqual(stats["maximum_ms"], 10.0)

    def test_rejects_empty_or_invalid_latency_samples(self):
        for samples in ([], [1.0, float("nan")], [1.0, -0.1]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                benchmark._latency_statistics(samples)


class GpuBaselineLineageTests(unittest.TestCase):
    def test_rejects_manifest_that_does_not_match_source_weights(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            weights, onnx_model = _create_model_pair(Path(temporary_directory))
            manifest_path = onnx_model.with_suffix(".json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_weights_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source_weights_sha256"):
                benchmark._verify_source_model(weights, onnx_model)


class GpuBaselineExecutionTests(unittest.TestCase):
    def test_rejects_environment_without_onnx_cuda_provider(self):
        with (
            patch.object(benchmark.torch.cuda, "is_available", return_value=True),
            patch.object(
                benchmark.ort,
                "get_available_providers",
                return_value=["CPUExecutionProvider"],
            ),
        ):
            with self.assertRaisesRegex(DeviceUnavailableError, "CUDAExecutionProvider"):
                benchmark.run_gpu_baseline(
                    "unused.png",
                    output_json="unused.json",
                    output_markdown="unused.md",
                )

    def test_runs_paired_strict_cuda_benchmark_and_writes_both_reports(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            weights, onnx_model = _create_model_pair(directory)
            image_path = directory / "input.png"
            image_path.write_bytes(b"test-image")
            output_json = directory / "result.json"
            output_markdown = directory / "result.md"
            calls: list[str] = []
            pytorch_detector = FakeDetector("pytorch", calls)
            onnx_detector = FakeDetector("onnx", calls)

            with (
                patch.object(benchmark.torch.cuda, "is_available", return_value=True),
                patch.object(
                    benchmark.ort,
                    "get_available_providers",
                    return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
                ),
                patch.object(benchmark, "ImageDetector", return_value=pytorch_detector),
                patch.object(benchmark, "OnnxDetector", return_value=onnx_detector) as onnx_factory,
                patch.object(
                    benchmark,
                    "load_image",
                    return_value=(np.zeros((12, 20, 3), dtype=np.uint8), 0.25),
                ),
                patch.object(
                    benchmark, "_software_hardware",
                    return_value={"gpu_name": "Mock GPU", "onnxruntime_available_providers": [
                        "CUDAExecutionProvider", "CPUExecutionProvider"
                    ]},
                ),
            ):
                report = benchmark.run_gpu_baseline(
                    image_path,
                    weights_path=weights,
                    onnx_path=onnx_model,
                    output_json=output_json,
                    output_markdown=output_markdown,
                    warmup_runs=1,
                    measured_runs=2,
                    trials=2,
                )

            self.assertEqual(
                calls,
                [
                    "pytorch", "onnx", "pytorch", "onnx", "onnx", "pytorch",
                    "onnx", "pytorch", "onnx", "pytorch", "pytorch", "onnx",
                ],
            )
            onnx_factory.assert_called_once_with(
                model_path=onnx_model.resolve(),
                device="cuda",
                precision="fp32",
                require_cuda=True,
            )
            self.assertEqual(report["settings"]["batch_size"], 1)
            self.assertEqual(report["settings"]["trial_runtime_order"], [
                ["pytorch", "onnx"], ["onnx", "pytorch"]
            ])
            self.assertEqual(
                len(report["configurations"]["pytorch"]["raw_full_call_latency_samples_ms"]),
                4,
            )
            self.assertEqual(
                report["configurations"]["pytorch"]["stage_latency"]["model_execution"]["mean_ms"],
                2.0,
            )
            self.assertEqual(
                report["configurations"]["onnx"]["stage_latency"]["model_execution"]["mean_ms"],
                3.0,
            )
            self.assertTrue(output_json.is_file())
            self.assertTrue(output_markdown.is_file())
            self.assertEqual(
                json.loads(output_json.read_text(encoding="utf-8"))["benchmark_type"],
                "phase16b_strict_cuda_pytorch_vs_onnx_fp32",
            )
            summary = output_markdown.read_text(encoding="utf-8")
            self.assertIn("Full-call latency", summary)
            self.assertIn("Model-execution latency", summary)
            self.assertIn("CUDAExecutionProvider", summary)

    def test_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            existing_json = directory / "existing.json"
            existing_json.write_text("keep", encoding="utf-8")

            with (
                patch.object(benchmark.torch.cuda, "is_available", return_value=True),
                patch.object(
                    benchmark.ort,
                    "get_available_providers",
                    return_value=["CUDAExecutionProvider"],
                ),
            ):
                with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                    benchmark.run_gpu_baseline(
                        directory / "missing.png",
                        output_json=existing_json,
                        output_markdown=directory / "new.md",
                    )

            self.assertEqual(existing_json.read_text(encoding="utf-8"), "keep")

    def test_cli_reports_inference_failures_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            stderr = io.StringIO()
            with patch.object(
                benchmark,
                "run_gpu_baseline",
                side_effect=InferenceError("input tensor allocation failed"),
            ):
                with redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as exit_context:
                        benchmark.main(["--output-dir", temporary_directory])

            self.assertEqual(exit_context.exception.code, 2)
            self.assertIn("input tensor allocation failed", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
