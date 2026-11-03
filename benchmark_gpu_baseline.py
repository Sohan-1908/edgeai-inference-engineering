"""Reproducible, strict-CUDA PyTorch versus ONNX Runtime image benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch

from edgeai.detector import ImageDetector
from edgeai.errors import DeviceUnavailableError, InferenceError, ModelLoadError
from edgeai.image_io import load_image
from edgeai.onnx_detector import DEFAULT_ONNX_PATH, OnnxDetector
from edgeai.config import DEFAULT_MODEL

DEFAULT_WARMUP_RUNS = 10
DEFAULT_MEASURED_RUNS = 30
DEFAULT_TRIALS = 3
INPUT_SIZE = 640
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.7
_STAGE_NAMES = {
    "preprocessing": "preprocess_ms",
    "model_execution": "runtime_ms",
    "postprocessing": "postprocess_ms",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_model(weights_path: Path, onnx_path: Path) -> dict[str, Any]:
    manifest_path = onnx_path.with_suffix(".json")
    if not manifest_path.is_file():
        raise ValueError(f"ONNX export manifest is required to verify source weights: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read ONNX export manifest '{manifest_path}': {exc}") from exc

    weights_hash = _sha256_file(weights_path)
    onnx_hash = _sha256_file(onnx_path)
    source_hash = manifest.get("source_weights_sha256")
    manifest_onnx_hash = manifest.get("sha256")
    if not isinstance(source_hash, str) or source_hash.lower() != weights_hash:
        raise ValueError("ONNX manifest source_weights_sha256 does not match the PyTorch checkpoint.")
    if not isinstance(manifest_onnx_hash, str) or manifest_onnx_hash.lower() != onnx_hash:
        raise ValueError("ONNX model SHA-256 does not match its export manifest.")
    if manifest.get("precision") != "FP32":
        raise ValueError("The ONNX export manifest must identify an FP32 model.")
    if manifest.get("batch_size") != 1 or manifest.get("input_size") != [INPUT_SIZE, INPUT_SIZE]:
        raise ValueError("The ONNX export must use batch size 1 and a 640x640 input.")

    return {
        "weights_sha256": weights_hash,
        "onnx_sha256": onnx_hash,
        "manifest_path": str(manifest_path),
        "source_weights_match": True,
        "onnx_manifest_hash_match": True,
        "precision": "FP32",
        "batch_size": 1,
        "input_size": [INPUT_SIZE, INPUT_SIZE],
    }


def _latency_statistics(samples: list[float]) -> dict[str, float]:
    if not samples or any(not np.isfinite(value) or value < 0 for value in samples):
        raise ValueError("Latency samples must be non-empty, finite, and non-negative.")
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(mean(samples)),
        "median_ms": float(median(samples)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
    }


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _driver_version() -> dict[str, str | None]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"version": None, "status": f"unavailable: {type(exc).__name__}"}
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not versions:
        return {"version": None, "status": "unavailable: nvidia-smi returned no driver version"}
    return {"version": versions[0], "status": "available"}


def _software_hardware() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "pytorch_cuda_runtime": torch.version.cuda,
        "onnxruntime": ort.__version__,
        "onnxruntime_available_providers": ort.get_available_providers(),
        "ultralytics": _package_version("ultralytics"),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_driver": _driver_version(),
    }


def _predict(detector: Any, image: np.ndarray, runtime: str) -> tuple[float, dict[str, float]]:
    started = perf_counter()
    result, _ = detector.predict_frame(
        image,
        CONFIDENCE,
        **(
            {"image_size": INPUT_SIZE, "iou_threshold": IOU_THRESHOLD}
            if runtime == "pytorch"
            else {"iou_threshold": IOU_THRESHOLD}
        ),
    )
    full_call_ms = (perf_counter() - started) * 1000.0
    timings = getattr(detector, "last_timings", {}) or {}
    stages = {}
    for stage_name, timing_key in _STAGE_NAMES.items():
        value = timings.get(timing_key)
        if value is not None:
            stages[stage_name] = float(value)
    return full_call_ms, stages


def _summarize_configuration(
    runtime: str,
    detector: Any,
    model_path: Path,
    model_sha256: str,
    trial_samples: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    records = [record for trial in trial_samples for record in trial]
    full_call_samples = [record["full_call_ms"] for record in records]
    stage_summaries = {}
    for stage_name in _STAGE_NAMES:
        samples = [
            record["stages"][stage_name]
            for record in records
            if stage_name in record["stages"]
        ]
        stage_summaries[stage_name] = _latency_statistics(samples) if samples else None

    trial_means = [mean(record["full_call_ms"] for record in trial) for trial in trial_samples]
    model_execution_samples = [
        record["stages"]["model_execution"]
        for record in records
        if "model_execution" in record["stages"]
    ]
    return {
        "runtime": runtime,
        "runtime_version": str(detector.runtime_version),
        "model_path": str(model_path),
        "model_format": "PyTorch checkpoint" if runtime == "pytorch" else "ONNX",
        "model_sha256": model_sha256,
        "device": detector.device,
        "execution_backend": "PyTorch CUDA" if runtime == "pytorch" else "ONNX Runtime CUDA",
        "execution_provider": None if runtime == "pytorch" else detector.execution_provider,
        "session_providers": (
            None if runtime == "pytorch" else detector.session.get_providers()
        ),
        "cpu_ep_fallback_disabled": None if runtime == "pytorch" else bool(detector.require_cuda),
        "precision": "FP32",
        "model_input_size": [INPUT_SIZE, INPUT_SIZE],
        "batch_size": 1,
        "model_file_size_bytes": model_path.stat().st_size,
        "full_inference_call_latency": _latency_statistics(full_call_samples),
        "stage_latency": stage_summaries,
        "pooled_full_call_throughput_fps": (
            len(full_call_samples) * 1000.0 / sum(full_call_samples)
        ),
        "pooled_model_execution_throughput_fps": (
            len(model_execution_samples) * 1000.0 / sum(model_execution_samples)
            if model_execution_samples and sum(model_execution_samples) > 0
            else None
        ),
        "trial_mean_full_call_latency_ms": trial_means,
        "raw_full_call_latency_samples_ms": full_call_samples,
        "raw_stage_latency_samples_ms": {
            stage: [
                record["stages"][stage]
                for record in records
                if stage in record["stages"]
            ]
            for stage in _STAGE_NAMES
        },
    }


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 16B GPU inference baseline",
        "",
        f"- Created (UTC): `{report['created_utc']}`",
        f"- Input: `{report['input']['path']}` ({report['input']['width']}x{report['input']['height']})",
        f"- Input SHA-256: `{report['input']['sha256']}`",
        f"- Source checkpoint SHA-256: `{report['model_lineage']['weights_sha256']}`",
        (
            f"- Settings: {report['settings']['input_size'][0]}x{report['settings']['input_size'][1]}, "
            f"batch 1, FP32, confidence {report['settings']['confidence_threshold']}, "
            f"IoU {report['settings']['iou_threshold']}, "
            f"{report['settings']['warmup_iterations_per_trial']} warm-ups and "
            f"{report['settings']['measured_iterations_per_trial']} measured iterations per trial, "
            f"{report['settings']['trials']} paired trials"
        ),
        "",
        "Full-call latency is an independently measured wall-clock interval around the detector call. "
        "Model-execution latency is the runtime-reported model stage: Ultralytics `speed.inference` "
        "for PyTorch and synchronous `session.run` for ONNX Runtime.",
        "",
        "| Runtime | Full call mean / median / p90 / p95 (ms) | Full-call min-max (ms) | Model execution mean / median / p90 / p95 (ms) | Pooled full-call FPS | Pooled model-execution FPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("pytorch", "onnx"):
        config = report["configurations"][name]
        full = config["full_inference_call_latency"]
        execution = config["stage_latency"]["model_execution"]
        if execution is None:
            execution_summary = "unavailable"
        else:
            execution_summary = (
                f"{execution['mean_ms']:.3f} / {execution['median_ms']:.3f} / "
                f"{execution['p90_ms']:.3f} / {execution['p95_ms']:.3f}"
            )
        model_fps = config["pooled_model_execution_throughput_fps"]
        model_fps_text = f"{model_fps:.3f}" if model_fps is not None else "unavailable"
        lines.append(
            f"| {name} | {full['mean_ms']:.3f} / {full['median_ms']:.3f} / "
            f"{full['p90_ms']:.3f} / {full['p95_ms']:.3f} | "
            f"{full['minimum_ms']:.3f}-{full['maximum_ms']:.3f} | "
            f"{execution_summary} | "
            f"{config['pooled_full_call_throughput_fps']:.3f} | {model_fps_text}"
        )
    lines.extend(
        [
            "",
            "Throughput is pooled as measured calls divided by the sum of their corresponding latency samples. "
            "Model execution and full-call throughput are distinct.",
            "",
            "The ONNX session uses `CUDAExecutionProvider` with CPU EP fallback disabled; initialization "
            "or execution errors fail the run rather than being reported as GPU measurements.",
            "",
            "GPU utilization and VRAM telemetry were not collected. No process-specific utilization or memory "
            "claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def run_gpu_baseline(
    image_path: str | Path,
    *,
    weights_path: str | Path = DEFAULT_MODEL,
    onnx_path: str | Path = DEFAULT_ONNX_PATH,
    output_json: str | Path,
    output_markdown: str | Path,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    measured_runs: int = DEFAULT_MEASURED_RUNS,
    trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    if warmup_runs < 1 or measured_runs < 1 or trials < 1:
        raise ValueError("warmup_runs, measured_runs, and trials must be positive.")
    if not torch.cuda.is_available():
        raise DeviceUnavailableError("The GPU baseline requires PyTorch CUDA; no CPU fallback is permitted.")
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available_providers:
        raise DeviceUnavailableError(
            f"The GPU baseline requires ONNX Runtime CUDAExecutionProvider; available providers: {available_providers}"
        )

    image_file = Path(image_path).expanduser().resolve()
    weights_file = Path(weights_path).expanduser().resolve()
    onnx_file = Path(onnx_path).expanduser().resolve()
    json_file = Path(output_json).expanduser()
    markdown_file = Path(output_markdown).expanduser()
    if json_file.resolve() == markdown_file.resolve():
        raise ValueError("JSON and Markdown output paths must be different.")
    for output_file in (json_file, markdown_file):
        if output_file.exists():
            raise FileExistsError(f"Refusing to overwrite existing benchmark output: {output_file}")
    for model_file in (weights_file, onnx_file):
        if not model_file.is_file():
            raise FileNotFoundError(f"Model file does not exist: {model_file}")

    lineage = _verify_source_model(weights_file, onnx_file)
    image, image_load_ms = load_image(image_file)
    pytorch_detector = ImageDetector(model_name=str(weights_file), device="cuda")
    if pytorch_detector.device != "cuda:0":
        raise DeviceUnavailableError(f"PyTorch initialized on {pytorch_detector.device}, expected cuda:0.")
    onnx_detector = OnnxDetector(
        model_path=onnx_file,
        device="cuda",
        precision="fp32",
        require_cuda=True,
    )
    if (
        onnx_detector.device != "cuda:0"
        or onnx_detector.execution_provider != "CUDAExecutionProvider"
        or not onnx_detector.require_cuda
    ):
        raise DeviceUnavailableError(
            "ONNX Runtime did not initialize with strict CUDAExecutionProvider and CPU fallback disabled."
        )
    if onnx_detector.input_size != INPUT_SIZE:
        raise ValueError(f"Expected a {INPUT_SIZE}x{INPUT_SIZE} ONNX input, got {onnx_detector.input_size}.")

    detectors = {
        "pytorch": (pytorch_detector, weights_file, lineage["weights_sha256"]),
        "onnx": (onnx_detector, onnx_file, lineage["onnx_sha256"]),
    }
    trial_samples: dict[str, list[list[dict[str, Any]]]] = {"pytorch": [], "onnx": []}
    for trial_index in range(trials):
        warmup_order = ("pytorch", "onnx") if trial_index % 2 == 0 else ("onnx", "pytorch")
        for runtime in warmup_order:
            detector = detectors[runtime][0]
            for _ in range(warmup_runs):
                _predict(detector, image, runtime)

        trial_order = ("pytorch", "onnx") if trial_index % 2 == 0 else ("onnx", "pytorch")
        for runtime in ("pytorch", "onnx"):
            trial_samples[runtime].append([])
        for iteration in range(measured_runs):
            order = trial_order if iteration % 2 == 0 else tuple(reversed(trial_order))
            for runtime in order:
                detector = detectors[runtime][0]
                full_call_ms, stages = _predict(detector, image, runtime)
                trial_samples[runtime][-1].append(
                    {"full_call_ms": full_call_ms, "stages": stages}
                )

    configurations = {
        runtime: _summarize_configuration(
            runtime,
            detectors[runtime][0],
            detectors[runtime][1],
            detectors[runtime][2],
            trial_samples[runtime],
        )
        for runtime in ("pytorch", "onnx")
    }
    report = {
        "benchmark_type": "phase16b_strict_cuda_pytorch_vs_onnx_fp32",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(image_file),
            "sha256": _sha256_file(image_file),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "decode_ms": image_load_ms,
        },
        "model_lineage": lineage,
        "settings": {
            "input_size": [INPUT_SIZE, INPUT_SIZE],
            "batch_size": 1,
            "precision": "FP32",
            "confidence_threshold": CONFIDENCE,
            "iou_threshold": IOU_THRESHOLD,
            "warmup_iterations_per_trial": warmup_runs,
            "measured_iterations_per_trial": measured_runs,
            "trials": trials,
            "total_measured_iterations_per_runtime": measured_runs * trials,
            "trial_runtime_order": [
                ["pytorch", "onnx"] if index % 2 == 0 else ["onnx", "pytorch"]
                for index in range(trials)
            ],
            "stage_timing_source": {
                "pytorch": "Ultralytics Result.speed: preprocess, inference, postprocess",
                "onnx": "prepare_onnx_input, synchronous InferenceSession.run, output decode/NMS",
            },
            "full_call_timing": "perf_counter wall time around detector.predict_frame; PyTorch predict_frame synchronizes CUDA before and after inference.",
            "percentile_method": "NumPy percentile with its default linear interpolation.",
        },
        "hardware_software": _software_hardware(),
        "memory_telemetry": {
            "collected": False,
            "note": "GPU utilization and VRAM telemetry were not collected; no process-specific claim is made.",
        },
        "configurations": configurations,
    }
    markdown = _markdown_summary(report)
    for output_file in (json_file, markdown_file):
        output_file.parent.mkdir(parents=True, exist_ok=True)
    report["output_paths"] = {
        "json": str(json_file.resolve()),
        "markdown": str(markdown_file.resolve()),
    }
    json_created = False
    markdown_created = False
    try:
        with json_file.open("x", encoding="utf-8") as stream:
            json_created = True
            stream.write(json.dumps(report, indent=2))
        with markdown_file.open("x", encoding="utf-8") as stream:
            markdown_created = True
            stream.write(markdown)
    except OSError:
        if markdown_created:
            markdown_file.unlink(missing_ok=True)
        if json_created:
            json_file.unlink(missing_ok=True)
        raise
    return report


def _timestamped_outputs(output_dir: Path) -> tuple[Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"phase16b_gpu_baseline_{timestamp}"
    return output_dir / f"{stem}.json", output_dir / f"{stem}.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("outputs/smoke_input.jpg"))
    parser.add_argument("--weights", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--onnx-model", type=Path, default=Path(DEFAULT_ONNX_PATH))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=DEFAULT_MEASURED_RUNS)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    args = parser.parse_args(argv)
    output_json, output_markdown = _timestamped_outputs(args.output_dir)
    try:
        report = run_gpu_baseline(
            args.image,
            weights_path=args.weights,
            onnx_path=args.onnx_model,
            output_json=output_json,
            output_markdown=output_markdown,
            warmup_runs=args.warmup_runs,
            measured_runs=args.runs,
            trials=args.trials,
        )
    except (
        DeviceUnavailableError,
        FileNotFoundError,
        FileExistsError,
        InferenceError,
        ModelLoadError,
        ValueError,
        OSError,
    ) as exc:
        parser.exit(2, f"Benchmark failed: {exc}\n")
    print(_markdown_summary(report))
    print(f"\nJSON: {report['output_paths']['json']}\nMarkdown: {report['output_paths']['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
