"""Collect separate ONNX Runtime CPU operator traces for FP32 and INT8 models."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edgeai.image_io import load_image
from edgeai.onnx_detector import OnnxDetector

ROOT = Path(__file__).resolve().parent
DEFAULT_FP32_MODEL = ROOT / "models" / "yolo11n.onnx"
DEFAULT_INT8_MODEL = ROOT / "models" / "yolo11n_int8.onnx"
DEFAULT_IMAGE_DIR = ROOT / "outputs" / "int8_benchmark_frames"
DEFAULT_BENCHMARK_REPORT = ROOT / "outputs" / "phase7b_cpu_benchmark.json"
DEFAULT_PROFILE_DIR = ROOT / "outputs" / "ort_profiles"
DEFAULT_IMAGE_COUNT = 4
DEFAULT_CPU_THREADS = 4
DEFAULT_WARMUP_RUNS = 10


def load_benchmark_images(
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    benchmark_report: str | Path = DEFAULT_BENCHMARK_REPORT,
) -> list[tuple[Path, Any]]:
    """Load the four exact image files referenced by the Phase 7B benchmark."""
    report_path = Path(benchmark_report)
    if not report_path.is_file():
        raise FileNotFoundError(f"Benchmark report does not exist: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read benchmark image manifest {report_path}: {exc}") from exc

    image_names = report.get("benchmark_images") if isinstance(report, dict) else None
    if not isinstance(image_names, list) or len(image_names) != DEFAULT_IMAGE_COUNT:
        raise ValueError(
            f"{report_path} must list exactly {DEFAULT_IMAGE_COUNT} benchmark_images."
        )

    root = Path(image_dir)
    loaded: list[tuple[Path, Any]] = []
    for saved_path in image_names:
        if not isinstance(saved_path, str) or not saved_path:
            raise ValueError("Benchmark image paths must be non-empty strings.")
        image_path = root / Path(saved_path).name
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Benchmark image is missing: {image_path} "
                f"(referenced by {report_path})"
            )
        image, _ = load_image(image_path)
        loaded.append((image_path.resolve(), image))
    return loaded


def _run_warmups(
    model_path: Path,
    precision: str,
    images: list[Any],
    *,
    cpu_threads: int,
    warmup_runs: int,
) -> None:
    detector = OnnxDetector(
        model_path,
        device="cpu",
        precision=precision,
        cpu_threads=cpu_threads,
        profile=False,
    )
    if detector.execution_provider != "CPUExecutionProvider":
        raise RuntimeError(
            f"Warm-up selected unexpected provider {detector.execution_provider!r} "
            f"for {model_path}."
        )
    if detector.input_size != 640:
        raise ValueError(
            f"Expected a fixed 640x640 ONNX input for {model_path}; "
            f"got {detector.input_size}x{detector.input_size}."
        )
    try:
        for index in range(warmup_runs):
            detector.predict_frame(images[index % len(images)])
    finally:
        del detector
        gc.collect()


def _profile_model(
    model_path: Path,
    precision: str,
    images: list[Any],
    *,
    cpu_threads: int,
    profile_prefix: Path,
) -> Path:
    detector = OnnxDetector(
        model_path,
        device="cpu",
        precision=precision,
        cpu_threads=cpu_threads,
        profile=True,
        profile_file_prefix=profile_prefix,
    )
    if detector.execution_provider != "CPUExecutionProvider":
        raise RuntimeError(
            f"Profiling selected unexpected provider {detector.execution_provider!r} "
            f"for {model_path}."
        )
    if detector.input_size != 640:
        raise ValueError(
            f"Expected a fixed 640x640 ONNX input for {model_path}; "
            f"got {detector.input_size}x{detector.input_size}."
        )
    try:
        for image in images:
            detector.predict_frame(image)
    finally:
        trace_name = detector.end_profiling()
        del detector
        gc.collect()

    if trace_name is None:
        raise RuntimeError(f"ONNX Runtime did not return a profiling trace for {model_path}.")
    trace_path = Path(trace_name).resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"ONNX Runtime trace was not created: {trace_path}")
    if trace_path.parent != profile_prefix.parent.resolve():
        raise RuntimeError(
            f"ONNX Runtime wrote trace outside the requested profile directory: {trace_path}"
        )
    return trace_path


def run_operator_profiling(
    *,
    fp32_model: str | Path = DEFAULT_FP32_MODEL,
    int8_model: str | Path = DEFAULT_INT8_MODEL,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    benchmark_report: str | Path = DEFAULT_BENCHMARK_REPORT,
    profile_dir: str | Path = DEFAULT_PROFILE_DIR,
    cpu_threads: int = DEFAULT_CPU_THREADS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
) -> dict[str, Any]:
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer.")
    if warmup_runs < 1:
        raise ValueError("warmup_runs must be a positive integer.")

    fp32_path = Path(fp32_model).expanduser().resolve()
    int8_path = Path(int8_model).expanduser().resolve()
    for model_path in (fp32_path, int8_path):
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model does not exist: {model_path}")

    images_with_paths = load_benchmark_images(image_dir, benchmark_report)
    image_paths = [path for path, _ in images_with_paths]
    images = [image for _, image in images_with_paths]
    destination = Path(profile_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    # ORT profiling begins with session creation, so warm-up uses separate,
    # unprofiled sessions to keep initialization and warm-up out of the traces.
    for model_path, precision in ((fp32_path, "fp32"), (int8_path, "int8")):
        _run_warmups(
            model_path,
            precision,
            images,
            cpu_threads=cpu_threads,
            warmup_runs=warmup_runs,
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    traces = {
        "FP32": _profile_model(
            fp32_path,
            "fp32",
            images,
            cpu_threads=cpu_threads,
            profile_prefix=destination / f"fp32_{stamp}",
        ),
        "INT8": _profile_model(
            int8_path,
            "int8",
            images,
            cpu_threads=cpu_threads,
            profile_prefix=destination / f"int8_{stamp}",
        ),
    }

    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profiling_configuration": {
            "execution_provider": "CPUExecutionProvider",
            "device": "cpu",
            "cpu_threads": cpu_threads,
            "input_size": [640, 640],
            "batch_size": 1,
            "image_count": len(images),
            "warmup_runs_per_model": warmup_runs,
            "profiling_runs_per_image": 1,
            "warmup_profiling_enabled": False,
        },
        "benchmark_images": [str(path) for path in image_paths],
        "models": {
            "FP32": {
                "path": str(fp32_path),
                "name": fp32_path.name,
                "precision": "FP32",
                "size_bytes": fp32_path.stat().st_size,
                "trace_path": str(traces["FP32"]),
            },
            "INT8": {
                "path": str(int8_path),
                "name": int8_path.name,
                "precision": "INT8",
                "size_bytes": int8_path.stat().st_size,
                "trace_path": str(traces["INT8"]),
            },
        },
    }
    manifest_path = destination / f"profile_manifest_{stamp}.json"
    with manifest_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")

    print("ONNX Runtime CPU profiling completed.")
    print(f"FP32 trace: {traces['FP32']}")
    print(f"INT8 trace: {traces['INT8']}")
    print(f"Manifest: {manifest_path}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-model", type=Path, default=DEFAULT_FP32_MODEL)
    parser.add_argument("--int8-model", type=Path, default=DEFAULT_INT8_MODEL)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--benchmark-report", type=Path, default=DEFAULT_BENCHMARK_REPORT)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    args = parser.parse_args(argv)
    run_operator_profiling(
        fp32_model=args.fp32_model,
        int8_model=args.int8_model,
        image_dir=args.image_dir,
        benchmark_report=args.benchmark_report,
        profile_dir=args.profile_dir,
        cpu_threads=args.cpu_threads,
        warmup_runs=args.warmup_runs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
