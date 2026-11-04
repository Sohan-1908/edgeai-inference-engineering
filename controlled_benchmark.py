"""Paired, interleaved CPU benchmark for the FP32 and QDQ ONNX models."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Sequence

import onnxruntime as ort
import numpy as np
import psutil

from edgeai.image_io import load_image
from edgeai.onnx_detector import OnnxDetector

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "controlled_benchmarks"
DEFAULT_MODEL_PATHS = {
    "FP32": PROJECT_ROOT / "models" / "yolo11n.onnx",
    "QDQ_INT8": PROJECT_ROOT / "models" / "yolo11n_int8.onnx",
}
DEFAULT_IMAGE_PATHS = tuple(
    PROJECT_ROOT / "outputs" / "int8_benchmark_frames" / name
    for name in (
        "benchmark_000_frame_000001.jpg",
        "benchmark_001_frame_000006.jpg",
        "benchmark_002_frame_000011.jpg",
        "benchmark_003_frame_000017.jpg",
    )
)
EXPECTED_IMAGE_SHA256 = {
    "benchmark_000_frame_000001.jpg": "b24b143a8c9a185220399ca493758b945eab3266e1d77de6a1f8eeb7a8ae6d3c",
    "benchmark_001_frame_000006.jpg": "0617627866facb84b5fdcfc5030366e1600166287ae590cc2e7e6a0d7e580e5b",
    "benchmark_002_frame_000011.jpg": "1740a11e257d014a80aa5d730961fb54f15417864d65f1caf45305877c458f95",
    "benchmark_003_frame_000017.jpg": "eab606d3232555c29fa8534c09c4922c92f20f227e8b4a121448666c847a3e6c",
}
MODEL_ORDER = ("FP32", "QDQ_INT8")
TRIAL_ORDERS = (
    ("FP32", "QDQ_INT8"),
    ("QDQ_INT8", "FP32"),
    ("FP32", "QDQ_INT8"),
)
WARMUPS_PER_MODEL_PER_TRIAL = 10
MEASURED_RUNS_PER_MODEL_PER_TRIAL = 50
TRIAL_COUNT = 3
CPU_THREADS = 4
INTER_OP_THREADS = 1
INPUT_SIZE = 640
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.7
HEAVY_CPU_AVERAGE_PERCENT = 80.0
LOW_AVAILABLE_RAM_BYTES = 1 * 1024**3
LOW_AVAILABLE_RAM_FRACTION = 0.10


class ControlledBenchmarkError(RuntimeError):
    """Raised when a benchmark run failed after writing its partial report."""


def execution_order(trial_index: int) -> tuple[str, str]:
    if not 0 <= trial_index < len(TRIAL_ORDERS):
        raise ValueError(f"trial_index must be in [0, {len(TRIAL_ORDERS) - 1}]")
    return TRIAL_ORDERS[trial_index]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latency_statistics(
    samples_ms: Sequence[float], *, include_throughput: bool = True
) -> dict[str, float | int | None]:
    if not samples_ms:
        raise ValueError("At least one latency sample is required")
    values = [float(value) for value in samples_ms]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Latency samples must be finite and non-negative")
    average = statistics.mean(values)
    result: dict[str, float | int | None] = {
        "sample_count": len(values),
        "mean_ms": average,
        "median_ms": statistics.median(values),
        "p95_ms": float(np.percentile(values, 95)),
        "standard_deviation_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "throughput_inferences_per_second": (
            1000.0 / average if average and include_throughput else None
        ),
    }
    return result


def _cpu_model() -> str | None:
    cpu_model = platform.processor() or None
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                cpu_model = str(winreg.QueryValueEx(key, "ProcessorNameString")[0])
        except OSError:
            pass
    return cpu_model


def system_snapshot() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1.0, percpu=True)
    snapshot: dict[str, Any] = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "cpu_utilization_percent_by_logical_processor": cpu_percent,
        "cpu_utilization_average_percent": (
            statistics.mean(cpu_percent) if cpu_percent else None
        ),
        "total_ram_bytes": memory.total,
        "available_ram_bytes": memory.available,
        "ram_used_percent": memory.percent,
    }
    try:
        frequency = psutil.cpu_freq()
    except (AttributeError, OSError):
        frequency = None
    if frequency is not None:
        snapshot["cpu_frequency_mhz"] = {
            "current": frequency.current,
            "min": frequency.min,
            "max": frequency.max,
        }
    return snapshot


def _environment_snapshot() -> dict[str, Any]:
    return {
        "captured_at": datetime.now().astimezone().isoformat(),
        "operating_system": platform.platform(),
        "cpu_model": _cpu_model(),
        "logical_processor_count": os.cpu_count(),
        "python_version": sys.version,
        "onnxruntime_version": ort.__version__,
        "onnxruntime_available_providers": ort.get_available_providers(),
        "psutil_version": psutil.__version__,
        "configured_execution_provider": "CPUExecutionProvider",
        "session_options": {
            "intra_op_num_threads": CPU_THREADS,
            "inter_op_num_threads": INTER_OP_THREADS,
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "enable_profiling": False,
        },
        "resource_warning_thresholds": {
            "heavy_cpu_average_percent": HEAVY_CPU_AVERAGE_PERCENT,
            "low_available_ram_bytes": LOW_AVAILABLE_RAM_BYTES,
            "low_available_ram_fraction_of_total": LOW_AVAILABLE_RAM_FRACTION,
        },
    }


def _new_output_directory(output_root: str | Path) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    candidate = root / timestamp
    suffix = 0
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = root / f"{timestamp}-{suffix:02d}"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def _check_system_warnings(
    trial_records: Sequence[dict[str, Any]],
    initial_snapshot: dict[str, Any] | None = None,
) -> list[str]:
    warnings: list[str] = []
    records: list[tuple[str, dict[str, Any]]] = []
    if initial_snapshot is not None:
        records.append(("Pre-run", {"system_before": initial_snapshot}))
    records.extend((f"Trial {item['trial']}", item) for item in trial_records)
    for label, record in records:
        for phase in ("before", "after"):
            snapshot = record.get(f"system_{phase}")
            if not snapshot:
                continue
            average_cpu = snapshot.get("cpu_utilization_average_percent")
            if average_cpu is not None and average_cpu >= HEAVY_CPU_AVERAGE_PERCENT:
                warnings.append(
                    f"{label} {phase}: average CPU utilization "
                    f"{average_cpu:.1f}% met/exceeded {HEAVY_CPU_AVERAGE_PERCENT:.0f}%."
                )
            total = snapshot.get("total_ram_bytes")
            available = snapshot.get("available_ram_bytes")
            if total and available is not None and (
                available < LOW_AVAILABLE_RAM_BYTES
                or available / total < LOW_AVAILABLE_RAM_FRACTION
            ):
                warnings.append(
                    f"{label} {phase}: available RAM "
                    f"{available / (1024**3):.2f} GiB met the configured low-memory threshold."
                )
    return warnings


def _aggregate_model_results(
    raw_measurements: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metric_names = (
        "preprocessing_ms",
        "onnxruntime_execution_ms",
        "postprocessing_ms",
        "total_inference_latency_ms",
    )
    for model in MODEL_ORDER:
        model_samples = [item for item in raw_measurements if item["model"] == model]
        result[model] = {
            "measurements": len(model_samples),
            "metrics": {
                metric: latency_statistics([item[metric] for item in model_samples])
                if metric == "total_inference_latency_ms"
                else latency_statistics(
                    [item[metric] for item in model_samples],
                    include_throughput=False,
                )
                for metric in metric_names
            },
            "trial_summaries": [],
        }
        for trial in range(TRIAL_COUNT):
            trial_samples = [
                item
                for item in model_samples
                if item["trial"] == trial + 1
            ]
            result[model]["trial_summaries"].append(
                {
                    "trial": trial + 1,
                    "execution_position": (
                        execution_order(trial).index(model) + 1
                    ),
                    "metrics": {
                        metric: latency_statistics(
                            [item[metric] for item in trial_samples],
                            include_throughput=(
                                metric == "total_inference_latency_ms"
                            ),
                        )
                        for metric in metric_names
                    },
                }
            )
    return result


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Controlled Paired FP32 vs QDQ INT8 CPU Benchmark",
        "",
        f"- **Status:** {report['status']}",
        f"- **Started:** {report.get('started_at', 'not started')}",
        f"- **Ended:** {report.get('ended_at', 'not completed')}",
        f"- **Output directory:** `{report['output_directory']}`",
        f"- **Models:** FP32 and QDQ INT8; CPUExecutionProvider only",
        f"- **Configuration:** {WARMUPS_PER_MODEL_PER_TRIAL} warm-ups and "
        f"{MEASURED_RUNS_PER_MODEL_PER_TRIAL} measured inferences per model per trial, "
        f"{TRIAL_COUNT} trials; four CPU intra-op threads and one inter-op thread",
        "",
        "## Trial order",
        "",
        "| Trial | Execution order |",
        "|---:|---|",
    ]
    for trial_index in range(TRIAL_COUNT):
        lines.append(
            f"| {trial_index + 1} | {' -> '.join(execution_order(trial_index))} |"
        )
    lines.extend(["", "## Aggregate results", ""])
    if report.get("aggregates"):
        lines.extend(
            [
                "| Model | Metric | Mean (ms) | Median (ms) | P95 (ms) | Std dev (ms) | Throughput (inferences/s) |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for model in MODEL_ORDER:
            for label, key in (
                ("Preprocessing", "preprocessing_ms"),
                ("ORT execution", "onnxruntime_execution_ms"),
                ("Postprocessing", "postprocessing_ms"),
                ("Total inference", "total_inference_latency_ms"),
            ):
                stats = report["aggregates"][model]["metrics"][key]
                throughput = stats["throughput_inferences_per_second"]
                throughput_text = "n/a" if throughput is None else f"{throughput:.2f}"
                lines.append(
                    f"| {model} | {label} | {stats['mean_ms']:.3f} | "
                    f"{stats['median_ms']:.3f} | {stats['p95_ms']:.3f} | "
                    f"{stats['standard_deviation_ms']:.3f} | {throughput_text} |"
                )
    else:
        lines.append("No complete model aggregates were available.")
    lines.extend(["", "## Trial-level total inference latency", ""])
    if report.get("aggregates"):
        lines.extend(
            [
                "| Trial | Model | Runs | Mean (ms) | Median (ms) | P95 (ms) | Std dev (ms) |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for trial in range(TRIAL_COUNT):
            for model in execution_order(trial):
                summary = report["aggregates"][model]["trial_summaries"][trial][
                    "metrics"
                ]["total_inference_latency_ms"]
                lines.append(
                    f"| {trial + 1} | {model} | {summary['sample_count']} | "
                    f"{summary['mean_ms']:.3f} | {summary['median_ms']:.3f} | "
                    f"{summary['p95_ms']:.3f} | "
                    f"{summary['standard_deviation_ms']:.3f} |"
                )
    lines.extend(["", "## Warnings and errors", ""])
    warnings = report.get("warnings", [])
    errors = report.get("errors", [])
    if warnings:
        lines.extend(f"- **Warning:** {warning}" for warning in warnings)
    if errors:
        lines.extend(f"- **Error:** {error['message']}" for error in errors)
    if not warnings and not errors:
        lines.append("No benchmark errors or configured resource-threshold warnings.")
    lines.extend(
        [
            "",
            "CPU and memory snapshots describe observed conditions only; they do not establish causation.",
            "All raw per-inference measurements are preserved in `raw_measurements.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _persist(
    output_directory: Path,
    report: dict[str, Any],
    raw_data: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    report["ended_at"] = datetime.now().astimezone().isoformat()
    report["output_files"] = {
        "report_json": str(output_directory / "controlled_benchmark_report.json"),
        "report_markdown": str(output_directory / "controlled_benchmark_report.md"),
        "raw_measurements": str(output_directory / "raw_measurements.json"),
        "environment": str(output_directory / "benchmark_environment.json"),
    }
    _write_json(output_directory / "raw_measurements.json", raw_data)
    _write_json(output_directory / "benchmark_environment.json", environment)
    _write_json(output_directory / "controlled_benchmark_report.json", report)
    (output_directory / "controlled_benchmark_report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )


def run_controlled_benchmark(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    image_paths: Sequence[str | Path] = DEFAULT_IMAGE_PATHS,
    model_paths: dict[str, str | Path] | None = None,
    expected_image_hashes: dict[str, str] | None = EXPECTED_IMAGE_SHA256,
    detector_factory: Callable[..., Any] = OnnxDetector,
    image_loader: Callable[..., Any] = load_image,
    snapshotter: Callable[[], dict[str, Any]] = system_snapshot,
) -> dict[str, Any]:
    output_directory = _new_output_directory(output_root)
    selected_models = model_paths or DEFAULT_MODEL_PATHS
    report: dict[str, Any] = {
        "benchmark_type": "paired_interleaved_cpu_fp32_vs_qdq_int8",
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "output_directory": str(output_directory.resolve()),
        "configuration": {
            "execution_provider": "CPUExecutionProvider",
            "intra_op_threads": CPU_THREADS,
            "inter_op_threads": INTER_OP_THREADS,
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "profiling_enabled_during_measurement": False,
            "input_size": [INPUT_SIZE, INPUT_SIZE],
            "confidence": CONFIDENCE,
            "iou_threshold": IOU_THRESHOLD,
            "warmups_per_model_per_trial": WARMUPS_PER_MODEL_PER_TRIAL,
            "measured_runs_per_model_per_trial": MEASURED_RUNS_PER_MODEL_PER_TRIAL,
            "trials": TRIAL_COUNT,
            "timing_unit": "milliseconds",
            "total_inference_latency_boundary": (
                "perf_counter_ns around the complete OnnxDetector.predict_frame call; "
                "session creation and image decoding excluded"
            ),
        },
        "trial_order": [
            {"trial": index + 1, "models": list(execution_order(index))}
            for index in range(TRIAL_COUNT)
        ],
        "models": {},
        "images": [],
        "aggregates": {},
        "trial_environment": [],
        "warnings": [],
        "errors": [],
    }
    raw_data: dict[str, Any] = {
        "timing_unit": "milliseconds",
        "timing_fields": {
            "preprocessing_ms": "OnnxDetector.last_timings['preprocess_ms']",
            "onnxruntime_execution_ms": "OnnxDetector.last_timings['runtime_ms']",
            "postprocessing_ms": "OnnxDetector.last_timings['postprocess_ms']",
            "total_inference_latency_ms": (
                "perf_counter_ns around the complete predict_frame call"
            ),
        },
        "measurements": [],
        "errors": report["errors"],
    }
    environment = _environment_snapshot()
    environment["initial_resource_snapshot"] = None
    environment["trial_resource_snapshots"] = report["trial_environment"]
    try:
        if len(image_paths) != 4:
            raise ValueError(f"Exactly four benchmark images are required; got {len(image_paths)}")
        if set(selected_models) != set(MODEL_ORDER):
            raise ValueError(f"model_paths must contain exactly {MODEL_ORDER}")
        model_hashes: dict[str, str] = {}
        for model in MODEL_ORDER:
            path = Path(selected_models[model]).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{model} model does not exist: {path}")
            model_hashes[model] = sha256_file(path)
            report["models"][model] = {
                "path": str(path),
                "sha256": model_hashes[model],
                "file_size_bytes": path.stat().st_size,
                "precision": "fp32" if model == "FP32" else "int8",
            }
        images: list[dict[str, Any]] = []
        for index, image_path in enumerate(image_paths):
            path = Path(image_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Benchmark image does not exist: {path}")
            actual_hash = sha256_file(path)
            expected_hash = (
                expected_image_hashes.get(path.name)
                if expected_image_hashes is not None
                else None
            )
            image_record = {
                "index": index,
                "path": str(path),
                "file_name": path.name,
                "sha256": actual_hash,
                "expected_sha256": expected_hash,
                "hash_verified_against_previous_report": (
                    expected_hash is not None and actual_hash == expected_hash
                ),
            }
            report["images"].append(image_record)
            if expected_hash is not None and actual_hash != expected_hash:
                raise ValueError(
                    f"Benchmark image hash mismatch for {path.name}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
            image, _ = image_loader(path)
            if image is None:
                raise ValueError(f"Could not decode benchmark image: {path}")
            image_record["shape"] = list(image.shape)
            images.append({"record": image_record, "image": image})
        environment["initial_resource_snapshot"] = snapshotter()

        trial_records: list[dict[str, Any]] = []
        for trial_index in range(TRIAL_COUNT):
            trial_number = trial_index + 1
            trial_record: dict[str, Any] = {
                "trial": trial_number,
                "model_order": list(execution_order(trial_index)),
                "started_at": datetime.now().astimezone().isoformat(),
                "system_before": snapshotter(),
                "session_runs": [],
            }
            trial_records.append(trial_record)
            report["trial_environment"].append(trial_record)
            for model in execution_order(trial_index):
                model_path = Path(selected_models[model]).expanduser().resolve()
                precision = "fp32" if model == "FP32" else "int8"
                session_started = perf_counter_ns()
                detector = detector_factory(
                    model_path,
                    device="cpu",
                    cpu_threads=CPU_THREADS,
                    precision=precision,
                    profile=False,
                )
                session_creation_ms = (perf_counter_ns() - session_started) / 1_000_000
                providers = list(detector.session.get_providers())
                if providers != ["CPUExecutionProvider"]:
                    raise RuntimeError(
                        f"{model} trial {trial_number} did not use CPUExecutionProvider only: "
                        f"{providers}"
                    )
                if int(detector.input_size) != INPUT_SIZE:
                    raise RuntimeError(
                        f"{model} has {detector.input_size} input size; expected {INPUT_SIZE}"
                    )
                if detector.profile:
                    raise RuntimeError(f"Profiling unexpectedly enabled for {model}")
                session_run = {
                    "trial": trial_number,
                    "model": model,
                    "session_creation_ms": session_creation_ms,
                    "execution_providers": providers,
                    "warmup_inferences": WARMUPS_PER_MODEL_PER_TRIAL,
                    "measured_inferences": 0,
                    "model_order_position": len(trial_record["session_runs"]) + 1,
                }
                trial_record["session_runs"].append(session_run)
                for warmup_index in range(WARMUPS_PER_MODEL_PER_TRIAL):
                    detector.predict_frame(
                        images[warmup_index % len(images)]["image"],
                        CONFIDENCE,
                        iou_threshold=IOU_THRESHOLD,
                    )
                for measurement_index in range(
                    MEASURED_RUNS_PER_MODEL_PER_TRIAL
                ):
                    image_index = (
                        trial_index * MEASURED_RUNS_PER_MODEL_PER_TRIAL
                        + measurement_index
                    ) % len(images)
                    image_record = images[image_index]["record"]
                    started_ns = perf_counter_ns()
                    detector.predict_frame(
                        images[image_index]["image"],
                        CONFIDENCE,
                        iou_threshold=IOU_THRESHOLD,
                    )
                    total_ms = (perf_counter_ns() - started_ns) / 1_000_000
                    timings = detector.last_timings
                    sample = {
                        "trial": trial_number,
                        "model_order_position": session_run[
                            "model_order_position"
                        ],
                        "model": model,
                        "measurement_index": measurement_index + 1,
                        "image_index": image_index,
                        "image_path": image_record["path"],
                        "image_sha256": image_record["sha256"],
                        "preprocessing_ms": float(timings["preprocess_ms"]),
                        "onnxruntime_execution_ms": float(timings["runtime_ms"]),
                        "postprocessing_ms": float(timings["postprocess_ms"]),
                        "total_inference_latency_ms": total_ms,
                        "timestamp": datetime.now().astimezone().isoformat(),
                    }
                    for field in (
                        "preprocessing_ms",
                        "onnxruntime_execution_ms",
                        "postprocessing_ms",
                        "total_inference_latency_ms",
                    ):
                        value = sample[field]
                        if not math.isfinite(value) or value < 0:
                            raise ValueError(
                                f"Invalid {field} measurement for {model}: {value}"
                            )
                    raw_data["measurements"].append(sample)
                    session_run["measured_inferences"] += 1
                del detector
                gc.collect()
            trial_record["ended_at"] = datetime.now().astimezone().isoformat()
            trial_record["system_after"] = snapshotter()
            environment["trial_resource_snapshots"] = report["trial_environment"]

        report["aggregates"] = _aggregate_model_results(raw_data["measurements"])
        report["status"] = "completed"
        report["warnings"] = _check_system_warnings(
            trial_records, environment["initial_resource_snapshot"]
        )
        if len(raw_data["measurements"]) != (
            TRIAL_COUNT * MEASURED_RUNS_PER_MODEL_PER_TRIAL * len(MODEL_ORDER)
        ):
            raise RuntimeError("The completed benchmark has an unexpected sample count.")
    except Exception as exc:
        report["status"] = "failed"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "occurred_at": datetime.now().astimezone().isoformat(),
        }
        report["errors"].append(error)
        raw_data["errors"] = report["errors"]
        report["warnings"] = _check_system_warnings(
            report["trial_environment"], environment["initial_resource_snapshot"]
        )
    finally:
        environment["trial_resource_snapshots"] = report["trial_environment"]
        _persist(output_directory, report, raw_data, environment)

    if report["status"] != "completed":
        raise ControlledBenchmarkError(
            f"Benchmark failed; partial results and error details were saved to "
            f"{output_directory}"
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory for the new timestamped run directory.",
    )
    args = parser.parse_args(argv)
    try:
        report = run_controlled_benchmark(output_root=args.output_root)
    except ControlledBenchmarkError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Status: {report['status']}")
    print(f"Output directory: {report['output_directory']}")
    for model in MODEL_ORDER:
        total = report["aggregates"][model]["metrics"][
            "total_inference_latency_ms"
        ]
        print(
            f"{model}: mean={total['mean_ms']:.3f} ms, "
            f"median={total['median_ms']:.3f} ms, "
            f"P95={total['p95_ms']:.3f} ms, "
            f"stddev={total['standard_deviation_ms']:.3f} ms"
        )
    for warning in report["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
