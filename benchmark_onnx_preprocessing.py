"""Profile CPU preprocessing stages used by the fixed-shape ONNX detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import cv2
import numpy as np

from edgeai.image_io import load_image
from edgeai.onnx_detector import INPUT_SIZE, prepare_onnx_input

DEFAULT_IMAGE = Path("outputs/smoke_input.jpg")
DEFAULT_ONNX_MODEL = Path("models/yolo11n.onnx")
DEFAULT_WARMUP_RUNS = 10
DEFAULT_MEASURED_RUNS = 30
DEFAULT_TRIALS = 3

_STAGE_NAMES = (
    "input_size_validation",
    "letterbox_geometry",
    "resize",
    "border_padding",
    "channel_reversal_transpose_batch_view",
    "contiguous_float32_conversion",
    "in_place_normalization",
)


def _latency_statistics(samples: list[float]) -> dict[str, float]:
    if not samples or not all(np.isfinite(samples)) or any(value < 0 for value in samples):
        raise ValueError("Latency samples must be non-empty, finite, and non-negative.")
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_stages(
    image: np.ndarray, size: int
) -> tuple[np.ndarray, float, int, int, dict[str, float], dict[str, tuple[int, int]], float]:
    stage_ms: dict[str, float] = {}
    stage_timestamps: dict[str, tuple[int, int]] = {}
    pipeline_start = perf_counter_ns()

    start = perf_counter_ns()
    if size not in {320, 480, 640}:
        raise ValueError("ONNX input size must be one of 320, 480, or 640")
    end = perf_counter_ns()
    stage_ms["input_size_validation"] = (end - start) / 1_000_000
    stage_timestamps["input_size_validation"] = (start, end)

    start = perf_counter_ns()
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    pad_width = size - resized_width
    pad_height = size - resized_height
    left = int(round(pad_width / 2 - 0.1))
    top = int(round(pad_height / 2 - 0.1))
    right = pad_width - left
    bottom = pad_height - top
    end = perf_counter_ns()
    stage_ms["letterbox_geometry"] = (end - start) / 1_000_000
    stage_timestamps["letterbox_geometry"] = (start, end)

    start = perf_counter_ns()
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    end = perf_counter_ns()
    stage_ms["resize"] = (end - start) / 1_000_000
    stage_timestamps["resize"] = (start, end)

    start = perf_counter_ns()
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    end = perf_counter_ns()
    stage_ms["border_padding"] = (end - start) / 1_000_000
    stage_timestamps["border_padding"] = (start, end)

    start = perf_counter_ns()
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)[None]
    end = perf_counter_ns()
    stage_ms["channel_reversal_transpose_batch_view"] = (end - start) / 1_000_000
    stage_timestamps["channel_reversal_transpose_batch_view"] = (start, end)

    start = perf_counter_ns()
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    end = perf_counter_ns()
    stage_ms["contiguous_float32_conversion"] = (end - start) / 1_000_000
    stage_timestamps["contiguous_float32_conversion"] = (start, end)

    start = perf_counter_ns()
    np.divide(tensor, np.float32(255.0), out=tensor)
    end = perf_counter_ns()
    stage_ms["in_place_normalization"] = (end - start) / 1_000_000
    stage_timestamps["in_place_normalization"] = (start, end)

    pipeline_ms = (perf_counter_ns() - pipeline_start) / 1_000_000
    return tensor, scale, left, top, stage_ms, stage_timestamps, pipeline_ms


def _verify_preprocessing_equivalence(image: np.ndarray, size: int) -> dict[str, Any]:
    profiled = _profile_stages(image, size)
    production_tensor, production_scale, production_left, production_top = prepare_onnx_input(
        image, size
    )
    tensor, scale, left, top = profiled[:4]
    if not np.array_equal(tensor, production_tensor):
        raise AssertionError("Profiled preprocessing tensor differs from prepare_onnx_input.")
    if (scale, left, top) != (production_scale, production_left, production_top):
        raise AssertionError("Profiled letterbox metadata differs from prepare_onnx_input.")
    if (
        tensor.shape != (1, 3, size, size)
        or tensor.dtype != np.float32
        or not tensor.flags.c_contiguous
        or not np.isfinite(tensor).all()
    ):
        raise AssertionError("Profiled tensor does not meet the ONNX input contract.")
    return {
        "verified_exact_tensor_equality": True,
        "verified_letterbox_metadata_equality": True,
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "tensor_c_contiguous": bool(tensor.flags.c_contiguous),
        "tensor_all_finite": bool(np.isfinite(tensor).all()),
        "letterbox_scale": scale,
        "pad_left": left,
        "pad_top": top,
    }


def profile_preprocessing(
    image: np.ndarray,
    *,
    input_size: int = INPUT_SIZE,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    measured_runs: int = DEFAULT_MEASURED_RUNS,
    trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    if warmup_runs < 1 or measured_runs < 1 or trials < 1:
        raise ValueError("warmup_runs, measured_runs, and trials must be positive.")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected a decoded uint8 BGR image with shape (height, width, 3).")

    equivalence = _verify_preprocessing_equivalence(image, input_size)
    raw_samples: list[dict[str, Any]] = []
    stage_samples = {name: [] for name in _STAGE_NAMES}
    total_samples: list[float] = []
    instrumented_wall_samples: list[float] = []
    trial_samples: list[dict[str, Any]] = []

    for trial_index in range(trials):
        for iteration in range(warmup_runs):
            _profile_stages(image, input_size)
            prepare_onnx_input(image, input_size)

        trial_rows = []
        for iteration in range(measured_runs):
            sample_timestamp = datetime.now(timezone.utc).isoformat()
            if (trial_index * measured_runs + iteration) % 2 == 0:
                profiled = _profile_stages(image, input_size)
                total_start = perf_counter_ns()
                prepare_onnx_input(image, input_size)
                total_ms = (perf_counter_ns() - total_start) / 1_000_000
                order = "stages_then_total"
            else:
                total_start = perf_counter_ns()
                prepare_onnx_input(image, input_size)
                total_ms = (perf_counter_ns() - total_start) / 1_000_000
                profiled = _profile_stages(image, input_size)
                order = "total_then_stages"

            _, _, _, _, stage_ms, stage_timestamps, pipeline_ms = profiled
            for name in _STAGE_NAMES:
                stage_samples[name].append(stage_ms[name])
            total_samples.append(total_ms)
            instrumented_wall_samples.append(pipeline_ms)
            row = {
                "trial": trial_index + 1,
                "iteration": iteration + 1,
                "timestamp_utc": sample_timestamp,
                "measurement_order": order,
                "stage_latencies_ms": stage_ms,
                "stage_monotonic_ns": {
                    name: {"start": values[0], "end": values[1]}
                    for name, values in stage_timestamps.items()
                },
                "sum_stage_latencies_ms": float(sum(stage_ms.values())),
                "instrumented_pipeline_wall_ms": pipeline_ms,
                "prepare_onnx_input_total_ms": total_ms,
                "total_minus_stage_sum_ms": float(total_ms - sum(stage_ms.values())),
            }
            raw_samples.append(row)
            trial_rows.append(row)

        trial_samples.append(
            {
                "trial": trial_index + 1,
                "stage_statistics_ms": {
                    name: _latency_statistics([row["stage_latencies_ms"][name] for row in trial_rows])
                    for name in _STAGE_NAMES
                },
                "prepare_onnx_input_total_ms": _latency_statistics(
                    [row["prepare_onnx_input_total_ms"] for row in trial_rows]
                ),
                "sum_stage_latencies_ms": _latency_statistics(
                    [row["sum_stage_latencies_ms"] for row in trial_rows]
                ),
            }
        )

    stage_statistics = {
        name: _latency_statistics(values) for name, values in stage_samples.items()
    }
    stage_mean_sum = float(sum(stats["mean_ms"] for stats in stage_statistics.values()))
    total_statistics = _latency_statistics(total_samples)
    instrumented_statistics = _latency_statistics(instrumented_wall_samples)
    total_mean = total_statistics["mean_ms"]
    return {
        "status": "completed",
        "input_contract": equivalence,
        "sample_count": len(raw_samples),
        "stage_statistics_ms": stage_statistics,
        "sum_of_stage_means_ms": stage_mean_sum,
        "prepare_onnx_input_total_statistics_ms": total_statistics,
        "instrumented_pipeline_wall_statistics_ms": instrumented_statistics,
        "mean_total_minus_sum_of_stage_means_ms": float(total_mean - stage_mean_sum),
        "mean_total_minus_sum_percentage": (
            float((total_mean - stage_mean_sum) / total_mean * 100.0)
            if total_mean
            else 0.0
        ),
        "trial_summaries": trial_samples,
        "raw_samples": raw_samples,
    }


def _environment() -> dict[str, str]:
    import onnxruntime
    import torch

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_bits": str(sys.maxsize.bit_length() + 1),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pytorch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "execution_scope": "CPU preprocessing only; no model or inference session is initialized",
    }


def _timestamped_output(output_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return output_dir / f"phase17b_onnx_preprocessing_{timestamp}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--onnx-model", type=Path, default=DEFAULT_ONNX_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=DEFAULT_MEASURED_RUNS)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    args = parser.parse_args(argv)

    image_path = args.image.expanduser().resolve()
    model_path = args.onnx_model.expanduser().resolve()
    if not image_path.is_file():
        parser.error(f"Image file does not exist: {image_path}")
    if not model_path.is_file():
        parser.error(f"ONNX model file does not exist: {model_path}")
    output_path = (args.output or _timestamped_output(Path("outputs"))).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        parser.error(f"Refusing to overwrite existing output: {output_path}")

    image, image_load_ms = load_image(image_path)
    profile = profile_preprocessing(
        image,
        input_size=INPUT_SIZE,
        warmup_runs=args.warmup_runs,
        measured_runs=args.runs,
        trials=args.trials,
    )
    manifest_path = model_path.with_suffix(".json")
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = {
        "benchmark_type": "phase17b_onnx_preprocessing_cpu_stage_profile",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": profile["status"],
        "configuration": {
            "image_path": str(image_path),
            "image_sha256": _sha256_file(image_path),
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "image_decode_ms": image_load_ms,
            "onnx_model_path": str(model_path),
            "onnx_model_sha256": _sha256_file(model_path),
            "onnx_manifest_path": str(manifest_path) if manifest else None,
            "onnx_manifest": manifest,
            "input_size": [INPUT_SIZE, INPUT_SIZE],
            "batch_size": 1,
            "precision": "FP32",
            "warmup_runs_per_trial": args.warmup_runs,
            "measured_runs_per_trial": args.runs,
            "trials": args.trials,
            "percentile_method": "NumPy percentile with its default linear interpolation",
        },
        "environment": _environment(),
        "methodology": {
            "preprocessing_implementation": "Independent timed decomposition of the existing edgeai.onnx_detector.prepare_onnx_input operations; production source is unchanged.",
            "stage_order": list(_STAGE_NAMES),
            "total_latency": "perf_counter_ns wall time around prepare_onnx_input, measured separately from stage timings.",
            "stage_latency": "perf_counter_ns around each corresponding operation in the decomposed preprocessing path.",
            "measurement_order": "Stage-profiled and whole-function measurements alternate order by sample.",
            "input_reuse": "The same decoded BGR uint8 image array is reused for every warm-up and measurement.",
            "model_execution": "Not performed; no ONNX Runtime session or model inference is initialized.",
        },
        "preprocessing": profile,
        "output_path": str(output_path.resolve()),
    }
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Preprocessing profile JSON: {output_path.resolve()}")
    print(f"Measured calls: {profile['sample_count']}")
    print(
        "Total mean: "
        f"{profile['prepare_onnx_input_total_statistics_ms']['mean_ms']:.6f} ms; "
        f"sum of stage means: {profile['sum_of_stage_means_ms']:.6f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
