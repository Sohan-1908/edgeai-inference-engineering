"""Profile ONNX output decoding and NMS using a captured model output."""

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

import numpy as np

from edgeai.errors import DeviceUnavailableError
from edgeai.detector_types import Detection
from edgeai.image_io import load_image
from edgeai.onnx_detector import (
    DEFAULT_IOU_THRESHOLD,
    INPUT_SIZE,
    MAX_DETECTIONS,
    OnnxDetector,
    _box_iou,
    _decode_yolo_output,
    prepare_onnx_input,
)

DEFAULT_IMAGE = Path("outputs/smoke_input.jpg")
DEFAULT_ONNX_MODEL = Path("models/yolo11n.onnx")
DEFAULT_WARMUP_RUNS = 10
DEFAULT_MEASURED_RUNS = 30
DEFAULT_TRIALS = 3

_STAGE_NAMES = (
    "output_conversion_shape_validation_orientation",
    "class_score_selection",
    "confidence_filtering",
    "box_conversion",
    "class_aware_greedy_nms",
    "maximum_detection_cap",
    "coordinate_rescaling_clipping_and_detection_construction",
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


def _profile_decode(
    output: np.ndarray,
    *,
    scale: float,
    pad_left: int,
    pad_top: int,
    image_width: int,
    image_height: int,
    names: dict[int, str],
    confidence: float,
    iou_threshold: float,
) -> tuple[list[Any], dict[str, float], float]:
    timings: dict[str, float] = {}
    pipeline_start = perf_counter_ns()

    start = perf_counter_ns()
    prediction = np.asarray(output, dtype=np.float32)
    if prediction.ndim == 3:
        if prediction.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got output shape {prediction.shape}")
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError(f"Expected a rank-2 YOLO output after removing batch, got {prediction.shape}")
    feature_count = 4 + len(names) if names else None
    if feature_count is not None and prediction.shape[0] == feature_count:
        prediction = prediction.T
    elif feature_count is None and prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.T
    if prediction.shape[1] < 5:
        raise ValueError(f"YOLO output has too few channels: {prediction.shape}")
    timings[_STAGE_NAMES[0]] = (perf_counter_ns() - start) / 1_000_000

    start = perf_counter_ns()
    class_scores = prediction[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]
    timings[_STAGE_NAMES[1]] = (perf_counter_ns() - start) / 1_000_000

    start = perf_counter_ns()
    keep = np.flatnonzero(scores >= confidence)
    if keep.size == 0:
        timings[_STAGE_NAMES[2]] = (perf_counter_ns() - start) / 1_000_000
        for name in _STAGE_NAMES[3:]:
            timings[name] = 0.0
        return [], timings, (perf_counter_ns() - pipeline_start) / 1_000_000
    prediction = prediction[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    timings[_STAGE_NAMES[2]] = (perf_counter_ns() - start) / 1_000_000

    start = perf_counter_ns()
    centers = prediction[:, :2]
    sizes = prediction[:, 2:4]
    boxes = np.column_stack((centers - sizes / 2, centers + sizes / 2))
    timings[_STAGE_NAMES[3]] = (perf_counter_ns() - start) / 1_000_000

    start = perf_counter_ns()
    selected = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        order = indices[np.argsort(scores[indices])[::-1]]
        while order.size:
            current = int(order[0])
            selected.append(current)
            rest = order[1:]
            if not rest.size:
                break
            order = rest[_box_iou(boxes[current], boxes[rest]) <= iou_threshold]
    timings[_STAGE_NAMES[4]] = (perf_counter_ns() - start) / 1_000_000

    start = perf_counter_ns()
    if len(selected) > MAX_DETECTIONS:
        selected = sorted(selected, key=lambda index: float(scores[index]), reverse=True)[:MAX_DETECTIONS]
    timings[_STAGE_NAMES[5]] = (perf_counter_ns() - start) / 1_000_000

    start = perf_counter_ns()
    detections = []
    for index in selected:
        x1, y1, x2, y2 = boxes[index]
        x1 = np.clip((x1 - pad_left) / scale, 0, image_width)
        y1 = np.clip((y1 - pad_top) / scale, 0, image_height)
        x2 = np.clip((x2 - pad_left) / scale, 0, image_width)
        y2 = np.clip((y2 - pad_top) / scale, 0, image_height)
        class_id = int(class_ids[index])
        detections.append(
            Detection(
                class_id=class_id,
                class_name=names.get(class_id, str(class_id)),
                confidence=float(scores[index]),
                bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
            )
        )
    timings[_STAGE_NAMES[6]] = (perf_counter_ns() - start) / 1_000_000
    return detections, timings, (perf_counter_ns() - pipeline_start) / 1_000_000


def _call_decode(
    output: np.ndarray,
    *,
    scale: float,
    pad_left: int,
    pad_top: int,
    image_width: int,
    image_height: int,
    names: dict[int, str],
    confidence: float,
    iou_threshold: float,
) -> list[Any]:
    return _decode_yolo_output(
        output,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        image_width=image_width,
        image_height=image_height,
        names=names,
        confidence=confidence,
        iou_threshold=iou_threshold,
    )


def profile_postprocessing(
    output: np.ndarray,
    *,
    scale: float,
    pad_left: int,
    pad_top: int,
    image_width: int,
    image_height: int,
    names: dict[int, str],
    confidence: float = 0.25,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    measured_runs: int = DEFAULT_MEASURED_RUNS,
    trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    if warmup_runs < 1 or measured_runs < 1 or trials < 1:
        raise ValueError("warmup_runs, measured_runs, and trials must be positive.")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0.0 and 1.0")

    kwargs = {
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "image_width": image_width,
        "image_height": image_height,
        "names": names,
        "confidence": confidence,
        "iou_threshold": iou_threshold,
    }
    reference = _call_decode(output, **kwargs)
    reference_count = len(reference)
    stage_samples = {name: [] for name in _STAGE_NAMES}
    total_samples: list[float] = []
    instrumented_wall_samples: list[float] = []
    raw_samples: list[dict[str, Any]] = []
    trial_summaries: list[dict[str, Any]] = []
    consistency_checks = 0

    for trial_index in range(trials):
        for _ in range(warmup_runs):
            staged, _, _ = _profile_decode(output, **kwargs)
            whole = _call_decode(output, **kwargs)
            if staged != reference or whole != reference:
                raise AssertionError("Warm-up output differs from the reference detections.")

        trial_rows = []
        for iteration in range(measured_runs):
            timestamp = datetime.now(timezone.utc).isoformat()
            if (trial_index * measured_runs + iteration) % 2 == 0:
                staged, stage_ms, instrumented_wall_ms = _profile_decode(output, **kwargs)
                start = perf_counter_ns()
                whole = _call_decode(output, **kwargs)
                total_ms = (perf_counter_ns() - start) / 1_000_000
                order = "stages_then_whole"
            else:
                start = perf_counter_ns()
                whole = _call_decode(output, **kwargs)
                total_ms = (perf_counter_ns() - start) / 1_000_000
                staged, stage_ms, instrumented_wall_ms = _profile_decode(output, **kwargs)
                order = "whole_then_stages"

            if staged != reference or whole != reference:
                raise AssertionError("Profiled output differs from the reference detections.")
            consistency_checks += 2
            for name in _STAGE_NAMES:
                stage_samples[name].append(stage_ms[name])
            total_samples.append(total_ms)
            instrumented_wall_samples.append(instrumented_wall_ms)
            row = {
                "trial": trial_index + 1,
                "iteration": iteration + 1,
                "timestamp_utc": timestamp,
                "measurement_order": order,
                "stage_latencies_ms": stage_ms,
                "stage_sum_ms": float(sum(stage_ms.values())),
                "instrumented_pipeline_wall_ms": instrumented_wall_ms,
                "whole_function_latency_ms": total_ms,
                "whole_minus_stage_sum_ms": float(total_ms - sum(stage_ms.values())),
                "reference_detection_count": reference_count,
                "profiled_detection_count": len(staged),
                "whole_function_detection_count": len(whole),
            }
            raw_samples.append(row)
            trial_rows.append(row)

        trial_summaries.append(
            {
                "trial": trial_index + 1,
                "stage_statistics_ms": {
                    name: _latency_statistics(
                        [row["stage_latencies_ms"][name] for row in trial_rows]
                    )
                    for name in _STAGE_NAMES
                },
                "whole_function_statistics_ms": _latency_statistics(
                    [row["whole_function_latency_ms"] for row in trial_rows]
                ),
                "stage_sum_statistics_ms": _latency_statistics(
                    [row["stage_sum_ms"] for row in trial_rows]
                ),
            }
        )

    stage_statistics = {
        name: _latency_statistics(values) for name, values in stage_samples.items()
    }
    total_statistics = _latency_statistics(total_samples)
    instrumented_statistics = _latency_statistics(instrumented_wall_samples)
    stage_mean_sum = float(sum(stats["mean_ms"] for stats in stage_statistics.values()))
    total_mean = total_statistics["mean_ms"]
    return {
        "sample_count": len(raw_samples),
        "stage_names": list(_STAGE_NAMES),
        "stage_statistics_ms": stage_statistics,
        "sum_of_stage_means_ms": stage_mean_sum,
        "whole_function_statistics_ms": total_statistics,
        "instrumented_pipeline_wall_statistics_ms": instrumented_statistics,
        "mean_whole_minus_stage_sum_ms": float(total_mean - stage_mean_sum),
        "mean_whole_minus_stage_sum_percentage": (
            float((total_mean - stage_mean_sum) / total_mean * 100.0)
            if total_mean
            else 0.0
        ),
        "reference_detection_count": reference_count,
        "outputs_consistent": True,
        "measured_output_consistency_checks": consistency_checks,
        "trial_summaries": trial_summaries,
        "raw_samples": raw_samples,
    }


def _environment() -> dict[str, str]:
    import cv2
    import onnxruntime

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_bits": str(sys.maxsize.bit_length() + 1),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "onnxruntime": onnxruntime.__version__,
    }


def _timestamped_output(output_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return output_dir / f"phase17c_postprocessing_{timestamp}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--onnx-model", type=Path, default=DEFAULT_ONNX_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=DEFAULT_MEASURED_RUNS)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
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

    image, decode_ms = load_image(image_path)
    detector = OnnxDetector(model_path, device="cuda", precision="fp32", require_cuda=True)
    if (
        detector.device != "cuda:0"
        or detector.execution_provider != "CUDAExecutionProvider"
        or not detector.require_cuda
    ):
        raise DeviceUnavailableError("Phase 17C requires strict ONNX CUDA execution.")

    input_tensor, scale, pad_left, pad_top = prepare_onnx_input(image, INPUT_SIZE)
    inference_start = perf_counter_ns()
    model_outputs = detector.session.run(
        detector.output_names,
        {detector.input.name: input_tensor},
    )
    output_capture_ms = (perf_counter_ns() - inference_start) / 1_000_000
    if not model_outputs:
        raise RuntimeError("ONNX Runtime returned no outputs.")
    raw_output = model_outputs[0]
    profile = profile_postprocessing(
        raw_output,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        image_width=int(image.shape[1]),
        image_height=int(image.shape[0]),
        names=detector.names,
        confidence=args.confidence,
        iou_threshold=args.iou_threshold,
        warmup_runs=args.warmup_runs,
        measured_runs=args.runs,
        trials=args.trials,
    )

    manifest_path = model_path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    report = {
        "benchmark_type": "phase17c_onnx_postprocessing_stage_profile",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "configuration": {
            "image_path": str(image_path),
            "image_sha256": _sha256_file(image_path),
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "image_decode_ms": decode_ms,
            "onnx_model_path": str(model_path),
            "onnx_model_sha256": _sha256_file(model_path),
            "onnx_manifest": manifest,
            "input_size": [INPUT_SIZE, INPUT_SIZE],
            "batch_size": 1,
            "precision": "FP32",
            "confidence_threshold": args.confidence,
            "iou_threshold": args.iou_threshold,
            "warmup_runs_per_trial": args.warmup_runs,
            "measured_runs_per_trial": args.runs,
            "trials": args.trials,
        },
        "runtime": {
            "device": detector.device,
            "execution_provider": detector.execution_provider,
            "cpu_fallback_disabled": detector.require_cuda,
            "captured_output_shape": list(raw_output.shape),
            "captured_output_dtype": str(raw_output.dtype),
            "capture_inference_latency_ms_excluded_from_profile": output_capture_ms,
        },
        "environment": _environment(),
        "methodology": {
            "path": "Captured ONNX Runtime output -> _decode_yolo_output equivalent stages -> Detection list.",
            "model_execution": "One strict CUDA session.run captured the representative output; this call is excluded from all postprocessing timings.",
            "postprocessing_input": "The same read-only captured model output is reused for every warm-up and measurement.",
            "whole_function": "perf_counter_ns around the existing production _decode_yolo_output function.",
            "stage_decomposition": "Separately timed decomposition of the exact operations in _decode_yolo_output; no production code is patched.",
            "measurement_order": "Stage-decomposed and whole-function measurements alternate order by sample.",
            "percentile_method": "NumPy percentile with its default linear interpolation.",
        },
        "postprocessing": profile,
        "output_path": str(output_path.resolve()),
    }
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Postprocessing profile JSON: {output_path.resolve()}")
    print(f"Measured calls: {profile['sample_count']}")
    print(f"Detection count: {profile['reference_detection_count']}")
    print(
        "Whole-function mean: "
        f"{profile['whole_function_statistics_ms']['mean_ms']:.6f} ms; "
        f"sum of stage means: {profile['sum_of_stage_means_ms']:.6f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
