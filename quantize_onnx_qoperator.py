"""Run an isolated static QOperator INT8 experiment for YOLO11n on CPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from analyze_ort_profile import load_trace
from diagnose_onnx_qdq import inspect_model
from edgeai.benchmark import _detection_matching
from edgeai.image_io import load_image
from edgeai.onnx_detector import DEFAULT_ONNX_PATH, OnnxDetector
from edgeai.onnx_int8 import DEFAULT_INT8_ONNX_PATH, ImageCalibrationReader

ROOT = Path(__file__).resolve().parent
DEFAULT_QDQ_METADATA = ROOT / "models" / "yolo11n_int8.json"
DEFAULT_EXPERIMENT_MODEL_DIR = ROOT / "models" / "experiments"
DEFAULT_EXPERIMENT_OUTPUT_DIR = ROOT / "outputs" / "experiments" / "qoperator"
DEFAULT_BENCHMARK_MANIFEST = ROOT / "outputs" / "phase7b_cpu_benchmark.json"
DEFAULT_MODEL_NAME = "yolo11n_qoperator_int8.onnx"
CPU_PROVIDER = "CPUExecutionProvider"
CPU_THREADS = 4
INTER_OP_THREADS = 1
WARMUPS_PER_TRIAL = 10
MEASURED_RUNS_PER_TRIAL = 50
TRIAL_COUNT = 3
BENCHMARK_CONFIDENCE = 0.25
DETECTION_MATCH_IOU = 0.5
DETECTOR_NMS_IOU = 0.7
QUANTIZED_OPERATOR_TYPES = {
    "QLinearConv",
    "ConvInteger",
    "QLinearMatMul",
    "MatMulInteger",
    "MatMulIntegerToFloat",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantization_api() -> tuple[Any, Any, Any, Any]:
    try:
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Static ONNX Runtime quantization is unavailable in this environment. "
            "Install a compatible onnxruntime quantization package before running "
            "the experiment; no quantization fallback was attempted."
        ) from exc
    return quantize_static, QuantFormat, QuantType, CalibrationMethod


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {description} {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return data


def load_experiment_inputs(
    *,
    fp32_model: str | Path = DEFAULT_ONNX_PATH,
    qdq_model: str | Path = DEFAULT_INT8_ONNX_PATH,
    qdq_metadata: str | Path = DEFAULT_QDQ_METADATA,
    benchmark_manifest: str | Path = DEFAULT_BENCHMARK_MANIFEST,
) -> dict[str, Any]:
    """Load exact calibration and benchmark manifests from the established runs."""
    fp32_path = Path(fp32_model).expanduser().resolve()
    qdq_path = Path(qdq_model).expanduser().resolve()
    metadata_path = Path(qdq_metadata).expanduser().resolve()
    benchmark_path = Path(benchmark_manifest).expanduser().resolve()
    for path in (fp32_path, qdq_path, metadata_path, benchmark_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment input does not exist: {path}")

    qdq_report = _read_json_object(metadata_path, description="QDQ metadata")
    calibration = qdq_report.get("calibration")
    dataset = calibration.get("dataset") if isinstance(calibration, dict) else None
    calibration_names = (
        dataset.get("calibration_image_paths") if isinstance(dataset, dict) else None
    )
    selected_nodes = qdq_report.get("graph_analysis", {}).get(
        "quantized_conv_node_names"
    )
    excluded_nodes = qdq_report.get("graph_analysis", {}).get(
        "float_only_conv_node_names"
    )
    if not isinstance(calibration_names, list) or not calibration_names:
        raise ValueError(
            f"QDQ metadata does not list calibration_image_paths: {metadata_path}"
        )
    if not isinstance(selected_nodes, list) or not selected_nodes:
        raise ValueError(
            f"QDQ metadata does not list selected Conv nodes: {metadata_path}"
        )
    if not isinstance(excluded_nodes, list):
        raise ValueError(
            f"QDQ metadata does not list excluded Conv nodes: {metadata_path}"
        )

    source_hash = _sha256(fp32_path)
    qdq_hash = _sha256(qdq_path)
    expected_source_hash = qdq_report.get("source_fp32_sha256")
    if expected_source_hash and expected_source_hash != source_hash:
        raise ValueError(
            "QDQ calibration metadata references a different FP32 model hash; "
            "refusing to mix calibration policies or source artifacts."
        )

    calibration_paths = [Path(path).expanduser().resolve() for path in calibration_names]
    for path in calibration_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Calibration image does not exist: {path}")
    calibration_count = calibration.get("image_count")
    if calibration_count != len(calibration_paths):
        raise ValueError(
            "Calibration image count in QDQ metadata does not match its image manifest."
        )

    benchmark = _read_json_object(benchmark_path, description="benchmark manifest")
    benchmark_names = benchmark.get("benchmark_images")
    if not isinstance(benchmark_names, list) or len(benchmark_names) != 4:
        raise ValueError(
            f"Benchmark manifest must list exactly four benchmark_images: {benchmark_path}"
        )
    benchmark_paths = [Path(path).expanduser().resolve() for path in benchmark_names]
    for path in benchmark_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark image does not exist: {path}")

    model = onnx.load(str(fp32_path), load_external_data=True)
    onnx.checker.check_model(model)
    source_conv_names = [
        node.name for node in model.graph.node if node.op_type == "Conv"
    ]
    source_conv_name_set = set(source_conv_names)
    selected_set = set(selected_nodes)
    excluded_set = set(excluded_nodes)
    if not selected_set.issubset(source_conv_name_set):
        raise ValueError("QDQ metadata selects Conv nodes absent from the source model.")
    if not excluded_set.issubset(source_conv_name_set):
        raise ValueError("QDQ metadata excludes Conv nodes absent from the source model.")
    if selected_set & excluded_set or selected_set | excluded_set != source_conv_name_set:
        raise ValueError(
            "QDQ selected and excluded Conv-node lists do not partition source Conv nodes."
        )

    return {
        "fp32_model": fp32_path,
        "qdq_model": qdq_path,
        "qdq_metadata_path": metadata_path,
        "benchmark_manifest_path": benchmark_path,
        "source_fp32_sha256": source_hash,
        "qdq_model_sha256": qdq_hash,
        "calibration_paths": calibration_paths,
        "selected_conv_names": selected_nodes,
        "excluded_conv_names": excluded_nodes,
        "source_conv_names": source_conv_names,
        "benchmark_paths": benchmark_paths,
        "historical_benchmark": {
            "source": str(benchmark_path),
            "onnxruntime_version": benchmark.get("hardware_software", {}).get(
                "onnxruntime"
            ),
            "cpu_threads": benchmark.get("hardware_software", {}).get("cpu_threads"),
            "mean_latency_ms": {
                name: benchmark.get("configurations", {})
                .get(name, {})
                .get("mean_latency_ms")
                for name in ("FP32", "INT8")
            },
        },
        "qdq_calibration_metadata": calibration,
    }


def _validate_experiment_output_path(
    output_model: str | Path,
    *,
    fp32_model: str | Path,
    qdq_model: str | Path,
    experiment_model_dir: str | Path = DEFAULT_EXPERIMENT_MODEL_DIR,
) -> Path:
    destination = Path(output_model).expanduser().resolve()
    experiment_root = Path(experiment_model_dir).expanduser().resolve()
    if destination.suffix.lower() != ".onnx":
        raise ValueError("QOperator experiment output must use the .onnx extension.")
    if destination in {
        Path(fp32_model).expanduser().resolve(),
        Path(qdq_model).expanduser().resolve(),
    }:
        raise ValueError("QOperator output cannot overwrite the FP32 or existing QDQ model.")
    if not destination.is_relative_to(experiment_root):
        raise ValueError(
            f"QOperator output must remain inside the experiment model directory "
            f"{experiment_root}; got {destination}."
        )
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing experiment model: {destination}"
        )
    return destination


def _graph_quantized_operator_names(model: onnx.ModelProto) -> list[str]:
    return [
        node.name or f"{node.op_type}_{index}"
        for index, node in enumerate(model.graph.node)
        if node.op_type in QUANTIZED_OPERATOR_TYPES
    ]


def inspect_qoperator_graph(
    model_path: str | Path, *, model_name: str | None = None
) -> dict[str, Any]:
    path = Path(model_path).expanduser().resolve()
    model = onnx.load(str(path), load_external_data=True)
    summary = inspect_model(model, model_name=model_name or path.name)
    counts = summary["operator_counts"]
    qlinear_nodes = [node for node in model.graph.node if node.op_type == "QLinearConv"]
    quantized_names = _graph_quantized_operator_names(model)
    summary["qoperator_diagnostics"] = {
        "conv_node_count": counts.get("Conv", 0),
        "qlinear_conv_node_count": counts.get("QLinearConv", 0),
        "conv_integer_node_count": counts.get("ConvInteger", 0),
        "quantize_linear_node_count": counts.get("QuantizeLinear", 0),
        "dequantize_linear_node_count": counts.get("DequantizeLinear", 0),
        "quantized_operator_names": quantized_names,
        "qlinear_conv_names": [node.name for node in qlinear_nodes],
        "remaining_float_conv_node_names": [
            node.name for node in model.graph.node if node.op_type == "Conv"
        ],
        "quantized_convolution_survives_graph": bool(qlinear_nodes)
        or counts.get("ConvInteger", 0) > 0,
    }
    return summary


def _cpu_session_smoke(
    model_path: str | Path,
    input_feed: dict[str, np.ndarray],
    *,
    cpu_threads: int = CPU_THREADS,
) -> dict[str, Any]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = cpu_threads
    options.inter_op_num_threads = INTER_OP_THREADS
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=[CPU_PROVIDER],
    )
    providers = session.get_providers()
    if not providers or providers[0] != CPU_PROVIDER:
        raise RuntimeError(
            f"Expected {CPU_PROVIDER} for validation, got {providers!r}."
        )
    outputs = session.run(None, input_feed)
    if not outputs:
        raise RuntimeError("CPU smoke inference returned no model outputs.")
    return {
        "execution_provider": providers[0],
        "execution_succeeded": True,
        "input_shapes": {name: list(value.shape) for name, value in input_feed.items()},
        "output_shapes": [list(value.shape) for value in outputs],
        "output_dtypes": [str(value.dtype) for value in outputs],
    }


def quantize_qoperator_model(
    *,
    fp32_model: str | Path,
    qdq_model: str | Path,
    output_model: str | Path,
    calibration_paths: list[str | Path],
    selected_conv_names: list[str],
    excluded_conv_names: list[str],
    calibration_data_reader: Any,
    experiment_model_dir: str | Path = DEFAULT_EXPERIMENT_MODEL_DIR,
    cpu_threads: int = CPU_THREADS,
) -> dict[str, Any]:
    """Generate and validate one isolated QOperator model; never fall back formats."""
    source = Path(fp32_model).expanduser().resolve()
    qdq = Path(qdq_model).expanduser().resolve()
    destination = _validate_experiment_output_path(
        output_model,
        fp32_model=source,
        qdq_model=qdq,
        experiment_model_dir=experiment_model_dir,
    )
    if not calibration_paths:
        raise ValueError("At least one calibration image is required.")
    if not selected_conv_names:
        raise ValueError("At least one supported Conv node must be selected.")
    source_hash = _sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    quantize_static, quant_format, quant_type, calibration_method = _quantization_api()
    started = perf_counter()
    try:
        quantize_static(
            model_input=source,
            model_output=destination,
            calibration_data_reader=calibration_data_reader,
            quant_format=quant_format.QOperator,
            op_types_to_quantize=["Conv"],
            nodes_to_quantize=selected_conv_names,
            nodes_to_exclude=excluded_conv_names,
            per_channel=True,
            reduce_range=False,
            activation_type=quant_type.QInt8,
            weight_type=quant_type.QInt8,
            calibrate_method=calibration_method.MinMax,
            calibration_providers=[CPU_PROVIDER],
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
            },
        )
        quantization_ms = (perf_counter() - started) * 1000.0
        if not destination.is_file():
            raise FileNotFoundError(
                f"ONNX Runtime quantization did not create its output model: {destination}"
            )
        quantized_model = onnx.load(str(destination), load_external_data=True)
        onnx.checker.check_model(quantized_model)
        graph_summary = inspect_qoperator_graph(destination)
        graph = graph_summary["qoperator_diagnostics"]
        qlinear_count = graph["qlinear_conv_node_count"]
        if qlinear_count == 0:
            raise RuntimeError(
                "QOperator conversion produced no QLinearConv nodes; refusing to "
                "label an unchanged or float-only model as an INT8 experiment."
            )

        calibration_data_reader.rewind()
        input_feed = calibration_data_reader.get_next()
        if not input_feed:
            raise RuntimeError("Calibration reader returned no input for CPU smoke inference.")
        validation = _cpu_session_smoke(
            destination, input_feed, cpu_threads=cpu_threads
        )
        if _sha256(source) != source_hash:
            raise RuntimeError("Source FP32 model hash changed during QOperator quantization.")

        source_names = set(
            node.name
            for node in onnx.load(str(source), load_external_data=True).graph.node
            if node.op_type == "Conv"
        )
        quantized_names = {
            name[:-6] if name.endswith("_quant") else name
            for name in graph["qlinear_conv_names"]
            if name
        }
        quantized_source_names = sorted(source_names & quantized_names)
        unconverted_selected = sorted(
            set(selected_conv_names) - set(quantized_source_names)
        )
        intended_excluded = sorted(set(excluded_conv_names))
        validation["onnx_checker"] = "passed"
        validation["graph_contains_qlinear_conv"] = True
        validation["quantized_conv_node_count"] = qlinear_count
        validation["quantized_conv_source_names"] = quantized_source_names

        return {
            "model_path": str(destination),
            "source_fp32_sha256": source_hash,
            "output_model_sha256": _sha256(destination),
            "quantization_duration_ms": quantization_ms,
            "quantization_configuration": {
                "format": "QOperator",
                "activation_type": "QInt8",
                "weight_type": "QInt8",
                "calibration_method": "MinMax",
                "calibration_provider": CPU_PROVIDER,
                "per_channel_weights": True,
                "symmetric_activations": True,
                "symmetric_weights": True,
                "reduce_range": False,
                "cpu_validation_threads": {
                    "intra_op": cpu_threads,
                    "inter_op": INTER_OP_THREADS,
                },
            },
            "calibration_image_count": len(calibration_paths),
            "calibration_image_paths": [
                str(Path(path).expanduser().resolve()) for path in calibration_paths
            ],
            "selected_conv_node_count": len(selected_conv_names),
            "quantized_conv_node_count": qlinear_count,
            "quantized_conv_node_names": quantized_source_names,
            "unsupported_or_unconverted_selected_conv_count": len(
                unconverted_selected
            ),
            "unsupported_or_unconverted_selected_conv_names": unconverted_selected,
            "deliberately_excluded_conv_node_count": len(intended_excluded),
            "deliberately_excluded_conv_node_names": intended_excluded,
            "generated_graph": graph,
            "validation": validation,
        }
    except Exception:
        if destination.is_file():
            destination.unlink()
        raise


def _optimized_graph(
    model_path: Path, output_path: Path, *, cpu_threads: int
) -> dict[str, Any]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(output_path)
    options.intra_op_num_threads = cpu_threads
    options.inter_op_num_threads = INTER_OP_THREADS
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=[CPU_PROVIDER],
    )
    providers = session.get_providers()
    if not providers or providers[0] != CPU_PROVIDER:
        raise RuntimeError(f"Expected {CPU_PROVIDER}; got {providers!r}.")
    del session
    gc.collect()
    if not output_path.is_file():
        raise FileNotFoundError(f"ORT did not save its optimized graph: {output_path}")
    return {
        "path": str(output_path.resolve()),
        "size_bytes": output_path.stat().st_size,
        "execution_provider": providers[0],
        "graph": inspect_qoperator_graph(output_path),
    }


def _aggregate_accuracy(
    reference: list[Any], candidate: list[Any], *, image_paths: list[str]
) -> dict[str, Any]:
    per_image = []
    for label, fp32_detections, variant_detections in zip(
        image_paths, reference, candidate
    ):
        per_image.append(
            {
                "image": label,
                **_detection_matching(
                    fp32_detections,
                    variant_detections,
                    iou_threshold=DETECTION_MATCH_IOU,
                ),
            }
        )
    matched = sum(item["matched_count"] for item in per_image)
    reference_count = sum(item["reference_detection_count"] for item in per_image)
    candidate_count = sum(item["candidate_detection_count"] for item in per_image)

    def matched_weighted_mean(key: str) -> float | None:
        values = [
            (item[key], item["matched_count"])
            for item in per_image
            if item[key] is not None and item["matched_count"]
        ]
        if not values:
            return None
        return sum(value * count for value, count in values) / sum(
            count for _, count in values
        )

    return {
        "reference_model": "FP32",
        "candidate_detection_count": candidate_count,
        "reference_detection_count": reference_count,
        "matched_count": matched,
        "missed_reference_count": sum(
            item["missed_reference_count"] for item in per_image
        ),
        "additional_candidate_count": sum(
            item["additional_candidate_count"] for item in per_image
        ),
        "class_agreement_rate": matched / max(candidate_count, 1),
        "mean_matched_iou": matched_weighted_mean("mean_matched_iou"),
        "mean_absolute_confidence_difference": matched_weighted_mean(
            "mean_absolute_confidence_difference"
        ),
        "mean_absolute_box_coordinate_difference_px": matched_weighted_mean(
            "mean_absolute_box_coordinate_difference_px"
        ),
        "matching_method": "Greedy same-class matching by IoU >= 0.5, matching the existing INT8 benchmark.",
        "per_image": per_image,
        "limitation": "Four benchmark images sampled from one video are limited validation, not general accuracy certification.",
    }


def _benchmark_models(
    model_paths: dict[str, Path],
    images: list[np.ndarray],
    image_labels: list[str],
    *,
    cpu_threads: int,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    configurations: dict[str, Any] = {}
    predictions: dict[str, list[Any]] = {}
    for name in ("FP32", "QDQ_INT8", "QOPERATOR_INT8"):
        model_path = model_paths[name]
        precision = "fp32" if name == "FP32" else "int8"
        detector = OnnxDetector(
            model_path,
            device="cpu",
            cpu_threads=cpu_threads,
            precision=precision,
        )
        if detector.execution_provider != CPU_PROVIDER:
            raise RuntimeError(
                f"{name} selected {detector.execution_provider}, expected {CPU_PROVIDER}."
            )
        latency_samples: list[float] = []
        trial_means: list[float] = []
        stage_values: dict[str, list[float]] = {
            "preprocess": [],
            "execution": [],
            "postprocess": [],
        }
        for trial in range(TRIAL_COUNT):
            for index in range(WARMUPS_PER_TRIAL):
                detector.predict_frame(
                    images[index % len(images)],
                    BENCHMARK_CONFIDENCE,
                    iou_threshold=DETECTOR_NMS_IOU,
                )
            trial_samples: list[float] = []
            for index in range(MEASURED_RUNS_PER_TRIAL):
                image_index = (trial * MEASURED_RUNS_PER_TRIAL + index) % len(images)
                _, elapsed_ms = detector.predict_frame(
                    images[image_index],
                    BENCHMARK_CONFIDENCE,
                    iou_threshold=DETECTOR_NMS_IOU,
                )
                trial_samples.append(float(elapsed_ms))
                latency_samples.append(float(elapsed_ms))
                for stage, key in (
                    ("preprocess", "preprocess_ms"),
                    ("execution", "runtime_ms"),
                    ("postprocess", "postprocess_ms"),
                ):
                    stage_values[stage].append(float(detector.last_timings[key]))
            trial_means.append(mean(trial_samples))

        current_predictions = []
        for image in images:
            detector.predict_frame(
                image,
                BENCHMARK_CONFIDENCE,
                iou_threshold=DETECTOR_NMS_IOU,
            )
            current_predictions.append(list(detector.last_detections))
        predictions[name] = current_predictions
        configurations[name] = {
            "model_path": str(model_path.resolve()),
            "model_size_bytes": model_path.stat().st_size,
            "model_size_mb": model_path.stat().st_size / 1_000_000,
            "runtime_version": detector.runtime_version,
            "execution_provider": detector.execution_provider,
            "intra_op_threads": cpu_threads,
            "inter_op_threads": INTER_OP_THREADS,
            "input_size": [detector.input_size, detector.input_size],
            "batch_size": 1,
            "warmups_per_trial": WARMUPS_PER_TRIAL,
            "measured_runs_per_trial": MEASURED_RUNS_PER_TRIAL,
            "trial_count": TRIAL_COUNT,
            "mean_latency_ms": mean(latency_samples),
            "median_latency_ms": median(latency_samples),
            "p95_latency_ms": float(np.percentile(latency_samples, 95)),
            "throughput_images_per_second": 1000.0 / mean(latency_samples),
            "stage_mean_ms": {
                stage: mean(values) for stage, values in stage_values.items()
            },
            "trial_mean_latency_ms": trial_means,
            "raw_latency_samples_ms": latency_samples,
        }
        del detector
        gc.collect()

    accuracy = {
        "QDQ_INT8": _aggregate_accuracy(
            predictions["FP32"], predictions["QDQ_INT8"], image_paths=image_labels
        ),
        "QOPERATOR_INT8": _aggregate_accuracy(
            predictions["FP32"],
            predictions["QOPERATOR_INT8"],
            image_paths=image_labels,
        ),
    }
    return {
        "benchmark_type": "three_way_fp32_qdq_qoperator_onnx_cpu",
        "settings": {
            "execution_provider": CPU_PROVIDER,
            "intra_op_threads": cpu_threads,
            "inter_op_threads": INTER_OP_THREADS,
            "benchmark_image_count": len(images),
            "benchmark_image_paths": image_labels,
            "confidence": BENCHMARK_CONFIDENCE,
            "detector_nms_iou_threshold": DETECTOR_NMS_IOU,
            "accuracy_matching_iou_threshold": DETECTION_MATCH_IOU,
            "warmups_per_trial": WARMUPS_PER_TRIAL,
            "measured_runs_per_trial": MEASURED_RUNS_PER_TRIAL,
            "trials": TRIAL_COUNT,
            "timing_method": "Wall-clock around OnnxDetector.predict_frame; stage metrics read from the detector. Session initialization and separately profiled inference are excluded.",
        },
        "configurations": configurations,
        "accuracy_vs_fp32": accuracy,
    }, predictions


def _profile_model(
    name: str,
    model_path: Path,
    images: list[np.ndarray],
    *,
    profile_dir: Path,
    cpu_threads: int,
) -> dict[str, Any]:
    prefix = profile_dir / name.lower()
    detector = OnnxDetector(
        model_path,
        device="cpu",
        cpu_threads=cpu_threads,
        precision="fp32" if name == "FP32" else "int8",
        profile=True,
        profile_file_prefix=prefix,
    )
    try:
        if detector.execution_provider != CPU_PROVIDER:
            raise RuntimeError(
                f"Profile for {name} selected {detector.execution_provider}, "
                f"expected {CPU_PROVIDER}."
            )
        for image in images:
            detector.predict_frame(
                image,
                BENCHMARK_CONFIDENCE,
                iou_threshold=DETECTOR_NMS_IOU,
            )
    finally:
        trace_name = detector.end_profiling()
        del detector
        gc.collect()
    if not trace_name:
        raise RuntimeError(f"ONNX Runtime did not return a trace for {name}.")
    trace_path = Path(trace_name).resolve()
    if not trace_path.is_file() or trace_path.parent != profile_dir.resolve():
        raise FileNotFoundError(
            f"Expected profiling trace under {profile_dir}, received {trace_path}."
        )
    trace = load_trace(trace_path)
    providers = sorted(
        {
            item["execution_provider"]
            for item in trace["operators"]
            if item["execution_provider"] != "unknown"
        }
    )
    quantized_runtime_ops = [
        item
        for item in trace["operators"]
        if item["operator"] in QUANTIZED_OPERATOR_TYPES
    ]
    return {
        "trace_path": str(trace_path),
        "execution_providers_observed": providers,
        "quantized_operator_events": quantized_runtime_ops,
        "operator_count": len(trace["operators"]),
        "top_operators_by_accumulated_profile_duration": trace["operators"][:20],
        "warning": "Profiled trace totals include profiling overhead and are not comparable to unprofiled end-to-end benchmark latency.",
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2)
        output.write("\n")


def run_qoperator_experiment(
    *,
    fp32_model: str | Path = DEFAULT_ONNX_PATH,
    qdq_model: str | Path = DEFAULT_INT8_ONNX_PATH,
    qdq_metadata: str | Path = DEFAULT_QDQ_METADATA,
    benchmark_manifest: str | Path = DEFAULT_BENCHMARK_MANIFEST,
    output_model: str | Path = DEFAULT_EXPERIMENT_MODEL_DIR / DEFAULT_MODEL_NAME,
    experiment_output_dir: str | Path = DEFAULT_EXPERIMENT_OUTPUT_DIR,
    cpu_threads: int = CPU_THREADS,
) -> dict[str, Any]:
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer.")
    inputs = load_experiment_inputs(
        fp32_model=fp32_model,
        qdq_model=qdq_model,
        qdq_metadata=qdq_metadata,
        benchmark_manifest=benchmark_manifest,
    )
    qoperator_path = _validate_experiment_output_path(
        output_model,
        fp32_model=inputs["fp32_model"],
        qdq_model=inputs["qdq_model"],
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(experiment_output_dir).expanduser().resolve() / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    optimized_dir = run_dir / "optimized_graphs"
    profile_dir = run_dir / "profiles"
    optimized_dir.mkdir()
    profile_dir.mkdir()

    reader = ImageCalibrationReader(
        inputs["calibration_paths"],
        onnx.load(str(inputs["fp32_model"]), load_external_data=True)
        .graph.input[0]
        .name,
    )
    qoperator_metadata = quantize_qoperator_model(
        fp32_model=inputs["fp32_model"],
        qdq_model=inputs["qdq_model"],
        output_model=qoperator_path,
        calibration_paths=inputs["calibration_paths"],
        selected_conv_names=inputs["selected_conv_names"],
        excluded_conv_names=inputs["excluded_conv_names"],
        calibration_data_reader=reader,
        cpu_threads=cpu_threads,
    )

    model_paths = {
        "FP32": inputs["fp32_model"],
        "QDQ_INT8": inputs["qdq_model"],
        "QOPERATOR_INT8": qoperator_path,
    }
    graph_diagnostics: dict[str, Any] = {}
    for name, model_path in model_paths.items():
        graph_diagnostics[name] = {
            "original": inspect_qoperator_graph(model_path),
            "optimized": _optimized_graph(
                model_path,
                optimized_dir / f"{name.lower()}_optimized.onnx",
                cpu_threads=cpu_threads,
            ),
        }

    images: list[np.ndarray] = []
    image_labels: list[str] = []
    for path in inputs["benchmark_paths"]:
        image, _ = load_image(path)
        images.append(image)
        image_labels.append(str(path))
    benchmark, _ = _benchmark_models(
        model_paths,
        images,
        image_labels,
        cpu_threads=cpu_threads,
    )
    profiling = {
        name: _profile_model(
            name,
            path,
            images,
            profile_dir=profile_dir,
            cpu_threads=cpu_threads,
        )
        for name, path in model_paths.items()
    }

    source_hash_after = _sha256(inputs["fp32_model"])
    qdq_hash_after = _sha256(inputs["qdq_model"])
    if source_hash_after != inputs["source_fp32_sha256"]:
        raise RuntimeError("Source FP32 model changed during the experiment.")
    if qdq_hash_after != inputs["qdq_model_sha256"]:
        raise RuntimeError("Existing QDQ INT8 model changed during the experiment.")
    historical_means = inputs["historical_benchmark"]["mean_latency_ms"]
    historical_comparison = {}
    for current_name, historical_name in (
        ("FP32", "FP32"),
        ("QDQ_INT8", "INT8"),
    ):
        historical_mean = historical_means.get(historical_name)
        current_mean = benchmark["configurations"][current_name]["mean_latency_ms"]
        historical_comparison[current_name] = {
            "phase7b_mean_latency_ms": historical_mean,
            "current_mean_latency_ms": current_mean,
            "current_to_phase7b_ratio": (
                current_mean / historical_mean
                if isinstance(historical_mean, (int, float)) and historical_mean > 0
                else None
            ),
        }
    metadata = {
        "experiment_type": "onnxruntime_static_qoperator_int8_yolo11n",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_fp32_model_path": str(inputs["fp32_model"]),
        "source_fp32_sha256": inputs["source_fp32_sha256"],
        "existing_qdq_model_path": str(inputs["qdq_model"]),
        "existing_qdq_model_sha256_before_experiment": inputs["qdq_model_sha256"],
        "existing_qdq_model_sha256_after_experiment": qdq_hash_after,
        "output_model_path": str(qoperator_path),
        "output_model_sha256": qoperator_metadata["output_model_sha256"],
        "onnxruntime_version": ort.__version__,
        "quantization_format": "QOperator",
        "quantization_types": {
            "activation": "QInt8",
            "weight": "QInt8",
        },
        "calibration_method": "MinMax",
        "calibration_provider": CPU_PROVIDER,
        "per_channel_weight_quantization": True,
        "calibration_image_count": len(inputs["calibration_paths"]),
        "calibration_image_paths": [
            str(path) for path in inputs["calibration_paths"]
        ],
        "calibration_metadata_source": str(inputs["qdq_metadata_path"]),
        "quantization_duration_ms": qoperator_metadata[
            "quantization_duration_ms"
        ],
        "source_conv_node_count": len(inputs["source_conv_names"]),
        "selected_conv_node_count": qoperator_metadata[
            "selected_conv_node_count"
        ],
        "quantized_conv_node_count": qoperator_metadata["quantized_conv_node_count"],
        "quantized_conv_node_names": qoperator_metadata["quantized_conv_node_names"],
        "unsupported_or_unconverted_selected_conv_count": qoperator_metadata[
            "unsupported_or_unconverted_selected_conv_count"
        ],
        "unsupported_or_unconverted_selected_conv_names": qoperator_metadata[
            "unsupported_or_unconverted_selected_conv_names"
        ],
        "deliberately_excluded_conv_node_count": qoperator_metadata[
            "deliberately_excluded_conv_node_count"
        ],
        "deliberately_excluded_conv_node_names": qoperator_metadata[
            "deliberately_excluded_conv_node_names"
        ],
        "validation": {
            **qoperator_metadata["validation"],
            "fp32_source_hash_preserved": (
                source_hash_after == inputs["source_fp32_sha256"]
            ),
            "qdq_model_hash_preserved": (
                qdq_hash_after == inputs["qdq_model_sha256"]
            ),
            "qdq_model_hash_after_experiment": qdq_hash_after,
            "execution_provider": CPU_PROVIDER,
            "optimized_graphs_and_reports": "passed",
            "benchmark_and_accuracy_validation": (
                "completed; no accuracy acceptance threshold was defined"
            ),
            "operator_profiling": "completed",
        },
        "runtime_warnings": [
            "ONNX Runtime 1.30.0 warned: 'Please use QuantFormat.QDQ for activation type QInt8 and weight type QInt8. Or it will lead to bad performance on x64.'",
            "ONNX Runtime recommended model preprocessing before quantization; preprocessing settings were not changed because this experiment must match the existing QDQ setup.",
        ],
        "historical_phase7b_comparison": {
            **inputs["historical_benchmark"],
            "current_vs_historical_mean_latency": historical_comparison,
            "interpretation": (
                "Current FP32/QDQ absolute means differ substantially from Phase 7B; "
                "the cause was not established. Use the same-run three-way comparison "
                "for this experiment and do not infer a cross-run speed change."
            ),
        },
        "benchmark_report": benchmark,
        "graph_diagnostics": graph_diagnostics,
        "operator_profiling": profiling,
        "limitations": [
            "Benchmark and accuracy comparisons use four images sampled from a single video and do not provide broad performance or accuracy certification.",
            "Operator trace durations are profiled measurements and must not be compared with unprofiled end-to-end benchmark timings.",
            "The current FP32 and QDQ absolute means are substantially higher than the earlier Phase 7B means; the reason is unresolved and limits comparisons across runs.",
            "QOperator conversion is restricted to the same Conv node names selected by the existing QDQ experiment; deliberately excluded Conv nodes remain floating point.",
            "Quantized operator names in an optimized graph establish graph structure; CPU execution traces are reported separately as runtime evidence.",
        ],
    }
    _write_json(run_dir / "quantization_metadata.json", metadata)
    _write_json(run_dir / "qoperator_experiment_report.json", metadata)
    print(f"QOperator model: {qoperator_path}")
    print(f"Experiment report: {run_dir / 'qoperator_experiment_report.json'}")
    for name, config in benchmark["configurations"].items():
        print(
            f"{name}: mean {config['mean_latency_ms']:.3f} ms, "
            f"median {config['median_latency_ms']:.3f} ms, "
            f"P95 {config['p95_latency_ms']:.3f} ms, "
            f"{config['throughput_images_per_second']:.2f} images/s"
        )
    print(
        "QOperator runtime quantized operators: "
        f"{profiling['QOPERATOR_INT8']['quantized_operator_events']}"
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-model", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--qdq-model", type=Path, default=DEFAULT_INT8_ONNX_PATH)
    parser.add_argument("--qdq-metadata", type=Path, default=DEFAULT_QDQ_METADATA)
    parser.add_argument(
        "--benchmark-manifest", type=Path, default=DEFAULT_BENCHMARK_MANIFEST
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=DEFAULT_EXPERIMENT_MODEL_DIR / DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        "--experiment-output-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_OUTPUT_DIR,
    )
    parser.add_argument("--cpu-threads", type=int, default=CPU_THREADS)
    args = parser.parse_args(argv)
    run_qoperator_experiment(
        fp32_model=args.fp32_model,
        qdq_model=args.qdq_model,
        qdq_metadata=args.qdq_metadata,
        benchmark_manifest=args.benchmark_manifest,
        output_model=args.output_model,
        experiment_output_dir=args.experiment_output_dir,
        cpu_threads=args.cpu_threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
