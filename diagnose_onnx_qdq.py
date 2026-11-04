"""Inspect ONNX QDQ graphs and save CPU-optimized graph diagnostics."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent
DEFAULT_FP32_MODEL = ROOT / "models" / "yolo11n.onnx"
DEFAULT_INT8_MODEL = ROOT / "models" / "yolo11n_int8.onnx"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "ort_optimized_graphs"
DEFAULT_PROFILE_REPORT = ROOT / "outputs" / "ort_operator_profile_comparison.json"
QDQ_OPS = {"QuantizeLinear", "DequantizeLinear"}
QUANTIZED_COMPUTE_OPS = {
    "ConvInteger",
    "MatMulInteger",
    "MatMulIntegerToFloat",
    "QLinearConv",
    "QLinearMatMul",
    "QLinearAdd",
    "QLinearMul",
}


def _attribute_int(node: onnx.NodeProto, name: str) -> int | None:
    for attribute in node.attribute:
        if attribute.name == name and attribute.type == onnx.AttributeProto.INT:
            return int(attribute.i)
    return None


def _tensor_type_name(value_info: onnx.ValueInfoProto) -> str | None:
    if not value_info.type.HasField("tensor_type"):
        return None
    try:
        return TensorProto.DataType.Name(value_info.type.tensor_type.elem_type)
    except ValueError:
        return None


def inspect_model(model: onnx.ModelProto, *, model_name: str) -> dict[str, Any]:
    """Summarize graph nodes and QDQ patterns without changing the model."""
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=False, data_prop=False
    )
    graph = inferred.graph
    producers = {output: node for node in graph.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    initializers = {item.name: item for item in graph.initializer}
    value_types = {
        value.name: type_name
        for value in (*graph.input, *graph.value_info, *graph.output)
        if (type_name := _tensor_type_name(value)) is not None
    }
    operator_counts = dict(sorted(Counter(node.op_type for node in graph.node).items()))
    conv_nodes = [node for node in graph.node if node.op_type == "Conv"]
    qdq_counts = {
        op_type: operator_counts.get(op_type, 0)
        for op_type in ("QuantizeLinear", "DequantizeLinear")
    }

    conv_details: list[dict[str, Any]] = []
    quantized_weight_conv_count = 0
    convs_with_activation_dq = 0
    convs_with_output_q = 0
    per_channel_weight_count = 0
    per_tensor_weight_count = 0
    convs_with_int8_weight_initializer = 0
    structural_qdq_conv_candidates = 0

    for conv in conv_nodes:
        activation_producer = producers.get(conv.input[0]) if conv.input else None
        weight_producer = producers.get(conv.input[1]) if len(conv.input) > 1 else None
        output_q_nodes = [
            consumer
            for output in conv.output
            for consumer in consumers.get(output, [])
            if consumer.op_type == "QuantizeLinear"
        ]
        activation_is_dq = bool(
            activation_producer and activation_producer.op_type == "DequantizeLinear"
        )
        if activation_is_dq:
            convs_with_activation_dq += 1
        if output_q_nodes:
            convs_with_output_q += 1

        weight_quantization: dict[str, Any] | None = None
        is_quantized_weight = False
        if weight_producer and weight_producer.op_type == "DequantizeLinear":
            weight_initializer = initializers.get(weight_producer.input[0])
            scale_initializer = initializers.get(weight_producer.input[1])
            zero_initializer = (
                initializers.get(weight_producer.input[2])
                if len(weight_producer.input) > 2
                else None
            )
            weight_type = (
                TensorProto.DataType.Name(weight_initializer.data_type)
                if weight_initializer
                else value_types.get(weight_producer.input[0])
            )
            scale_count = (
                int(numpy_helper.to_array(scale_initializer).size)
                if scale_initializer
                else None
            )
            axis = _attribute_int(weight_producer, "axis")
            is_quantized_weight = weight_type in {"INT8", "UINT8"}
            if is_quantized_weight:
                quantized_weight_conv_count += 1
                convs_with_int8_weight_initializer += weight_type == "INT8"
                if scale_count is not None and scale_count > 1:
                    per_channel_weight_count += 1
                elif scale_count == 1:
                    per_tensor_weight_count += 1
            weight_quantization = {
                "dequantize_node": weight_producer.name,
                "stored_weight_type": weight_type,
                "stored_weight_shape": (
                    list(weight_initializer.dims) if weight_initializer else None
                ),
                "scale_shape": (
                    list(scale_initializer.dims) if scale_initializer else None
                ),
                "scale_count": scale_count,
                "zero_point_type": (
                    TensorProto.DataType.Name(zero_initializer.data_type)
                    if zero_initializer
                    else None
                ),
                "axis": axis,
                "granularity": (
                    "per-channel"
                    if scale_count is not None and scale_count > 1
                    else "per-tensor"
                    if scale_count == 1
                    else "unknown"
                ),
            }

        conv_details.append(
            {
                "name": conv.name,
                "input_tensor_types": [
                    value_types.get(name) for name in conv.input
                ],
                "input_producers": [
                    (
                        producers[name].op_type
                        if name in producers
                        else "initializer"
                        if name in initializers
                        else "graph-input-or-unknown"
                    )
                    for name in conv.input
                ],
                "output_tensor_types": [
                    value_types.get(name) for name in conv.output
                ],
                "activation_input_from_dequantize_linear": activation_is_dq,
                "weight_input_from_dequantize_linear": bool(
                    weight_producer and weight_producer.op_type == "DequantizeLinear"
                ),
                "quantized_weight_initializer": is_quantized_weight,
                "weight_quantization": weight_quantization,
                "output_quantize_nodes": [node.name for node in output_q_nodes],
            }
        )
        if activation_is_dq and is_quantized_weight and output_q_nodes:
            structural_qdq_conv_candidates += 1

    return {
        "model_name": model_name,
        "ir_version": int(model.ir_version),
        "opsets": {
            entry.domain or "ai.onnx": int(entry.version)
            for entry in model.opset_import
        },
        "node_count": len(graph.node),
        "operator_counts": operator_counts,
        "conv_count": len(conv_nodes),
        "qdq_node_counts": qdq_counts,
        "quantized_compute_node_counts": {
            op_type: operator_counts[op_type]
            for op_type in sorted(QUANTIZED_COMPUTE_OPS)
            if operator_counts.get(op_type, 0)
        },
        "conv_qdq_adjacency": {
            "conv_activation_inputs_from_dequantize": convs_with_activation_dq,
            "conv_weight_inputs_from_dequantize": sum(
                item["weight_input_from_dequantize_linear"]
                for item in conv_details
            ),
            "conv_outputs_followed_by_quantize": convs_with_output_q,
            "conv_with_int8_or_uint8_weight_initializer_after_dequantize": (
                quantized_weight_conv_count
            ),
            "conv_weight_inputs_stored_as_int8_count": (
                convs_with_int8_weight_initializer
            ),
            "per_channel_weight_count": per_channel_weight_count,
            "per_tensor_weight_count": per_tensor_weight_count,
            "structural_qdq_conv_candidate_count": structural_qdq_conv_candidates,
        },
        "representative_convolutions": [
            item
            for item in conv_details
            if item["quantized_weight_initializer"]
        ][:3]
        or conv_details[:3],
        "all_convolutions": conv_details,
    }


def compare_graphs(
    original: dict[str, Any], optimized: dict[str, Any]
) -> dict[str, Any]:
    original_ops = original["operator_counts"]
    optimized_ops = optimized["operator_counts"]
    op_types = sorted(set(original_ops) | set(optimized_ops))
    changes = {
        op: {
            "original_count": original_ops.get(op, 0),
            "optimized_count": optimized_ops.get(op, 0),
            "delta": optimized_ops.get(op, 0) - original_ops.get(op, 0),
        }
        for op in op_types
        if original_ops.get(op, 0) != optimized_ops.get(op, 0)
    }
    original_quantized_count = sum(
        original["quantized_compute_node_counts"].values()
    )
    int8_fused_nodes = optimized["quantized_compute_node_counts"]
    remaining_qdq = optimized["qdq_node_counts"]
    fusion_observed = sum(int8_fused_nodes.values()) > original_quantized_count and (
        sum(remaining_qdq.values()) < sum(original["qdq_node_counts"].values())
    )
    return {
        "original_node_count": original["node_count"],
        "optimized_node_count": optimized["node_count"],
        "operator_count_changes": changes,
        "original_qdq_node_counts": original["qdq_node_counts"],
        "optimized_qdq_node_counts": remaining_qdq,
        "original_quantized_compute_node_counts": original[
            "quantized_compute_node_counts"
        ],
        "optimized_quantized_compute_node_counts": int8_fused_nodes,
        "qdq_to_quantized_operator_fusion_observed": fusion_observed,
        "fusion_assessment": (
            "Optimized graph contains recognized quantized compute node types and "
            "has fewer QDQ nodes; inspect graph connectivity to attribute rewrites."
            if fusion_observed
            else "No graph-level QDQ-to-quantized-operator rewrite was confirmed by "
            "the recognized node types and QDQ counts."
        ),
    }


def _cpu_features() -> dict[str, Any]:
    features: dict[str, bool] = {}
    module = None
    for module_name in (
        "numpy._core._multiarray_umath",
        "numpy.core._multiarray_umath",
    ):
        try:
            module = importlib.import_module(module_name)
            break
        except ImportError:
            continue
    detected = getattr(module, "__cpu_features__", {}) if module else {}
    for feature in (
        "SSE42",
        "AVX",
        "AVX2",
        "AVX_VNNI",
        "AVX512F",
        "AVX512_VNNI",
    ):
        if feature in detected:
            features[feature] = bool(detected[feature])

    cpu_model = platform.processor() or None
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            cpu_model = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            winreg.CloseKey(key)
        except OSError:
            pass

    return {
        "model": cpu_model,
        "architecture": platform.machine(),
        "logical_processor_count": os.cpu_count(),
        "instruction_sets": features,
        "instruction_set_detection_source": (
            "NumPy runtime CPU feature detection; omitted flags are not reported."
        ),
    }


def probe_qlinearconv_cpu() -> dict[str, Any]:
    """Run a tiny QLinearConv model to verify CPU EP kernel availability."""
    x_info = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 1, 1, 1])
    y_info = helper.make_tensor_value_info("y", TensorProto.INT8, [1, 1, 1, 1])
    initializers = [
        numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), "x_scale"),
        numpy_helper.from_array(np.asarray(0, dtype=np.int8), "x_zero"),
        numpy_helper.from_array(
            np.asarray([[[[4]]]], dtype=np.int8), "weight"
        ),
        numpy_helper.from_array(np.asarray(0.25, dtype=np.float32), "weight_scale"),
        numpy_helper.from_array(np.asarray(0, dtype=np.int8), "weight_zero"),
        numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), "y_scale"),
        numpy_helper.from_array(np.asarray(0, dtype=np.int8), "y_zero"),
    ]
    qconv = helper.make_node(
        "QLinearConv",
        [
            "x",
            "x_scale",
            "x_zero",
            "weight",
            "weight_scale",
            "weight_zero",
            "y_scale",
            "y_zero",
        ],
        ["y"],
        name="cpu_qlinearconv_capability_probe",
    )
    graph = helper.make_graph(
        [qconv], "qlinearconv_cpu_probe", [x_info], [y_info], initializers
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = min(model.ir_version, 10)
    try:
        onnx.checker.check_model(model)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        providers = session.get_providers()
        output = session.run(
            ["y"], {"x": np.asarray([[[[2]]]], dtype=np.int8)}
        )[0]
        return {
            "operator": "QLinearConv",
            "execution_provider": providers[0] if providers else None,
            "available": bool(providers and providers[0] == "CPUExecutionProvider"),
            "execution_succeeded": True,
            "output_dtype": str(output.dtype),
            "output_shape": list(output.shape),
        }
    except (RuntimeError, ValueError, onnx.checker.ValidationError) as exc:
        return {
            "operator": "QLinearConv",
            "execution_provider": "CPUExecutionProvider",
            "available": False,
            "execution_succeeded": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_diagnostic_report(
    *,
    models: dict[str, dict[str, Any]],
    graph_summaries: dict[str, dict[str, dict[str, Any]]],
    runtime: dict[str, Any],
    cpu: dict[str, Any],
    cpu_threads: int,
    profile_evidence: dict[str, Any] | None,
    qlinearconv_probe: dict[str, Any],
) -> dict[str, Any]:
    comparisons = {
        precision: compare_graphs(pair["original"], pair["optimized"])
        for precision, pair in graph_summaries.items()
    }
    int8_original = graph_summaries["INT8"]["original"]
    int8_comparison = comparisons["INT8"]
    return {
        "report_type": "onnx_qdq_graph_and_cpu_runtime_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "onnxruntime_version": runtime["onnxruntime_version"],
            "available_execution_providers": runtime["available_execution_providers"],
            "session_execution_provider": "CPUExecutionProvider",
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "cpu_threads": {
                "intra_op": cpu_threads,
                "inter_op": 1,
                "logical_processors": cpu.get("logical_processor_count"),
            },
        },
        "cpu": cpu,
        "numpy_version": np.__version__,
        "models": models,
        "graphs": graph_summaries,
        "original_vs_optimized": comparisons,
        "quantized_kernel_probe": qlinearconv_probe,
        "prior_operator_profile_evidence": profile_evidence,
        "findings": {
            "int8_conv_weights_stored_as_int8": (
                int8_original["conv_qdq_adjacency"][
                    "conv_weight_inputs_stored_as_int8_count"
                ]
                > 0
            ),
            "int8_qdq_conv_weight_count": int8_original["conv_qdq_adjacency"][
                "conv_with_int8_or_uint8_weight_initializer_after_dequantize"
            ],
            "int8_per_channel_weight_count": int8_original["conv_qdq_adjacency"][
                "per_channel_weight_count"
            ],
            "int8_structural_qdq_conv_candidate_count": int8_original[
                "conv_qdq_adjacency"
            ]["structural_qdq_conv_candidate_count"],
            "int8_optimized_graph_qdq_fusion_confirmed": int8_comparison[
                "qdq_to_quantized_operator_fusion_observed"
            ],
            "cpu_qlinearconv_kernel_probe_succeeded": qlinearconv_probe[
                "execution_succeeded"
            ],
        },
        "limitations_and_warnings": [
            "A successful synthetic QLinearConv CPU probe confirms that this installed runtime can execute that operator; it does not prove that the inspected QDQ model is fused or that all quantized convolution shapes use the same kernel.",
            "Optimized graph node types and QDQ counts are evidence of graph rewrites, but fusion attribution also depends on graph connectivity and execution traces.",
            "The existing operator profile is a four-image profiled sample. Its Conv and conversion timings include profiling overhead and are not an end-to-end benchmark or a causal explanation.",
            "INT8 weight initializer types and scale shapes establish quantized storage and granularity, not that Conv consumes integer tensors at runtime; DequantizeLinear outputs may restore float inputs.",
            "CPU instruction-set flags are obtained from NumPy runtime detection. Missing flags are not interpreted as unsupported unless explicitly reported false.",
            "ONNX Runtime warned that optimized graphs saved above ORT_ENABLE_EXTENDED may contain hardware-specific transformations; use these artifacts only as diagnostics for this runtime and machine.",
        ],
    }


def _read_profile_evidence(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read operator profile report {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"Operator profile report must be a JSON object: {path}")
    provider = report.get("execution_provider_validation", {})
    int8_operators = report.get("trace_summaries", {}).get("INT8", {}).get(
        "operators", []
    )
    return {
        "source": str(path.resolve()),
        "execution_provider_validation": provider,
        "int8_conv_operator": next(
            (item for item in int8_operators if item.get("operator") == "Conv"),
            None,
        ),
        "int8_conversion_operators": [
            item
            for item in int8_operators
            if item.get("operator") in QDQ_OPS
        ],
    }


def _optimized_session(
    model_path: Path, output_path: Path, *, cpu_threads: int
) -> dict[str, Any]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(output_path)
    options.intra_op_num_threads = cpu_threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    providers = session.get_providers()
    if not providers or providers[0] != "CPUExecutionProvider":
        raise RuntimeError(
            f"Expected CPUExecutionProvider for {model_path}, got {providers!r}."
        )
    del session
    if not output_path.is_file():
        raise FileNotFoundError(
            f"ONNX Runtime did not save its optimized graph: {output_path}"
        )
    return {
        "provider": providers[0],
        "optimized_graph_path": str(output_path.resolve()),
    }


def run_diagnostic(
    *,
    fp32_model: str | Path = DEFAULT_FP32_MODEL,
    int8_model: str | Path = DEFAULT_INT8_MODEL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    profile_report: str | Path = DEFAULT_PROFILE_REPORT,
    cpu_threads: int = 4,
) -> dict[str, Any]:
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer.")
    source_models = {
        "FP32": Path(fp32_model).expanduser().resolve(),
        "INT8": Path(int8_model).expanduser().resolve(),
    }
    for path in source_models.values():
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model does not exist: {path}")

    base_dir = Path(output_dir).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = base_dir / run_name
    run_dir.mkdir()

    models: dict[str, dict[str, Any]] = {}
    graph_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for precision, source_path in source_models.items():
        original_model = onnx.load(str(source_path), load_external_data=True)
        optimized_path = run_dir / f"{precision.lower()}_optimized.onnx"
        _optimized_session(source_path, optimized_path, cpu_threads=cpu_threads)
        optimized_model = onnx.load(str(optimized_path), load_external_data=True)
        models[precision] = {
            "name": source_path.name,
            "path": str(source_path),
            "size_bytes": source_path.stat().st_size,
            "optimized_graph_path": str(optimized_path.resolve()),
            "optimized_size_bytes": optimized_path.stat().st_size,
        }
        graph_summaries[precision] = {
            "original": inspect_model(original_model, model_name=source_path.name),
            "optimized": inspect_model(
                optimized_model, model_name=optimized_path.name
            ),
        }

    profile_path = Path(profile_report).expanduser().resolve()
    report = build_diagnostic_report(
        models=models,
        graph_summaries=graph_summaries,
        runtime={
            "onnxruntime_version": ort.__version__,
            "available_execution_providers": ort.get_available_providers(),
        },
        cpu=_cpu_features(),
        cpu_threads=cpu_threads,
        profile_evidence=_read_profile_evidence(profile_path),
        qlinearconv_probe=probe_qlinearconv_cpu(),
    )
    report_path = run_dir / "onnx_qdq_diagnostic.json"
    with report_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")

    print(f"ONNX Runtime: {ort.__version__}; CPUExecutionProvider; threads: {cpu_threads}")
    print(f"CPU: {report['cpu']['model']}; ISA: {report['cpu']['instruction_sets']}")
    for precision in ("FP32", "INT8"):
        original = graph_summaries[precision]["original"]
        optimized = graph_summaries[precision]["optimized"]
        print(
            f"{precision}: Conv {original['conv_count']} -> {optimized['conv_count']}; "
            f"Q/DQ {original['qdq_node_counts']} -> {optimized['qdq_node_counts']}; "
            f"quantized compute nodes "
            f"{optimized['quantized_compute_node_counts']}"
        )
    print(
        "INT8 QDQ-to-quantized-op rewrite confirmed: "
        f"{report['findings']['int8_optimized_graph_qdq_fusion_confirmed']}"
    )
    print(
        "Synthetic CPU QLinearConv probe succeeded: "
        f"{report['findings']['cpu_qlinearconv_kernel_probe_succeeded']}"
    )
    print(f"Optimized graphs and report: {run_dir}")
    print(f"Diagnostic report: {report_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-model", type=Path, default=DEFAULT_FP32_MODEL)
    parser.add_argument("--int8-model", type=Path, default=DEFAULT_INT8_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile-report", type=Path, default=DEFAULT_PROFILE_REPORT)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    run_diagnostic(
        fp32_model=args.fp32_model,
        int8_model=args.int8_model,
        output_dir=args.output_dir,
        profile_report=args.profile_report,
        cpu_threads=args.cpu_threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
