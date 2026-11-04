"""Compare ONNX Runtime operator-profile traces for FP32 and INT8 models."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from numbers import Real
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "ort_operator_profile_comparison.json"
CONVERSION_OPERATORS = {
    "Cast",
    "CastLike",
    "DequantizeLinear",
    "DynamicQuantizeLinear",
    "QuantizeLinear",
}
QUANTIZED_COMPUTE_PREFIXES = (
    "QLinear",
    "QAttention",
)
QUANTIZED_COMPUTE_OPERATORS = {
    "ConvInteger",
    "MatMulInteger",
    "MatMulIntegerToFloat",
}


def _duration_us(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    duration = float(value)
    if not math.isfinite(duration) or duration < 0:
        return None
    return duration


def _tensor_dtypes(args: dict[str, Any]) -> set[str]:
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).lower()
                if normalized in {
                    "float",
                    "float16",
                    "double",
                    "bfloat16",
                    "int8",
                    "uint8",
                    "int16",
                    "int32",
                    "int64",
                    "bool",
                }:
                    found.add(normalized)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(args.get("input_type_shape"))
    visit(args.get("output_type_shape"))
    return found


def parse_trace_events(events: Any, *, source: str | None = None) -> dict[str, Any]:
    """Aggregate only completed ONNX Runtime node kernel-time trace events.

    ORT trace `dur` is expressed in microseconds. Session, node fence, and
    unrelated events are deliberately excluded to avoid counting a node more
    than once.
    """
    if not isinstance(events, list):
        raise ValueError("ONNX Runtime trace root must be a JSON event list.")

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    skipped: defaultdict[str, int] = defaultdict(int)
    session_events = 0
    node_events = 0
    kernel_candidates = 0

    for event in events:
        if not isinstance(event, dict):
            skipped["malformed_event"] += 1
            continue
        category = event.get("cat")
        if category == "Session":
            session_events += 1
            continue
        if category != "Node":
            skipped["unrelated_category"] += 1
            continue
        node_events += 1
        name = event.get("name")
        if not isinstance(name, str) or not name.endswith("_kernel_time"):
            skipped["non_kernel_node_event"] += 1
            continue
        if event.get("ph") != "X":
            skipped["incomplete_kernel_event"] += 1
            continue
        kernel_candidates += 1
        duration = _duration_us(event.get("dur"))
        if duration is None:
            skipped["invalid_duration"] += 1
            continue
        args = event.get("args")
        if not isinstance(args, dict):
            skipped["missing_args"] += 1
            continue
        operator = args.get("op_name")
        if not isinstance(operator, str) or not operator.strip():
            skipped["missing_operator"] += 1
            continue
        provider_value = args.get("provider")
        provider = (
            provider_value.strip()
            if isinstance(provider_value, str) and provider_value.strip()
            else "unknown"
        )
        key = (operator, provider)
        group = groups.setdefault(
            key,
            {
                "operator": operator,
                "execution_provider": provider,
                "execution_count": 0,
                "total_duration_us": 0.0,
                "tensor_dtypes": set(),
            },
        )
        group["execution_count"] += 1
        group["total_duration_us"] += duration
        group["tensor_dtypes"].update(_tensor_dtypes(args))

    operators = []
    for group in groups.values():
        count = group["execution_count"]
        total_us = group["total_duration_us"]
        operators.append(
            {
                "operator": group["operator"],
                "execution_provider": group["execution_provider"],
                "execution_count": count,
                "total_duration_us": total_us,
                "total_duration_ms": total_us / 1000.0,
                "average_duration_us": total_us / count,
                "average_duration_ms": total_us / count / 1000.0,
                "tensor_dtypes": sorted(group["tensor_dtypes"]),
            }
        )
    operators.sort(
        key=lambda item: (-item["total_duration_us"], item["operator"], item["execution_provider"])
    )

    return {
        "source": source,
        "trace_event_count": len(events),
        "session_event_count": session_events,
        "node_event_count": node_events,
        "kernel_event_candidate_count": kernel_candidates,
        "included_kernel_event_count": sum(item["execution_count"] for item in operators),
        "skipped_event_counts": dict(sorted(skipped.items())),
        "operators": operators,
    }


def load_trace(path: str | Path) -> dict[str, Any]:
    trace_path = Path(path).expanduser().resolve()
    try:
        events = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read ONNX Runtime trace {trace_path}: {exc}") from exc
    parsed = parse_trace_events(events, source=str(trace_path))
    parsed["trace_path"] = str(trace_path)
    return parsed


def _is_quantized_compute(operator: str) -> bool:
    return operator in QUANTIZED_COMPUTE_OPERATORS or operator.startswith(
        QUANTIZED_COMPUTE_PREFIXES
    )


def _model_operator_insights(trace: dict[str, Any]) -> dict[str, Any]:
    operators = trace["operators"]
    conversions = [item for item in operators if item["operator"] in CONVERSION_OPERATORS]
    quantized_compute = [
        item for item in operators if _is_quantized_compute(item["operator"])
    ]
    float_typed = [
        item
        for item in operators
        if any(dtype in {"float", "float16", "double", "bfloat16"} for dtype in item["tensor_dtypes"])
    ]
    providers = sorted(
        {item["execution_provider"] for item in operators if item["execution_provider"] != "unknown"}
    )
    return {
        "observed_execution_providers": providers,
        "conversion_operators": conversions,
        "quantized_compute_operators": quantized_compute,
        "quantized_compute_observed_on_cpu": any(
            item["execution_provider"] == "CPUExecutionProvider"
            for item in quantized_compute
        ),
        "float_typed_operators": float_typed,
        "float_typed_operator_count": len(float_typed),
        "note": (
            "Float-typed nodes and conversion operators are reported for graph review. "
            "Their presence alone does not establish that they are unexpected or a bottleneck."
        ),
    }


def build_comparison_report(
    fp32_trace: dict[str, Any],
    int8_trace: dict[str, Any],
    *,
    fp32_model: dict[str, Any],
    int8_model: dict[str, Any],
    profiling_configuration: dict[str, Any],
    top_n: int = 20,
) -> dict[str, Any]:
    if top_n < 1:
        raise ValueError("top_n must be positive.")

    fp32_ops = {
        (item["operator"], item["execution_provider"]): item
        for item in fp32_trace["operators"]
    }
    int8_ops = {
        (item["operator"], item["execution_provider"]): item
        for item in int8_trace["operators"]
    }
    comparisons = []
    for operator, provider in sorted(set(fp32_ops) | set(int8_ops)):
        fp32 = fp32_ops.get((operator, provider))
        int8 = int8_ops.get((operator, provider))
        fp32_total = fp32["total_duration_ms"] if fp32 else 0.0
        int8_total = int8["total_duration_ms"] if int8 else 0.0
        fp32_count = fp32["execution_count"] if fp32 else 0
        int8_count = int8["execution_count"] if int8 else 0
        comparisons.append(
            {
                "operator": operator,
                "execution_provider": provider,
                "fp32_execution_count": fp32_count,
                "fp32_total_duration_ms": fp32_total,
                "fp32_average_duration_ms": fp32["average_duration_ms"] if fp32 else None,
                "int8_execution_count": int8_count,
                "int8_total_duration_ms": int8_total,
                "int8_average_duration_ms": int8["average_duration_ms"] if int8 else None,
                "total_duration_delta_ms_int8_minus_fp32": int8_total - fp32_total,
                "present_in_both": fp32 is not None and int8 is not None,
            }
        )

    fp32_insights = _model_operator_insights(fp32_trace)
    int8_insights = _model_operator_insights(int8_trace)
    cpu_threads = profiling_configuration.get("cpu_threads")
    image_count = profiling_configuration.get("image_count")
    providers_verified = (
        profiling_configuration.get("execution_provider") == "CPUExecutionProvider"
        and fp32_insights["observed_execution_providers"] == ["CPUExecutionProvider"]
        and int8_insights["observed_execution_providers"] == ["CPUExecutionProvider"]
    )
    shared_slower_int8 = [
        item
        for item in comparisons
        if item["present_in_both"]
        and item["execution_provider"] == "CPUExecutionProvider"
        and item["total_duration_delta_ms_int8_minus_fp32"] > 0
    ]
    shared_slower_int8.sort(
        key=lambda item: item["total_duration_delta_ms_int8_minus_fp32"],
        reverse=True,
    )

    return {
        "report_type": "onnxruntime_cpu_fp32_vs_int8_operator_profile",
        "models": {
            "FP32": fp32_model,
            "INT8": int8_model,
        },
        "profiling_configuration": profiling_configuration,
        "trace_summaries": {
            "FP32": fp32_trace,
            "INT8": int8_trace,
        },
        "execution_provider_validation": {
            "expected_provider": "CPUExecutionProvider",
            "all_recorded_node_events_cpu": providers_verified,
            "FP32_observed_providers": fp32_insights["observed_execution_providers"],
            "INT8_observed_providers": int8_insights["observed_execution_providers"],
            "quantized_compute_operators_observed": bool(
                int8_insights["quantized_compute_operators"]
            ),
            "quantized_compute_operators": int8_insights["quantized_compute_operators"],
            "quantized_compute_operators_on_cpu": int8_insights[
                "quantized_compute_observed_on_cpu"
            ],
        },
        "top_operators_by_total_duration": {
            "FP32": fp32_trace["operators"][:top_n],
            "INT8": int8_trace["operators"][:top_n],
        },
        "fp32_vs_int8_operators": comparisons,
        "operator_insights": {
            "FP32": fp32_insights,
            "INT8": int8_insights,
            "shared_operators_with_higher_int8_total_duration": shared_slower_int8,
            "possible_int8_conversion_overhead": int8_insights["conversion_operators"],
        },
        "limitations_and_warnings": [
            "ORT kernel event durations are summed operator times in microseconds converted to milliseconds; parallel execution can make these sums exceed wall-clock duration.",
            "Only completed Node events named *_kernel_time are included. Session, fence, and unrelated events are excluded to avoid double counting.",
            "Profiled runs include one inference per benchmark image after warm-up in separate unprofiled sessions. Profiling overhead and small sample count limit timing generalization.",
            "Operator frequency and summed time are descriptive, not proof of an end-to-end bottleneck or a causal explanation for INT8 being slower.",
            "Q/DQ events are conversion operators, not evidence that quantized convolution kernels executed. Quantized compute is reported separately.",
            "Float-typed operations are listed for review; the trace alone cannot determine whether they are unexpected for this partially quantized graph.",
            "The original INT8 benchmark uses four images from one video; conclusions must not be generalized to other datasets or hardware.",
        ],
    }


def _format_operator_rows(operators: list[dict[str, Any]], limit: int) -> list[str]:
    return [
        (
            f"  {item['operator']} [{item['execution_provider']}]: "
            f"{item['total_duration_ms']:.3f} ms total, "
            f"{item['execution_count']} executions, "
            f"{item['average_duration_ms']:.4f} ms average"
        )
        for item in operators[:limit]
    ]


def print_summary(report: dict[str, Any], *, top_n: int = 10) -> None:
    print("ONNX Runtime CPU operator profile comparison")
    config = report["profiling_configuration"]
    print(
        f"Images: {config.get('image_count', 'unknown')}; "
        f"warm-up calls/model: {config.get('warmup_runs_per_model', 'unknown')}; "
        f"CPU threads: {config.get('cpu_threads', 'unknown')}"
    )
    for precision in ("FP32", "INT8"):
        model = report["models"][precision]
        print(
            f"\n{precision}: {model['name']} "
            f"({model['size_bytes'] / 1_000_000:.2f} MB)"
        )
        for row in _format_operator_rows(
            report["top_operators_by_total_duration"][precision], top_n
        ):
            print(row)

    provider = report["execution_provider_validation"]
    print(
        "\nQuantized compute operators observed: "
        f"{provider['quantized_compute_operators_observed']}"
    )
    print(
        "Quantized compute operators observed on CPU: "
        f"{provider['quantized_compute_operators_on_cpu']}"
    )
    if not provider["quantized_compute_operators_observed"]:
        print(
            "No quantized compute operator was present in the trace; Q/DQ events, "
            "if present, are conversions rather than quantized-kernel evidence."
        )
    print(
        "Interpret operator totals as trace evidence, not standalone proof of "
        "an end-to-end bottleneck."
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read profiling manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Profiling manifest root must be a JSON object.")
    if not isinstance(manifest.get("models"), dict):
        raise ValueError("Profiling manifest must contain a models object.")
    return manifest


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-operators", type=int, default=20)
    args = parser.parse_args(argv)
    if args.top_operators < 1:
        parser.error("--top-operators must be positive.")

    manifest = _read_manifest(args.manifest)
    models = manifest["models"]
    config = manifest.get("profiling_configuration")
    if not isinstance(config, dict):
        raise ValueError("Profiling manifest must contain profiling_configuration.")
    fp32_model = models.get("FP32")
    int8_model = models.get("INT8")
    if not isinstance(fp32_model, dict) or not isinstance(int8_model, dict):
        raise ValueError("Profiling manifest must contain FP32 and INT8 model records.")
    fp32_trace = load_trace(fp32_model["trace_path"])
    int8_trace = load_trace(int8_model["trace_path"])
    report = build_comparison_report(
        fp32_trace,
        int8_trace,
        fp32_model=fp32_model,
        int8_model=int8_model,
        profiling_configuration=config,
        top_n=args.top_operators,
    )
    _write_report(args.output.expanduser().resolve(), report)
    print_summary(report)
    print(f"\nComparison report: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
