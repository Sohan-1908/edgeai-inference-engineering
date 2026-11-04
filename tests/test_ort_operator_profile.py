import tempfile
import unittest
from pathlib import Path

from analyze_ort_profile import (
    build_comparison_report,
    load_trace,
    parse_trace_events,
)


def _node_event(
    name,
    op_name,
    duration,
    *,
    provider="CPUExecutionProvider",
    phase="X",
    input_types=None,
):
    args = {"op_name": op_name, "provider": provider}
    if input_types is not None:
        args["input_type_shape"] = input_types
    return {
        "cat": "Node",
        "name": name,
        "ph": phase,
        "dur": duration,
        "args": args,
    }


class OrtTraceParsingTests(unittest.TestCase):
    def test_parses_valid_operator_events_and_groups_by_operator_and_provider(self):
        events = [
            _node_event("conv_kernel_time", "Conv", 100, input_types=[{"float": [1, 3]}]),
            _node_event("conv_2_kernel_time", "Conv", 300),
            _node_event("quant_kernel_time", "QuantizeLinear", 50),
        ]

        parsed = parse_trace_events(events)
        conv = next(item for item in parsed["operators"] if item["operator"] == "Conv")

        self.assertEqual(parsed["included_kernel_event_count"], 3)
        self.assertEqual(conv["execution_count"], 2)
        self.assertEqual(conv["total_duration_us"], 400)
        self.assertEqual(conv["average_duration_us"], 200)
        self.assertEqual(conv["tensor_dtypes"], ["float"])

    def test_malformed_and_missing_fields_are_skipped_safely(self):
        parsed = parse_trace_events(
            [
                None,
                _node_event("no_duration_kernel_time", "Conv", "not-a-number"),
                {"cat": "Node", "name": "missing_args_kernel_time", "ph": "X", "dur": 4},
                _node_event("missing_op_kernel_time", "", 4),
                _node_event("missing_provider_kernel_time", "Add", 10, provider=None),
                _node_event("nonfinite_kernel_time", "Mul", float("inf")),
            ]
        )

        self.assertEqual(parsed["included_kernel_event_count"], 1)
        self.assertEqual(parsed["operators"][0]["execution_provider"], "unknown")
        self.assertEqual(parsed["skipped_event_counts"]["malformed_event"], 1)
        self.assertEqual(parsed["skipped_event_counts"]["invalid_duration"], 2)
        self.assertEqual(parsed["skipped_event_counts"]["missing_args"], 1)
        self.assertEqual(parsed["skipped_event_counts"]["missing_operator"], 1)

    def test_session_fence_and_unrelated_events_are_not_counted(self):
        parsed = parse_trace_events(
            [
                {"cat": "Session", "name": "session_initialization", "ph": "X", "dur": 500},
                _node_event("conv_fence_before", "Conv", 100),
                _node_event("conv_kernel_time", "Conv", 25),
                {"cat": "Other", "name": "metadata", "ph": "X", "dur": 999},
            ]
        )

        self.assertEqual(parsed["session_event_count"], 1)
        self.assertEqual(parsed["included_kernel_event_count"], 1)
        self.assertEqual(parsed["operators"][0]["total_duration_us"], 25)
        self.assertEqual(parsed["skipped_event_counts"]["non_kernel_node_event"], 1)

    def test_trace_loader_rejects_non_list_json_root(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text('{"events":[]}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "event list"):
                load_trace(path)


class OrtComparisonReportTests(unittest.TestCase):
    def test_comparison_report_includes_shared_conversion_cpu_and_float_details(self):
        fp32 = parse_trace_events(
            [
                _node_event(
                    "conv_kernel_time",
                    "Conv",
                    1000,
                    input_types=[{"float": [1, 3]}],
                ),
                _node_event("add_kernel_time", "Add", 100),
            ]
        )
        int8 = parse_trace_events(
            [
                _node_event("conv_kernel_time", "Conv", 1800),
                _node_event("dq_kernel_time", "DequantizeLinear", 450),
                _node_event("qconv_kernel_time", "QLinearConv", 70),
            ]
        )
        report = build_comparison_report(
            fp32,
            int8,
            fp32_model={"name": "fp32.onnx", "size_bytes": 1000},
            int8_model={"name": "int8.onnx", "size_bytes": 500},
            profiling_configuration={
                "execution_provider": "CPUExecutionProvider",
                "cpu_threads": 4,
                "image_count": 4,
            },
        )

        self.assertTrue(report["execution_provider_validation"]["all_recorded_node_events_cpu"])
        self.assertTrue(report["execution_provider_validation"]["quantized_compute_operators_on_cpu"])
        self.assertIn("DequantizeLinear", [
            item["operator"]
            for item in report["operator_insights"]["INT8"]["conversion_operators"]
        ])
        self.assertEqual(
            report["operator_insights"]["FP32"]["float_typed_operators"][0]["operator"],
            "Conv",
        )
        conv = next(
            item
            for item in report["fp32_vs_int8_operators"]
            if item["operator"] == "Conv"
        )
        self.assertEqual(conv["total_duration_delta_ms_int8_minus_fp32"], 0.8)

    def test_comparison_report_requires_positive_top_count(self):
        empty = parse_trace_events([])
        with self.assertRaisesRegex(ValueError, "top_n"):
            build_comparison_report(
                empty,
                empty,
                fp32_model={},
                int8_model={},
                profiling_configuration={},
                top_n=0,
            )


if __name__ == "__main__":
    unittest.main()
