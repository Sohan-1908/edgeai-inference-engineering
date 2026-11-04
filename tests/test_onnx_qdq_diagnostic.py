import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from diagnose_onnx_qdq import (
    build_diagnostic_report,
    compare_graphs,
    inspect_model,
)


def _qdq_conv_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 2])
    initializers = [
        numpy_helper.from_array(np.asarray(0.25, dtype=np.float32), "x_scale"),
        numpy_helper.from_array(np.asarray(0, dtype=np.int8), "x_zero"),
        numpy_helper.from_array(
            np.asarray([[[[1.0]]], [[[0.5]]]], dtype=np.float32), "float_weight"
        ),
        numpy_helper.from_array(
            np.asarray([[[[4]]], [[[2]]]], dtype=np.int8), "quantized_weight"
        ),
        numpy_helper.from_array(
            np.asarray([0.125, 0.25], dtype=np.float32), "weight_scale"
        ),
        numpy_helper.from_array(np.asarray([0, 0], dtype=np.int8), "weight_zero"),
        numpy_helper.from_array(np.asarray(0.25, dtype=np.float32), "y_scale"),
        numpy_helper.from_array(np.asarray(0, dtype=np.int8), "y_zero"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear", ["x", "x_scale", "x_zero"], ["x_q"], name="x_quantize"
        ),
        helper.make_node(
            "DequantizeLinear", ["x_q", "x_scale", "x_zero"], ["x_dq"], name="x_dq"
        ),
        helper.make_node(
            "DequantizeLinear",
            ["quantized_weight", "weight_scale", "weight_zero"],
            ["weight_dq"],
            name="weight_dq",
            axis=0,
        ),
        helper.make_node("Conv", ["x_dq", "weight_dq"], ["conv_out"], name="conv"),
        helper.make_node(
            "QuantizeLinear",
            ["conv_out", "y_scale", "y_zero"],
            ["conv_q"],
            name="conv_quantize",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["conv_q", "y_scale", "y_zero"],
            ["y"],
            name="conv_dequantize",
        ),
    ]
    graph = helper.make_graph(nodes, "qdq_conv_test", [x], [y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    return model


class OnnxQdqGraphInspectionTests(unittest.TestCase):
    def test_inspects_qdq_conv_int8_storage_and_per_channel_scale(self):
        summary = inspect_model(_qdq_conv_model(), model_name="synthetic.onnx")

        self.assertEqual(summary["conv_count"], 1)
        self.assertEqual(summary["qdq_node_counts"]["QuantizeLinear"], 2)
        self.assertEqual(summary["qdq_node_counts"]["DequantizeLinear"], 3)
        self.assertEqual(
            summary["conv_qdq_adjacency"][
                "conv_with_int8_or_uint8_weight_initializer_after_dequantize"
            ],
            1,
        )
        conv = summary["representative_convolutions"][0]
        self.assertEqual(conv["input_tensor_types"], ["FLOAT", "FLOAT"])
        self.assertEqual(conv["output_tensor_types"], ["FLOAT"])
        self.assertTrue(conv["activation_input_from_dequantize_linear"])
        self.assertEqual(
            conv["weight_quantization"]["stored_weight_type"], "INT8"
        )
        self.assertEqual(conv["weight_quantization"]["granularity"], "per-channel")
        self.assertEqual(
            summary["conv_qdq_adjacency"]["per_channel_weight_count"], 1
        )
        self.assertEqual(
            summary["conv_qdq_adjacency"]["structural_qdq_conv_candidate_count"], 1
        )

    def test_graph_comparison_reports_operator_rewrite_and_qdq_removal(self):
        original = {
            "node_count": 4,
            "operator_counts": {"Conv": 1, "DequantizeLinear": 2, "QuantizeLinear": 1},
            "conv_count": 1,
            "qdq_node_counts": {"QuantizeLinear": 1, "DequantizeLinear": 2},
            "quantized_compute_node_counts": {},
        }
        optimized = {
            "node_count": 1,
            "operator_counts": {"QLinearConv": 1},
            "conv_count": 0,
            "qdq_node_counts": {"QuantizeLinear": 0, "DequantizeLinear": 0},
            "quantized_compute_node_counts": {"QLinearConv": 1},
        }

        comparison = compare_graphs(original, optimized)

        self.assertTrue(comparison["qdq_to_quantized_operator_fusion_observed"])
        self.assertEqual(
            comparison["operator_count_changes"]["QLinearConv"]["optimized_count"], 1
        )

    def test_existing_quantized_operator_does_not_count_as_new_fusion(self):
        graph = {
            "node_count": 3,
            "operator_counts": {
                "QLinearConv": 1,
                "QuantizeLinear": 1,
                "DequantizeLinear": 1,
            },
            "conv_count": 0,
            "qdq_node_counts": {"QuantizeLinear": 1, "DequantizeLinear": 1},
            "quantized_compute_node_counts": {"QLinearConv": 1},
        }
        optimized = {
            **graph,
            "node_count": 1,
            "operator_counts": {"QLinearConv": 1},
            "qdq_node_counts": {"QuantizeLinear": 0, "DequantizeLinear": 0},
        }

        comparison = compare_graphs(graph, optimized)

        self.assertFalse(comparison["qdq_to_quantized_operator_fusion_observed"])

    def test_report_generation_includes_runtime_and_graph_findings(self):
        summary = inspect_model(_qdq_conv_model(), model_name="synthetic.onnx")
        graphs = {
            precision: {"original": summary, "optimized": summary}
            for precision in ("FP32", "INT8")
        }
        report = build_diagnostic_report(
            models={
                "FP32": {"name": "fp32.onnx", "size_bytes": 100},
                "INT8": {"name": "int8.onnx", "size_bytes": 50},
            },
            graph_summaries=graphs,
            runtime={
                "onnxruntime_version": "test-version",
                "available_execution_providers": ["CPUExecutionProvider"],
            },
            cpu={
                "model": "test CPU",
                "logical_processor_count": 8,
                "instruction_sets": {"AVX2": True, "AVX512F": False},
            },
            cpu_threads=4,
            profile_evidence={"source": "profile.json"},
            qlinearconv_probe={
                "operator": "QLinearConv",
                "execution_provider": "CPUExecutionProvider",
                "available": True,
                "execution_succeeded": True,
            },
        )

        self.assertEqual(report["runtime"]["cpu_threads"]["intra_op"], 4)
        self.assertTrue(report["findings"]["int8_conv_weights_stored_as_int8"])
        self.assertTrue(report["findings"]["cpu_qlinearconv_kernel_probe_succeeded"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text(json.dumps(report), encoding="utf-8")
            saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(saved["report_type"], "onnx_qdq_graph_and_cpu_runtime_diagnostic")


if __name__ == "__main__":
    unittest.main()
