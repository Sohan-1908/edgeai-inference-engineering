import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from quantize_onnx_qoperator import (
    _validate_experiment_output_path,
    inspect_qoperator_graph,
    quantize_qoperator_model,
)


def _source_model() -> onnx.ModelProto:
    image = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    quantized_conv_out = helper.make_tensor_value_info(
        "selected_out", TensorProto.FLOAT, [1, 1, 2, 2]
    )
    excluded_conv_out = helper.make_tensor_value_info(
        "excluded_out", TensorProto.FLOAT, [1, 1, 2, 2]
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv", ["x", "selected_weight"], ["selected_out"], name="selected_conv"
            ),
            helper.make_node(
                "Conv",
                ["x", "excluded_weight"],
                ["excluded_out"],
                name="excluded_conv",
            ),
        ],
        "source_conv_graph",
        [image],
        [quantized_conv_out, excluded_conv_out],
        [
            numpy_helper.from_array(
                np.asarray([[[[0.75]]]], dtype=np.float32), "selected_weight"
            ),
            numpy_helper.from_array(
                np.asarray([[[[0.5]]]], dtype=np.float32), "excluded_weight"
            ),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    return model


def _qoperator_model() -> onnx.ModelProto:
    image = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    selected_out = helper.make_tensor_value_info(
        "selected_out", TensorProto.FLOAT, [1, 1, 2, 2]
    )
    excluded_out = helper.make_tensor_value_info(
        "excluded_out", TensorProto.FLOAT, [1, 1, 2, 2]
    )
    initializers = [
        numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), "x_scale"),
        numpy_helper.from_array(np.asarray(0, dtype=np.int8), "x_zero"),
        numpy_helper.from_array(
            np.asarray([[[[8]]]], dtype=np.int8), "selected_weight_q"
        ),
        numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), "weight_scale"),
        numpy_helper.from_array(np.asarray(0, dtype=np.int8), "weight_zero"),
        numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), "y_scale"),
        numpy_helper.from_array(np.asarray(0, dtype=np.int8), "y_zero"),
        numpy_helper.from_array(
            np.asarray([[[[0.5]]]], dtype=np.float32), "excluded_weight"
        ),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear", ["x", "x_scale", "x_zero"], ["x_q"], name="input_quant"
        ),
        helper.make_node(
            "QLinearConv",
            [
                "x_q",
                "x_scale",
                "x_zero",
                "selected_weight_q",
                "weight_scale",
                "weight_zero",
                "y_scale",
                "y_zero",
            ],
            ["selected_q"],
            name="selected_conv_quant",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["selected_q", "y_scale", "y_zero"],
            ["selected_out"],
            name="selected_output_dequant",
        ),
        helper.make_node(
            "Conv",
            ["x", "excluded_weight"],
            ["excluded_out"],
            name="excluded_conv",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "qoperator_with_excluded_float_conv",
        [image],
        [selected_out, excluded_out],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    return model


class _OneInputCalibrationReader:
    def rewind(self):
        return None

    def get_next(self):
        return {"x": np.asarray([[[[0.2, 0.4], [0.6, 0.8]]]], dtype=np.float32)}


def _fake_quantization_api():
    def quantize_static(*, model_output, **kwargs):
        if kwargs["quant_format"] != "QOperator":
            raise AssertionError("Expected QOperator quantization.")
        if kwargs["per_channel"] is not True:
            raise AssertionError("Expected per-channel weights.")
        onnx.save(_qoperator_model(), str(model_output))

    return (
        quantize_static,
        SimpleNamespace(QOperator="QOperator"),
        SimpleNamespace(QInt8="QInt8"),
        SimpleNamespace(MinMax="MinMax"),
    )


class QOperatorGenerationTests(unittest.TestCase):
    def _fixture(self, directory: str):
        root = Path(directory)
        models = root / "models"
        experiment_dir = models / "experiments"
        experiment_dir.mkdir(parents=True)
        fp32_path = models / "fp32.onnx"
        qdq_path = models / "qdq.onnx"
        onnx.save(_source_model(), str(fp32_path))
        onnx.save(_source_model(), str(qdq_path))
        calibration = root / "calibration.jpg"
        calibration.write_bytes(b"synthetic-calibration-reference")
        return fp32_path, qdq_path, experiment_dir, calibration

    def test_generation_metadata_graph_inspection_and_cpu_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            fp32, qdq, experiment_dir, calibration = self._fixture(directory)
            source_hash = hashlib.sha256(fp32.read_bytes()).hexdigest()
            output = experiment_dir / "qoperator.onnx"
            with patch(
                "quantize_onnx_qoperator._quantization_api",
                return_value=_fake_quantization_api(),
            ) as api:
                report = quantize_qoperator_model(
                    fp32_model=fp32,
                    qdq_model=qdq,
                    output_model=output,
                    calibration_paths=[calibration],
                    selected_conv_names=["selected_conv"],
                    excluded_conv_names=["excluded_conv"],
                    calibration_data_reader=_OneInputCalibrationReader(),
                    experiment_model_dir=experiment_dir,
                )

            self.assertTrue(output.is_file())
            self.assertEqual(hashlib.sha256(fp32.read_bytes()).hexdigest(), source_hash)
            self.assertEqual(report["source_fp32_sha256"], source_hash)
            self.assertEqual(
                report["output_model_sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            self.assertEqual(report["quantization_configuration"]["format"], "QOperator")
            self.assertTrue(report["quantization_configuration"]["per_channel_weights"])
            self.assertEqual(report["quantized_conv_node_count"], 1)
            self.assertEqual(report["unsupported_or_unconverted_selected_conv_count"], 0)
            self.assertEqual(report["deliberately_excluded_conv_node_count"], 1)
            self.assertEqual(report["validation"]["execution_provider"], "CPUExecutionProvider")
            self.assertTrue(report["validation"]["execution_succeeded"])
            self.assertTrue(report["validation"]["onnx_checker"] == "passed")
            self.assertEqual(
                report["generated_graph"]["remaining_float_conv_node_names"],
                ["excluded_conv"],
            )
            self.assertEqual(api.call_count, 1)

    def test_selected_but_unconverted_conv_is_reported_as_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            fp32, qdq, experiment_dir, calibration = self._fixture(directory)
            output = experiment_dir / "partially_converted.onnx"
            with patch(
                "quantize_onnx_qoperator._quantization_api",
                return_value=_fake_quantization_api(),
            ):
                report = quantize_qoperator_model(
                    fp32_model=fp32,
                    qdq_model=qdq,
                    output_model=output,
                    calibration_paths=[calibration],
                    selected_conv_names=["selected_conv", "excluded_conv"],
                    excluded_conv_names=[],
                    calibration_data_reader=_OneInputCalibrationReader(),
                    experiment_model_dir=experiment_dir,
                )

            self.assertEqual(
                report["unsupported_or_unconverted_selected_conv_count"], 1
            )
            self.assertEqual(
                report["unsupported_or_unconverted_selected_conv_names"],
                ["excluded_conv"],
            )

    def test_output_path_must_be_isolated_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            fp32, qdq, experiment_dir, _ = self._fixture(directory)
            with self.assertRaisesRegex(ValueError, "cannot overwrite"):
                _validate_experiment_output_path(
                    fp32,
                    fp32_model=fp32,
                    qdq_model=qdq,
                    experiment_model_dir=experiment_dir,
                )
            with self.assertRaisesRegex(ValueError, "inside the experiment"):
                _validate_experiment_output_path(
                    Path(directory) / "outside.onnx",
                    fp32_model=fp32,
                    qdq_model=qdq,
                    experiment_model_dir=experiment_dir,
                )

    def test_existing_experiment_model_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            fp32, qdq, experiment_dir, _ = self._fixture(directory)
            output = experiment_dir / "existing.onnx"
            output.write_bytes(b"preserve")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                _validate_experiment_output_path(
                    output,
                    fp32_model=fp32,
                    qdq_model=qdq,
                    experiment_model_dir=experiment_dir,
                )
            self.assertEqual(output.read_bytes(), b"preserve")

    def test_quantization_unavailable_fails_explicitly_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            fp32, qdq, experiment_dir, calibration = self._fixture(directory)
            output = experiment_dir / "unavailable.onnx"
            with patch(
                "quantize_onnx_qoperator._quantization_api",
                side_effect=RuntimeError("QOperator unavailable"),
            ):
                with self.assertRaisesRegex(RuntimeError, "QOperator unavailable"):
                    quantize_qoperator_model(
                        fp32_model=fp32,
                        qdq_model=qdq,
                        output_model=output,
                        calibration_paths=[calibration],
                        selected_conv_names=["selected_conv"],
                        excluded_conv_names=["excluded_conv"],
                        calibration_data_reader=_OneInputCalibrationReader(),
                        experiment_model_dir=experiment_dir,
                    )
            self.assertFalse(output.exists())

    def test_graph_inspector_reports_quantized_and_float_conv_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qoperator.onnx"
            onnx.save(_qoperator_model(), str(path))

            report = inspect_qoperator_graph(path)

            self.assertEqual(report["qoperator_diagnostics"]["qlinear_conv_node_count"], 1)
            self.assertEqual(report["qoperator_diagnostics"]["conv_node_count"], 1)
            self.assertEqual(
                report["qoperator_diagnostics"]["quantize_linear_node_count"], 1
            )
            self.assertEqual(
                report["qoperator_diagnostics"]["dequantize_linear_node_count"], 1
            )
            self.assertTrue(
                report["qoperator_diagnostics"]["quantized_convolution_survives_graph"]
            )
            self.assertEqual(
                report["qoperator_diagnostics"]["remaining_float_conv_node_names"],
                ["excluded_conv"],
            )


if __name__ == "__main__":
    unittest.main()
