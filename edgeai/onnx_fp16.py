"""FP16 conversion and strict CUDA-only ONNX Runtime support."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import onnx
import onnxruntime as ort
import torch

from edgeai.errors import DeviceUnavailableError, ModelLoadError
from edgeai.onnx_export import DEFAULT_ONNX_PATH, _sha256, validate_onnx_artifact

DEFAULT_FP16_ONNX_PATH = Path("models/yolo11n_fp16.onnx")


def load_onnx_detector(
    precision: str,
    *,
    fp32_path: str | Path = DEFAULT_ONNX_PATH,
    fp16_path: str | Path = DEFAULT_FP16_ONNX_PATH,
    allow_fp32_fallback: bool = False,
):
    """Load requested precision; permit a reported FP32 fallback only by opt-in."""
    from edgeai.onnx_detector import OnnxDetector

    if precision == "fp32":
        return OnnxDetector(fp32_path, device="auto"), None
    if precision != "fp16":
        raise ValueError("precision must be fp32 or fp16")
    try:
        return OnnxDetector(fp16_path, device="cuda", precision="fp16", require_cuda=True), None
    except Exception as exc:
        if not allow_fp32_fallback:
            raise
        detector = OnnxDetector(fp32_path, device="cuda")
        message = f"Explicit FP16-to-FP32 fallback: {type(exc).__name__}: {exc}"
        detector.fallback_error = message
        return detector, message


def convert_onnx_to_fp16(
    fp32_model_path: str | Path = DEFAULT_ONNX_PATH,
    output_path: str | Path = DEFAULT_FP16_ONNX_PATH,
) -> dict:
    """Convert internal graph tensors to FP16 while keeping public I/O FP32.

    This uses Microsoft's ONNX float16 converter and its supported-op block
    list. The original FP32 graph is read only and is never overwritten.
    A strict CUDA-only session and a real inference call validate the result.
    """
    source = Path(fp32_model_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise ModelLoadError(f"FP32 ONNX model file does not exist: {source}")
    if source == destination:
        raise ValueError("FP16 output must be a separate file; the FP32 model cannot be overwritten.")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise DeviceUnavailableError(
            f"CUDAExecutionProvider is unavailable; installed providers: {ort.get_available_providers()}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = _sha256(source)
    try:
        from onnxconverter_common import float16

        graph = onnx.load(str(source))
        started = perf_counter()
        fp16_graph = float16.convert_float_to_float16(graph, keep_io_types=True)
        onnx.checker.check_model(fp16_graph)
        onnx.save(fp16_graph, str(destination))
        conversion_ms = (perf_counter() - started) * 1000.0

        report = validate_onnx_artifact(
            destination,
            providers=["CUDAExecutionProvider"],
            disable_cpu_fallback=True,
        )
        session_options = ort.SessionOptions()
        session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session_options.enable_profiling = True
        session = ort.InferenceSession(
            str(destination),
            sess_options=session_options,
            providers=["CUDAExecutionProvider"],
        )
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA provider was not selected: {session.get_providers()}")
        input_meta = session.get_inputs()[0]
        input_tensor = np.zeros([1, 3, 640, 640], dtype=np.float32)
        test_outputs = session.run([session.get_outputs()[0].name], {input_meta.name: input_tensor})
        profile_path = Path(session.end_profiling())
        try:
            profile_events = json.loads(profile_path.read_text(encoding="utf-8"))
            node_providers = sorted({
                event.get("args", {}).get("provider")
                for event in profile_events
                if event.get("cat") == "Node" and event.get("args", {}).get("provider")
            })
        finally:
            profile_path.unlink(missing_ok=True)
        if not node_providers or any(provider != "CUDAExecutionProvider" for provider in node_providers):
            raise RuntimeError(f"Observed ONNX node execution providers were not CUDA-only: {node_providers}")
        if test_outputs[0].shape != tuple(report["outputs"][0]["shape"]):
            raise ValueError(f"FP16 smoke inference returned unexpected shape {test_outputs[0].shape}")
        if _sha256(source) != source_hash:
            raise RuntimeError("The source FP32 ONNX model changed during conversion.")

        fp16_initializers = sum(
            initializer.data_type == onnx.TensorProto.FLOAT16 for initializer in fp16_graph.graph.initializer
        )
        report.update(
            {
                "source_fp32_model": str(source),
                "source_fp32_sha256": source_hash,
                "conversion_method": "onnxconverter-common.float16.convert_float_to_float16",
                "converter_version": "1.16.0",
                "conversion_ms": conversion_ms,
                "precision": "FP16 internal graph with FP32 model input/output",
                "fp16_initializer_count": fp16_initializers,
                "execution_provider_verified": session.get_providers()[0],
                "observed_node_execution_providers": node_providers,
                "cpu_ep_fallback_disabled": True,
                "cuda_smoke_inference": "passed",
            }
        )
        destination.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    except Exception as exc:
        if isinstance(exc, (ModelLoadError, DeviceUnavailableError, ValueError)):
            raise
        raise ModelLoadError(
            f"FP16 ONNX conversion or strict CUDA validation failed: {type(exc).__name__}: {exc}"
        ) from exc
