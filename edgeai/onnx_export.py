"""Reproducible FP32 export and structural validation for YOLO11n."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from time import perf_counter

import onnx
import onnxruntime as ort

from edgeai.errors import ModelLoadError

DEFAULT_ONNX_PATH = Path("models/yolo11n.onnx")
INPUT_SIZE = 640
EXPORT_OPSET = 17


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_onnx_artifact(
    model_path: str | Path,
    *,
    input_size: int = INPUT_SIZE,
    providers: list[str] | None = None,
    disable_cpu_fallback: bool = False,
) -> dict:
    """Check the graph and its runtime-visible fixed batch-one I/O shapes."""
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise ModelLoadError(f"ONNX model file does not exist: {path}")
    try:
        graph = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(graph)
        selected_providers = providers or ["CPUExecutionProvider"]
        if disable_cpu_fallback:
            selected_providers = [
                provider for provider in selected_providers if provider != "CPUExecutionProvider"
            ]
        options = ort.SessionOptions()
        if disable_cpu_fallback:
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=selected_providers
        )
        actual_providers = session.get_providers()
        if providers and actual_providers[0] != providers[0]:
            raise ValueError(
                f"Requested provider {providers[0]} was not selected: {actual_providers}"
            )
        inputs = [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in session.get_inputs()
        ]
        outputs = [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in session.get_outputs()
        ]
        expected_shape = [1, 3, input_size, input_size]
        if len(inputs) != 1 or inputs[0]["shape"] != expected_shape:
            raise ValueError(f"Expected one input shaped {expected_shape}, got {inputs}")
        if not outputs or len(outputs[0]["shape"]) != 3 or outputs[0]["shape"][0] != 1:
            raise ValueError(f"Expected batch-one YOLO detection output, got {outputs}")
        return {
            "path": str(path),
            "input": inputs[0],
            "outputs": outputs,
            "opsets": {item.domain or "ai.onnx": item.version for item in graph.opset_import},
            "file_size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "session_providers": actual_providers,
            "cpu_ep_fallback_disabled": disable_cpu_fallback,
        }
    except Exception as exc:
        if isinstance(exc, ModelLoadError):
            raise
        raise ModelLoadError(f"ONNX model validation failed for '{path}': {exc}") from exc


def export_yolo11n(
    weights_path: str | Path = "yolo11n.pt",
    output_path: str | Path = DEFAULT_ONNX_PATH,
    *,
    input_size: int = INPUT_SIZE,
) -> dict:
    """Export a copied checkpoint so the original PyTorch file is untouched.

    The export is deliberately static, FP32, batch one, square, opset 17,
    with NMS outside the graph so confidence and IoU thresholds remain runtime
    settings.
    """
    weights = Path(weights_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if input_size not in {320, 480, 640}:
        raise ValueError("input_size must be one of the supported sizes: 320, 480, 640")
    if not weights.is_file():
        raise ModelLoadError(f"PyTorch weights do not exist: {weights}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = _sha256(weights)
    try:
        from ultralytics import YOLO

        with tempfile.TemporaryDirectory(prefix="edgeai_export_", dir=destination.parent) as tmp:
            temporary_weights = Path(tmp) / weights.name
            shutil.copy2(weights, temporary_weights)
            model = YOLO(str(temporary_weights))
            started = perf_counter()
            exported = model.export(
                format="onnx",
                imgsz=input_size,
                batch=1,
                dynamic=False,
                half=False,
                nms=False,
                simplify=False,
                opset=EXPORT_OPSET,
                device="cpu",
                verbose=False,
            )
            export_ms = (perf_counter() - started) * 1000.0
            exported_path = Path(exported).resolve()
            if not exported_path.is_file():
                raise FileNotFoundError(f"Exporter did not create the expected artifact: {exported_path}")
            shutil.copy2(exported_path, destination)
        if _sha256(weights) != source_hash:
            raise RuntimeError("The source PyTorch weights changed during export.")
        report = validate_onnx_artifact(destination, input_size=input_size)
        report.update(
            {
                "source_weights": str(weights),
                "source_weights_sha256": source_hash,
                "export_ms": export_ms,
                "precision": "FP32",
                "batch_size": 1,
                "input_size": [input_size, input_size],
                "dynamic_shapes": False,
                "nms_embedded": False,
                "opset_requested": EXPORT_OPSET,
            }
        )
        manifest_path = destination.with_suffix(".json")
        manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    except Exception as exc:
        if isinstance(exc, ModelLoadError):
            raise
        raise ModelLoadError(f"YOLO11n ONNX export failed: {exc}") from exc
