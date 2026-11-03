"""Calibrated, partial static QDQ INT8 quantization for YOLO11n ONNX."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from edgeai.errors import ImageInputError, ModelLoadError, VideoInputError
from edgeai.image_io import load_image
from edgeai.onnx_detector import DEFAULT_IOU_THRESHOLD, DEFAULT_ONNX_PATH, OnnxDetector, prepare_onnx_input

DEFAULT_INT8_ONNX_PATH = Path("models/yolo11n_int8.onnx")
DEFAULT_CALIBRATION_DIR = Path("outputs/int8_calibration")
DEFAULT_CALIBRATION_VIDEO = Path("outputs/cuda_video_smoke.mp4")
CALIBRATION_FRAME_COUNT = 16
FLOAT_ONLY_NODE_PREFIXES = ("/model.10/", "/model.23/")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uniform_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count < 1 or sample_count < 1:
        return []
    count = min(frame_count, sample_count)
    # Integer arithmetic makes selected frame indices stable across runs/platforms.
    return sorted({round(index * (frame_count - 1) / max(count - 1, 1)) for index in range(count)})


def sample_calibration_frames(
    video_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_CALIBRATION_DIR,
    sample_count: int = CALIBRATION_FRAME_COUNT,
) -> tuple[list[Path], dict]:
    """Decode and persist deterministic, evenly spaced frames from a local video."""
    source = Path(video_path).expanduser().resolve()
    if source.suffix.lower() not in {".mp4", ".avi"}:
        raise VideoInputError(f"Unsupported calibration video format: {source.suffix}")
    if not source.is_file():
        raise VideoInputError(f"Calibration video does not exist: {source}")
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise VideoInputError(f"Could not open calibration video: {source}")
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if frame_count < 1:
            raise VideoInputError(f"Calibration video has no readable frames: {source}")
        indices = _uniform_indices(frame_count, sample_count)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        saved = []
        for ordinal, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                raise VideoInputError(f"Could not decode calibration frame {frame_index} from {source}")
            target = destination / f"calibration_{ordinal:03d}_frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise ImageInputError(f"Could not save calibration frame: {target}")
            saved.append(target.resolve())
    finally:
        capture.release()
    return saved, {
        "source_video": str(source),
        "source_video_sha256": _sha256(source),
        "source_frame_count": frame_count,
        "selected_frame_indices": indices,
        "calibration_image_count": len(saved),
        "calibration_image_paths": [str(path) for path in saved],
        "sampling": "deterministic evenly spaced frame indices using rounded linear spacing",
    }


class ImageCalibrationReader(CalibrationDataReader):
    """Feed preprocessed images using the same preprocessing as OnnxDetector."""

    def __init__(self, image_paths: list[str | Path], input_name: str):
        if not image_paths:
            raise ValueError("At least one calibration image is required.")
        self.image_paths = [Path(path).expanduser().resolve() for path in image_paths]
        self.input_name = input_name
        self._index = 0

    def get_next(self):
        if self._index >= len(self.image_paths):
            return None
        image_path = self.image_paths[self._index]
        image, _ = load_image(image_path)
        tensor, _, _, _ = prepare_onnx_input(image)
        self._index += 1
        return {self.input_name: tensor}

    def rewind(self):
        self._index = 0


def _quantized_conv_names(model: onnx.ModelProto) -> list[str]:
    return [
        node.name for node in model.graph.node
        if node.op_type == "Conv" and not any(node.name.startswith(prefix) for prefix in FLOAT_ONLY_NODE_PREFIXES)
    ]


def quantize_yolo11n_static(
    calibration_images: list[str | Path],
    *,
    fp32_model_path: str | Path = DEFAULT_ONNX_PATH,
    output_path: str | Path = DEFAULT_INT8_ONNX_PATH,
    calibration_metadata: dict | None = None,
) -> dict:
    """Create S8S8 static QDQ model for selected Conv nodes, preserving float-only heads."""
    source = Path(fp32_model_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise ModelLoadError(f"FP32 ONNX model file does not exist: {source}")
    if source == destination or destination.name.lower() in {"yolo11n.onnx", "yolo11n_fp16.onnx"}:
        raise ValueError("INT8 output must be separate; FP32 and FP16 artifacts cannot be overwritten.")
    if len(calibration_images) < 4:
        raise ValueError("Use at least four varied calibration images for this static calibration experiment.")
    source_hash = _sha256(source)
    graph = onnx.load(str(source))
    onnx.checker.check_model(graph)
    input_name = graph.graph.input[0].name
    conv_nodes = [node for node in graph.graph.node if node.op_type == "Conv"]
    quantize_nodes = _quantized_conv_names(graph)
    float_only_nodes = [
        node.name for node in conv_nodes
        if any(node.name.startswith(prefix) for prefix in FLOAT_ONLY_NODE_PREFIXES)
    ]
    if not quantize_nodes:
        raise ModelLoadError("No suitable Conv nodes were found for partial INT8 quantization.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    reader = ImageCalibrationReader(calibration_images, input_name)
    started = perf_counter()
    quantize_static(
        model_input=source,
        model_output=destination,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv"],
        nodes_to_exclude=float_only_nodes,
        per_channel=True,
        reduce_range=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=["CPUExecutionProvider"],
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )
    quantization_ms = (perf_counter() - started) * 1000.0
    try:
        quantized_graph = onnx.load(str(destination))
        onnx.checker.check_model(quantized_graph)
        qdq_count = sum(node.op_type in {"QuantizeLinear", "DequantizeLinear"} for node in quantized_graph.graph.node)
        if qdq_count == 0:
            raise ModelLoadError("Quantized graph contains no QDQ operators; refusing to report an FP32 model as INT8.")
        # Runtime validation and smoke inference are CPU-only by design for this phase.
        detector = OnnxDetector(destination, device="cpu", precision="int8")
        smoke_image, _ = load_image(calibration_images[0])
        detector.predict_frame(smoke_image, confidence=0.25, iou_threshold=DEFAULT_IOU_THRESHOLD)
        if _sha256(source) != source_hash:
            raise ModelLoadError("FP32 source ONNX model changed during quantization.")
        report = {
            "path": str(destination),
            "source_fp32_model": str(source),
            "source_fp32_sha256": source_hash,
            "quantization_method": "ONNX Runtime static S8S8 QDQ with MinMax calibration",
            "onnxruntime_version": ort.__version__,
            "sha256": _sha256(destination),
            "onnx_opset": next((item.version for item in quantized_graph.opset_import if item.domain in ("", "ai.onnx")), None),
            "quantization_ms": quantization_ms,
            "file_size_bytes": destination.stat().st_size,
            "fp32_file_size_bytes": source.stat().st_size,
            "calibration": {
                "image_count": len(calibration_images),
                "preprocessing": "same 640x640 letterbox, BGR-to-RGB, NCHW, float32 normalization to [0,1] as OnnxDetector",
                "dataset": calibration_metadata or {"calibration_image_paths": [str(Path(p).resolve()) for p in calibration_images]},
                "method": "MinMax static activation calibration; per-channel symmetric INT8 weights and symmetric INT8 activations",
            },
            "graph_analysis": {
                "original_conv_count": len(conv_nodes),
                "quantized_conv_node_names": quantize_nodes,
                "quantized_conv_count": len(quantize_nodes),
                "float_only_conv_node_names": float_only_nodes,
                "float_only_conv_count": len(float_only_nodes),
                "qdq_node_count": qdq_count,
                "non_conv_compute_ops_preserved_in_float": sorted({
                    node.op_type for node in graph.graph.node if node.op_type not in {"Conv", "Constant"}
                }),
                "reason_float_only": "Keep the YOLO11n attention block and all detection-head/DFL convolutions in FP32 to reduce accuracy risk. Non-Conv ops are not quantized by this selected-op policy.",
            },
            "validation": {
                "onnx_checker": "passed",
                "cpu_session": "passed",
                "execution_provider": detector.execution_provider,
                "input_shape": detector.input.shape,
                "output_shapes": [item.shape for item in detector.session.get_outputs()],
                "smoke_inference": "passed",
            },
        }
        destination.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def create_default_int8_model(
    *,
    calibration_video: str | Path = DEFAULT_CALIBRATION_VIDEO,
    fp32_model_path: str | Path = DEFAULT_ONNX_PATH,
    output_path: str | Path = DEFAULT_INT8_ONNX_PATH,
    output_dir: str | Path = DEFAULT_CALIBRATION_DIR,
    sample_count: int = CALIBRATION_FRAME_COUNT,
) -> dict:
    images, metadata = sample_calibration_frames(calibration_video, output_dir=output_dir, sample_count=sample_count)
    return quantize_yolo11n_static(images, fp32_model_path=fp32_model_path, output_path=output_path,
                                   calibration_metadata=metadata)
