"""Pretrained Ultralytics YOLO model loading."""

from time import perf_counter

from edgeai.device import ultralytics_device
from edgeai.errors import ModelLoadError


def load_yolo_model(model_name: str, device: str):
    """Load model weights once and return ``(model, elapsed_ms)``.

    The model is moved to the selected device here so model setup and image
    prediction are accounted for separately.
    """
    started = perf_counter()
    try:
        from ultralytics import YOLO

        model = YOLO(model_name)
        target = "cuda:0" if device == "cuda:0" else "cpu"
        model.model.to(target)
        if device == "cuda:0":
            import torch

            torch.cuda.synchronize(0)
        elapsed_ms = (perf_counter() - started) * 1000.0
        return model, elapsed_ms
    except Exception as exc:
        raise ModelLoadError(f"Could not load YOLO model '{model_name}': {exc}") from exc
