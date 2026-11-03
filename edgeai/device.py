"""Execution-device selection with an explicit CPU fallback."""

import torch

from edgeai.errors import DeviceUnavailableError


def resolve_device(preference: str = "auto") -> str:
    """Return ``cpu`` or ``cuda:0`` based on preference and availability.

    ``auto`` prefers CUDA when PyTorch can use it. An explicit CUDA request
    fails clearly when CUDA is unavailable instead of silently changing device.
    """
    normalized = preference.strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")

    if normalized == "cpu":
        return "cpu"
    cuda_available = torch.cuda.is_available()
    if normalized == "cuda":
        if not cuda_available:
            raise DeviceUnavailableError(
                "CUDA was requested, but torch.cuda.is_available() is False."
            )
        return "cuda:0"
    return "cuda:0" if cuda_available else "cpu"


def ultralytics_device(device: str) -> str | int:
    """Translate our readable device name to Ultralytics' predict argument."""
    return 0 if device == "cuda:0" else "cpu"
