"""Strict CUDA FP32 versus FP16 ONNX Runtime benchmarks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort
import torch

from edgeai.benchmark import _detection_matching, _hardware_info
from edgeai.errors import VideoInputError
from edgeai.image_io import load_image
from edgeai.onnx_detector import DEFAULT_IOU_THRESHOLD, DEFAULT_ONNX_PATH, OnnxDetector
from edgeai.onnx_fp16 import DEFAULT_FP16_ONNX_PATH
from edgeai.video import VideoProcessor


def _strict_available() -> None:
    if not torch.cuda.is_available() or "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "FP16 comparison requires PyTorch CUDA and ONNX Runtime CUDAExecutionProvider; "
            f"torch.cuda.is_available={torch.cuda.is_available()}, providers={ort.get_available_providers()}"
        )


def _save_report(report: dict, output_path: str | Path) -> dict:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(output.resolve())
    return report


def _varied_images(paths: list[str | Path] | None, video_path: str | Path | None) -> tuple[list[np.ndarray], list[str]]:
    images, labels = [], []
    for path in paths or []:
        image, _ = load_image(path)
        images.append(image)
        labels.append(str(Path(path).resolve()))
    if video_path is not None:
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise VideoInputError(f"Could not open video for image collection: {video_path}")
            count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
            for index in np.linspace(0, count - 1, num=min(8, count), dtype=int):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok and frame is not None:
                    images.append(frame)
                    labels.append(f"{Path(video_path).resolve()}#frame={index}")
        finally:
            capture.release()
    if len(images) < 2:
        raise ValueError("Prediction consistency requires at least two varied images or video frames.")
    return images, labels


def _cuda_memory(detector: OnnxDetector) -> dict:
    # ORT allocations do not appear in PyTorch's CUDA allocator counters.
    return {"peak_allocated_mb": None, "peak_reserved_mb": None,
            "note": "Not measurable via PyTorch allocator; ONNX Runtime owns these CUDA allocations."}


def run_fp16_image_benchmark(
    image_paths: list[str | Path] | None = None,
    *,
    video_for_variety: str | Path | None = None,
    fp32_path: str | Path = DEFAULT_ONNX_PATH,
    fp16_path: str | Path = DEFAULT_FP16_ONNX_PATH,
    output_path: str | Path = "outputs/phase7a_image_benchmark.json",
    warmups: int = 10,
    runs_per_trial: int = 50,
    trials: int = 3,
    confidence: float = 0.25,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict:
    _strict_available()
    if warmups < 10 or runs_per_trial < 50 or trials < 3:
        raise ValueError("Use at least 10 warm-ups and 50 measured calls in each of at least 3 trials.")
    images, labels = _varied_images(image_paths, video_for_variety)
    configs, consistency, loaded = {}, [], {}
    for precision, model in (("FP32", fp32_path), ("FP16", fp16_path)):
        detector = OnnxDetector(model, device="cuda", precision=precision.lower(), require_cuda=True)
        loaded[precision] = detector
        all_total, all_stages, trial_means, last = [], {k: [] for k in ("preprocess", "execution", "postprocess")}, [], None
        for trial in range(trials):
            for n in range(warmups):
                detector.predict_frame(images[n % len(images)], confidence, iou_threshold=iou_threshold)
            samples = []
            for n in range(runs_per_trial):
                image = images[(trial * runs_per_trial + n) % len(images)]
                torch.cuda.synchronize(0)
                result, elapsed = detector.predict_frame(image, confidence, iou_threshold=iou_threshold)
                torch.cuda.synchronize(0)
                samples.append(elapsed)
                for key, field in (("preprocess", "preprocess_ms"), ("execution", "runtime_ms"), ("postprocess", "postprocess_ms")):
                    all_stages[key].append(detector.last_timings[field])
                last = detector.last_detections
            all_total.extend(samples)
            trial_means.append(mean(samples))
        configs[precision] = {
            "model": str(Path(model).resolve()), "model_file_size_bytes": detector.model_size_bytes,
            "session_initialization_ms": detector.model_load_ms, "device": detector.device,
            "execution_provider": detector.execution_provider, "precision": detector.precision,
            "cpu_ep_fallback_disabled": detector.require_cuda,
            "warmups_per_trial": warmups, "measured_runs_per_trial": runs_per_trial,
            "trials": trials, "trial_mean_latency_ms": trial_means,
            "mean_latency_ms": mean(all_total), "median_latency_ms": median(all_total),
            "p95_latency_ms": float(np.percentile(all_total, 95)),
            "throughput_images_per_second": 1000 / mean(all_total),
            "median_trial_mean_latency_ms": median(trial_means),
            "stage_mean_ms": {key: mean(values) for key, values in all_stages.items()},
            "raw_latency_samples_ms": all_total, "gpu_memory": _cuda_memory(detector),
            "last_detections": [d.to_dict() for d in last or []],
        }
    for image, label in zip(images, labels):
        outputs = {}
        for precision, detector in loaded.items():
            detector.predict_frame(image, confidence, iou_threshold=iou_threshold)
            outputs[precision] = detector.last_detections
        consistency.append({"image": label, **_detection_matching(outputs["FP32"], outputs["FP16"])})
    report = {
        "benchmark_type": "strict_cuda_onnx_fp32_vs_fp16_image", "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_software": _hardware_info(), "input_images": labels,
        "settings": {"input_size": [640, 640], "batch_size": 1, "confidence": confidence,
                     "iou_threshold": iou_threshold, "warmups_per_trial": warmups,
                     "runs_per_trial": runs_per_trial, "trials": trials},
        "timing_method": "CUDA synchronized around each complete predict call; stage timers report preprocessing, ORT session.run execution, and post-processing separately. Model/session initialization excluded from latency.",
        "consistency_method": "Greedy same-class matching by IoU >= 0.5; per-image counts, confidence deltas, box-coordinate deltas and IoU reported.",
        "configurations": configs, "per_image_consistency": consistency,
    }
    return _save_report(report, output_path)


def run_fp16_video_benchmark(
    video_path: str | Path,
    *,
    fp32_path: str | Path = DEFAULT_ONNX_PATH,
    fp16_path: str | Path = DEFAULT_FP16_ONNX_PATH,
    output_dir: str | Path = "outputs/phase7a_video",
    output_path: str | Path = "outputs/phase7a_video_benchmark.json",
    warmup_frames: int = 3,
    confidence: float = 0.25,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict:
    _strict_available()
    frame_detections, runs, outputs = {"FP32": [], "FP16": []}, {}, {}
    for precision, model in (("FP32", fp32_path), ("FP16", fp16_path)):
        detector = OnnxDetector(model, device="cuda", precision=precision.lower(), require_cuda=True)
        run = VideoProcessor(detector).process_video(
            video_path, output_dir=Path(output_dir) / precision.lower(), confidence=confidence,
            warmup_frames=warmup_frames, track=False, image_size=640, iou_threshold=iou_threshold,
            frame_callback=lambda index, detections, key=precision: frame_detections[key].append(detections),
        )
        runs[precision] = run.to_dict()
        output = cv2.VideoCapture(run.annotated_video_path)
        try:
            outputs[precision] = {"opened": output.isOpened(),
                                  "frames": int(round(output.get(cv2.CAP_PROP_FRAME_COUNT))),
                                  "width": int(output.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  "height": int(output.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                  "fps": output.get(cv2.CAP_PROP_FPS)}
        finally:
            output.release()
    matches = [_detection_matching(a, b) | {"frame_index": i}
               for i, (a, b) in enumerate(zip(frame_detections["FP32"], frame_detections["FP16"]))]
    report = {"benchmark_type": "strict_cuda_onnx_fp32_vs_fp16_video", "created_utc": datetime.now(timezone.utc).isoformat(),
              "hardware_software": _hardware_info(), "input_video": str(Path(video_path).resolve()),
              "tracking": False, "settings": {"confidence": confidence, "iou_threshold": iou_threshold,
                                                "input_size": [640, 640], "warmup_frames": warmup_frames},
              "runs": runs, "output_video_validation": outputs, "frame_detection_consistency": matches}
    return _save_report(report, output_path)
