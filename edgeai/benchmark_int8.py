"""Controlled CPU benchmark and consistency checks for static QDQ INT8."""

from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort
import psutil
import torch

from edgeai.benchmark import _detection_matching
from edgeai.errors import VideoInputError
from edgeai.image_io import load_image
from edgeai.onnx_detector import DEFAULT_ONNX_PATH, OnnxDetector
from edgeai.onnx_int8 import DEFAULT_INT8_ONNX_PATH, _uniform_indices


def sample_benchmark_video_frames(
    video_path: str | Path,
    *,
    output_dir: str | Path = "outputs/int8_benchmark_frames",
    excluded_indices: set[int] | None = None,
    sample_count: int = 12,
) -> tuple[list[Path], dict]:
    """Save deterministic clip frames while excluding calibration frame indices."""
    source = Path(video_path).expanduser().resolve()
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise VideoInputError(f"Could not open benchmark video: {source}")
        total = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        excluded = excluded_indices or set()
        candidates = [index for index in range(total) if index not in excluded]
        if not candidates:
            raise VideoInputError("No benchmark frames remain after excluding calibration frames.")
        slots = _uniform_indices(len(candidates), min(sample_count, len(candidates)))
        indices = [candidates[slot] for slot in slots]
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        paths = []
        for output_index, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                raise VideoInputError(f"Could not decode benchmark frame {frame_index} from {source}")
            target = destination / f"benchmark_{output_index:03d}_frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise VideoInputError(f"Could not write benchmark frame: {target}")
            paths.append(target.resolve())
    finally:
        capture.release()
    return paths, {"source_video": str(source), "frame_count": total,
                   "selected_indices": indices, "excluded_calibration_indices": sorted(excluded),
                   "benchmark_image_count": len(paths)}


def run_int8_cpu_benchmark(
    image_paths: list[str | Path],
    *,
    fp32_path: str | Path = DEFAULT_ONNX_PATH,
    int8_path: str | Path = DEFAULT_INT8_ONNX_PATH,
    output_path: str | Path = "outputs/phase7b_cpu_benchmark.json",
    warmups: int = 10,
    runs_per_trial: int = 50,
    trials: int = 3,
    cpu_threads: int = 4,
    confidence: float = 0.25,
    iou_threshold: float = 0.7,
    image_manifest: dict | None = None,
) -> dict:
    if warmups < 10 or runs_per_trial < 50 or trials < 3:
        raise ValueError("INT8 benchmark requires at least 10 warm-ups, 50 runs, and 3 trials.")
    if len(image_paths) < 4 or cpu_threads < 1:
        raise ValueError("Provide at least four varied images and a positive CPU thread count.")
    images, labels = [], []
    for path in image_paths:
        image, _ = load_image(path)
        images.append(image)
        labels.append(str(Path(path).expanduser().resolve()))
    process = psutil.Process()
    configs, predictions = {}, {}
    for precision, model_path in (("FP32", fp32_path), ("INT8", int8_path)):
        gc.collect()
        rss_before = process.memory_info().rss
        detector = OnnxDetector(model_path, device="cpu", cpu_threads=cpu_threads,
                                precision="int8" if precision == "INT8" else "fp32")
        rss_after_init = process.memory_info().rss
        samples, trial_means = [], []
        stages = {key: [] for key in ("preprocess", "execution", "postprocess")}
        peak_rss = rss_after_init
        for trial in range(trials):
            for index in range(warmups):
                detector.predict_frame(images[index % len(images)], confidence, iou_threshold=iou_threshold)
            measured = []
            for index in range(runs_per_trial):
                image = images[(trial * runs_per_trial + index) % len(images)]
                _, elapsed = detector.predict_frame(image, confidence, iou_threshold=iou_threshold)
                measured.append(elapsed)
                peak_rss = max(peak_rss, process.memory_info().rss)
                for stage, key in (("preprocess", "preprocess_ms"),
                                   ("execution", "runtime_ms"),
                                   ("postprocess", "postprocess_ms")):
                    stages[stage].append(float(detector.last_timings[key]))
            samples.extend(measured)
            trial_means.append(mean(measured))
        predictions[precision] = {}
        for image, label in zip(images, labels):
            detector.predict_frame(image, confidence, iou_threshold=iou_threshold)
            predictions[precision][label] = detector.last_detections
        configs[precision] = {
            "model": str(Path(model_path).resolve()),
            "runtime": "onnxruntime", "runtime_version": ort.__version__,
            "execution_provider": detector.execution_provider, "device": detector.device,
            "precision": precision, "model_initialization_ms": detector.model_load_ms,
            "model_file_size_bytes": detector.model_size_bytes,
            "input_size": [640, 640], "batch_size": 1,
            "warmups_per_trial": warmups, "measured_runs_per_trial": runs_per_trial,
            "trial_count": trials, "trial_mean_latency_ms": trial_means,
            "mean_latency_ms": mean(samples), "median_latency_ms": median(samples),
            "p95_latency_ms": float(np.percentile(samples, 95)),
            "throughput_images_per_second": 1000.0 / mean(samples),
            "median_trial_mean_latency_ms": median(trial_means),
            "stage_mean_ms": {key: mean(values) for key, values in stages.items()},
            "raw_latency_samples_ms": samples,
            "process_memory": {
                "rss_before_session_mb": rss_before / (1024**2),
                "rss_after_session_init_mb": rss_after_init / (1024**2),
                "peak_rss_during_benchmark_mb": peak_rss / (1024**2),
                "peak_growth_after_init_mb": (peak_rss - rss_after_init) / (1024**2),
                "measurement": "psutil process RSS; includes Python, runtime and all still-loaded sessions.",
            },
        }
        del detector
        gc.collect()
    consistency = []
    for label in labels:
        consistency.append({"image": label,
                            **_detection_matching(predictions["FP32"][label], predictions["INT8"][label])})
    report = {
        "benchmark_type": "onnx_cpu_fp32_vs_static_int8",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_software": {
            "platform": __import__("platform").platform(),
            "processor": __import__("platform").processor(),
            "pytorch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "onnxruntime": ort.__version__,
            "onnxruntime_available_providers": ort.get_available_providers(),
            "cpu_threads": cpu_threads,
        },
        "benchmark_images": labels, "image_manifest": image_manifest,
        "settings": {"input_size": [640, 640], "batch_size": 1, "confidence": confidence,
                     "iou_threshold": iou_threshold, "warmups_per_trial": warmups,
                     "measured_runs_per_trial": runs_per_trial, "trials": trials},
        "timing_method": "Wall-clock around full OnnxDetector.predict_frame. Stage means separate preprocessing, ORT execution, and post-processing. Session initialization excluded from latency. Same image sequence/order used in both configurations.",
        "consistency_method": "Greedy same-class matching by IoU >= 0.5; per-image detection counts, matched/unmatched counts, mean IoU, confidence and box-coordinate differences.",
        "configurations": configs, "per_image_consistency": consistency,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(destination.resolve())
    return report
