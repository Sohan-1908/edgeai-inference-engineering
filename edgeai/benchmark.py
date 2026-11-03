"""Controlled PyTorch / ONNX Runtime image and video comparisons."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort
import torch

from edgeai.config import DEFAULT_MODEL
from edgeai.detector import ImageDetector
from edgeai.errors import ModelLoadError
from edgeai.image_io import load_image
from edgeai.onnx_detector import DEFAULT_IOU_THRESHOLD, DEFAULT_ONNX_PATH, OnnxDetector
from edgeai.postprocess import extract_detections
from edgeai.video import VideoProcessor

IMAGE_SIZE = 640
DEFAULT_CONFIDENCE = 0.25


def _detection_matching(reference, candidate, iou_threshold: float = 0.5) -> dict:
    """Greedily match same-class boxes by IoU and summarize differences."""
    pairs = []
    for ref_index, ref in enumerate(reference):
        for cand_index, cand in enumerate(candidate):
            if ref.class_id != cand.class_id:
                continue
            a = np.asarray(ref.bbox_xyxy, dtype=np.float64)
            b = np.asarray(cand.bbox_xyxy, dtype=np.float64)
            top_left = np.maximum(a[:2], b[:2])
            bottom_right = np.minimum(a[2:], b[2:])
            intersection = float(np.prod(np.maximum(bottom_right - top_left, 0)))
            area_a = float(np.prod(np.maximum(a[2:] - a[:2], 0)))
            area_b = float(np.prod(np.maximum(b[2:] - b[:2], 0)))
            iou = intersection / max(area_a + area_b - intersection, 1e-9)
            if iou >= iou_threshold:
                pairs.append((iou, ref_index, cand_index))
    pairs.sort(reverse=True)
    used_ref, used_candidate, matched = set(), set(), []
    for iou, ref_index, cand_index in pairs:
        if ref_index not in used_ref and cand_index not in used_candidate:
            used_ref.add(ref_index)
            used_candidate.add(cand_index)
            matched.append((iou, reference[ref_index], candidate[cand_index]))
    return {
        "reference_detection_count": len(reference),
        "candidate_detection_count": len(candidate),
        "matched_count": len(matched),
        "missed_reference_count": len(reference) - len(matched),
        "additional_candidate_count": len(candidate) - len(matched),
        "class_agreement_rate": len(matched) / max(len(candidate), 1),
        "match_rate_reference": len(matched) / max(len(reference), 1),
        "iou_threshold": iou_threshold,
        "mean_matched_iou": mean([item[0] for item in matched]) if matched else None,
        "mean_absolute_confidence_difference": (
            mean([abs(a.confidence - b.confidence) for _, a, b in matched]) if matched else None
        ),
        "mean_absolute_box_coordinate_difference_px": (
            mean(
                [
                    float(np.mean(np.abs(np.asarray(a.bbox_xyxy) - np.asarray(b.bbox_xyxy))))
                    for _, a, b in matched
                ]
            )
            if matched
            else None
        ),
    }


def _hardware_info() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "pytorch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "onnxruntime": ort.__version__,
        "onnxruntime_available_providers": ort.get_available_providers(),
        "opencv": cv2.__version__,
    }


def _configuration(
    name: str,
    detector,
    measured_samples,
    trial_means,
    stage_samples,
    detections,
    confidence: float,
    iou_threshold: float,
):
    return {
        "name": name,
        "status": "ok",
        "runtime": detector.runtime_name,
        "runtime_version": detector.runtime_version,
        "device": detector.device,
        "execution_provider": detector.execution_provider,
        "model": detector.model_name,
        "model_load_or_session_init_ms": detector.model_load_ms,
        "model_file_size_bytes": detector.model_size_bytes,
        "input_size": [IMAGE_SIZE, IMAGE_SIZE],
        "batch_size": 1,
        "precision": "FP32",
        "confidence_threshold": confidence,
        "iou_threshold": iou_threshold,
        "warmup_runs_per_trial": 10,
        "measured_runs_per_trial": len(measured_samples) // max(len(trial_means), 1),
        "trial_count": len(trial_means),
        "trial_mean_latencies_ms": trial_means,
        "mean_latency_ms": mean(measured_samples),
        "median_latency_ms": median(measured_samples),
        "p95_latency_ms": float(np.percentile(measured_samples, 95)),
        "median_trial_mean_latency_ms": median(trial_means),
        "throughput_images_per_second": 1000.0 / median(trial_means),
        "raw_latency_samples_ms": measured_samples,
        "preprocessing_mean_ms": mean(stage_samples["preprocess"]) if stage_samples["preprocess"] else None,
        "runtime_execution_mean_ms": mean(stage_samples["runtime"]) if stage_samples["runtime"] else None,
        "postprocessing_mean_ms": mean(stage_samples["postprocess"]) if stage_samples["postprocess"] else None,
        "detections": [item.to_dict() for item in detections],
        "fallback_error": getattr(detector, "fallback_error", None),
        "gpu_memory": getattr(detector, "gpu_memory_report", None),
    }


def run_image_benchmark(
    image_path: str | Path,
    *,
    weights_path: str | Path = DEFAULT_MODEL,
    onnx_path: str | Path = DEFAULT_ONNX_PATH,
    output_path: str | Path = "outputs/onnx_image_benchmark.json",
    runs_per_trial: int = 30,
    trials: int = 3,
    cpu_threads: int = 4,
    confidence: float = DEFAULT_CONFIDENCE,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict:
    """Benchmark PyTorch/ORT on CPU and GPU (when ORT CUDA can initialize)."""
    if runs_per_trial < 30:
        raise ValueError("runs_per_trial must be at least 30 for benchmark comparisons")
    if trials < 1 or cpu_threads < 1:
        raise ValueError("trials and cpu_threads must be positive")
    image, image_load_ms = load_image(image_path)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(cpu_threads)
    results = []
    detectors = {}
    last_detections = {}
    try:
        specs = [("pytorch_cpu", "pytorch", "cpu"), ("onnx_cpu", "onnx", "cpu")]
        if torch.cuda.is_available():
            specs.append(("pytorch_cuda", "pytorch", "cuda"))
        else:
            results.append({"name": "pytorch_cuda", "status": "skipped", "error": "PyTorch CUDA is unavailable."})
        if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
            specs.append(("onnx_cuda", "onnx", "cuda"))
        else:
            results.append(
                {
                    "name": "onnx_cuda",
                    "status": "skipped_cuda_unavailable",
                    "error": f"CUDAExecutionProvider unavailable; available providers: {ort.get_available_providers()}",
                }
            )
        for name, runtime, device in specs:
            started_load = perf_counter()
            try:
                if runtime == "pytorch":
                    detector = ImageDetector(model_name=str(weights_path), device=device)
                else:
                    detector = OnnxDetector(model_path=onnx_path, device=device, cpu_threads=cpu_threads)
            except Exception as exc:
                results.append({"name": name, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
                continue
            if runtime == "onnx" and device == "cuda" and detector.device != "cuda:0":
                results.append(
                    {
                        "name": name,
                        "status": "skipped_cuda_initialization_failed",
                        "error": detector.fallback_error,
                        "fallback_configuration": "onnx_cpu",
                    }
                )
                continue
            if detector.model_load_ms == 0:
                detector.model_load_ms = (perf_counter() - started_load) * 1000.0
            detectors[name] = detector
            samples = []
            trial_means = []
            stages = {"preprocess": [], "runtime": [], "postprocess": []}
            gpu_baseline = None
            gpu_peak_allocated = None
            gpu_peak_reserved = None
            last_result = None
            for _ in range(trials):
                for _ in range(10):
                    last_result, _ = detector.predict_frame(
                        image,
                        confidence,
                        **(
                            {"image_size": IMAGE_SIZE, "iou_threshold": iou_threshold}
                            if runtime == "pytorch"
                            else {"iou_threshold": iou_threshold}
                        ),
                    )
                if detector.device == "cuda:0" and getattr(detector, "gpu_memory_supported", True):
                    torch.cuda.synchronize(0)
                    torch.cuda.reset_peak_memory_stats(0)
                    gpu_baseline = torch.cuda.memory_allocated(0) / (1024**2)
                trial_samples = []
                for _ in range(runs_per_trial):
                    last_result, elapsed_ms = detector.predict_frame(
                        image,
                        confidence,
                        **(
                            {"image_size": IMAGE_SIZE, "iou_threshold": iou_threshold}
                            if runtime == "pytorch"
                            else {"iou_threshold": iou_threshold}
                        ),
                    )
                    trial_samples.append(elapsed_ms)
                    sample_timings = getattr(detector, "last_timings", {}) or {}
                    result_speed = getattr(last_result, "speed", {}) or {}
                    stages["preprocess"].append(
                        float(sample_timings.get("preprocess_ms", result_speed.get("preprocess", 0.0)))
                    )
                    stages["runtime"].append(
                        float(sample_timings.get("runtime_ms", result_speed.get("inference", elapsed_ms)))
                    )
                    stages["postprocess"].append(
                        float(sample_timings.get("postprocess_ms", result_speed.get("postprocess", 0.0)))
                    )
                if detector.device == "cuda:0" and getattr(detector, "gpu_memory_supported", True):
                    torch.cuda.synchronize(0)
                    gpu_peak_allocated = max(
                        gpu_peak_allocated or 0.0, torch.cuda.max_memory_allocated(0) / (1024**2)
                    )
                    gpu_peak_reserved = max(
                        gpu_peak_reserved or 0.0, torch.cuda.max_memory_reserved(0) / (1024**2)
                    )
                samples.extend(trial_samples)
                trial_means.append(mean(trial_samples))
            detections = (
                detector.last_detections
                if runtime == "onnx"
                else extract_detections(last_result)
            )
            last_detections[name] = detections
            detector.gpu_memory_report = (
                {
                    "baseline_allocated_mb": gpu_baseline,
                    "peak_allocated_mb": gpu_peak_allocated,
                    "peak_reserved_mb": gpu_peak_reserved,
                    "measurement": "PyTorch CUDA allocator counters; includes model memory.",
                }
                if detector.device == "cuda:0" and getattr(detector, "gpu_memory_supported", True)
                else None
            )
            results.append(
                _configuration(
                    name, detector, samples, trial_means, stages, detections, confidence, iou_threshold
                )
            )
    finally:
        torch.set_num_threads(old_threads)

    reference = last_detections.get("pytorch_cpu", [])
    for item in results:
        if item.get("status") == "ok":
            item["consistency_vs_pytorch_cpu"] = _detection_matching(reference, last_detections[item["name"]])
            if item["name"].startswith("onnx") and item["device"] == "cuda:0":
                item["gpu_memory_note"] = (
                    "Not measured: ONNX Runtime CUDA allocations are not visible to PyTorch allocator counters."
                )
    report = {
        "benchmark_type": "image_runtime_comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_image": str(Path(image_path).expanduser().resolve()),
        "input_dimensions": {"width": image.shape[1], "height": image.shape[0]},
        "image_load_ms": image_load_ms,
        "hardware_software": _hardware_info(),
        "consistency_method": "Greedy same-class matching by IoU >= 0.5; confidence and box-coordinate deviations are summarized for matched pairs.",
        "consistency_tolerances": {
            "minimum_iou_for_match": 0.5,
            "expected_mean_absolute_confidence_difference": 0.03,
            "expected_mean_absolute_box_coordinate_difference_px": 3.0,
            "note": "These are review thresholds, not pixel-perfect requirements; a deviation should trigger inspection of preprocessing, output decoding, and NMS.",
        },
        "timing_scope": "Each latency sample is the synchronized full prediction call. Stage means separate preprocessing, runtime execution, and postprocessing. Three trial means are summarized by their median; raw samples are retained.",
        "configurations": results,
    }
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(output.resolve())
    return report


def run_video_benchmark(
    video_path: str | Path,
    *,
    weights_path: str | Path = DEFAULT_MODEL,
    onnx_path: str | Path = DEFAULT_ONNX_PATH,
    output_dir: str | Path = "outputs/onnx_video_benchmark",
    output_path: str | Path = "outputs/onnx_video_benchmark.json",
    device: str = "cuda",
    warmup_frames: int = 3,
    cpu_threads: int = 4,
) -> dict:
    """Run both runtimes on one video without tracking and compare frame detections."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(cpu_threads)
    per_frame = {"pytorch": [], "onnx": []}
    try:
        onnx_detector = OnnxDetector(onnx_path, device=device, cpu_threads=cpu_threads)
        actual_device = "cuda" if onnx_detector.device == "cuda:0" else "cpu"
        pytorch_detector = ImageDetector(model_name=str(weights_path), device=actual_device)
        results = {}
        for name, detector in (("pytorch", pytorch_detector), ("onnx", onnx_detector)):
            processor = VideoProcessor(detector)
            run = processor.process_video(
                video_path,
                output_dir=Path(output_dir) / name,
                confidence=DEFAULT_CONFIDENCE,
                warmup_frames=warmup_frames,
                track=False,
                image_size=IMAGE_SIZE,
                iou_threshold=DEFAULT_IOU_THRESHOLD,
                frame_callback=lambda index, detections, key=name: per_frame[key].append(detections),
            )
            results[name] = run.to_dict()
        frame_matches = []
        for index, (pt_detections, ort_detections) in enumerate(zip(per_frame["pytorch"], per_frame["onnx"])):
            matched = _detection_matching(pt_detections, ort_detections)
            matched["frame_index"] = index
            frame_matches.append(matched)
        report = {
            "benchmark_type": "video_runtime_comparison",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input_video": str(Path(video_path).expanduser().resolve()),
            "runtime_device": actual_device,
            "tracking": False,
            "hardware_software": _hardware_info(),
            "runs": results,
            "frame_detection_consistency": {
                "frames_compared": len(frame_matches),
                "mean_matched_detections_per_frame": (
                    mean([item["matched_count"] for item in frame_matches]) if frame_matches else None
                ),
                "mean_match_rate_vs_pytorch": (
                    mean([item["match_rate_reference"] for item in frame_matches]) if frame_matches else None
                ),
                "per_frame": frame_matches,
            },
        }
    finally:
        torch.set_num_threads(previous_threads)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(output.resolve())
    return report
