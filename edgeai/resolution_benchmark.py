"""CUDA-only resolution sweeps using the existing PyTorch and ONNX detectors."""

from __future__ import annotations

import gc
import hashlib
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
from edgeai.config import DEFAULT_CONFIDENCE, DEFAULT_MODEL
from edgeai.detector import ImageDetector
from edgeai.errors import DeviceUnavailableError, ImageInputError, ModelLoadError, VideoInputError
from edgeai.image_io import load_image
from edgeai.onnx_detector import DEFAULT_IOU_THRESHOLD, DEFAULT_ONNX_PATH, OnnxDetector
from edgeai.onnx_export import export_yolo11n, validate_onnx_artifact
from edgeai.postprocess import extract_detections
from edgeai.video import VideoProcessor

RESOLUTION_SIZES = (640, 480, 320)
DEFAULT_RESOLUTION_JSON = Path("outputs/resolution_benchmark.json")
DEFAULT_RESOLUTION_MARKDOWN = Path("outputs/resolution_benchmark.md")


def validate_resolutions(resolutions) -> tuple[int, ...]:
    values = tuple(int(value) for value in resolutions)
    if not values or len(values) != len(set(values)):
        raise ValueError("resolutions must contain unique supported values")
    unsupported = [value for value in values if value not in RESOLUTION_SIZES]
    if unsupported:
        raise ValueError(f"Unsupported resolutions {unsupported}; allowed values are {RESOLUTION_SIZES}")
    if 640 not in values:
        raise ValueError("640x640 is required as the consistency reference")
    return values


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise DeviceUnavailableError("Resolution benchmark requires CUDA; PyTorch reports CUDA unavailable.")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise DeviceUnavailableError(
            "Resolution benchmark requires ONNX Runtime CUDAExecutionProvider; "
            f"available providers: {ort.get_available_providers()}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fresh_report_path(path: str | Path) -> Path:
    destination = Path(path).expanduser()
    if not destination.exists():
        return destination
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return destination.with_name(f"{destination.stem}_{stamp}{destination.suffix}")


def ensure_resolution_models(
    weights_path: str | Path = DEFAULT_MODEL,
    baseline_onnx_path: str | Path = DEFAULT_ONNX_PATH,
    *,
    resolutions=RESOLUTION_SIZES,
) -> dict[int, dict]:
    """Use existing 640 ONNX and export missing fixed-shape variants without overwriting files."""
    sizes = validate_resolutions(resolutions)
    weights = Path(weights_path).expanduser().resolve()
    baseline = Path(baseline_onnx_path).expanduser().resolve()
    if not weights.is_file():
        raise ModelLoadError(f"PyTorch checkpoint does not exist: {weights}")
    if not baseline.is_file():
        raise ModelLoadError(f"FP32 640 ONNX baseline does not exist: {baseline}")
    weight_hash = _sha256(weights)
    baseline_manifest = baseline.with_suffix(".json")
    if baseline_manifest.is_file():
        manifest = json.loads(baseline_manifest.read_text(encoding="utf-8"))
        recorded_hash = manifest.get("source_weights_sha256")
        if recorded_hash and recorded_hash != weight_hash:
            raise ModelLoadError("The 640 ONNX baseline was exported from different PyTorch weights.")

    models = {}
    for size in sizes:
        if size == 640:
            path = baseline
            export_ms = 0.0
            created = False
            manifest = validate_onnx_artifact(path, input_size=size)
        else:
            path = baseline.with_name(f"{baseline.stem}_{size}{baseline.suffix}")
            manifest_path = path.with_suffix(".json")
            if path.exists():
                if not manifest_path.is_file():
                    raise ModelLoadError(
                        f"Resolution model already exists without a manifest; refusing to overwrite: {path}"
                    )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("source_weights_sha256") != weight_hash:
                    raise ModelLoadError(
                        f"Existing {size} ONNX variant was exported from different weights: {path}"
                    )
                validated = validate_onnx_artifact(path, input_size=size)
                manifest.update(validated)
                export_ms = float(manifest.get("export_ms", 0.0))
                created = False
            else:
                manifest = export_yolo11n(weights, path, input_size=size)
                export_ms = float(manifest["export_ms"])
                created = True
        models[size] = {
            "path": str(path.resolve()),
            "sha256": manifest.get("sha256", _sha256(path)),
            "file_size_bytes": path.stat().st_size,
            "input_shape": manifest["input"]["shape"],
            "opset": manifest.get("opsets", {}).get("ai.onnx", manifest.get("opset_requested")),
            "source_weights_sha256": manifest.get("source_weights_sha256", weight_hash),
            "export_ms": export_ms,
            "created_for_this_run": created,
        }
    return models


def sample_video_frames(
    video_path: str | Path,
    *,
    output_dir: str | Path = "outputs/resolution_benchmark/benchmark_images",
    sample_count: int = 8,
) -> tuple[list[Path], dict]:
    """Persist deterministic evenly spaced source frames for the shared image benchmark set."""
    source = Path(video_path).expanduser().resolve()
    if source.suffix.lower() not in {".mp4", ".avi"} or not source.is_file():
        raise VideoInputError(f"Invalid resolution benchmark video: {source}")
    if sample_count < 1:
        raise ValueError("sample_count must be at least one")
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise VideoInputError(f"Could not open resolution benchmark video: {source}")
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if frame_count < 1:
            raise VideoInputError(f"Resolution benchmark video has no frames: {source}")
        indices = sorted({round(i * (frame_count - 1) / max(min(frame_count, sample_count) - 1, 1))
                          for i in range(min(frame_count, sample_count))})
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        paths = []
        for ordinal, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                raise VideoInputError(f"Could not decode source frame {frame_index} from {source}")
            output = destination / f"source_{ordinal:03d}_frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise VideoInputError(f"Could not save sampled frame: {output}")
            paths.append(output.resolve())
    finally:
        capture.release()
    return paths, {
        "video_path": str(source),
        "video_sha256": _sha256(source),
        "video_frame_count": frame_count,
        "selected_frame_indices": indices,
        "sampled_image_paths": [str(path) for path in paths],
        "sampling": "deterministic evenly spaced frame indices",
    }


def _sync_cuda() -> None:
    torch.cuda.synchronize(0)


def run_image_trials(
    detector,
    images: list[np.ndarray],
    *,
    resolution: int,
    warmups: int = 10,
    runs_per_trial: int = 50,
    trials: int = 3,
    confidence: float = DEFAULT_CONFIDENCE,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict:
    """Measure synchronized full prediction calls; exposed for focused test coverage."""
    if not images:
        raise ImageInputError("Resolution benchmark requires at least one decoded image.")
    if resolution not in RESOLUTION_SIZES or warmups < 0 or runs_per_trial < 1 or trials < 1:
        raise ValueError("Invalid resolution benchmark settings.")
    samples, trial_means = [], []
    stage_samples = {"preprocess": [], "runtime": [], "postprocess": []}
    peak_allocated = peak_reserved = baseline_allocated = baseline_reserved = None
    final_result = None
    final_detections = None
    for trial in range(trials):
        for index in range(warmups):
            detector.predict_frame(images[index % len(images)], confidence,
                                   image_size=resolution, iou_threshold=iou_threshold)
        _sync_cuda()
        if detector.runtime_name == "pytorch":
            torch.cuda.reset_peak_memory_stats(0)
            baseline_allocated = torch.cuda.memory_allocated(0) / (1024**2)
            baseline_reserved = torch.cuda.memory_reserved(0) / (1024**2)
        measured = []
        for index in range(runs_per_trial):
            image = images[(trial * runs_per_trial + index) % len(images)]
            _sync_cuda()
            result, call_ms = detector.predict_frame(
                image, confidence, image_size=resolution, iou_threshold=iou_threshold
            )
            _sync_cuda()
            measured.append(float(call_ms))
            last_timings = getattr(detector, "last_timings", {}) or {}
            result_speed = getattr(result, "speed", {}) or {}
            stage_samples["preprocess"].append(float(last_timings.get("preprocess_ms", result_speed.get("preprocess", 0.0))))
            stage_samples["runtime"].append(float(last_timings.get("runtime_ms", result_speed.get("inference", call_ms))))
            stage_samples["postprocess"].append(float(last_timings.get("postprocess_ms", result_speed.get("postprocess", 0.0))))
            final_result = result
        if detector.runtime_name == "pytorch":
            allocated = torch.cuda.max_memory_allocated(0) / (1024**2)
            reserved = torch.cuda.max_memory_reserved(0) / (1024**2)
            peak_allocated = max(peak_allocated or 0.0, allocated)
            peak_reserved = max(peak_reserved or 0.0, reserved)
        samples.extend(measured)
        trial_means.append(mean(measured))
    if detector.runtime_name.startswith("onnxruntime"):
        final_detections = list(getattr(detector, "last_detections", []))
        memory = {"baseline_allocated_mb": None, "peak_allocated_mb": None,
                  "peak_reserved_mb": None,
                  "measurement": "Not measurable through PyTorch counters; ONNX Runtime owns its CUDA allocations."}
    else:
        final_detections = extract_detections(final_result)
        memory = {"baseline_allocated_mb": baseline_allocated,
                  "baseline_reserved_mb": baseline_reserved,
                  "peak_allocated_mb": peak_allocated,
                  "peak_reserved_mb": peak_reserved,
                  "measurement": "PyTorch CUDA allocator counters; includes model memory."}
    return {
        "warmups_per_trial": warmups,
        "measured_runs_per_trial": runs_per_trial,
        "trials": trials,
        "trial_mean_latencies_ms": trial_means,
        "raw_latency_samples_ms": samples,
        "mean_latency_ms": mean(samples),
        "median_latency_ms": median(samples),
        "p95_latency_ms": float(np.percentile(samples, 95)),
        "median_trial_mean_latency_ms": median(trial_means),
        "throughput_images_per_second": 1000.0 / mean(samples),
        "stage_mean_ms": {key: mean(value) for key, value in stage_samples.items()},
        "gpu_memory": memory,
        "detections": final_detections,
    }


def _aggregate_consistency(per_input: list[dict]) -> dict:
    matched = sum(item["matched_count"] for item in per_input)
    reference = sum(item["reference_detection_count"] for item in per_input)
    candidate = sum(item["candidate_detection_count"] for item in per_input)

    def weighted(key):
        available = [item for item in per_input if item.get(key) is not None and item["matched_count"]]
        denominator = sum(item["matched_count"] for item in available)
        return (sum(item[key] * item["matched_count"] for item in available) / denominator) if denominator else None

    return {
        "input_count": len(per_input),
        "reference_detection_count": reference,
        "candidate_detection_count": candidate,
        "matched_count": matched,
        "missed_reference_count": reference - matched,
        "additional_candidate_count": candidate - matched,
        "class_agreement_rate": matched / max(candidate, 1),
        "match_rate_reference": matched / max(reference, 1),
        "mean_matched_iou": weighted("mean_matched_iou"),
        "mean_absolute_confidence_difference": weighted("mean_absolute_confidence_difference"),
        "mean_absolute_box_coordinate_difference_px": weighted("mean_absolute_box_coordinate_difference_px"),
        "per_input": per_input,
    }


def _run_one_config(
    runtime: str,
    resolution: int,
    image_paths: list[Path],
    images: list[np.ndarray],
    *,
    weights_path: str | Path,
    onnx_path: Path,
    video_path: str | Path | None,
    output_video_dir: Path,
    warmups: int,
    runs_per_trial: int,
    trials: int,
    video_warmup_frames: int,
    confidence: float,
    iou_threshold: float,
    cpu_threads: int,
) -> tuple[dict, list[list], dict | None]:
    if runtime == "pytorch":
        detector = ImageDetector(model_name=str(weights_path), device="cuda")
        if detector.device != "cuda:0":
            raise DeviceUnavailableError(f"PyTorch CUDA benchmark initialized on {detector.device}, expected cuda:0")
        initialization_ms = detector.model_load_ms
        selected_provider = detector.execution_provider
    else:
        detector = OnnxDetector(onnx_path, device="cuda", cpu_threads=cpu_threads,
                                precision="fp32", require_cuda=True)
        if detector.device != "cuda:0" or detector.execution_provider != "CUDAExecutionProvider":
            raise DeviceUnavailableError(
                f"ONNX CUDA benchmark did not select CUDAExecutionProvider: device={detector.device}, "
                f"provider={detector.execution_provider}, fallback_error={detector.fallback_error}"
            )
        initialization_ms = detector.model_load_ms
        selected_provider = detector.execution_provider

    image_metrics = run_image_trials(
        detector, images, resolution=resolution, warmups=warmups,
        runs_per_trial=runs_per_trial, trials=trials, confidence=confidence,
        iou_threshold=iou_threshold,
    )
    per_image_detections = []
    for path, image in zip(image_paths, images):
        result, _ = detector.predict_frame(image, confidence, image_size=resolution, iou_threshold=iou_threshold)
        detections = detector.last_detections if runtime == "onnx" else extract_detections(result)
        per_image_detections.append(detections)

    video_report = None
    per_frame_detections = None
    if video_path is not None:
        collected = []
        video_result = VideoProcessor(detector).process_video(
            video_path,
            output_dir=output_video_dir / runtime / str(resolution),
            confidence=confidence,
            warmup_frames=video_warmup_frames,
            track=False,
            image_size=resolution,
            iou_threshold=iou_threshold,
            frame_callback=lambda index, detections: collected.append(detections),
        )
        video_report = video_result.to_dict()
        video_capture = cv2.VideoCapture(video_result.annotated_video_path)
        try:
            video_report["output_validation"] = {
                "opened": video_capture.isOpened(),
                "frame_count": int(round(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))),
                "width": int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(video_capture.get(cv2.CAP_PROP_FPS)),
            }
        finally:
            video_capture.release()
        per_frame_detections = collected

    config = {
        "runtime": runtime,
        "precision": "FP32",
        "device": detector.device,
        "execution_provider": selected_provider,
        "resolution": [resolution, resolution],
        "batch_size": 1,
        "model": detector.model_name,
        "model_path": str(detector.model_path.resolve()) if runtime == "onnx" else str(Path(weights_path).resolve()),
        "model_file_size_bytes": detector.model_size_bytes,
        "model_initialization_ms": initialization_ms,
        "fallback_error": getattr(detector, "fallback_error", None),
        "image_benchmark": {key: value for key, value in image_metrics.items() if key != "detections"},
        "image_detections": [
            {"image": str(path.resolve()), "detections": [item.to_dict() for item in dets]}
            for path, dets in zip(image_paths, per_image_detections)
        ],
        "video_benchmark": video_report,
    }
    del detector
    gc.collect()
    return config, per_image_detections, per_frame_detections


def _markdown_report(report: dict) -> str:
    lines = [
        "# Input Resolution Benchmark",
        "",
        f"Generated: {report['created_utc']}",
        "",
        "## Configuration",
        "",
        f"- Device: {report['hardware_software'].get('gpu_name')}",
        f"- Image set: {len(report['source_images'])} images; batch size 1; confidence {report['settings']['confidence']}; NMS IoU {report['settings']['iou_threshold']}.",
        f"- Input sizes: {', '.join(f'{size}x{size}' for size in report['settings']['resolutions'])}.",
        f"- Image timing: {report['settings']['warmups_per_trial']} warm-ups, {report['settings']['measured_runs_per_trial']} measured runs × {report['settings']['trials']} trials.",
        "- PyTorch and ONNX use the same source images. ONNX uses separate static FP32 exports because each ONNX graph has a fixed input shape.",
        "",
        "## Image benchmark",
        "",
        "| Runtime / resolution | Init (ms) | Mean / median / P95 (ms) | Throughput (images/s) | GPU allocated / reserved peak (MiB) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, item in report["image_results"].items():
        m = item["image_benchmark"]
        memory = m["gpu_memory"]
        if memory["peak_allocated_mb"] is None:
            memory_text = "Not measurable for ORT"
        else:
            memory_text = f"{memory['peak_allocated_mb']:.2f} / {memory['peak_reserved_mb']:.2f}"
        lines.append(
            f"| {key} | {item['model_initialization_ms']:.2f} | "
            f"{m['mean_latency_ms']:.2f} / {m['median_latency_ms']:.2f} / {m['p95_latency_ms']:.2f} | "
            f"{m['throughput_images_per_second']:.2f} | {memory_text} |"
        )
    lines += ["", "## Detection consistency", "",
              "Each lower-resolution result is matched against the 640×640 result from the same runtime using same-class boxes at IoU ≥ 0.5. This measures consistency with that runtime's 640 baseline, not ground-truth accuracy.", "",
              "| Runtime / resolution | Matched | Unmatched reference | Additional | Mean matched IoU | Mean confidence delta | Mean box delta (px) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, value in report["consistency"].items():
        if key.endswith("_640"):
            continue
        lines.append(
            f"| {key} | {value['matched_count']} | {value['missed_reference_count']} | "
            f"{value['additional_candidate_count']} | {value['mean_matched_iou'] or 0:.4f} | "
            f"{value['mean_absolute_confidence_difference'] or 0:.4f} | "
            f"{value['mean_absolute_box_coordinate_difference_px'] or 0:.3f} |"
        )
    lines += ["", "## Video processing", ""]
    if report["video_results"]:
        lines += ["Input FPS is reported separately from model-only throughput and full video-pipeline FPS.", "",
                  "| Runtime / resolution | Input FPS | Model-only FPS | End-to-end FPS | Measured frames | Output valid |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for key, run in report["video_results"].items():
            perf = run["performance"]
            valid = run["output_validation"].get("opened") and run["output_validation"].get("frame_count") == perf["processed_frames"]
            lines.append(f"| {key} | {perf['input_video_fps'] or 0:.2f} | {perf['model_only_fps']:.2f} | "
                         f"{perf['end_to_end_fps']:.2f} | {perf['measured_frames']} | {valid} |")
    else:
        lines.append("No video was supplied; video FPS was not measured.")
    lines += ["", "## Method and limitations", "",
              "Latency covers the detector prediction call including preprocessing, execution and post-processing; stage means are present in the JSON. CUDA was synchronized around inference. PyTorch allocator memory includes model memory; ONNX Runtime allocations are not visible through PyTorch counters. Export time for the additional ONNX models is reported separately from model/session initialization and inference. Smaller inputs can lose small-object detail. These comparisons do not use ground-truth labels.", ""]
    return "\n".join(lines)


def run_resolution_benchmark(
    image_paths: list[str | Path],
    *,
    weights_path: str | Path = DEFAULT_MODEL,
    onnx_path: str | Path = DEFAULT_ONNX_PATH,
    video_path: str | Path | None = None,
    resolutions=RESOLUTION_SIZES,
    output_json: str | Path = DEFAULT_RESOLUTION_JSON,
    output_markdown: str | Path = DEFAULT_RESOLUTION_MARKDOWN,
    output_video_dir: str | Path = "outputs/resolution_benchmark/videos",
    warmups: int = 10,
    runs_per_trial: int = 50,
    trials: int = 3,
    video_warmup_frames: int = 10,
    cpu_threads: int = 4,
    confidence: float = DEFAULT_CONFIDENCE,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict:
    _require_cuda()
    sizes = validate_resolutions(resolutions)
    if warmups < 10 or runs_per_trial < 50 or trials < 3:
        raise ValueError("Resolution benchmark requires at least 10 warm-ups and 50 inferences in each of 3 trials.")
    if video_warmup_frames < 0 or cpu_threads < 1:
        raise ValueError("Invalid video warm-up or CPU thread count.")
    if not image_paths:
        raise ImageInputError("Provide at least one source image for the resolution benchmark.")
    images, image_records = [], []
    for input_path in image_paths:
        image, load_ms = load_image(input_path)
        images.append(image)
        image_records.append({"path": str(Path(input_path).expanduser().resolve()),
                              "dimensions": [int(image.shape[1]), int(image.shape[0])],
                              "load_ms": load_ms})
    if video_path is not None:
        video_source = Path(video_path).expanduser().resolve()
        if not video_source.is_file():
            raise VideoInputError(f"Resolution benchmark video does not exist: {video_source}")
    else:
        video_source = None
    model_metadata = ensure_resolution_models(weights_path, onnx_path, resolutions=sizes)
    configs, detections, video_detections = {}, {}, {}
    for resolution in sizes:
        onnx_variant = Path(model_metadata[resolution]["path"])
        for runtime in ("pytorch", "onnx"):
            key = f"{runtime}_cuda_{resolution}"
            config, image_result, video_result = _run_one_config(
                runtime, resolution, [Path(path).expanduser().resolve() for path in image_paths], images,
                weights_path=weights_path, onnx_path=onnx_variant, video_path=video_source,
                output_video_dir=Path(output_video_dir), warmups=warmups,
                runs_per_trial=runs_per_trial, trials=trials,
                video_warmup_frames=video_warmup_frames, confidence=confidence,
                iou_threshold=iou_threshold, cpu_threads=cpu_threads,
            )
            configs[key] = config
            detections[key] = image_result
            if video_result is not None:
                video_detections[key] = video_result

    consistency = {}
    for runtime in ("pytorch", "onnx"):
        reference_key = f"{runtime}_cuda_640"
        reference = detections[reference_key]
        for resolution in sizes:
            key = f"{runtime}_cuda_{resolution}"
            input_matches = []
            for index, (ref, candidate) in enumerate(zip(reference, detections[key])):
                match = _detection_matching(ref, candidate, iou_threshold=0.5)
                match.update({"image": image_records[index]["path"], "source_image_index": index})
                input_matches.append(match)
            consistency[key] = _aggregate_consistency(input_matches)

    video_consistency = {}
    if video_detections:
        for runtime in ("pytorch", "onnx"):
            reference_key = f"{runtime}_cuda_640"
            reference_frames = video_detections[reference_key]
            for resolution in sizes:
                key = f"{runtime}_cuda_{resolution}"
                per_frame = []
                for frame_index, (reference, candidate) in enumerate(zip(reference_frames, video_detections[key])):
                    item = _detection_matching(reference, candidate, iou_threshold=0.5)
                    item["frame_index"] = frame_index
                    per_frame.append(item)
                video_consistency[key] = _aggregate_consistency(per_frame)

    json_path = _fresh_report_path(output_json)
    markdown_path = _fresh_report_path(output_markdown)
    report = {
        "benchmark_type": "input_resolution_cuda_fp32",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_software": _hardware_info(),
        "source_images": image_records,
        "source_video": str(video_source) if video_source else None,
        "settings": {
            "resolutions": list(sizes), "batch_size": 1,
            "confidence": confidence, "iou_threshold": iou_threshold,
            "warmups_per_trial": warmups, "measured_runs_per_trial": runs_per_trial,
            "trials": trials, "video_warmup_frames": video_warmup_frames,
            "cuda_synchronization": "before and after each measured prediction call",
            "preprocessing": {
                "pytorch": "Ultralytics predict with rect=False and imgsz equal to each tested resolution",
                "onnx": "same project letterbox helper: preserve aspect ratio, pad with 114, BGR-to-RGB, NCHW float32 /255",
            },
        },
        "model_variants": model_metadata,
        "image_results": configs,
        "consistency_method": "Each 480/320 configuration is matched against the 640 configuration from the same runtime using same-class IoU >= 0.5; this is baseline consistency, not ground-truth accuracy.",
        "consistency": consistency,
        "video_results": {key: config["video_benchmark"] for key, config in configs.items() if config["video_benchmark"] is not None},
        "video_consistency": video_consistency,
        "measurement_notes": {
            "latency": "Synchronized detector prediction call including preprocess, runtime execution and postprocess; stage means are reported separately.",
            "throughput": "Images per second computed as 1000 / mean full prediction latency.",
            "video_fps": "model_only_fps is based on model execution timing; end_to_end_fps covers measured video decoding, prediction, annotation, writing, and release; input_video_fps is source metadata.",
            "gpu_memory": "PyTorch allocator allocated/reserved high-water marks include model memory. ONNX Runtime GPU allocations are not measurable with PyTorch allocator counters.",
            "export": "Additional static ONNX export times are recorded in model_variants and excluded from model initialization and prediction latency.",
        },
        "image_results_path": str(json_path.resolve()),
        "human_report_path": str(markdown_path.resolve()),
        "report_path": str(json_path.resolve()),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return report
