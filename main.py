"""Command-line entry point for pretrained YOLO image detection."""

import argparse
import json
import sys
from pathlib import Path

from edgeai.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MEASURED_RUNS,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WARMUP_RUNS,
)
from edgeai.detector import ImageDetector
from edgeai.errors import (
    DeviceUnavailableError,
    ImageInputError,
    InferenceError,
    ModelLoadError,
    VideoInputError,
    VideoProcessingError,
    WebcamInputError,
)
from edgeai.image_io import validate_image_path
from edgeai.video import VideoProcessor
from edgeai.video_io import validate_video_path
from edgeai.onnx_detector import DEFAULT_IOU_THRESHOLD, DEFAULT_ONNX_PATH, OnnxDetector
from edgeai.onnx_export import export_yolo11n
from edgeai.benchmark import run_image_benchmark, run_video_benchmark
from edgeai.onnx_fp16 import DEFAULT_FP16_ONNX_PATH, convert_onnx_to_fp16, load_onnx_detector
from edgeai.benchmark_fp16 import run_fp16_image_benchmark, run_fp16_video_benchmark
from edgeai.onnx_int8 import (
    DEFAULT_CALIBRATION_VIDEO,
    DEFAULT_INT8_ONNX_PATH,
    create_default_int8_model,
    quantize_yolo11n_static,
    sample_calibration_frames,
)
from edgeai.benchmark_int8 import run_int8_cpu_benchmark, sample_benchmark_video_frames
from edgeai.resolution_benchmark import (
    DEFAULT_RESOLUTION_MARKDOWN,
    DEFAULT_RESOLUTION_JSON,
    RESOLUTION_SIZES,
    run_resolution_benchmark,
    sample_video_frames as sample_resolution_video_frames,
)
from edgeai.webcam import SUPPORTED_WEBCAM_SIZES, run_webcam


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect objects in an image or video file.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Path to a JPG/JPEG/PNG image")
    input_group.add_argument("--video", help="Path to an MP4/AVI video")
    input_group.add_argument("--webcam", action="store_true", help="Run interactive webcam inference")
    input_group.add_argument("--export-onnx", action="store_true", help="Export YOLO weights to fixed FP32 ONNX")
    input_group.add_argument("--convert-fp16", action="store_true", help="Convert FP32 ONNX to a separate strict-CUDA FP16 model")
    input_group.add_argument("--quantize-int8", action="store_true", help="Calibrate and create a separate static INT8 QDQ model")
    parser.add_argument("--track", action="store_true", help="Enable ByteTrack for video inference")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Annotated output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ultralytics model name/path (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--runtime", choices=("pytorch", "onnx"), default="pytorch")
    parser.add_argument("--onnx-model", default=str(DEFAULT_ONNX_PATH), help="ONNX model path")
    parser.add_argument("--fp16-model", default=str(DEFAULT_FP16_ONNX_PATH), help="FP16 ONNX model path")
    parser.add_argument("--int8-model", default=str(DEFAULT_INT8_ONNX_PATH), help="Static INT8 ONNX model path")
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32", help="ONNX precision; FP16 requires CUDA and INT8 is CPU-only")
    parser.add_argument("--allow-fp32-fallback", action="store_true", help="Explicitly allow FP16 request to fall back to FP32 after a CUDA/model error")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold from 0 to 1")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, choices=SUPPORTED_WEBCAM_SIZES, default=640, help="Model input size (ONNX models have fixed shapes)")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index")
    parser.add_argument("--camera-width", type=int, default=1280, help="Requested webcam capture width")
    parser.add_argument("--camera-height", type=int, default=720, help="Requested webcam capture height")
    parser.add_argument("--webcam-warmup-runs", type=int, default=10, help="Webcam inference warm-up calls excluded from steady-state metrics")
    parser.add_argument("--webcam-max-frames", type=int, default=None, help="Stop webcam after this many measured frames (useful for a bounded run)")
    parser.add_argument("--webcam-duration-seconds", type=float, default=None, help="Stop a webcam run after this many measured seconds")
    parser.add_argument("--webcam-capture-mode", choices=("threaded", "sequential"), default="threaded", help="Capture continuously into a one-frame latest buffer, or capture synchronously for comparison")
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=DEFAULT_MEASURED_RUNS, help="Measured inference runs")
    parser.add_argument("--warmup-frames", type=int, default=3, help="Video frames reserved for warm-up")
    parser.add_argument("--benchmark", action="store_true", help="Compare PyTorch and ONNX Runtime")
    parser.add_argument("--benchmark-fp16", action="store_true", help="Compare strict CUDA ONNX FP32 and FP16")
    parser.add_argument("--benchmark-int8", action="store_true", help="Compare ONNX FP32 and static INT8 on CPU")
    parser.add_argument("--benchmark-resolutions", action="store_true", help="Compare PyTorch FP32 and ONNX FP32 CUDA at 640, 480 and 320 input sizes")
    parser.add_argument("--resolution-video", help="Optional video for end-to-end FPS measurements in the resolution benchmark")
    parser.add_argument("--resolution-warmups", type=int, default=10, help="Warm-up predictions per resolution benchmark trial")
    parser.add_argument("--resolution-runs", type=int, default=50, help="Measured predictions per resolution benchmark trial")
    parser.add_argument("--resolution-trials", type=int, default=3, help="Repeated trials per resolution benchmark configuration")
    parser.add_argument("--resolution-video-warmups", type=int, default=10, help="Video warm-up frames per runtime/resolution")
    parser.add_argument("--resolution-markdown", default=str(DEFAULT_RESOLUTION_MARKDOWN), help="Human-readable resolution benchmark report path")
    parser.add_argument("--benchmark-images", nargs="+", help="Additional varied images for precision consistency/benchmarking")
    parser.add_argument("--variety-video", help="Optional video sampled for separate benchmark images")
    parser.add_argument("--calibration-video", default=str(DEFAULT_CALIBRATION_VIDEO), help="Local video used for deterministic static quantization calibration")
    parser.add_argument("--calibration-images", nargs="+", help="Explicit calibration images; at least four are required")
    parser.add_argument("--benchmark-output", default=None, help="JSON report path for --benchmark")
    parser.add_argument("--benchmark-runs", type=int, default=30, help="Measured runs per image benchmark trial")
    parser.add_argument("--benchmark-trials", type=int, default=3, help="Repeated image benchmark trials")
    parser.add_argument("--cpu-threads", type=int, default=4, help="CPU threads used in ONNX benchmarks")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.webcam:
            incompatible = (
                args.convert_fp16 or args.benchmark or args.benchmark_fp16
                or args.benchmark_int8 or args.benchmark_resolutions or args.quantize_int8
                or args.export_onnx
            )
            if incompatible:
                raise ValueError("Webcam mode cannot be combined with model export, quantization, or benchmark modes.")
            if args.precision != "fp32":
                raise ValueError("Webcam mode currently supports FP32 only.")
            if args.device == "cpu":
                raise ValueError("Webcam inference requires CUDA; CPU execution is not enabled in this phase.")
            if args.track and args.runtime == "onnx":
                raise ValueError("ByteTrack webcam mode is supported with the PyTorch runtime only.")
            if args.runtime == "onnx":
                onnx_path = Path(args.onnx_model)
                default_model = Path(DEFAULT_ONNX_PATH).resolve()
                if args.imgsz != 640:
                    onnx_path = onnx_path.with_name(f"{onnx_path.stem}_{args.imgsz}.onnx")
                if not onnx_path.is_file() and Path(args.onnx_model).resolve() == default_model:
                    raise ModelLoadError(
                        f"ONNX model for {args.imgsz}x{args.imgsz} was not found: {onnx_path}. "
                        f"Export it with: python main.py --export-onnx --imgsz {args.imgsz} "
                        f"--onnx-model {onnx_path}"
                    )
                detector = OnnxDetector(
                    model_path=onnx_path, device="cuda", precision="fp32", require_cuda=True,
                )
                if detector.input_size != args.imgsz:
                    raise ModelLoadError(
                        f"Selected ONNX graph has fixed input {detector.input_size}x{detector.input_size}, "
                        f"but --imgsz requested {args.imgsz}. Use its matching static export."
                    )
            else:
                detector = ImageDetector(model_name=args.model, device="cuda")
            result = run_webcam(
                detector,
                camera_index=args.camera_index,
                capture_width=args.camera_width,
                capture_height=args.camera_height,
                inference_size=args.imgsz,
                confidence=args.conf,
                iou_threshold=args.iou,
                tracking=args.track,
                warmup_runs=args.webcam_warmup_runs,
                max_frames=args.webcam_max_frames,
                duration_seconds=args.webcam_duration_seconds,
                capture_mode=args.webcam_capture_mode,
            )
            report = result.to_dict()
        elif args.convert_fp16:
            report = convert_onnx_to_fp16(args.onnx_model, args.fp16_model)
        elif args.benchmark_resolutions:
            if args.track:
                raise ValueError("Resolution benchmark is detection-only; do not set --track.")
            if args.precision != "fp32":
                raise ValueError("Resolution benchmark compares FP32 only; do not request FP16 or INT8.")
            if args.device == "cpu":
                raise ValueError("Resolution benchmark is CUDA-only; CPU execution is not an automatic fallback.")
            source_images = [*(args.benchmark_images or [])]
            sample_video = args.variety_video or args.video or args.resolution_video
            video_for_fps = args.resolution_video or args.video or args.variety_video
            if args.image:
                source_images.insert(0, args.image)
            elif args.video:
                sampled, _ = sample_resolution_video_frames(args.video)
                source_images.extend(sampled)
            if sample_video:
                sampled, _ = sample_resolution_video_frames(sample_video)
                source_images.extend(sampled)
            if not source_images:
                raise ValueError("Provide --image, --video, or --benchmark-images for resolution benchmarking.")
            source_images = list(dict.fromkeys(str(Path(path).expanduser().resolve()) for path in source_images))
            report = run_resolution_benchmark(
                source_images,
                weights_path=args.model,
                onnx_path=args.onnx_model,
                video_path=video_for_fps,
                resolutions=RESOLUTION_SIZES,
                output_json=args.benchmark_output or str(DEFAULT_RESOLUTION_JSON),
                output_markdown=args.resolution_markdown,
                warmups=args.resolution_warmups,
                runs_per_trial=args.resolution_runs,
                trials=args.resolution_trials,
                video_warmup_frames=args.resolution_video_warmups,
                cpu_threads=args.cpu_threads,
                confidence=args.conf,
                iou_threshold=args.iou,
            )
        elif args.quantize_int8:
            if args.calibration_images:
                report = quantize_yolo11n_static(
                    args.calibration_images,
                    fp32_model_path=args.onnx_model,
                    output_path=args.int8_model,
                    calibration_metadata={
                        "source": "explicit user-supplied local images",
                        "calibration_image_paths": args.calibration_images,
                    },
                )
            else:
                report = create_default_int8_model(
                    calibration_video=args.calibration_video,
                    fp32_model_path=args.onnx_model,
                    output_path=args.int8_model,
                )
        elif args.benchmark_int8:
            if args.track:
                raise ValueError("INT8 CPU benchmark does not use tracking.")
            if not args.image:
                raise ValueError("--benchmark-int8 requires --image as its primary image.")
            benchmark_images = [args.image, *(args.benchmark_images or [])]
            image_manifest = None
            if args.variety_video:
                int8_manifest_path = Path(args.int8_model).with_suffix(".json")
                excluded = set()
                if int8_manifest_path.is_file():
                    saved = json.loads(int8_manifest_path.read_text(encoding="utf-8"))
                    excluded = set(saved.get("calibration", {}).get("dataset", {}).get("selected_frame_indices", []))
                sampled_paths, image_manifest = sample_benchmark_video_frames(
                    args.variety_video, excluded_indices=excluded,
                )
                benchmark_images.extend(sampled_paths)
            report = run_int8_cpu_benchmark(
                benchmark_images,
                fp32_path=args.onnx_model,
                int8_path=args.int8_model,
                output_path=args.benchmark_output or "outputs/phase7b_cpu_benchmark.json",
                warmups=args.warmup_runs,
                runs_per_trial=args.benchmark_runs,
                trials=args.benchmark_trials,
                cpu_threads=args.cpu_threads,
                confidence=args.conf,
                iou_threshold=args.iou,
                image_manifest=image_manifest,
            )
        elif args.benchmark_fp16:
            if args.track:
                raise ValueError("FP16 video benchmark requires tracking disabled.")
            if args.image:
                report = run_fp16_image_benchmark(
                    args.benchmark_images or [args.image], video_for_variety=args.variety_video, fp32_path=args.onnx_model,
                    fp16_path=args.fp16_model, output_path=args.benchmark_output or "outputs/phase7a_image_benchmark.json",
                    warmups=args.warmup_runs, runs_per_trial=args.benchmark_runs,
                    trials=args.benchmark_trials, confidence=args.conf, iou_threshold=args.iou,
                )
            elif args.video:
                report = run_fp16_video_benchmark(
                    args.video, fp32_path=args.onnx_model, fp16_path=args.fp16_model,
                    output_dir=args.output_dir, output_path=args.benchmark_output or "outputs/phase7a_video_benchmark.json",
                    warmup_frames=args.warmup_frames, confidence=args.conf, iou_threshold=args.iou,
                )
            else:
                raise ValueError("--benchmark-fp16 requires --image or --video.")
        elif args.benchmark:
            if args.track:
                raise ValueError("Tracking is not included in ONNX Runtime benchmarks.")
            if args.export_onnx:
                raise ValueError("Choose either --export-onnx or --benchmark.")
            if args.image:
                report = run_image_benchmark(
                    args.image,
                    weights_path=args.model,
                    onnx_path=args.onnx_model,
                    output_path=args.benchmark_output or "outputs/onnx_image_benchmark.json",
                    runs_per_trial=args.benchmark_runs,
                    trials=args.benchmark_trials,
                    cpu_threads=args.cpu_threads,
                    confidence=args.conf,
                    iou_threshold=args.iou,
                )
            elif args.video:
                report = run_video_benchmark(
                    args.video,
                    weights_path=args.model,
                    onnx_path=args.onnx_model,
                    output_dir=args.output_dir,
                    output_path=args.benchmark_output or "outputs/onnx_video_benchmark.json",
                    device=args.device,
                    warmup_frames=args.warmup_frames,
                    cpu_threads=args.cpu_threads,
                )
            else:
                raise ValueError("--benchmark requires --image or --video.")
        elif args.export_onnx:
            report = export_yolo11n(args.model, args.onnx_model)
        elif args.image:
            if args.track:
                raise ValueError("--track can only be used with --video")
            # Validate metadata before downloading or loading model weights.
            validate_image_path(args.image)
            if args.runtime == "onnx":
                if args.precision == "fp16":
                    detector, _ = load_onnx_detector("fp16", fp32_path=args.onnx_model, fp16_path=args.fp16_model,
                                                     allow_fp32_fallback=args.allow_fp32_fallback)
                elif args.precision == "int8":
                    if args.device == "cuda":
                        raise ValueError("This INT8 model is calibrated for CPU; CUDA INT8 is not enabled or benchmarked.")
                    detector = OnnxDetector(args.int8_model, device="cpu", precision="int8", cpu_threads=args.cpu_threads)
                else:
                    detector = OnnxDetector(model_path=args.onnx_model, device=args.device)
                result = detector.detect_image(
                    args.image,
                    output_dir=args.output_dir,
                    confidence=args.conf,
                    warmup_runs=args.warmup_runs,
                    measured_runs=args.runs,
                    iou_threshold=args.iou,
                )
            else:
                if args.precision in {"fp16", "int8"}:
                    raise ValueError("FP16 and INT8 precision are available only with --runtime onnx.")
                detector = ImageDetector(model_name=args.model, device=args.device)
                result = detector.detect_image(
                    args.image,
                    output_dir=args.output_dir,
                    confidence=args.conf,
                    warmup_runs=args.warmup_runs,
                    measured_runs=args.runs,
                    image_size=args.imgsz,
                    iou_threshold=args.iou,
                )
            report = result.to_dict()
            report["performance"]["inference_measurement_scope"] = (
                "Mean inference latency is the full prediction call: preprocessing + runtime execution + post-processing. The separated fields report stage means; image file loading and annotation/save are separate."
            )
            report["performance"]["gpu_memory_measurement_scope"] = (
                "PyTorch allocated and reserved CUDA memory high-water marks during measured inference after warm-up; includes model memory."
                if result.device == "cuda:0" and getattr(detector, "gpu_memory_supported", True)
                else "Unavailable for this runtime/device; ONNX Runtime allocations are not reported through PyTorch memory counters."
            )
        else:
            if args.track and args.runtime == "onnx":
                raise ValueError("ONNX tracking is not supported; choose --runtime pytorch for --track.")
            validate_video_path(args.video)
            if args.runtime == "onnx":
                if args.precision == "fp16":
                    detector, _ = load_onnx_detector("fp16", fp32_path=args.onnx_model, fp16_path=args.fp16_model,
                                                     allow_fp32_fallback=args.allow_fp32_fallback)
                elif args.precision == "int8":
                    if args.device == "cuda":
                        raise ValueError("This INT8 model is calibrated for CPU; CUDA INT8 is not enabled or benchmarked.")
                    detector = OnnxDetector(args.int8_model, device="cpu", precision="int8", cpu_threads=args.cpu_threads)
                else:
                    detector = OnnxDetector(model_path=args.onnx_model, device=args.device)
            else:
                if args.precision in {"fp16", "int8"}:
                    raise ValueError("FP16 and INT8 precision are available only with --runtime onnx.")
                detector = ImageDetector(model_name=args.model, device=args.device)
            video_processor = VideoProcessor(detector)
            result = video_processor.process_video(
                args.video,
                output_dir=args.output_dir,
                confidence=args.conf,
                warmup_frames=args.warmup_frames,
                track=args.track,
                image_size=args.imgsz,
                iou_threshold=args.iou,
            )
            report = result.to_dict()
            report["performance"]["metric_definitions"] = {
                "video_decoding_ms": "Sum of OpenCV frame-read times over all frames.",
                "warmup_inference_ms": "Sum of prediction-call durations for warm-up frames; excluded from measured latency statistics.",
                "inference_mean_ms and inference_median_ms": "Wall-clock time around the Ultralytics predict/track call. In tracking mode this includes ByteTrack and is not the model-only latency.",
                "model_inference_mean_ms and model_inference_median_ms": "Ultralytics' synchronized per-frame network-forward profiler time, excluding ByteTrack, decode, annotation, and writing.",
                "model_only_fps": "Measured frame count divided by summed network-forward time.",
                "tracking_processing_mean_ms": "Measured time inside Ultralytics' ByteTrack.update call. This is null in detection-only mode.",
                "tracking_inference_confidence": "Confidence floor passed to YOLO in tracking mode so ByteTrack can use its low-confidence association stage; capped at 0.10 even if --conf is higher. Detection-only mode uses --conf directly.",
                "total_detections": "Frame-level detection instances across the processed video; the same object in multiple frames counts multiple times.",
                "unique_tracking_ids": "Distinct ByteTrack IDs observed across the video; this is not the detection-instance count.",
                "measured_pipeline_ms": "Elapsed time from before decoding the first measured frame through processing the last measured frame and releasing resources; excludes warm-up and model loading.",
                "end_to_end_fps": "Measured frame count divided by measured_pipeline_ms; includes measured-frame decode, inference, annotation, writing, and resource release.",
                "whole_video_processing_fps": "All successfully processed frames divided by total_processing_ms; includes warm-up and full-file processing overhead.",
                "total_processing_ms": "Wall time for processing and writing the full video, including warm-up and resource release; excludes model loading.",
                "input_video_fps": "FPS reported by the source video metadata; output uses this FPS when valid.",
                "gpu_memory": "PyTorch allocated and reserved high-water marks over measured inference; includes model memory and excludes other processes' VRAM.",
                "runtime_execution_provider": "The ONNX Runtime session provider selected for the session, or the PyTorch CPU/CUDA device. A CUDA setup error is included in fallback_error if ONNX Runtime fell back to CPU.",
            }
    except (
        DeviceUnavailableError,
        ImageInputError,
        InferenceError,
        ModelLoadError,
        VideoInputError,
        VideoProcessingError,
        WebcamInputError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
