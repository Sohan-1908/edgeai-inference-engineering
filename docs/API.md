# FastAPI inference service

[Back to the README](../README.md)

## Overview and requirements

The API loads one existing PyTorch FP32 or ONNX FP32 detector at startup. It requires CUDA and fails startup if CUDA or the selected CUDA provider is unavailable; there is no CPU fallback. The selected runtime and device cannot be changed per request.

| Route | Purpose |
| --- | --- |
| `GET /health` | Readiness, detector initialization, active runtime/device, and model input resolution |
| `GET /api/v1/config` | Runtime configuration, supported configurations, and upload limits |
| `POST /api/v1/detect` | JPG/JPEG/PNG image detection |
| `POST /api/v1/detect/video` | MP4/AVI annotated video inference |

Interactive API documentation is served at `http://127.0.0.1:8000/docs`.

## Start locally

Activate the project environment, select the desired runtime, and start one Uvicorn worker:

```powershell
.\.venv\Scripts\Activate.ps1
$env:EDGEAI_API_RUNTIME = "pytorch" # or "onnx"
$env:EDGEAI_API_DEVICE = "cuda"
$env:EDGEAI_API_IMGSZ = "640"       # 640, 480, or 320
$env:EDGEAI_API_CONF = "0.25"
$env:EDGEAI_API_IOU = "0.70"
python -m uvicorn edgeai.api:app --host 127.0.0.1 --port 8000
```

For ONNX, the static graph must match `EDGEAI_API_IMGSZ`: `models/yolo11n.onnx` for 640, `models/yolo11n_480.onnx` for 480, and `models/yolo11n_320.onnx` for 320. Keep one Uvicorn worker; multiple workers each load a model and use additional GPU memory.

## Configuration variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `EDGEAI_API_RUNTIME` | Select `pytorch` or `onnx` | `pytorch` |
| `EDGEAI_API_DEVICE` | API execution device; only `cuda` is accepted | `cuda` |
| `EDGEAI_API_IMGSZ` | Model input size: 640, 480, or 320 | `640` |
| `EDGEAI_API_CONF` | Detection confidence threshold | `0.25` |
| `EDGEAI_API_IOU` | NMS IoU threshold | `0.70` |
| `EDGEAI_API_MODEL` | PyTorch model name/path | `yolo11n.pt` |
| `EDGEAI_API_ONNX_MODEL` | ONNX model path | `models/yolo11n.onnx` |
| `EDGEAI_API_MAX_UPLOAD_BYTES` | Maximum image upload size | 20 MiB |
| `EDGEAI_API_MAX_VIDEO_BYTES` | Maximum video upload size | 100 MiB |
| `EDGEAI_API_MAX_VIDEO_SECONDS` | Maximum video duration | 120 seconds |
| `EDGEAI_API_MAX_VIDEO_FRAMES` | Maximum decoded video frames | 3,600 |

Compose also reads `EDGEAI_API_HOST_PORT` and `EDGEAI_DASHBOARD_HOST_PORT` for host ports; see the [README Docker section](../README.md#docker-deployment).

## Image inference

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/detect" `
  -F "file=@outputs/smoke_input.jpg" `
  -F "include_annotated_image=true"
```

The JSON response includes image dimensions, model input resolution, detection count, class ID/name, confidence, and absolute `[x_min, y_min, x_max, y_max]` pixel coordinates in the original image, whose origin is the top-left. Timing fields separate image decoding, detector call, runtime preprocessing/model/post-processing where available, annotation/JPEG encoding, and total request processing. `include_annotated_image=true` includes an annotated JPEG as Base64; otherwise `annotated_image_base64` is `null`.

Image uploads default to a 20 MiB limit, configurable with `EDGEAI_API_MAX_UPLOAD_BYTES`. The API validates the filename extension and file signature and processes uploads without permanently saving them.

## Video inference and metadata

`POST /api/v1/detect/video` streams the annotated video in the response body; it does not embed video bytes in JSON. The content type is `video/mp4` or `video/x-msvideo`, with an attachment `Content-Disposition` filename. OpenCV must support the input decoder and output encoder. Dimensions and source FPS are preserved when usable media metadata is available.

Example request that saves both the annotated result and response headers:

```powershell
curl.exe --fail --show-error -D outputs\video_response_headers.txt `
  -o outputs\edgeai_annotated.mp4 `
  -F "file=@outputs\cuda_video_smoke.mp4" `
  http://127.0.0.1:8000/api/v1/detect/video
```

The compact JSON metadata is carried in the `X-EdgeAI-Metadata` response header:

| Metadata fields | Meaning |
| --- | --- |
| `source_frame_count`, `processed_frame_count`, `warmup_frames` | Source, completed, and warm-up frame counts |
| `source_fps`, `output_fps`, `video_width`, `video_height` | Input/output frame rates and video dimensions |
| `model_input_resolution`, `runtime`, `device` | Model and execution configuration |
| `warmup_inference_ms` | Time spent in warm-up inference |
| `mean_inference_latency_ms`, `median_inference_latency_ms`, `p95_inference_latency_ms` | Full per-frame inference-call latency summaries |
| `model_inference_mean_ms`, `model_inference_median_ms` | Network-forward timing summaries |
| `tracking_processing_mean_ms`, `tracking_processing_median_ms` | Direct ByteTrack update timing summaries; `null` when tracking is disabled |
| `annotation_ms`, `output_writing_ms`, `video_decoding_ms` | Time attributed to video decoding, annotation, and output writing |
| `measured_pipeline_ms`, `end_to_end_processing_fps` | Measured pipeline interval and throughput, excluding warm-up |
| `whole_video_processing_fps`, `total_processing_time_ms` | Whole-video throughput and processing duration, including warm-up |
| `tracking_enabled`, `tracking_inference_confidence` | Tracking state and inference confidence floor; confidence is `null` when tracking is disabled |
| `total_detections`, `unique_tracking_ids` | Total frame-level detections and distinct tracking IDs |

The tracking-specific fields `tracking_processing_mean_ms`, `tracking_processing_median_ms`, and `tracking_inference_confidence` are `null` when tracking is disabled. A successful whole-file response has equal source and processed frame counts.

Add `-F "track=true"` to enable ByteTrack on PyTorch. ONNX tracking is unsupported; an ONNX tracking request returns HTTP 422 rather than silently disabling tracking. Each video request resets tracker state before processing.

## Upload validation and processing

Video uploads are streamed into request-scoped temporary files, not retained in the project. The temporary directory is cleaned on failure or after the annotated result stream completes. Defaults are 100 MiB, 120 seconds, and 3,600 frames. Configure these with `EDGEAI_API_MAX_VIDEO_BYTES`, `EDGEAI_API_MAX_VIDEO_SECONDS`, and `EDGEAI_API_MAX_VIDEO_FRAMES`.

The service validates that OpenCV can open and decode the upload and checks dimensions, FPS, and frame-count metadata. If FPS metadata is missing or invalid, the existing processor uses a 30 FPS output fallback. Limits are enforced again during processing. When a frame count is declared, an early decode stop is rejected rather than returned as a partial download. If frame-count metadata is unavailable and the decoder reports a truncated stream as ordinary end-of-file, corruption may not be distinguishable from a complete short video.

Inference runs in a worker thread and shares a lock across image/video inference, so health/config requests remain available during detector work. The API does not expose model paths or let callers change runtime settings per request. The documented launch command binds to localhost; configure network exposure deliberately if remote clients are needed. Webcam streaming, authentication, and persistent jobs are not provided.

## Errors and operational limitations

Common endpoint errors are:

| Status | Meaning |
| --- | --- |
| 400 | Empty, corrupt, or unreadable image/video; or a video that ends before its declared frame count |
| 413 | Upload size, video duration, or frame limit exceeded |
| 415 | Unsupported image/video extension, type, or signature |
| 422 | ByteTrack requested while the configured runtime is ONNX |
| 500 | Inference, video processing, or output encoding failed |
| 503 | Detector has not initialized |

Error responses contain client-safe details. Internal inference failures are logged server-side; internal tracebacks are not returned to clients. CUDA or selected-runtime initialization failures stop service startup; they do not trigger CPU fallback.

OpenCV codec availability determines which MP4/AVI files can be decoded and encoded. If a video has no usable FPS metadata, the output FPS fallback is 30. A truncated video with no reliable declared frame count may be indistinguishable from a complete shorter video when the decoder reports ordinary EOF.
