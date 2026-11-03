# Inference Pipeline

This document describes the implemented image, video, and webcam paths. It distinguishes runtime-owned operations from EdgeAI's ONNX preprocessing and decoder.

## Image inference

The CLI and API accept JPG/JPEG/PNG images. Inputs are validated and decoded to OpenCV BGR arrays. The detector returns model results, which EdgeAI converts to plain detection records with class ID/name, confidence, and absolute `xyxy` coordinates in the original image. Annotation draws boxes and labels on a copy; CLI output is saved locally. The API can optionally encode an annotated JPEG in its response.

### PyTorch / Ultralytics

1. `ImageDetector` loads and reuses an Ultralytics YOLO model.
2. The decoded BGR image and confidence/input-size/IoU settings are passed to Ultralytics `predict`.
3. Ultralytics performs its model-input preprocessing, model execution, and prediction postprocessing.
4. For CUDA timing, `ImageDetector.predict_frame` synchronizes CUDA before and after the prediction call. Its wall-clock duration includes the prediction path but not file decoding or drawing.
5. EdgeAI extracts detections from the Ultralytics result, annotates the image, and saves or returns it.

Ultralytics' reported preprocessing, inference, and postprocessing timings are recorded separately where available. They are not assumed to add up exactly to the external wall-clock call time.

### ONNX Runtime

`OnnxDetector.predict_frame` performs the following sequence:

1. Validate the frame, confidence threshold, IoU threshold, and the fixed input size.
2. Letterbox with OpenCV resize and constant border padding.
3. Reverse BGR to RGB, transpose HWC to CHW, add a batch dimension, convert to C-contiguous `float32`, and normalize in place to `[0, 1]`.
4. Call synchronous `InferenceSession.run`. The session's provider is selected during initialization; a strict CUDA session requires `CUDAExecutionProvider` and disables CPU node fallback.
5. Decode YOLO output: normalize output orientation, select the maximum class score per candidate, filter by confidence, convert center/size boxes to corner coordinates, perform class-aware greedy NMS, cap detections, rescale and clip coordinates to the original image, then create `Detection` records.
6. Share the common annotation/output behavior with the PyTorch path.

ONNX exports use static 640, 480, or 320 square input shapes. A mismatched input size is rejected rather than resized to another exported shape.

## Video inference

`VideoProcessor` opens the source with OpenCV and reads frames until the stream ends or a configured limit is reached. It uses reported source FPS/frame counts when available and applies a 30 FPS fallback for output when FPS metadata is missing. For short clips it reduces the requested warm-up count so at least one frame can be measured.

For each frame the pipeline:

1. Reads/decodes the frame and checks dimensions, frame count, duration, and optional limits.
2. Runs the shared detector for warm-up or measured inference.
3. Optionally submits detections through a per-video ByteTrack session.
4. Annotates the frame and writes it through an OpenCV video writer.
5. Reports counts, source/output FPS, inference distributions, pipeline/whole-video throughput, decoding, annotation, and output-writing measurements where collected.

Tracking is PyTorch-only. The tracker receives a low-confidence candidate set (at most the configured minimum association threshold) so ByteTrack can apply its high/low association thresholds; tracking time is measured distinctly from the detector's model execution.

### API video flow

FastAPI validates MP4/AVI extension and MIME hint, copies the bounded upload into a request-specific temporary directory, probes the video, then invokes the same `VideoProcessor` with API limits and three warm-up frames. It streams annotated video bytes and returns JSON timing/count metadata in `X-EdgeAI-Metadata`. Tracking requested with ONNX is rejected. Temporary request files are cleaned on errors and after the response stream completes.

## Webcam inference

Webcam is a CLI feature; the dashboard does not stream webcam input. The webcam path requires CUDA and uses the existing capture/inference loop. The configured capture mode can be threaded (latest-frame buffer) or sequential. Optional PyTorch ByteTrack operates on the webcam stream. Camera capture and inference metrics are documented in [WEBCAM.md](WEBCAM.md).

## Timing boundaries

- **Model execution**: runtime/model stage. For ONNX this is synchronous `session.run`; for PyTorch the runtime-reported inference stage is used when available.
- **Full detector call**: external wall-clock time around `predict_frame`; PyTorch CUDA synchronization brackets the call.
- **Preprocessing/postprocessing**: runtime-reported or explicitly measured detector stages, where available.
- **Video end-to-end / whole-video FPS**: different video pipeline scopes; neither is source FPS.
- **API total request time**: request processing scope and not identical to detector-call latency.

See [benchmark results](benchmark_results.md) for measured examples and limitations.
