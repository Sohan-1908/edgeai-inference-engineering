# Webcam inference and measurements

[Back to the README](../README.md)

## Requirements and commands

Webcam mode captures camera frames and runs the existing PyTorch or ONNX FP32 detector. It requires CUDA and never falls back to CPU. The display shows detections, rolling processing FPS, and inference latency. Press **Q** or **Esc** to stop. The camera is released and OpenCV windows are closed on exit or error.

```powershell
# PyTorch FP32 + CUDA, camera 0, 640x640 model input
python main.py --webcam --device cuda

# Select a camera and request a capture size
python main.py --webcam --camera-index 1 --camera-width 1280 --camera-height 720 --device cuda

# ONNX FP32 + CUDAExecutionProvider
python main.py --webcam --runtime onnx --device cuda --imgsz 480

# Optional ByteTrack IDs (PyTorch only)
python main.py --webcam --runtime pytorch --device cuda --imgsz 640 --track
```

Requested capture dimensions (`--camera-width`, `--camera-height`) describe what OpenCV asks the camera to provide. A camera may choose another supported size; actual captured dimensions are printed in the session summary. `--imgsz` is the separate square model input and supports 640, 480, or 320. PyTorch letterboxes each frame to that size. ONNX uses static input shapes: `models/yolo11n.onnx` for 640, `models/yolo11n_480.onnx` for 480, and `models/yolo11n_320.onnx` for 320. Export a missing variant, for example:

```powershell
python main.py --export-onnx --imgsz 480 --onnx-model models/yolo11n_480.onnx
```

ONNX tracking is unsupported; ByteTrack is available with PyTorch only.

## Warm-up and metric definitions

Model initialization and warm-up are reported separately. Warm-up calls do not count toward steady-state latency or throughput.

- **Inference latency:** Detector-call timing, including runtime preprocessing and post-processing. PyTorch synchronizes CUDA around measured calls; ONNX uses synchronous `session.run` timing.
- **End-to-end FPS:** Processed frames per wall-clock second, including camera reads, inference, annotation, and display.
- **Capture FPS:** Successfully read camera frames per second during the measured interval. In threaded mode it measures the background reader even if inference is slower; in sequential mode reads are paced by inference.
- **Inference FPS:** Processed frames divided by the sum of full detector-call latencies.
- **Frame age at inference:** Time since OpenCV returned the frame. Camera firmware/backend buffering before that point is not observable.
- **Approximate capture-to-display delay:** Ends after the OpenCV display call and key poll, not at the monitor pixels.

FPS is a processing measurement and does not imply real-time operation. Compare it with the actual camera capture rate and account for the display environment.

Bound a run with `--webcam-max-frames 100` or `--webcam-duration-seconds 8`. The default `--webcam-capture-mode threaded` uses a dedicated capture thread and one latest-frame slot. If inference falls behind, a newer frame can replace an unread older frame. This bounds application-side buffering and favors a current view over a backlog, but intentionally drops frames. The capture thread is stopped and joined, and its camera handle is released on exit, disconnection, or inference errors. `--webcam-capture-mode sequential` remains available for comparisons. A camera driver may buffer internally; the application cannot observe upstream buffering.

## Webcam smoke benchmark

On an RTX 3050 Laptop GPU (4 GB VRAM, driver 610.78), camera 0 provided 1280x720 frames. These runs used 640x640 FP32 inference, 10 warm-up predictions, 60 measured webcam frames, and the display:

| Runtime | Model initialization | Warm-up | Mean / median / P95 inference | End-to-end FPS |
| --- | ---: | ---: | ---: | ---: |
| PyTorch CUDA | 1,992.8 ms | 10,997.7 ms | 118.03 / 105.82 / 235.84 ms | 5.27 |
| ONNX Runtime CUDA | 1,159.9 ms | 1,843.3 ms | 56.71 / 40.83 / 123.89 ms | 8.06 |

These are short live-camera measurements from one session per runtime, not stable performance promises; neither run reached 30 FPS. The raw summaries are `outputs/webcam_pytorch_cuda.json` and `outputs/webcam_onnx_cuda.json`. Camera mode, lighting, driver, GPU load, frame size, and display environment can affect results.

## Concurrent capture pipeline (Phase 9B)

The comparison used camera 0 at a requested and observed 1280x720 capture size. Each row averages two six-second runs with display enabled and 10 warm-up calls. Confidence was 0.25 and NMS IoU was 0.70. Latency columns average each run's mean, median, and P95; FPS, frame age, and drop percentage average the two run summaries.

| Runtime / input | Capture mode | Mean / median / P95 inference | Capture FPS | Inference FPS | End-to-end FPS | Captured / processed frames | Replaced | Mean age at inference | Approx. frame-to-display |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX CUDA 640 | Sequential | 46.02 / 34.78 / 100.95 ms | 11.93 | 22.52 | 11.93 | 72.5 / 72.5 | 0% | 0.84 ms | 59.30 ms |
| ONNX CUDA 640 | Threaded | 38.23 / 32.01 / 71.40 ms | 14.50 | 26.21 | 14.75 | 93.0 / 89.0 | 5.35% | 8.21 ms | 57.69 ms |
| ONNX CUDA 480 | Sequential | 53.33 / 36.52 / 133.75 ms | 8.05 | 18.75 | 8.05 | 49.0 / 49.0 | 0% | 0.90 ms | 69.32 ms |
| ONNX CUDA 480 | Threaded | 37.14 / 28.40 / 87.21 ms | 15.05 | 27.41 | 14.53 | 96.5 / 88.0 | 9.90% | 7.74 ms | 56.50 ms |
| PyTorch CUDA 640 | Sequential | 73.44 / 49.42 / 174.67 ms | 7.41 | 14.71 | 7.41 | 45.0 / 45.0 | 0% | 0.05 ms | 86.87 ms |
| PyTorch CUDA 640 | Threaded | 84.10 / 53.72 / 189.51 ms | 15.06 | 11.89 | 9.56 | 96.0 / 57.5 | 41.15% | 30.74 ms | 131.98 ms |

For this short repeat set, threaded capture raised average end-to-end FPS from 11.93 to 14.75 for ONNX 640, 8.05 to 14.53 for ONNX 480, and 7.41 to 9.56 for PyTorch 640. These are observed pipeline differences, not faster YOLO execution. PyTorch threaded inference latency was higher on average than sequential (84.10 vs. 73.44 ms), and inference FPS was lower. The threaded reader captured about 15 FPS in all three configurations; because the model could not consume every frame, it replaced roughly 5%, 10%, and 41% of measured frames respectively. Average frame age at inference increased from under 1 ms in sequential mode to 8 ms for ONNX 640, 8 ms for ONNX 480, and 31 ms for PyTorch. The buffer remains bounded at one pending frame, but latest-frame replacement does not make each displayed frame instantaneous. No configuration reached 30 FPS end-to-end.

Average initialization times were 1.2 s for ONNX and 1.4 s for PyTorch, reported separately. Warm-up averaged 1.9-2.7 s for ONNX and 9.7-11.2 s for PyTorch, and was excluded from steady-state rates and latency summaries. Raw per-run summaries are in `outputs/phase9b_webcam/benchmark.json` and individual JSON files in that folder. These short runs are device- and camera-specific; they are not stable throughput guarantees.

To compare capture policies with the same runtime, size, camera, and duration:

```powershell
python main.py --webcam --runtime onnx --device cuda --imgsz 640 --webcam-capture-mode sequential --webcam-duration-seconds 8
python main.py --webcam --runtime onnx --device cuda --imgsz 640 --webcam-capture-mode threaded --webcam-duration-seconds 8
```
