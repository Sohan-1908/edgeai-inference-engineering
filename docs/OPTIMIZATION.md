# Optimization experiments

[Back to the README](../README.md)

The experiments below compare precision, input resolution, and annotation rendering. Results are specific to their recorded hardware, inputs, and measurement methods. Equal detection counts or close matches on small test collections do not establish equivalent detection accuracy.

## FP16 ONNX Runtime CUDA (Phase 7A)

FP16 stores most internal model weights and activations with fewer bits. This may reduce model size, but does not guarantee faster inference. EdgeAI converts the existing FP32 ONNX model with `onnxconverter-common==1.16.0`, keeps graph input/output in FP32, and writes a separate `models/yolo11n_fp16.onnx`. The original `.pt` and FP32 ONNX models are not changed. Conversion follows the ONNX Runtime [FP16 guidance](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html).

```powershell
python main.py --convert-fp16 --onnx-model models/yolo11n.onnx --fp16-model models/yolo11n_fp16.onnx
python main.py --image outputs\smoke_input.jpg --runtime onnx --precision fp16
python main.py --video outputs\cuda_video_smoke.mp4 --runtime onnx --precision fp16
```

FP16 requires ONNX Runtime's CUDA provider and disables CPU node fallback. If a CUDA operator cannot run, model loading/inference reports the error. FP32 fallback is opt-in with `--allow-fp32-fallback` and recorded in `fallback_error`. The graph takes a 1x3x640x640 FP32 tensor, batch size 1, and uses the same confidence and NMS thresholds. The project does not install a separate CUDA Toolkit; the existing PyTorch CUDA runtime supplies compatible libraries to ONNX Runtime.

### Method

The image benchmark uses one still image plus sampled frames from a video (or additional paths with `--benchmark-images`). It keeps model architecture, 640x640 input, batch size, image sequence, confidence 0.25, and IoU threshold 0.70 constant. It uses at least 10 warm-ups and 50 measured calls in each of at least three trials, synchronizing CUDA around execution. Session initialization is separate; preprocessing, ONNX execution, and post-processing are reported separately.

```powershell
python main.py --image outputs\smoke_input.jpg --variety-video outputs\cuda_video_smoke.mp4 --benchmark-fp16 --benchmark-runs 50 --benchmark-trials 3 --warmup-runs 10
python main.py --video outputs\cuda_video_smoke.mp4 --benchmark-fp16 --warmup-frames 3
```

The video benchmark disables tracking and processes both modes on the same source clip. Video output metrics distinguish source FPS from processing FPS. ORT CUDA memory is unavailable through PyTorch allocator counters. Profiling during conversion verified that every profiled model node was assigned to `CUDAExecutionProvider`; CPU node fallback was disabled.

### Image results

Environment: Windows 11, Python 3.12.10, PyTorch 2.14.0+cu130, ONNX Runtime GPU 1.30.0, ONNX 1.23.0, and an NVIDIA RTX 3050 Laptop GPU (4 GB, driver 610.78). The FP32 model was 10,701,975 bytes; converted FP16 was 5,409,312 bytes (49.5% smaller). Conversion took 458 ms, passed ONNX validation and a real CUDA smoke inference, and retained its FP32 input/output types and 1x3x640x640 input shape.

The image benchmark used `outputs/smoke_input.jpg` plus eight sampled frames from the 62-frame test clip; each mode had 10 warm-ups and 3x50 measured predictions. Across nine images, all 169 detections matched at IoU >= 0.5, with no missed or additional detections. Mean matched IoU was 0.9892; mean absolute confidence difference 0.00087; mean absolute box-coordinate difference 0.093 px.

| ONNX CUDA precision | Session initialization | Mean / median / P95 latency | Median trial mean | Throughput | Stage means: preprocess / execution / postprocess |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 1,032.5 ms | 55.02 / 39.39 / 130.63 ms | 54.65 ms | 18.18 images/s | 10.93 / 33.39 / 10.70 ms |
| FP16 | 987.6 ms | 69.83 / 51.82 / 149.55 ms | 72.38 ms | 14.32 images/s | 15.54 / 41.98 / 12.31 ms |

FP16 did not improve speed in this test: mean prediction latency was 26.9% higher, and throughput calculated from pooled mean latency was 21.2% lower. It did reduce model file size by about half. FP16 slightly shifted confidence and box values, as expected from reduced-precision rounding.

### Video results

The video benchmark used the same 62-frame, 640x360, 30 FPS clip, no tracking, 640x640 model input, confidence 0.25, and IoU 0.70. Both annotated outputs decoded as 62 frames at 640x360 and 30 FPS.

| Metric | FP32 CUDA | FP16 CUDA |
| --- | ---: | ---: |
| Measured frames (after warm-up) | 59 | 59 |
| Mean / median model execution | 24.74 / 21.58 ms | 34.61 / 26.32 ms |
| End-to-end processing FPS | 19.75 | 15.27 |
| Total clip processing time | 5.23 s | 9.03 s |
| Session initialization | 1.08 s | 0.72 s |
| Detection instances | 1,238 | 1,238 |

Across 62 corresponding frames, 1,237 detections matched at IoU >= 0.5, with one FP32 reference detection missed and one FP16 additional detection. Mean matched IoU was 0.9887, mean absolute confidence difference 0.00091, and mean absolute coordinate difference 0.082 px. FP16 was slower in this video trial too. These measurements show a model-size reduction, not a speed win for this machine and ONNX Runtime setup. Raw samples are `outputs/phase7a_image_benchmark.json` and `outputs/phase7a_video_benchmark.json`.

## Static INT8 ONNX Runtime CPU (Phase 7B)

Static quantization uses representative inputs to measure activation ranges ahead of inference. EdgeAI uses ONNX Runtime's static S8S8 QDQ quantizer and keeps the input/output interface in FP32. QDQ means `QuantizeLinear`/`DequantizeLinear` nodes wrap selected compute operations; some operations therefore still run in floating point. See the [ONNX Runtime quantization guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

The exported YOLO11n graph is opset 17 and has 355 nodes: 88 Conv, 78 Sigmoid, 81 Mul, 21 Concat, plus attention MatMul/Softmax, Resize, MaxPool, reshape, and other tensor operations. The recipe quantizes 56 backbone/neck Conv nodes with per-channel symmetric INT8 weights and MinMax-calibrated symmetric INT8 activations. It keeps the model.10 attention block, all model.23 detection head/DFL Conv nodes, and all non-Conv operations in FP32. This is a conservative experiment, not a claim that the selected layers are universally safe.

Create the separate model using 16 deterministic, evenly spaced frames from the local 62-frame sample video. Calibration frames are saved under `outputs/int8_calibration/` with a manifest in `models/yolo11n_int8.json`. Calibration uses the same 640x640 letterbox, BGR-to-RGB conversion, NCHW layout, and float32 normalization as inference. The benchmark samples different frame indices from that video, plus the still image, so its clip frames do not overlap calibration frames.

```powershell
python main.py --quantize-int8
python main.py --quantize-int8 --calibration-video outputs\cuda_video_smoke.mp4
python main.py --image outputs\smoke_input.jpg --runtime onnx --precision int8
python main.py --image outputs\smoke_input.jpg --variety-video outputs\cuda_video_smoke.mp4 --benchmark-int8 --benchmark-runs 50 --benchmark-trials 3 --warmup-runs 10 --cpu-threads 4
```

INT8 is CPU-only in this phase. With `--device auto`, the INT8 model is explicitly loaded on CPU; `--device cuda` is rejected, so it cannot quietly use CUDA or CPU fallback. FP32 and FP16 model files remain untouched.

### Method and results

The benchmark fixes batch size 1, 640x640, confidence 0.25, IoU 0.70, CPU thread count, and image sequence. It runs 10 warm-ups and 3x50 measured calls per model. Session initialization is separate. Each latency wraps the full prediction call, with preprocessing, ORT execution, and post-processing stage means also recorded. Process memory uses `psutil` RSS samples; this includes Python and ORT and is not a model-only allocation counter.

The run used the local Intel CPU (PyTorch reported AVX2 as its CPU capability), ONNX Runtime 1.30.0, and 4 CPU threads. The 16 calibration frames came from indices `0,4,8,12,16,20,24,28,33,37,41,45,49,53,57,61`. Evaluation used `outputs/smoke_input.jpg` plus 12 other sampled video frames. FP32 was 10,701,975 bytes and the partial QDQ INT8 model was 5,199,888 bytes (51.4% smaller). The final CLI quantization run took 15,895.6 ms; an earlier identical conversion measured 3,645.8 ms, showing calibration time varied between runs. The quantized graph passed ONNX validation and CPU smoke inference.

| CPU model | Initialization | Mean / median / P95 latency | Median trial mean | Throughput | Stage means: preprocess / execution / postprocess | Peak process RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 230.7 ms | 153.46 / 150.45 / 198.38 ms | 157.00 ms | 6.52 images/s | 10.90 / 133.75 / 8.81 ms | 676.91 MiB |
| Partial INT8 QDQ | 533.3 ms | 253.91 / 245.12 / 334.04 ms | 243.95 ms | 3.94 images/s | 11.14 / 234.53 / 8.24 ms | 680.39 MiB |

INT8 was 65.5% slower by mean latency and throughput fell 39.6%; session initialization also took longer. RSS peaks were similar and are process-wide. Across 13 evaluation images, FP32 produced 244 detections and INT8 220; 217 matched at IoU >= 0.5, with 27 FP32 detections unmatched and 3 additional INT8 detections. Mean matched IoU was 0.9704, mean absolute confidence difference 0.0397, and mean box-coordinate difference 0.393 px. This is a meaningful recall/count change for a small, unlabeled test collection; it does not establish dataset-level mAP. Accuracy and speed together make this INT8 model unsuitable as a drop-in optimization for this workload. Keep FP32 as the CPU baseline unless broader, task-specific evaluation and different CPU hardware show otherwise.

The optional CUDA probe used `CUDAExecutionProvider` with CPU node fallback explicitly disabled. ORT rejected the graph because some nodes were assigned to the default CPU provider. No mixed-provider fallback performance test was run and no CUDA INT8 performance claim is made. CUDA INT8 remains unsupported for this app configuration unless the graph is converted and validated for a supported GPU quantization runtime.

## Input resolution benchmark (Phase 8)

The resolution benchmark compares PyTorch FP32 + CUDA with ONNX FP32 + CUDAExecutionProvider at 640x640, 480x480, and 320x320, using the same nine source images (one still plus eight sampled clip frames). It uses 10 warm-ups and three trials of 50 measured calls, batch size 1, confidence 0.25, and NMS IoU 0.70. CUDA is synchronized for timed predictions. Each configuration reports initialization, mean/median/P95, throughput, timing-stage means, and memory where measurable. PyTorch uses `imgsz` with Ultralytics letterboxing (`rect=False`); ONNX requires a separate static FP32 export for each input size and uses the project letterbox helper. New 480/320 exports are separate files; the original 640 model remains the reference and is not overwritten.

```powershell
python main.py --image outputs\smoke_input.jpg --variety-video outputs\cuda_video_smoke.mp4 --resolution-video outputs\cuda_video_smoke.mp4 --benchmark-resolutions
```

CUDA is required. The PyTorch benchmark explicitly selects `cuda`; ONNX initializes a strict CUDA session with CPU node fallback disabled. It errors rather than quietly falling back. JSON preserves raw samples and metadata, and the Markdown report summarizes image throughput, detection consistency, and video FPS. If a report path already exists, a timestamped path is chosen.

### Image measurements

Environment: Windows 11, Python 3.12.10, PyTorch 2.14.0+cu130, ONNX Runtime 1.30.0, NVIDIA RTX 3050 Laptop GPU. The source video had 62 frames, 640x360, and 30 FPS. The new static ONNX exports were 10,628,475 bytes at 480 (export 4.20 s) and 10,575,971 bytes at 320 (export 2.81 s); the existing 640 model was 10,701,975 bytes. Export times are not included in model/session initialization or inference.

| Runtime / resolution | Init (ms) | Mean / median / P95 (ms) | Throughput (images/s) | Peak allocated / reserved (MiB) |
| --- | ---: | ---: | ---: | ---: |
| PyTorch CUDA 640 | 1,894.5 | 75.03 / 55.05 / 170.18 | 13.33 | 75.78 / 98.00 |
| ONNX CUDA 640 | 762.0 | 60.92 / 49.64 / 136.30 | 16.41 | Not measurable |
| PyTorch CUDA 480 | 334.5 | 52.35 / 42.49 / 124.07 | 19.10 | 65.48 / 98.00 |
| ONNX CUDA 480 | 770.9 | 47.47 / 34.10 / 115.08 | 21.07 | Not measurable |
| PyTorch CUDA 320 | 173.9 | 73.22 / 45.27 / 179.14 | 13.66 | 58.13 / 98.00 |
| ONNX CUDA 320 | 631.8 | 47.87 / 31.20 / 114.18 | 20.89 | Not measurable |

The 480 input had the lowest mean latency in both runtimes in this run. Going from 480 to 320 did not improve mean latency: PyTorch was slower at 320 and ONNX was effectively similar. Results vary between trials and the P95 values are wide; these are not general speed guarantees. ONNX Runtime GPU allocations are not visible in PyTorch memory counters.

Against each runtime's 640 reference on the nine-image set, 480 matched 96 detections, missed 71 reference detections, and added 3; mean matched IoU was 0.933 and mean confidence difference 0.144. At 320, 43 matched, 124 were unmatched, and 2 were additional; mean matched IoU was 0.912 and confidence difference 0.306. Detections at each resolution match closely between PyTorch and ONNX, but lower sizes differ substantially from the 640 baseline. This measures consistency, not ground-truth accuracy or mAP.

### Video measurements

All settings processed 62 frames with 10 warm-up frames and produced valid outputs retaining 640x360 and 30 FPS. Values below are model-only FPS / end-to-end FPS:

| Runtime / resolution | 640 | 480 | 320 |
| --- | ---: | ---: | ---: |
| PyTorch CUDA | 17.01 / 11.73 | 19.65 / 13.72 | 15.81 / 11.25 |
| ONNX CUDA | 28.94 / 15.25 | 29.64 / 18.11 | 27.37 / 19.56 |

None reached the source 30 FPS end-to-end. The image-only report files are `outputs/resolution_benchmark.json` and `outputs/resolution_benchmark.md`. Raw video trials and per-frame comparisons are in `outputs/resolution_benchmark_with_video.json`; the report is `outputs/resolution_benchmark_with_video.md`.

## Annotation renderer benchmark

This isolated experiment uses identical decoded frames and deterministic synthetic detections; it does not load a model or run inference. Frame decoding and detection generation are outside timed calls. It compares a benchmark-only simple OpenCV renderer (green rectangles and labels above boxes) with production `edgeai.postprocess.annotate_image`. Six representative frames come from `outputs/cuda_video_smoke.mp4`; each renderer/density/frame combination receives two warm-ups and seven measurements. Environment: Python 3.12.10, OpenCV 5.0.0, Windows 11.

| Detections | Renderer | Calls | Mean (ms) | Median (ms) | P95 (ms) | Total (ms) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 5 | Baseline | 42 | 0.224 | 0.190 | 0.333 | 9.405 |
| 5 | Production | 42 | 0.265 | 0.222 | 0.472 | 11.124 |
| 20 | Baseline | 42 | 0.989 | 0.773 | 2.228 | 41.543 |
| 20 | Production | 42 | 1.191 | 0.954 | 2.818 | 50.032 |
| 50 | Baseline | 42 | 2.695 | 1.963 | 5.881 | 113.186 |
| 50 | Production | 42 | 5.415 | 4.415 | 9.861 | 227.444 |

Production renderer mean overhead vs. baseline:

- **5 detections:** +0.041 ms/call (+18.3% mean latency).
- **20 detections:** +0.202 ms/call (+20.4% mean latency).
- **50 detections:** +2.720 ms/call (+100.9% mean latency).

This isolates renderer cost for the measured frames and controlled detection counts; it does not explain variance between full video inference runs. Detection locations and identities are synthetic and deterministic, and baseline styling/label content differ from production styling. The comparison measures complete renderer cost, not only collision-search overhead. Results are hardware/load dependent. The runner is `benchmarks/benchmark_annotation.py`; its report and CSV are `benchmarks/results/annotation_benchmark.md` and `benchmarks/results/annotation_benchmark.csv`.
