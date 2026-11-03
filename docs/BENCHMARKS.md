# Benchmark methodology and results

[Back to the README](../README.md)

Benchmark results are specific to their recorded inputs, hardware, software versions, and run protocols. They do not establish dataset-level accuracy or general real-time 30 FPS performance. Counts alone do not prove equal classes, boxes, confidence scores, or accuracy.

## Video metric definitions

- **Input video FPS:** Rate read from source metadata; it is not processing speed.
- **Video decoding time:** Sum of OpenCV frame-read durations over the clip.
- **Model inference mean/median (model-forward):** Synchronized per-frame network-forward time. It excludes tracker update, decoding, annotation, and output writing.
- **Inference-call mean/median:** Synchronized wall time around the full per-frame `predict` or `track` API call. Tracking mode includes tracking callback work as well as preprocessing and prediction post-processing.
- **Tracking processing mean/median:** Direct time inside each measured `ByteTrack.update` call. It isolates the main association step, not every bit of Python tracking-mode overhead.
- **Model-only FPS:** Measured frames divided by summed model-forward time.
- **Measured-pipeline FPS (`end_to_end_processing_fps`):** Measured frames divided by elapsed measured pipeline time. It includes measured-frame decode, inference, annotation, writing, and resource release; excludes warm-up and model loading.
- **Whole-video processing FPS (`whole_video_processing_fps`):** Whole source video frames divided by whole-video processing time; includes warm-up.
- **Client request latency:** For the API runtime runner, wall time from upload start through complete annotated-video download.
- **Total detections / unique IDs:** Frame-level detection instances versus distinct tracking IDs seen over the whole video.
- **GPU memory:** PyTorch allocated/reserved high-water marks over measured frames include the loaded model and exclude other applications. ONNX Runtime CUDA allocations are not visible through PyTorch's counters.

These metrics measure different parts of execution. Direct tracker-update time is not the total causal overhead of enabling tracking. Do not treat source FPS, model-only FPS, measured-pipeline FPS, whole-video FPS, and client FPS as interchangeable.

## Detection-only versus tracking (Phase 5)

This comparison used the same 62-frame, 640x360, 30 FPS Ultralytics sample video, `yolo11n.pt`, CUDA device, confidence 0.10, and three warm-up frames for each run. Both output files decoded successfully with all 62 frames at the original resolution and FPS.

| Metric | Detection only | Detection + ByteTrack |
| --- | ---: | ---: |
| Measured frames | 59 | 59 |
| Frame-level detection instances | 1,743 | 1,198 |
| Unique tracking IDs | — | 24 |
| Visible tracked objects in final frame | — | 21 |
| Model-forward mean / median | 61.80 / 43.62 ms | 56.24 / 39.25 ms |
| Full predict/track call mean / median | 74.56 / 54.09 ms | 80.99 / 59.02 ms |
| ByteTrack `update()` mean / median | — | 9.40 / 6.81 ms |
| Measured end-to-end FPS | 11.48 | 10.93 |
| Peak PyTorch allocated / reserved | 66.36 / 78.00 MiB | 66.36 / 78.00 MiB |

The tracking run's full per-frame call was 6.43 ms slower on average and measured end-to-end FPS was lower. Network-forward timing varied between the single runs, so the lower model-forward value in the tracking row is not evidence that tracking makes inference faster. Direct tracker-update time is not the total causal overhead of enabling tracking. Annotated videos are `outputs/phase5_detection/cuda_video_smoke_annotated.mp4` and `outputs/phase5_tracking/cuda_video_smoke_annotated.mp4`.

## Tracking performance investigation (Phase 15D)

Five consecutive requests were made to the Dockerized `POST /api/v1/detect/video` endpoint with the same `demo_video.mp4` and configuration:

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM).
- Runtime/device: PyTorch CUDA, `cuda:0`.
- Model/input: YOLO11n, input resolution 640.
- Video: 62 frames, 640x360, source FPS 30.
- Tracking: ByteTrack enabled, tracking inference confidence 0.1.
- Detection confidence and NMS settings: unchanged from existing API configuration.

Run 1 was designated the settling run and excluded from steady-state statistics. The table gives the arithmetic mean, median, minimum, and maximum of the per-run values for Runs 2-5. Values are rounded to two decimal places. FPS statistics are averages/statistics over the four individual run rates, not pooled FPS.

| Per-run metric (Runs 2-5) | Mean | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Model inference mean (ms) | 15.01 | 14.94 | 12.73 | 17.43 |
| Tracker update mean (ms) | 2.11 | 2.10 | 2.07 | 2.19 |
| Inference-call mean (ms) | 22.17 | 22.00 | 19.33 | 25.37 |
| Inference-call median (ms) | 18.15 | 17.43 | 16.33 | 21.40 |
| Measured pipeline FPS | 38.17 | 38.22 | 33.74 | 42.49 |
| Whole-video processing FPS | 34.04 | 34.13 | 30.19 | 37.71 |
| Total processing time (ms) | 1,833.66 | 1,818.32 | 1,644.26 | 2,053.73 |
| Measured pipeline time (ms) | 1,557.09 | 1,545.46 | 1,388.69 | 1,748.76 |
| Warm-up inference time (ms) | 238.26 | 235.28 | 219.00 | 263.46 |

Across all five requests, each processed 62 frames and reported 1,383 total detections, 27 unique tracking IDs, PyTorch CUDA runtime, and tracking enabled. Four steady-state runs averaged 38.17 measured-pipeline FPS and 34.04 whole-video processing FPS. Model inference is network-forward time; tracker update is direct ByteTrack update timing; inference-call time wraps the full per-frame prediction/tracking call. Tracker-update timing is not equivalent to the total causal overhead of enabling tracking.

Tracker-update means remained relatively consistent across the final steady-state runs (approximately 2.07-2.19 ms). Model-inference means varied more (approximately 12.73-17.43 ms), contributing to the observed pipeline-throughput variation.

An earlier tracking-enabled Run 3 reached approximately 26.40 FPS. The slowdown was not consistently reproduced; its exact cause remains unknown. No optimization was shown to fix it.

GPU telemetry was collected with `nvidia-smi` at approximately one-second intervals during the five-request monitoring session:

- GPU temperature: approximately 47-50 C.
- Peak sampled GPU utilization: 83%; this was not sustained utilization.
- GPU memory use: approximately 392-447 MiB.
- Graphics clock reached approximately 1,170 MHz.
- Memory clock reached approximately 5,501 MHz.
- Peak reported power: approximately 12.31 W.

These are session observations, not per-run averages. CPU utilization was not recorded. The memory readings are host-wide observations, not process-specific peak VRAM. Temperature and memory observations showed no obvious thermal or memory-capacity pressure during the session, but do not establish the cause of the performance variation.

## PyTorch CUDA versus ONNX CUDA API benchmark (Phase 13C)

The `benchmark_runtime_comparison.py` runner compares PyTorch CUDA and ONNX CUDA against the same video through the existing API. It sets `track=false`, sends one unmeasured video request per runtime, and performs five measured requests by default. It checks `/health` and `/api/v1/config`, requires the requested runtime to be active on CUDA, validates the video response and `X-EdgeAI-Metadata`, and records failures rather than treating failed inference as a successful run. API warm-up frames (up to three) are excluded from inference-latency samples but remain part of end-to-end processing/request latency.

The recorded `phase13c-01` comparison used the same 62-frame video hash with 640 input resolution, confidence 0.25, NMS IoU 0.7, and tracking disabled. Both runtimes completed all five measured requests:

| Runtime | Successful / measured | Failed | Client latency mean / median / P95 (ms) | API per-run mean inference: mean / median / P95-of-run-means (ms) | Per-run API FPS: mean | Per-run client FPS: mean | Pooled client FPS | Whole-video API pooled FPS | Frames total | Detection counts by run | Mean host GPU-memory snapshots (MiB) |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| PyTorch CUDA | 5 / 5 | 0 | 1,610.77 / 1,615.74 / 1,749.25 | 19.86 / 19.48 / 22.35 | 42.26 | 38.76 | 38.49 | 40.90 | 310 | `1238,1238,1238,1238,1238` | 482.10 |
| ONNX CUDA | 5 / 5 | 0 | 1,489.70 / 1,562.08 / 1,656.06 | 17.72 / 18.04 / 19.61 | 45.10 | 42.07 | 41.62 | 43.93 | 310 | `1238,1238,1238,1238,1238` | 418.90 |

For this one paired session, ONNX mean client request latency was 7.52% lower than PyTorch and pooled client throughput was 8.13% higher. Percentages were calculated from unrounded means. These results are not a general performance guarantee.

### Metric interpretation and limitations

- Mean per-run API/client FPS is an arithmetic average of individual run rates. Pooled client FPS is total frames divided by total client request time. These are different calculations.
- Whole-video API pooled FPS is total processed frames divided by summed API total-processing time; that total includes warm-up. It differs from measured-phase API FPS.
- Mean/median/P95 of API per-run mean inference latency describe a distribution of run-level means. The API does not expose raw frame-latency samples, so a pooled frame-level P95 cannot be calculated.
- The report separately summarizes API per-run P95 values. Their mean, median, and P95 are summaries of per-run P95 values, not a P95 across individual frames.
- Detection counts were consistent by measured run index. Equal counts do not establish matching classes, boxes, confidence scores, or detection accuracy.
- Client request latency includes upload, API processing, and downloading the annotated video.
- `nvidia-smi` memory/utilization values are host-wide before/after request snapshots, not process-specific usage or peak VRAM. Other GPU activity can affect observations.
- Runtime changes require manually recreating the API container; the benchmark does not change containers.

Full records are in [runtime_benchmark.md](../runtime_benchmark.md) and [runtime_benchmark.csv](../runtime_benchmark.csv). To regenerate the report without API requests or new measurements:

```powershell
python benchmark_runtime_comparison.py --regenerate-report
```

Report regeneration preserves measured and warm-up records.

### Manual runtime comparison procedure

Use the same video, comparison ID, and other inference settings for both runtime invocations. Runtime switching and container recreation are explicit; wait for the API health check after each recreation.

```powershell
$env:EDGEAI_API_RUNTIME = "pytorch"
docker compose up -d --force-recreate api
python benchmark_runtime_comparison.py --video outputs\cuda_video_smoke.mp4 --runtime pytorch --runs 5 --comparison-id phase13c-01

$env:EDGEAI_API_RUNTIME = "onnx"
docker compose up -d --force-recreate api
python benchmark_runtime_comparison.py --video outputs\cuda_video_smoke.mp4 --runtime onnx --runs 5 --comparison-id phase13c-01
```

By default, the runner appends per-request and aggregate records to `runtime_benchmark.csv` and regenerates `runtime_benchmark.md`. Use a new comparison ID for each benchmark session.

## Repeated image and video runtime measurements (Phase 6)

### Image benchmark method

Run the supported image runtime/device configurations with:

```powershell
python main.py --image outputs\smoke_input.jpg --benchmark --benchmark-runs 30 --benchmark-trials 3 --cpu-threads 4
```

The benchmark uses the same image, weights, 640x640 letterboxing, batch size 1, confidence 0.25, IoU 0.70, and 10 warm-up runs per trial for each runtime/device pair. Each of three trials has 30 measured calls. Mean, median, and P95 use the pooled 90 measured samples; throughput is based on the median of the three trial mean latencies. Each call is timed around preprocessing, runtime execution, and post-processing; stage means are also reported. Session/model initialization is measured separately. PyTorch CUDA timing synchronizes around inference. PyTorch CUDA memory comes from allocator counters; ONNX Runtime CUDA memory is unavailable through those counters.

The recorded image comparison used `outputs/smoke_input.jpg` (810x1080), `yolo11n.pt` (5,613,764 bytes), and `models/yolo11n.onnx` (10,701,975 bytes). Environment: Windows 11, Python 3.12.10, PyTorch 2.14.0+cu130, ONNX Runtime 1.30.0, RTX 3050 Laptop GPU, driver 610.78. ONNX loaded on CPU and CUDA; CUDA sessions reported `CUDAExecutionProvider` as selected.

| Configuration | Mean / median / P95 latency | Median trial mean | Throughput | Stage means: pre / runtime / post | Peak GPU allocated / reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch CPU | 179.56 / 174.23 / 194.96 ms | 178.60 ms | 5.60 images/s | 5.16 / 170.54 / 2.31 ms | — |
| ONNX Runtime CPU | 59.27 / 42.54 / 151.44 ms | 42.15 ms | 23.72 images/s | 4.87 / 51.85 / 2.56 ms | — |
| PyTorch CUDA | 61.66 / 46.40 / 163.55 ms | 64.44 ms | 15.52 images/s | 5.26 / 48.01 / 7.27 ms | 75.78 / 98.00 MiB |
| ONNX Runtime CUDA | 52.77 / 41.88 / 120.72 ms | 49.29 ms | 20.29 images/s | 13.24 / 32.75 / 6.78 ms | Not measurable |

Each configuration matched all five PyTorch CPU reference detections at IoU >= 0.5. ONNX CPU had mean matched IoU 0.9999998, mean confidence difference 0.0000003, and mean box-coordinate difference 0.00003 px. ONNX CUDA had mean IoU 0.9998635, confidence difference 0.0002163, and box-coordinate difference 0.009 px. For this particular image and these 3x30 trials, ONNX had lower median trial latency on both CPU and CUDA. The 151 ms ONNX CPU P95 shows occasional long calls. One image is not a general performance or accuracy guarantee.

### Fixed-video comparison

The video comparison used the same CUDA device and 62-frame, 640x360, 30 FPS clip in detection-only mode:

```powershell
python main.py --video outputs\cuda_video_smoke.mp4 --benchmark --device cuda
```

| Metric | PyTorch CUDA | ONNX Runtime CUDA |
| --- | ---: | ---: |
| Processed / output frames | 62 / 62 | 62 / 62 |
| Total detections | 1,238 | 1,238 |
| Model-forward mean / median | 39.53 / 35.67 ms | 29.97 / 23.65 ms |
| Full inference-call mean / median | 49.68 / 44.82 ms | 49.00 / 40.18 ms |
| Measured end-to-end FPS | 17.23 | 17.09 |
| Full-video processing time, including warm-up | 15.87 s | 4.77 s |
| Model load / ORT session initialization | 1.52 s | 8.29 s |
| Peak allocated / reserved PyTorch CUDA memory | 75.78 / 98.00 MiB | Not measurable |

Every frame had a 100% same-class detection match at IoU >= 0.5; mean matched IoU across frames was 0.999787. End-to-end speeds are effectively the same in this run despite lower ONNX model-forward latency. Full-video time includes three warm-up frames: PyTorch warm-up took 12.21 s while ONNX warm-up took 1.25 s; session initialization is separate from both processing totals. Those cold-start differences explain why full-video processing time should not be read as steady-state speed. Neither run reached the source's 30 FPS. Raw image samples and per-frame video matches are in `outputs/phase6_image_benchmark.json` and `outputs/phase6_video_benchmark.json`; annotated videos are under `outputs/phase6_video_outputs/`.
