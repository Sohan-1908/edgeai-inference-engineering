# PyTorch CUDA vs ONNX CUDA runtime benchmark

Request latency is measured by this client from upload start through complete annotated-video download. API timing/FPS values are separately reported from `X-EdgeAI-Metadata`.

## Results

### Comparison `phase13c-01`

- Video SHA-256: `a8b0df3073740f808aa0aca3cff7787140837bfa24e01c934ba2cca474fd03cc`
- API inference settings: 640px, confidence `0.25`, NMS IoU `0.7`
- Tracking is disabled for all benchmark requests.
- Each runtime invocation includes one unmeasured video request. Every request also uses the server's existing up-to-three-frame warm-up: those frames are excluded from inference-latency samples but remain part of end-to-end processing/request latency.
- Mean/median API inference latency are the corresponding statistics of per-run means; P95 of per-run means is not a frame-level P95. The API provides per-run inference P95 summaries, so the report separately gives their mean/median/P95. A pooled P95 across individual frame-latency samples cannot be reconstructed because those samples are not returned.
- Mean API/client FPS are arithmetic means of per-run FPS. Pooled client FPS is total processed frames divided by total client request time. Whole-video API FPS is total processed frames divided by summed API total-processing time; that API total includes warm-up and is distinct from API end-to-end FPS, which excludes warm-up frames/time.

| Runtime | Runs: successful / measured | Failed | Client request latency: mean / median / P95 (ms) | API per-run mean inference: mean / median / P95-of-run-means (ms) | Per-run API P95: mean / median / P95 (ms) | API FPS: mean per-run | Client FPS: mean per-run | Client FPS: pooled | API FPS: whole-video pooled | Frames total | Detection counts by run | GPU memory: mean host snapshots (MiB) |
|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| pytorch | 5 / 5 | 0 | 1610.77 / 1615.74 / 1749.25 | 19.86 / 19.48 / 22.35 | 36.58 / 33.20 / 46.86 | 42.26 | 38.76 | 38.49 | 40.90 | 310 | `1238,1238,1238,1238,1238` | 482.10 |
| onnx | 5 / 5 | 0 | 1489.70 / 1562.08 / 1656.06 | 17.72 / 18.04 / 19.61 | 33.30 / 31.78 / 45.15 | 45.10 | 42.07 | 41.62 | 43.93 | 310 | `1238,1238,1238,1238,1238` | 418.90 |

#### Runtime comparison

- ONNX mean client request latency: **7.52% improvement** relative to PyTorch; the percentage is computed from unrounded means.
- ONNX pooled client throughput: **8.13% higher** than PyTorch; computed as total frames / total client time.
- Detection-count consistency by measured run index: **consistent** (PyTorch `1238,1238,1238,1238,1238`; ONNX `1238,1238,1238,1238,1238`). Matching counts do not establish matching classes, boxes, confidence scores, or detection accuracy.

#### GPU memory and limitations

GPU memory/utilization values are host-wide `nvidia-smi` snapshots immediately before/after each request, not process-specific usage or peak VRAM. Client request latency includes upload, processing, and downloading the annotated video. Runs use the same video hash and API-reported resolution/confidence/IoU settings, but runtime changes require manual API container recreation and each runtime is measured in a separate invocation.

## Per-run records

| Comparison | Runtime | Type | Run | Status | Client latency (ms) | API inference mean / median / run P95 (ms) | API end-to-end FPS (measured phase) | Processed frames | Detections | GPU memory before / after (MiB) | Error |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| phase13c-01 | pytorch | warmup | 0 | ok | 6502.01 | 17.99 / 12.63 / 46.96 | 45.32 | 62 | 1238 | 381.0 / 504.0 |  |
| phase13c-01 | pytorch | measured | 1 | ok | 1561.43 | 19.48 / 17.06 / 33.20 | 42.88 | 62 | 1238 | 499.0 / 489.0 |  |
| phase13c-01 | pytorch | measured | 2 | ok | 1733.60 | 21.20 / 19.65 / 32.22 | 39.92 | 62 | 1238 | 489.0 / 481.0 |  |
| phase13c-01 | pytorch | measured | 3 | ok | 1753.16 | 22.63 / 20.46 / 49.66 | 36.86 | 62 | 1238 | 477.0 / 477.0 |  |
| phase13c-01 | pytorch | measured | 4 | ok | 1389.93 | 16.82 / 14.55 / 32.14 | 48.86 | 62 | 1238 | 477.0 / 477.0 |  |
| phase13c-01 | pytorch | measured | 5 | ok | 1615.74 | 19.18 / 18.01 / 35.69 | 42.75 | 62 | 1238 | 477.0 / 478.0 |  |
| phase13c-01 | onnx | warmup | 0 | ok | 3000.03 | 20.88 / 18.32 / 44.04 | 37.92 | 62 | 1238 | 348.0 / 425.0 |  |
| phase13c-01 | onnx | measured | 1 | ok | 1307.49 | 15.35 / 13.75 / 24.17 | 52.18 | 62 | 1238 | 425.0 / 425.0 |  |
| phase13c-01 | onnx | measured | 2 | ok | 1596.77 | 19.49 / 14.89 / 46.20 | 40.88 | 62 | 1238 | 425.0 / 417.0 |  |
| phase13c-01 | onnx | measured | 3 | ok | 1670.89 | 19.64 / 16.60 / 40.96 | 39.53 | 62 | 1238 | 417.0 / 416.0 |  |
| phase13c-01 | onnx | measured | 4 | ok | 1311.29 | 16.09 / 14.97 / 23.37 | 49.15 | 62 | 1238 | 416.0 / 416.0 |  |
| phase13c-01 | onnx | measured | 5 | ok | 1562.08 | 18.04 / 16.17 / 31.78 | 43.74 | 62 | 1238 | 416.0 / 416.0 |  |
