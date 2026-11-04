# Benchmark Results

## Scope and evidence

The figures below come from the authoritative Phase 16B CUDA baseline and its unchanged Phase 17A repeat. The original Phase 16B artifact remains the reference; Phase 17A is a repeatability check, not a replacement baseline.

The compared methods use the same YOLO11n checkpoint and corresponding ONNX export, source image, and inference configuration.

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB |
| Source image | Original local experiment input, decoded 810 × 1080 |
| Model input / batch | 640 × 640 / 1 |
| Precision | FP32 |
| Confidence / IoU | 0.25 / 0.7 |
| Warm-up / measured / trials | 10 / 30 calls per trial / 3 paired trials |
| PyTorch device | `cuda:0` |
| ONNX provider | `CUDAExecutionProvider`, CPU fallback disabled |

The original Phase 16B JSON records the input, checkpoint, and ONNX hashes. Phase 17A verified that the same artifacts and input were used.

The original input image and raw experiment reports are local artifacts and are not included in the public repository. The measurements below preserve the recorded results.

## Paired CUDA results

Latency values are milliseconds. Full-call latency is an independently timed wall-clock interval around `detector.predict_frame`. Model execution is the runtime-reported model stage: Ultralytics `speed.inference` for PyTorch and synchronous `session.run` for ONNX Runtime. Throughput is pooled from total measured calls divided by the sum of corresponding samples.

| Runtime | Run | Full-call mean | Median | P90 | P95 | Model execution mean | Full-call throughput |
|---|---|---:|---:|---:|---:|---:|---:|
| PyTorch CUDA | Phase 16B baseline | 82.232 | 76.060 | 86.670 | 93.407 | 64.814 | 12.161 FPS |
| ONNX Runtime CUDA | Phase 16B baseline | 55.734 | 54.469 | 65.727 | 68.322 | 36.652 | 17.942 FPS |
| PyTorch CUDA | Phase 17A repeat | 80.606 | 79.066 | 89.265 | 95.220 | 63.477 | 12.406 FPS |
| ONNX Runtime CUDA | Phase 17A repeat | 59.732 | 57.905 | 68.993 | 71.201 | 37.861 | 16.742 FPS |

Under this paired baseline method, ONNX Runtime had lower full-call mean latency in both runs. The Phase 17A full-call mean differed from Phase 16B by -1.98% for PyTorch and +7.17% for ONNX Runtime.

The runtime ordering repeated in these two experiments. This is not a universal performance guarantee or a claim of accuracy equivalence.

The recorded benchmark command was:

```powershell
.\.venv\Scripts\python.exe benchmark_gpu_baseline.py --image outputs\smoke_input.jpg --weights yolo11n.pt --onnx-model models\yolo11n.onnx --output-dir outputs --warmup-runs 10 --runs 30 --trials 3
```

For exact reproduction, restore the original benchmark input and model artifacts. The command uses a local experiment input that is not distributed with the repository. The benchmark runner creates timestamped JSON and Markdown reports under `--output-dir`.

## Why Phase 16C is not directly comparable

The corrected-clock Phase 16C profile reported full-call means of 21.774 ms for PyTorch and 29.996 ms for ONNX Runtime, reversing the ordering in both paired Phase 16B-format runs. The cause was not established.

Phase 16C used separate runtime worker processes and sequential runtime profiles, rather than Phase 16B's single-process paired/interleaved schedule. It also sampled telemetry concurrently at a nominal 500 ms interval; only six telemetry samples per runtime fell in measured phases.

Phase 16B and Phase 17A did not collect continuous GPU telemetry. These differences are methodology and context differences, not proof of an optimization or a causal explanation.

Phase 16C's NVML measurements were device-wide, not process-specific. Its measured-phase device utilization averages were 42.3% (PyTorch) and 27.7% (ONNX); device memory used was about 694.7 and 658.7 MiB out of 4096 MiB respectively.

Six samples per runtime are not instantaneous peaks. WDDM did not provide reliable process-specific GPU memory, and GPU utilization alone does not establish a bottleneck.

## Isolated ONNX stage profiles

These profiles used different measurement scopes from the end-to-end paired benchmark. They isolate CPU preprocessing or decoding work. Do not add their figures to Phase 16B/17A latency or interpret them as end-to-end improvements.

### Phase 17B preprocessing

The final profile used the same 810 × 1080 input and 640 × 640, batch-1 FP32 ONNX preprocessing; it performed 10 warm-ups and 90 measured calls. It did not include model execution. The profiled and production preprocessing outputs matched exactly.

| Stage | Mean | Median | P90 | P95 |
|---|---:|---:|---:|---:|
| Contiguous float32 conversion | 4.459 | 3.660 | 6.768 | 7.224 |
| In-place normalization | 2.565 | 2.102 | 4.436 | 4.815 |
| Resize | 1.631 | 1.243 | 2.780 | 3.995 |
| Border padding | 1.129 | 0.975 | 1.501 | 1.906 |
| Letterbox geometry | 0.033 | 0.024 | 0.033 | 0.044 |
| Channel/layout/batch view | 0.023 | 0.021 | 0.027 | 0.029 |
| Input-size validation | 0.003 | 0.003 | 0.004 | 0.004 |

Whole preprocessing averaged 10.704 ms; the sum of stage means was 9.843 ms. This 0.861 ms difference comes from distinct executions and timing scopes; it is not assigned to a single stage or solely to profiling overhead.

### Phase 17C output decoding and NMS

The profile reused one captured FP32 output of shape `[1, 84, 8400]` from a CUDA ONNX session. It performed 10 warm-ups and 90 measured CPU postprocessing calls.

The reference decoder produced five detections; the decomposed and production paths agreed exactly for measured and warm-up calls.

| Stage | Mean | Median | P90 | P95 |
|---|---:|---:|---:|---:|
| Class-score selection | 6.601 | 5.805 | 8.635 | 9.091 |
| Class-aware greedy NMS | 1.584 | 0.988 | 4.125 | 4.913 |
| Coordinate restoration, clipping, detection construction | 0.709 | 0.450 | 1.247 | 1.708 |
| Confidence filtering | 0.178 | 0.151 | 0.249 | 0.289 |
| Box conversion | 0.189 | 0.117 | 0.162 | 0.201 |
| Output conversion/validation/orientation | 0.041 | 0.017 | 0.024 | 0.028 |
| Maximum-detection cap | 0.005 | 0.002 | 0.002 | 0.003 |

The production decoder averaged 8.693 ms, while summed stage means were 9.307 ms. These are isolated CPU timings for one captured output, not proof of an end-to-end bottleneck.

## Decision and limitations

Phase 18 made a **NO-GO** decision on production optimization.

Conversion and class-score selection are candidates for further controlled study, but isolated profiles do not establish that optimizing either reduces full-call latency. An optimization would require correctness/parity checks and repeated before/after paired Phase 16B-style end-to-end measurements.

No speedup is claimed.

The benchmark covers one input image and one hardware/software configuration. Latencies depend on runtime versions, device state, model artifacts, input, and measurement method.

Phase 16B did not collect GPU telemetry; Phase 16C telemetry was sparse and device-wide. No benchmark result establishes dataset-level accuracy, process-specific peak VRAM, or general real-time 30 FPS performance.

For environment setup and reproduction instructions, see the [Setup and Reproduction Guide](setup_and_reproduction.md).