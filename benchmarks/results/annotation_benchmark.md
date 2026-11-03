# Annotation renderer benchmark

## Method

- Source: `outputs\cuda_video_smoke.mp4`
- SHA-256: `a8b0df3073740f808aa0aca3cff7787140837bfa24e01c934ba2cca474fd03cc`
- Video: 62 frames, 640×360, 30.000 FPS
- Representative decoded frame indices (zero-based): `[0, 12, 24, 37, 49, 61]`
- Renderer warmups: 2 per renderer, density, and frame
- Measured repetitions: 7 per renderer, density, and frame
- Detection densities: 5, 20, 50; deterministic synthetic detections
- Identical in-memory frames and detection records are passed to both renderers.
- Frame decoding and detection generation happen outside timed calls; no model is loaded and no inference is run.
- The baseline is a benchmark-only simple OpenCV renderer: green rectangles and labels above boxes.
- The production renderer is `edgeai.postprocess.annotate_image`.
- Environment: Python 3.12.10, OpenCV 5.0.0, Windows-11-10.0.26300-SP0

## Results

| Detections | Renderer | Calls | Mean (ms) | Median (ms) | P95 (ms) | Total (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 5 | baseline | 42 | 0.224 | 0.190 | 0.333 | 9.405 |
| 5 | production | 42 | 0.265 | 0.222 | 0.472 | 11.124 |
| 20 | baseline | 42 | 0.989 | 0.773 | 2.228 | 41.543 |
| 20 | production | 42 | 1.191 | 0.954 | 2.818 | 50.032 |
| 50 | baseline | 42 | 2.695 | 1.963 | 5.881 | 113.186 |
| 50 | production | 42 | 5.415 | 4.415 | 9.861 | 227.444 |

## Production overhead vs baseline

- **5 detections:** +0.041 ms/call (+18.3% mean latency).
- **20 detections:** +0.202 ms/call (+20.4% mean latency).
- **50 detections:** +2.720 ms/call (+100.9% mean latency).

## Interpretation and limitations

This isolates renderer cost for the measured frames and controlled detection counts; it does not explain variance between full video inference runs. Detection locations and identities are synthetic and deterministic, while the pixels come from the real sample video. Baseline styling and label content differ from production styling, so the comparison measures complete renderer cost, not only collision-search overhead. Results are hardware/load dependent and are not a substitute for end-to-end video timing.
