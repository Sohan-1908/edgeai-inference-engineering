# Setup and Reproduction

This guide separates recorded environment facts and completed experiments from setup and runtime commands that are documented but still require local validation.

## Verified benchmark environment

The Phase 16B/17A experiment reports record the following environment:

- Windows 11 build `10.0.26300`, 64-bit Python 3.12.10.
- NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB; driver 610.78.
- PyTorch 2.14.0+cu130 with CUDA runtime 13.0.
- ONNX Runtime 1.30.0 with `CUDAExecutionProvider` available.
- Ultralytics 8.4.197, NumPy 2.5.4, and OpenCV 5.0.0.

These are recorded experiment-environment details, not a guarantee that every dependency combination will work on another machine.

## Local environment setup

The repository provides separate dependency manifests:

- `requirements.txt`: minimal CPU dependencies for the standalone cloud demo.
- `requirements-cloud.txt`: equivalent dependencies for local cloud-demo setup.
- `requirements-gpu.txt`: pinned CUDA/local development environment.

For local GPU development on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
```

A compatible NVIDIA driver and CUDA-enabled PyTorch installation are required for GPU inference. The FastAPI service also requires CUDA. The standalone `cloud_app.py` uses CPU ONNX Runtime and does not require CUDA.

No dependencies were installed as part of the Phase 20 documentation work.

## CLI commands

The following are documented examples for the current `main.py` interface. They are not a claim that every command has been validated on a clean installation.

The image examples use the tracked `edgeai_annotated.jpg` file in the repository root. Provide your own video file if `demo_video.mp4` is unavailable.

```powershell
# Image inference
python main.py --image edgeai_annotated.jpg --device auto

# Video inference
python main.py --video demo_video.mp4 --device auto

# ONNX Runtime on CUDA
python main.py --image edgeai_annotated.jpg --runtime onnx --device cuda --onnx-model models\yolo11n.onnx

# Optional PyTorch video tracking
python main.py --video demo_video.mp4 --runtime pytorch --device cuda --track

# Export a 640 static FP32 ONNX model
python main.py --export-onnx --model yolo11n.pt --imgsz 640
```

CLI file inference supports CPU or CUDA where the selected runtime permits it. Webcam inference requires CUDA and a connected camera:

```powershell
python main.py --webcam --device cuda
```

## API configuration

FastAPI requires CUDA; its detector is initialized during application startup. It does not fall back to CPU. The supported runtime choices are PyTorch FP32 and strict CUDA ONNX FP32. Use one Uvicorn worker so multiple model copies do not consume additional GPU memory.

| Variable | Default | Description |
|---|---|---|
| `EDGEAI_API_RUNTIME` | `pytorch` | `pytorch` or `onnx` |
| `EDGEAI_API_DEVICE` | `cuda` | Only `cuda` is accepted by the API |
| `EDGEAI_API_IMGSZ` | `640` | `640`, `480`, or `320` |
| `EDGEAI_API_CONF` | `0.25` | Confidence threshold |
| `EDGEAI_API_IOU` | `0.70` | NMS IoU threshold |
| `EDGEAI_API_MODEL` | `yolo11n.pt` | PyTorch checkpoint path/name |
| `EDGEAI_API_ONNX_MODEL` | `models/yolo11n.onnx` | ONNX model path |
| `EDGEAI_API_MAX_UPLOAD_BYTES` | 20 MiB | Image upload limit |
| `EDGEAI_API_MAX_VIDEO_BYTES` | 100 MiB | Video upload limit |
| `EDGEAI_API_MAX_VIDEO_SECONDS` | 120 | Video duration limit |
| `EDGEAI_API_MAX_VIDEO_FRAMES` | 3600 | Decoded frame limit |
| `EDGEAI_API_BASE_URL` | `http://127.0.0.1:8000` | Streamlit backend URL |

Run the API and dashboard in separate PowerShell terminals.

**Terminal 1 — API:**

```powershell
$env:EDGEAI_API_RUNTIME = "pytorch" # or "onnx"
$env:EDGEAI_API_DEVICE = "cuda"
python -m uvicorn edgeai.api:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Dashboard:**

```powershell
$env:EDGEAI_API_BASE_URL = "http://127.0.0.1:8000"
python -m streamlit run app.py
```

The documented endpoints are:

- `GET /health`
- `GET /api/v1/config`
- `POST /api/v1/detect`
- `POST /api/v1/detect/video`

FastAPI documentation is available at `/docs`. Image upload accepts JPG/JPEG/PNG; video accepts MP4/AVI as supported by installed OpenCV codecs. The video response streams annotated bytes and reports metrics in `X-EdgeAI-Metadata`.

Live API/dashboard connectivity remains unverified in the recorded sandbox validation.

## Optional container configuration

`compose.yaml` defines separate API and dashboard services and requests an NVIDIA GPU for the API. Docker Compose runtime was not available during the latest validation, so these are documented commands, not verified container execution:

```powershell
docker compose config
docker compose up --build -d
docker compose ps
docker compose logs -f api
```

Default host ports are 8000 for the API and 8501 for the dashboard. `EDGEAI_API_HOST_PORT` and `EDGEAI_DASHBOARD_HOST_PORT` can change the host port mappings. Model artifact mounts in Compose must resolve locally; Docker Desktop/engine and NVIDIA container GPU support are required.

## Reproduce the Phase 16B CUDA baseline

The exact command recorded in Phase 16B and repeated unchanged in Phase 17A was:

```powershell
.\.venv\Scripts\python.exe benchmark_gpu_baseline.py --image outputs\smoke_input.jpg --weights yolo11n.pt --onnx-model models\yolo11n.onnx --output-dir outputs --warmup-runs 10 --runs 30 --trials 3
```

**Recorded execution status:** The Phase 16B baseline and Phase 17A repeat both completed successfully (exit code 0).

Each run used the same 810 × 1080 image, YOLO11n checkpoint and corresponding ONNX export, 640 × 640 input, batch 1, FP32, confidence 0.25, IoU 0.7, 10 warm-ups, 30 measured calls per trial and 3 paired trials. ONNX used `CUDAExecutionProvider` with CPU fallback disabled.

The original `outputs\smoke_input.jpg` benchmark input and generated timestamped reports are local experiment artifacts and are not included as portable reproduction assets in this repository. To reproduce the exact experiment, restore the original input image and model artifacts used in the recorded runs. The runner generates timestamped JSON and Markdown reports under `--output-dir`, preserving previous reports.

The recorded Phase 16B and Phase 17A results are summarized in [Benchmark Results](benchmark_results.md).

## Validation status and commands

The documented full test command is:

```powershell
python -m unittest discover -s tests -v
```

The latest validation records **153 tests passed cumulatively**, **19 API tests unvalidated**, and **zero completed assertion failures**. The full suite did not complete because asyncio/TestClient socket setup stalled in the Windows sandbox. FastAPI and Streamlit live health/connectivity checks were blocked by sandbox localhost restrictions. Docker runtime was not available.

These results are not a full integration-test pass.

The following health checks are documented but were not successfully executed in the sandbox:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/config
```

Run the full suite and live checks in a local Windows environment that permits asyncio socket-pair creation and localhost networking. Container validation additionally requires Docker Desktop/engine and NVIDIA container GPU support.
