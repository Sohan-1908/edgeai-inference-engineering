# Setup and Reproduction

This guide separates recorded environment facts and commands that completed from setup/runtime commands that are documented but still require local validation.

## Verified benchmark environment

The Phase 16B/17A experiment reports record:

- Windows 11 build `10.0.26300`, 64-bit Python 3.12.10.
- NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB; driver 610.78.
- PyTorch 2.14.0+cu130 with CUDA runtime 13.0.
- ONNX Runtime 1.30.0 with `CUDAExecutionProvider` available.
- Ultralytics 8.4.197, NumPy 2.5.4, and OpenCV 5.0.0.

The project `requirements.txt` pins its dependencies and PyTorch CUDA wheel index. `requirements-api.txt` and `requirements-dashboard.txt` separate the API and frontend dependency sets. A compatible NVIDIA driver/runtime combination is required for GPU inference. No dependencies were installed as part of the Phase 20 documentation work.

## Local environment setup

Documented Windows PowerShell setup (not executed during Phase 20):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The repository includes `yolo11n.pt` and static ONNX artifacts/manifests in `models/`. If recreating or changing artifacts, use the CLI export command and check the matching input shape; model settings and hashes are part of the authoritative benchmark records. A first Ultralytics run may download pretrained weights if the checkpoint is not present.

Available static ONNX sizes are 640 (`models/yolo11n.onnx`), 480 (`models/yolo11n_480.onnx`), and 320 (`models/yolo11n_320.onnx`). API ONNX configuration must point at the export matching `EDGEAI_API_IMGSZ`. FP16 and INT8 artifacts are separate experiments; they are not the Phase 16B FP32 comparison.

## CLI commands

These commands reflect the current `main.py` options. They are documented examples; not all were executed as part of the Phase 19 validation.

```powershell
# Image and file video
python main.py --image outputs\smoke_input.jpg --device auto
python main.py --video demo_video.mp4 --device auto

# ONNX Runtime on CUDA, with matching fixed-shape export
python main.py --image outputs\smoke_input.jpg --runtime onnx --device cuda --onnx-model models\yolo11n.onnx

# Optional PyTorch video tracking
python main.py --video demo_video.mp4 --runtime pytorch --device cuda --track

# Export a 640 static FP32 ONNX model from the checkpoint
python main.py --export-onnx --model yolo11n.pt --imgsz 640
```

CLI file inference can use CPU or CUDA where supported. Webcam inference requires CUDA and a connected camera:

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

Run API and dashboard in separate PowerShell terminals:

```powershell
$env:EDGEAI_API_RUNTIME = "pytorch" # or "onnx"
$env:EDGEAI_API_DEVICE = "cuda"
python -m uvicorn edgeai.api:app --host 127.0.0.1 --port 8000
```

```powershell
$env:EDGEAI_API_BASE_URL = "http://127.0.0.1:8000"
python -m streamlit run app.py
```

The documented endpoints are `GET /health`, `GET /api/v1/config`, `POST /api/v1/detect`, and `POST /api/v1/detect/video`. FastAPI docs are at `/docs`. Image upload accepts JPG/JPEG/PNG; video accepts MP4/AVI as supported by installed OpenCV codecs. The video response streams annotated bytes and reports metrics in `X-EdgeAI-Metadata`.

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

The exact command recorded in Phase 16B and repeated unchanged in Phase 17A is:

```powershell
.\.venv\Scripts\python.exe benchmark_gpu_baseline.py --image outputs\smoke_input.jpg --weights yolo11n.pt --onnx-model models\yolo11n.onnx --output-dir outputs --warmup-runs 10 --runs 30 --trials 3
```

**Recorded execution status:** the Phase 16B baseline and Phase 17A repeat both completed successfully (exit code 0). Each used the same 810 × 1080 image, YOLO11n checkpoint and corresponding ONNX export, 640 × 640 input, batch 1, FP32, confidence 0.25, IoU 0.7, 10 warm-ups, 30 measured calls per trial and 3 paired trials. ONNX used `CUDAExecutionProvider` with CPU fallback disabled. The runner generates timestamped outputs so prior reports are retained; record the paths printed at completion.

The current runner creates UTC-timestamped JSON and Markdown names under `--output-dir` (`phase16b_gpu_baseline_<timestamp>.json` and `.md`), rather than overwriting a fixed result path.

- Designated baseline report: [Phase 16B](../outputs/phase16b_gpu_baseline_20261103T103243252509Z.md) and [JSON](../outputs/phase16b_gpu_baseline_20261103T103243252509Z.json).
- Repeat: [Phase 17A analysis](../outputs/phase17a_repeatability_analysis.md) and [JSON](../outputs/phase16b_gpu_baseline_20261103T111116898836Z.json).

## Validation status and commands

The documented full test command is:

```powershell
python -m unittest discover -s tests -v
```

The latest validation records 153 tests passed cumulatively, 19 API tests unvalidated, and zero completed assertion failures. The full suite did not complete because asyncio/TestClient socket setup stalled in the Windows sandbox. FastAPI and Streamlit live health/connectivity checks were blocked by sandbox localhost restrictions. Docker runtime was not available. These are not full integration passes.

The following health checks are documented but were not successfully executed in the sandbox:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/config
```

Run the full suite and live checks on a local Windows environment that permits asyncio socket-pair creation and localhost networking. Container validation additionally requires Docker Desktop/engine and NVIDIA container GPU support. See [Phase 19C validation report](../outputs/phase19c_final_validation_report.md).
