# EdgeAI CPU cloud demo setup

This repository includes a standalone Streamlit image demo that runs the
tracked `models/yolo11n.onnx` model with ONNX Runtime's
`CPUExecutionProvider`. It is separate from the API-backed dashboard in
`app.py`; it does not require a FastAPI service, CUDA, or a GPU.

For the full project overview and local GPU setup, see the [main
README](README.md).

## Files and repository layout

Deploy the repository as checked in; do not copy files into another project.
The app expects these paths relative to the repository root:

```text
cloud_app.py
requirements.txt
models/yolo11n.onnx
.streamlit/config.toml   # Optional; applies the checked-in dark theme and upload limit
```

The root `requirements.txt` contains the minimal CPU demo dependencies.
`requirements-cloud.txt` has the same dependency declarations and is provided
for an explicitly named cloud-only install in local environments. Use
`requirements.txt` for Streamlit Community Cloud's root-level dependency
discovery. Do not select `requirements-gpu.txt` for this demo; it installs
CUDA-enabled PyTorch, ONNX Runtime GPU, and the broader local project stack.

## Deploy on Streamlit Community Cloud

1. Make the repository available on GitHub and open Streamlit Community Cloud.
2. Create an app using repository `Sohan-1908/edgeai-inference-engineering`,
   branch `main`, and main file path `cloud_app.py`.
3. Keep the entry point at the repository root. The root
   `requirements.txt`, `models/` artifact, and optional `.streamlit/`
   configuration are part of this app's expected repository layout.
4. In Advanced settings, choose Python 3.12, matching the project's documented
   local environment. The hosted deployment has not been validated; if package
   resolution fails, review the build log and select a Python version supported
   by the declared packages before retrying.
5. Deploy. The app uses `models/yolo11n.onnx`, which is tracked in this
   repository.

No secrets, API URL, or environment variables are required by `cloud_app.py`.
The app accepts JPG, JPEG, or PNG image uploads and runs detection on the
Streamlit host's CPU. It is a demonstration, not a GPU deployment or a
performance benchmark.

## Run locally on Windows

From the repository root, create an isolated Python 3.12 environment and install
the cloud-only dependency set:

```powershell
py -3.12 -m venv .venv-cloud
.\.venv-cloud\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-cloud.txt
python -m streamlit run cloud_app.py
```

Alternatively, install `requirements.txt`; it currently declares the same
four packages and version ranges as `requirements-cloud.txt`. The local app
uses the checked-in `.streamlit/config.toml` when run from the repository root.

## Dependency responsibilities

| File | Intended use |
|---|---|
| `requirements.txt` | Minimal CPU-only standalone cloud demo; recognized at the repository root by Streamlit Community Cloud. |
| `requirements-cloud.txt` | Same minimal CPU-only set, named explicitly for local cloud-demo setup. |
| `requirements-gpu.txt` | Full pinned Windows/Python 3.12 CUDA development environment; not for the CPU cloud demo. |
| `requirements-api.txt` | FastAPI inference service/container dependencies. |
| `requirements-dashboard.txt` | HTTP dashboard/container dependencies for `app.py`. |

The hosted app executes inference in-process using ONNX Runtime CPU. It is not
equivalent to local CUDA inference, and successful local execution does not
certify a hosted deployment.
