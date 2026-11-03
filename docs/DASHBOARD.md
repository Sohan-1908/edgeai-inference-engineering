# Streamlit dashboard

[Back to the README](../README.md)

## Architecture

The request path is **Streamlit -> FastAPI -> existing inference engine**. Streamlit calls `/health`, `/api/v1/config`, `/api/v1/detect`, and `/api/v1/detect/video` over HTTP. It does not load YOLO/ONNX models, import internal inference modules to execute detection, or perform inference locally.

The API controls the active runtime and device. The dashboard displays the active runtime and backend configurations but does not offer a per-request runtime selector because the API does not support one.

## Start the dashboard

Start the CUDA-enabled FastAPI backend in one terminal, then launch Streamlit in another. For the backend setup and environment options, see [API documentation](./API.md).

```powershell
.\.venv\Scripts\Activate.ps1
$env:EDGEAI_API_BASE_URL = "http://127.0.0.1:8000" # optional; this is the default
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Configuration and connection status

`EDGEAI_API_BASE_URL` configures the API endpoint. The default is `http://127.0.0.1:8000` for a local dashboard. Docker Compose sets the dashboard container's URL to `http://api:8000`.

The sidebar shows API connection status, runtime/device, model input resolution, supported backend configurations, and upload limits. Use **Refresh connection** after changing the backend URL or restarting FastAPI. Backend settings are selected through the API environment variables `EDGEAI_API_RUNTIME`, `EDGEAI_API_DEVICE`, `EDGEAI_API_IMGSZ`, `EDGEAI_API_CONF`, and `EDGEAI_API_IOU`.

## Image workflow

The **Image Inference** tab accepts JPG, JPEG, or PNG files. Submit the image to the backend to display the annotated image, detection count, class/confidence/bounding-box table, and timing fields returned by the API. The frontend does not execute inference.

## Video workflow

The **Video Inference** tab accepts MP4 or AVI files. ByteTrack is optional when the active API runtime is PyTorch; ONNX tracking remains unsupported and the API rejects that combination. The dashboard displays the returned annotated video and metadata and offers **Download annotated video** to save the result.

Video metrics are parsed from the `X-EdgeAI-Metadata` response header. Missing or malformed metadata is reported, and fields are marked unavailable rather than fabricated. Forms defer inference until submission, so ordinary Streamlit reruns do not resend a file.

## HTTP client behavior and limitations

The reusable client in `edgeai/api_client.py` handles connection failures, non-2xx responses, and safely parsed metadata, with separate connection, request, and video-read timeouts. Only small health/config values are kept in Streamlit session state; upload and result bytes are not persisted to disk by the dashboard. Uploaded and output media still occupy memory during the active session.

The dashboard file-upload cap follows the API setting up to the 100 MiB Streamlit server cap configured in `.streamlit/config.toml`. Codec support depends on the OpenCV backend used by FastAPI. The dashboard does not provide authentication, remote deployment, webcam streaming, or persistent job history.
