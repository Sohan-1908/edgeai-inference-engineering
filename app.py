"""Streamlit client UI for the EdgeAI FastAPI service (no local inference)."""

from __future__ import annotations

import base64
import os
from math import ceil
from typing import Any

import streamlit as st

from edgeai.api_client import ApiClientError, EdgeAIApiClient


DEFAULT_API_URL = os.getenv("EDGEAI_API_BASE_URL", "http://127.0.0.1:8000")
MIB = 1024 * 1024

st.set_page_config(
    page_title="EdgeAI — Real-Time Vision Inference",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #0b1018; }
        [data-testid="stAppViewContainer"] > .main { background: #0b1018; }
        .block-container { max-width: 1480px; padding-top: 1.4rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] { background: #101823; border-right: 1px solid #1c2a3b; }
        [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
        h1, h2, h3 { letter-spacing: -0.025em; }
        h1 { font-size: clamp(2rem, 4vw, 3.15rem) !important; font-weight: 700 !important; }
        h2 { font-size: 1.45rem !important; }
        [data-testid="stMetric"] {
            background: #111a27; border: 1px solid #1e2d40; border-radius: 12px;
            padding: 1rem 1.1rem; min-height: 108px;
        }
        [data-testid="stMetricLabel"] { color: #93a6bc !important; }
        [data-testid="stMetricValue"] { color: #edf4ff !important; }
        div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
            border-radius: 9px; font-weight: 650; min-height: 2.65rem;
        }
        div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
            background: #3478f6; border-color: #3478f6;
        }
        div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
            background: #2868dc; border-color: #2868dc;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #75aaff; border-bottom-color: #4285f4;
        }
        .edgeai-kicker { color: #77a9ff; font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
        .edgeai-subtitle { color: #9aabc0; font-size: 1.04rem; margin-top: -.8rem; }
        .edgeai-card { background: #111a27; border: 1px solid #1e2d40; border-radius: 12px; padding: 1rem 1.15rem; }
        .edgeai-muted { color: #92a4ba; }
        .edgeai-status { display: inline-flex; align-items: center; gap: .45rem; padding: .35rem .7rem; border: 1px solid #29435d; border-radius: 99px; background: #122239; color: #9fc5ff; font-size: .84rem; font-weight: 650; }
        @media (max-width: 720px) { .block-container { padding-left: 1rem; padding-right: 1rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _read_api_state(base_url: str, refresh: bool) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    state_key = "edgeai_api_state"
    saved = st.session_state.get(state_key, {})
    if refresh or saved.get("base_url") != base_url:
        try:
            client = EdgeAIApiClient(base_url)
            health = client.get_health()
            config = client.get_config() if health.get("detector_initialized") else None
            saved = {
                "base_url": base_url,
                "health": health,
                "config": config,
                "error": None,
            }
        except (ApiClientError, ValueError) as exc:
            saved = {"base_url": base_url, "health": None, "config": None, "error": str(exc)}
        st.session_state[state_key] = saved
    return saved.get("health"), saved.get("config"), saved.get("error")


def _file_size_limit_mib(config: dict[str, Any] | None, key: str, fallback_bytes: int) -> int:
    value = config.get(key) if config else None
    return max(1, ceil((int(value) if value else fallback_bytes) / MIB))


def _format_number(value: Any, suffix: str = "") -> str:
    if value is None:
        return "Unavailable"
    try:
        return f"{float(value):,.2f}{suffix}"
    except (TypeError, ValueError):
        return "Unavailable"


def _show_api_error(message: str) -> None:
    st.error(message, icon="⚠️")
    st.caption("Check the backend terminal and the API URL in the sidebar, then refresh the connection.")


def _render_image_tab(client: EdgeAIApiClient, ready: bool, config: dict[str, Any] | None) -> None:
    st.subheader("Image inference")
    st.markdown('<p class="edgeai-muted">Upload a still image to detect objects with the active backend model.</p>', unsafe_allow_html=True)
    limit_mib = _file_size_limit_mib(config, "max_upload_bytes", 20 * MIB)
    with st.form("image_inference_form", clear_on_submit=False):
        upload = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            key="image_upload",
            max_upload_size=limit_mib,
            help=f"JPG, JPEG or PNG. API limit: {limit_mib} MiB.",
        )
        submitted = st.form_submit_button("Run image detection", type="primary", disabled=not ready)

    if not ready:
        st.info("Connect to a ready FastAPI backend before running inference.")
        return
    if submitted:
        if upload is None:
            st.warning("Choose an image before submitting.")
            return
        try:
            with st.spinner("Sending image to EdgeAI…"):
                result = client.detect_image(
                    upload.getvalue(),
                    upload.name,
                    content_type=upload.type or "application/octet-stream",
                    include_annotated_image=True,
                )
        except ApiClientError as exc:
            _show_api_error(str(exc))
            return

        st.success("Image processed successfully.")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Detections", result.get("number_of_detections", "Unavailable"))
        timings = result.get("timings") if isinstance(result.get("timings"), dict) else {}
        metric_cols[1].metric("Inference", _format_number(timings.get("inference_ms"), " ms"))
        metric_cols[2].metric("Preprocessing", _format_number(timings.get("preprocessing_ms"), " ms"))
        metric_cols[3].metric("Request total", _format_number(timings.get("total_request_processing_ms"), " ms"))

        image_payload = result.get("annotated_image_base64")
        if image_payload:
            try:
                annotated = base64.b64decode(image_payload, validate=True)
                left, right = st.columns([1.2, 1])
                with left:
                    st.image(annotated, caption="Annotated result", width="stretch")
                with right:
                    st.markdown("#### Detection details")
                    detections = result.get("detections")
                    if detections:
                        st.dataframe(detections, width="stretch", hide_index=True)
                    else:
                        st.info("No objects were detected in this image.")
            except (ValueError, TypeError):
                st.warning("The API returned an annotated image that could not be decoded.")
        else:
            st.warning("The API response did not include an annotated image.")


def _render_video_tab(client: EdgeAIApiClient, ready: bool, config: dict[str, Any] | None) -> None:
    st.subheader("Video inference")
    st.markdown('<p class="edgeai-muted">Process a complete MP4 or AVI clip and download its annotated output.</p>', unsafe_allow_html=True)
    runtime = (config or {}).get("runtime")
    runtime_label = runtime.upper() if runtime else "Unavailable"
    st.caption(f"Active runtime: {runtime_label}. Runtime selection is controlled by the FastAPI backend environment.")
    limit_mib = _file_size_limit_mib(config, "max_video_bytes", 100 * MIB)
    with st.form("video_inference_form", clear_on_submit=False):
        upload = st.file_uploader(
            "Choose a video",
            type=["mp4", "avi"],
            key="video_upload",
            max_upload_size=limit_mib,
            help=f"MP4 or AVI, depending on installed OpenCV codecs. API limit: {limit_mib} MiB.",
        )
        track = st.checkbox(
            "Enable ByteTrack IDs",
            value=False,
            disabled=(not ready or runtime != "pytorch"),
            help="Tracking is supported by the API only when its active runtime is PyTorch.",
        )
        submitted = st.form_submit_button("Run video detection", type="primary", disabled=not ready)

    if not ready:
        st.info("Connect to a ready FastAPI backend before running inference.")
        return
    if track and runtime == "onnx":
        st.warning("ByteTrack is unavailable with ONNX. Select PyTorch in the backend configuration to use tracking.")
        return
    if submitted:
        if upload is None:
            st.warning("Choose a video before submitting.")
            return
        try:
            with st.spinner("Processing video on the EdgeAI backend… Large clips may take some time."):
                result = client.detect_video(
                    upload.getvalue(),
                    upload.name,
                    content_type=upload.type or "application/octet-stream",
                    track=track,
                    runtime=runtime,
                )
        except ApiClientError as exc:
            _show_api_error(str(exc))
            return

        if result.metadata_error:
            st.warning(result.metadata_error)
        metadata = result.metadata or {}
        st.success("Video processed successfully.")
        st.video(result.content, format=result.content_type)
        st.download_button(
            "Download annotated video",
            data=result.content,
            file_name=result.file_name,
            mime=result.content_type,
            type="primary",
            on_click="ignore",
            help="Save the streamed annotated result to your device.",
        )

        metric_rows = [
            ("Frames processed", metadata.get("processed_frame_count")),
            ("Input FPS", _format_number(metadata.get("source_fps"))),
            ("Output FPS", _format_number(metadata.get("output_fps"))),
            ("Mean inference", _format_number(metadata.get("mean_inference_latency_ms"), " ms")),
            ("Median inference", _format_number(metadata.get("median_inference_latency_ms"), " ms")),
            ("P95 inference", _format_number(metadata.get("p95_inference_latency_ms"), " ms")),
            ("End-to-end FPS", _format_number(metadata.get("end_to_end_processing_fps"))),
            ("Total processing", _format_number(metadata.get("total_processing_time_ms"), " ms")),
            ("Detections", metadata.get("total_detections")),
            ("Unique track IDs", metadata.get("unique_tracking_ids") if metadata.get("tracking_enabled") else "Not tracked"),
        ]
        cols = st.columns(5)
        for index, (label, value) in enumerate(metric_rows):
            if value is None:
                display_value = "Unavailable"
            elif isinstance(value, float):
                display_value = _format_number(value)
            else:
                display_value = str(value)
            cols[index % 5].metric(label, display_value)
        st.caption(
            f"{metadata.get('video_width', 'Unavailable')} × {metadata.get('video_height', 'Unavailable')} px · "
            f"input {metadata.get('model_input_resolution', 'Unavailable')} · "
            f"{str(metadata.get('runtime', runtime_label)).upper()} / {metadata.get('device', 'Unavailable')} · "
            f"Tracking {'on' if metadata.get('tracking_enabled') else 'off'}"
        )


def main() -> None:
    _apply_styles()
    with st.sidebar:
        st.markdown("### ◈ EdgeAI")
        st.caption("Vision inference console")
        base_url = st.text_input("FastAPI base URL", value=DEFAULT_API_URL, key="api_base_url")
        refresh = st.button("Refresh connection", width="stretch")
        health, config, error = _read_api_state(base_url, refresh)
        is_ready = bool(health and health.get("status") == "ready" and health.get("detector_initialized"))
        if error:
            st.error("Backend offline")
            st.caption(error)
        elif is_ready:
            st.success("Backend connected")
        else:
            st.warning("Backend is not ready")
        if health:
            st.markdown("---")
            st.markdown("**Active configuration**")
            st.write(f"Runtime · `{health.get('active_runtime') or '—'}`")
            st.write(f"Device · `{health.get('active_device') or '—'}`")
            st.write(f"Input · `{health.get('model_input_resolution') or '—'}` px")
        if config:
            available = config.get("supported_configurations") or []
            st.markdown("**Available runtime configurations**")
            if available:
                for item in available:
                    st.caption(
                        f"{str(item.get('runtime', 'unknown')).upper()} · "
                        f"{item.get('device', 'unknown')} · "
                        f"sizes {', '.join(map(str, item.get('input_resolutions') or [])) or '—'}"
                    )
            st.caption(
                f"Image cap {config.get('max_upload_bytes', '—')} B · "
                f"Video cap {config.get('max_video_bytes', '—')} B"
            )
        st.markdown("---")
        st.caption("Inference is sent to FastAPI. This dashboard does not load models.")

    st.markdown('<div class="edgeai-kicker">EdgeAI · Inference platform</div>', unsafe_allow_html=True)
    st.title("Real-Time Vision Inference")
    st.markdown('<p class="edgeai-subtitle">Run image and video detection through your connected GPU-backed API.</p>', unsafe_allow_html=True)

    if config:
        supported = config.get("supported_configurations") or []
        modes = "PyTorch · ONNX" if len(supported) > 1 else (str(config.get("runtime", "—")).upper())
        cards = st.columns(4)
        cards[0].metric("API status", "Ready" if is_ready else "Not ready")
        cards[1].metric("Runtime", str(config.get("runtime", "Unavailable")).upper())
        cards[2].metric("Device", health.get("active_device", "Unavailable") if health else "Unavailable")
        cards[3].metric("Model input", f"{config.get('input_resolution', 'Unavailable')} px")
        st.caption(f"Supported inference modes · {modes} · image and video endpoints")
    elif error:
        st.info("Enter a reachable API base URL in the sidebar and refresh the connection to load backend status.")

    image_tab, video_tab = st.tabs(["Image Inference", "Video Inference"])
    try:
        client = EdgeAIApiClient(base_url)
    except ValueError:
        # The form controls are disabled while offline; retain a safe client object
        # so an empty URL remains a visible connection error instead of a crash.
        client = EdgeAIApiClient(DEFAULT_API_URL)
    with image_tab:
        _render_image_tab(client, is_ready, config)
    with video_tab:
        _render_video_tab(client, is_ready, config)


if __name__ == "__main__":
    main()
