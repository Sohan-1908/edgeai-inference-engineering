import ast
import json
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import onnxruntime as ort
import streamlit as st
import supervision as sv


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL = Path(__file__).parent / "models" / "yolo11n.onnx"
SIZE = 640

MAX_VIDEO_SIZE = 50 * 1024 * 1024
MAX_VIDEO_DURATION = 60
MAX_VIDEO_FRAMES = 1800

FFMPEG_TIMEOUT_SECONDS = 180
VIDEO_WARMUP_FRAMES = 3

BYTE_TRACKER_INPUT_CONFIDENCE = 0.10
BYTE_TRACKER_HIGH_THRESHOLD = 0.25
BYTE_TRACKER_NEW_TRACK_THRESHOLD = 0.25
BYTE_TRACKER_BUFFER = 30
BYTE_TRACKER_MATCH_THRESHOLD = 0.80


COCO = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# --------------------------------------------------
# Streamlit page configuration and styling
# --------------------------------------------------

st.set_page_config(
    page_title="EdgeAI · Cloud Demo",
    page_icon="◈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1018;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
    }

    [data-testid="stSidebar"] {
        background: #101823;
    }

    [data-testid="stMetric"] {
        background: #111a27;
        border: 1px solid #26364b;
        border-radius: 12px;
        padding: 14px;
    }

    .kicker {
        color: #77a9ff;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .15em;
    }

    .note {
        background: #111a27;
        border: 1px solid #26364b;
        border-radius: 12px;
        padding: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Load ONNX model
# --------------------------------------------------


@st.cache_resource
def load_model(path):

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 2

    sess = ort.InferenceSession(
        path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    inp = sess.get_inputs()[0]

    if inp.shape != [1, 3, SIZE, SIZE]:
        raise RuntimeError(f"Expected [1,3,{SIZE},{SIZE}], got {inp.shape}")

    raw = sess.get_modelmeta().custom_metadata_map.get("names")

    try:
        parsed = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            parsed = None

    if isinstance(parsed, dict):
        names = {int(k): str(v) for k, v in parsed.items()}
    elif isinstance(parsed, list):
        names = {i: str(v) for i, v in enumerate(parsed)}
    else:
        names = dict(enumerate(COCO))

    return (
        sess,
        inp.name,
        [o.name for o in sess.get_outputs()],
        names,
    )


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------


def preprocess(im):

    h, w = im.shape[:2]

    scale = min(SIZE / h, SIZE / w)

    rw = round(w * scale)
    rh = round(h * scale)

    resized = cv2.resize(
        im,
        (rw, rh),
        interpolation=cv2.INTER_LINEAR,
    )

    pw = SIZE - rw
    ph = SIZE - rh

    left = round(pw / 2 - 0.1)
    top = round(ph / 2 - 0.1)

    padded = cv2.copyMakeBorder(
        resized,
        top,
        ph - top,
        left,
        pw - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    x = padded[:, :, ::-1].transpose(2, 0, 1)[None]

    tensor = np.ascontiguousarray(x, dtype=np.float32) / 255.0

    return tensor, scale, left, top


# --------------------------------------------------
# Intersection over Union
# --------------------------------------------------


def box_iou(one, many):

    lt = np.maximum(one[:2], many[:, :2])
    rb = np.minimum(one[2:], many[:, 2:])

    inter = np.prod(
        np.maximum(rb - lt, 0),
        axis=1,
    )

    a = np.prod(np.maximum(one[2:] - one[:2], 0))

    b = np.prod(
        np.maximum(many[:, 2:] - many[:, :2], 0),
        axis=1,
    )

    return inter / np.maximum(a + b - inter, 1e-9)


# --------------------------------------------------
# YOLO output decoding and NMS
# --------------------------------------------------


def decode(
    out,
    scale,
    left,
    top,
    w,
    h,
    names,
    conf,
    threshold,
):

    p = np.asarray(out, dtype=np.float32)

    if p.ndim == 3:
        p = p[0]

    if p.shape[0] == 4 + len(names):
        p = p.T
    elif p.shape[0] < p.shape[1]:
        p = p.T

    scores0 = p[:, 4:]

    ids = np.argmax(scores0, axis=1)

    scores = scores0[
        np.arange(len(ids)),
        ids,
    ]

    keep = np.flatnonzero(scores >= conf)

    if not len(keep):
        return []

    p = p[keep]
    scores = scores[keep]
    ids = ids[keep]

    c = p[:, :2]
    sz = p[:, 2:4]

    boxes = np.column_stack((c - sz / 2, c + sz / 2))

    selected = []

    for cid in np.unique(ids):
        order = np.flatnonzero(ids == cid)
        order = order[np.argsort(scores[order])[::-1]]

        while len(order):
            i = int(order[0])
            selected.append(i)

            rest = order[1:]

            if not len(rest):
                break

            order = rest[box_iou(boxes[i], boxes[rest]) <= threshold]

    result = []

    for i in sorted(
        selected,
        key=lambda j: float(scores[j]),
        reverse=True,
    )[:100]:
        x1, y1, x2, y2 = boxes[i]
        cid = int(ids[i])

        result.append(
            {
                "class": names.get(cid, str(cid)),
                "class_id": cid,
                "confidence": float(scores[i]),
                "box": (
                    float(np.clip((x1 - left) / scale, 0, w)),
                    float(np.clip((y1 - top) / scale, 0, h)),
                    float(np.clip((x2 - left) / scale, 0, w)),
                    float(np.clip((y2 - top) / scale, 0, h)),
                ),
            }
        )

    return result


# --------------------------------------------------
# Convert ONNX detections into Supervision format
# --------------------------------------------------


def to_supervision_detections(detections):

    if not detections:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.asarray(
            [d["box"] for d in detections],
            dtype=np.float32,
        ),
        confidence=np.asarray(
            [d["confidence"] for d in detections],
            dtype=np.float32,
        ),
        class_id=np.asarray(
            [d["class_id"] for d in detections],
            dtype=int,
        ),
    )


# --------------------------------------------------
# ByteTrack initialization
# --------------------------------------------------


def create_tracker(fps):

    tracker = sv.ByteTrack(
        track_activation_threshold=BYTE_TRACKER_HIGH_THRESHOLD,
        lost_track_buffer=BYTE_TRACKER_BUFFER,
        minimum_matching_threshold=BYTE_TRACKER_MATCH_THRESHOLD,
        frame_rate=max(1, round(fps)),
        minimum_consecutive_frames=1,
    )

    # Supervision defaults to activation threshold + 0.1.
    # Override this to match Ultralytics new_track_thresh=0.25.
    tracker.det_thresh = BYTE_TRACKER_NEW_TRACK_THRESHOLD

    return tracker


# --------------------------------------------------
# Draw bounding boxes and tracking IDs
# --------------------------------------------------


def annotate(im, dets):

    out = im.copy()

    for d in dets:
        x1, y1, x2, y2 = map(
            lambda x: int(round(x)),
            d["box"],
        )

        color = (214, 130, 54)

        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        if d.get("tracker_id") is not None:
            label = f"ID {d['tracker_id']} | {d['class']} {d['confidence']:.2f}"
        else:
            label = f"{d['class']} {d['confidence']:.2f}"

        (_, th), base = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )

        y = max(th + base + 6, y1)

        cv2.rectangle(
            out,
            (
                x1,
                max(0, y - th - base - 6),
            ),
            (
                x1 + min(400, len(label) * 11 + 8),
                y,
            ),
            color,
            -1,
        )

        cv2.putText(
            out,
            label,
            (x1 + 4, y - base - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return out


# --------------------------------------------------
# Run ByteTrack on one frame
# --------------------------------------------------


def track_frame(tracker, detections, names):

    supervision_detections = to_supervision_detections(detections)

    tracked = tracker.update_with_detections(supervision_detections)

    result = []

    if tracked.tracker_id is None:
        return result

    for i in range(len(tracked)):
        cid = int(tracked.class_id[i]) if tracked.class_id is not None else -1

        confidence = (
            float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
        )

        result.append(
            {
                "class": names.get(cid, str(cid)),
                "class_id": cid,
                "confidence": confidence,
                "box": tuple(float(x) for x in tracked.xyxy[i]),
                "tracker_id": int(tracked.tracker_id[i]),
            }
        )

    return result


# --------------------------------------------------
# Video inference and FFmpeg encoding
# --------------------------------------------------


def process_video(
    uploaded_file,
    model_bundle,
    confidence,
    iou_threshold,
    progress_callback=None,
):
    """Run ONNX CPU inference, ByteTrack, and H.264 encoding."""

    if uploaded_file.size > MAX_VIDEO_SIZE:
        raise ValueError("Video exceeds the 50 MB upload limit.")

    session, input_name, outputs, names = model_bundle

    start_time = time.perf_counter()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        suffix = Path(uploaded_file.name).suffix.lower()

        if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            suffix = ".mp4"

        input_path = temp_dir / f"input{suffix}"
        output_path = temp_dir / "annotated_video.mp4"
        stderr_path = temp_dir / "ffmpeg_stderr.log"

        input_path.write_bytes(uploaded_file.getvalue())

        cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():
            cap.release()
            raise ValueError("Unable to open the uploaded video.")

        ffmpeg = None

        frames_processed = 0
        total_detections = 0
        total_tracked_detections = 0
        inference_time_total = 0.0

        unique_track_ids = set()

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
                raise ValueError("Invalid video metadata or unsupported video.")

            if reported_frames > MAX_VIDEO_FRAMES:
                raise ValueError("Video exceeds the 1800-frame limit.")

            if reported_frames > 0 and reported_frames / fps > MAX_VIDEO_DURATION:
                raise ValueError("Video duration exceeds 60 seconds.")

            output_width = width + (width % 2)
            output_height = height + (height % 2)

            ok, first_frame = cap.read()

            if not ok or first_frame is None:
                raise ValueError("Video contains no readable frames.")

            # Warm up ONNX Runtime.
            warmup_tensor, _, _, _ = preprocess(first_frame)

            for _ in range(VIDEO_WARMUP_FRAMES):
                session.run(
                    outputs,
                    {input_name: warmup_tensor},
                )

            # A new tracker is created for every uploaded video.
            tracker = create_tracker(fps)

            ffmpeg_cmd = [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{output_width}x{output_height}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]

            with open(stderr_path, "wb") as stderr_file:
                ffmpeg = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                )

                frame = first_frame

                while frame is not None:
                    if frames_processed >= MAX_VIDEO_FRAMES:
                        raise ValueError("Video exceeds the 1800-frame limit.")

                    if frames_processed >= MAX_VIDEO_DURATION * fps:
                        raise ValueError("Video duration exceeds 60 seconds.")

                    tensor, scale, left, top = preprocess(frame)

                    inference_start = time.perf_counter()

                    pred = session.run(
                        outputs,
                        {input_name: tensor},
                    )

                    inference_time_total += time.perf_counter() - inference_start

                    # Keep low-confidence candidates for ByteTrack.
                    raw_detections = decode(
                        pred[0],
                        scale,
                        left,
                        top,
                        width,
                        height,
                        names,
                        min(
                            confidence,
                            BYTE_TRACKER_INPUT_CONFIDENCE,
                        ),
                        iou_threshold,
                    )

                    total_detections += len(raw_detections)

                    tracked_detections = track_frame(
                        tracker,
                        raw_detections,
                        names,
                    )

                    total_tracked_detections += len(tracked_detections)

                    for detection in tracked_detections:
                        unique_track_ids.add(detection["tracker_id"])

                    annotated = annotate(
                        frame,
                        tracked_detections,
                    )

                    if output_width != width or output_height != height:
                        annotated = cv2.copyMakeBorder(
                            annotated,
                            0,
                            output_height - height,
                            0,
                            output_width - width,
                            cv2.BORDER_CONSTANT,
                            value=(0, 0, 0),
                        )

                    if ffmpeg.poll() is not None:
                        raise RuntimeError("FFmpeg stopped unexpectedly.")

                    ffmpeg.stdin.write(np.ascontiguousarray(annotated).tobytes())

                    frames_processed += 1

                    if progress_callback is not None:
                        progress_callback(
                            frames_processed,
                            reported_frames,
                        )

                    ok, next_frame = cap.read()

                    frame = next_frame if ok else None

                ffmpeg.stdin.close()

                return_code = ffmpeg.wait(timeout=FFMPEG_TIMEOUT_SECONDS)

                if return_code != 0:
                    error_message = stderr_path.read_text(errors="replace")

                    raise RuntimeError(
                        "FFmpeg encoding failed: " + error_message[-2000:]
                    )

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("FFmpeg encoding exceeded its time limit.") from exc

        finally:
            cap.release()

            if ffmpeg is not None and ffmpeg.poll() is None:
                ffmpeg.terminate()

                try:
                    ffmpeg.wait(timeout=5)

                except subprocess.TimeoutExpired:
                    ffmpeg.kill()
                    ffmpeg.wait()

        if frames_processed == 0:
            raise RuntimeError("No video frames were processed.")

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not produce a valid output video.")

        processing_seconds = time.perf_counter() - start_time

        output_bytes = output_path.read_bytes()

        metrics = {
            "reported_frames": reported_frames,
            "processed_frames": frames_processed,
            "width": width,
            "height": height,
            "fps": fps,
            "duration": frames_processed / fps,
            "processing_seconds": processing_seconds,
            "avg_inference_ms": (inference_time_total / frames_processed) * 1000,
            "processing_fps": (frames_processed / processing_seconds),
            "total_detections": total_detections,
            "tracked_detections": total_tracked_detections,
            "unique_track_ids": len(unique_track_ids),
            "provider": "CPUExecutionProvider",
        }

        return output_bytes, metrics


# --------------------------------------------------
# Main application UI
# --------------------------------------------------

st.markdown(
    '<div class="kicker">EDGEAI · PUBLIC CLOUD DEMO</div>',
    unsafe_allow_html=True,
)

st.title("Vision inference, running on CPU.")

st.caption("YOLO11n · ONNX Runtime · Image and Video Detection")

st.markdown(
    """
    <div class="note">
        Hosted inference runs on <b>CPU</b>.
        It does not access your laptop GPU, and its timings
        are not comparable to local CUDA benchmarks.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Check model availability
# --------------------------------------------------

if not MODEL.is_file():
    st.error("Missing models/yolo11n.onnx")
    st.stop()


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------

with st.sidebar:
    st.markdown("### ◈ EdgeAI")

    confidence = st.slider(
        "Confidence threshold",
        0.05,
        0.95,
        0.25,
        0.05,
    )

    iou_threshold = st.slider(
        "NMS IoU threshold",
        0.1,
        0.9,
        0.7,
        0.05,
    )

    st.caption("YOLO11n FP32 · 640×640 · ONNX Runtime CPU")


# --------------------------------------------------
# Image and Video tabs
# --------------------------------------------------

image_tab, video_tab = st.tabs(["Image Inference", "Video Inference"])


# --------------------------------------------------
# Image inference
# --------------------------------------------------

with image_tab:
    upload = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    if upload:
        if upload.size > 10 * 1024 * 1024:
            st.error("Image must be under 10 MB.")

        else:
            image = cv2.imdecode(
                np.frombuffer(
                    upload.getvalue(),
                    np.uint8,
                ),
                cv2.IMREAD_COLOR,
            )

            if image is None:
                st.error("Could not decode image.")

            else:
                st.image(
                    cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB,
                    ),
                    caption="Input image",
                    width="stretch",
                )

                if st.button(
                    "Run detection",
                    type="primary",
                    width="stretch",
                    key="image_detect",
                ):
                    try:
                        with st.spinner("Running ONNX inference on CPU..."):
                            model_bundle = load_model(str(MODEL))

                            session, input_name, outputs, names = model_bundle

                            tensor, scale, left, top = preprocess(image)

                            start = time.perf_counter()

                            pred = session.run(
                                outputs,
                                {input_name: tensor},
                            )

                            runtime = (time.perf_counter() - start) * 1000

                            detections = decode(
                                pred[0],
                                scale,
                                left,
                                top,
                                image.shape[1],
                                image.shape[0],
                                names,
                                confidence,
                                iou_threshold,
                            )

                            result = annotate(
                                image,
                                detections,
                            )

                        st.success("Inference completed")

                        a, b, c = st.columns(3)

                        a.metric(
                            "Objects detected",
                            len(detections),
                        )

                        b.metric(
                            "ONNX Runtime",
                            f"{runtime:.1f} ms",
                        )

                        c.metric("Provider", "CPU")

                        left_col, right_col = st.columns([1.3, 1])

                        with left_col:
                            st.image(
                                cv2.cvtColor(
                                    result,
                                    cv2.COLOR_BGR2RGB,
                                ),
                                caption="Annotated result",
                                width="stretch",
                            )

                            ok, encoded = cv2.imencode(
                                ".jpg",
                                result,
                            )

                            if ok:
                                st.download_button(
                                    "Download annotated image",
                                    encoded.tobytes(),
                                    "edgeai_annotated.jpg",
                                    "image/jpeg",
                                )

                        with right_col:
                            st.subheader("Detection details")

                            if detections:
                                for index, detection in enumerate(
                                    detections,
                                    start=1,
                                ):
                                    x1, y1, x2, y2 = detection["box"]

                                    st.markdown(
                                        f"""
                                        **Detection {index} — {detection["class"]}**

                                        Confidence: `{detection["confidence"]:.3f}`

                                        Bounding Box: `({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})`

                                        ---
                                        """
                                    )

                            else:
                                st.info("No objects detected above this threshold.")

                    except Exception as exc:
                        st.exception(exc)


# --------------------------------------------------
# Video inference
# --------------------------------------------------

with video_tab:
    st.markdown(
        """
        Upload a video to run frame-by-frame YOLO11n inference
        with persistent ByteTrack IDs.

        The output is encoded as an H.264 MP4 video.
        """
    )

    st.caption("Limits: 50 MB · 60 seconds · 1,800 frames")

    video_upload = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="video_upload",
    )

    if video_upload:
        st.video(video_upload.getvalue())

        if video_upload.size > MAX_VIDEO_SIZE:
            st.error("Video exceeds the 50 MB upload limit.")

        elif st.button(
            "Run video inference",
            type="primary",
            width="stretch",
            key="video_detect",
        ):
            progress = st.progress(
                0.0,
                text="Initializing video inference...",
            )

            try:
                model_bundle = load_model(str(MODEL))

                def update_progress(done, total):

                    if total > 0:
                        ratio = min(done / total, 1.0)

                        progress.progress(
                            ratio,
                            text=f"Processing frame {done}/{total}",
                        )

                    else:
                        progress.progress(
                            0.0,
                            text=f"Processed {done} frames",
                        )

                with st.spinner("Processing video on CPU..."):
                    video_bytes, metrics = process_video(
                        video_upload,
                        model_bundle,
                        confidence,
                        iou_threshold,
                        update_progress,
                    )

                progress.progress(
                    1.0,
                    text="Video processing completed.",
                )

                st.success("Video inference and tracking completed.")

                st.subheader("Processing metrics")

                a, b, c = st.columns(3)

                a.metric(
                    "Processed frames",
                    metrics["processed_frames"],
                )

                b.metric(
                    "Average inference",
                    f"{metrics['avg_inference_ms']:.1f} ms/frame",
                )

                c.metric(
                    "Processing FPS",
                    f"{metrics['processing_fps']:.2f}",
                )

                d, e, f = st.columns(3)

                d.metric(
                    "Raw detections",
                    metrics["total_detections"],
                )

                e.metric(
                    "Unique track IDs",
                    metrics["unique_track_ids"],
                )

                f.metric(
                    "Tracked detections",
                    metrics["tracked_detections"],
                )

                g, h, i = st.columns(3)

                g.metric(
                    "Processing time",
                    f"{metrics['processing_seconds']:.2f} s",
                )

                h.metric(
                    "Video FPS",
                    f"{metrics['fps']:.2f}",
                )

                i.metric(
                    "Provider",
                    "CPU",
                )

                st.subheader("Video information")

                st.write(f"**Resolution:** {metrics['width']} × {metrics['height']}")

                st.write(f"**Processed duration:** {metrics['duration']:.2f} seconds")

                st.write(f"**Reported frames:** {metrics['reported_frames']}")

                st.subheader("Annotated video")

                st.video(video_bytes)

                st.download_button(
                    "Download annotated video",
                    data=video_bytes,
                    file_name="edgeai_annotated_video.mp4",
                    mime="video/mp4",
                    type="primary",
                    width="stretch",
                )

            except Exception as exc:
                progress.empty()

                st.error(f"Video processing failed: {exc}")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "EdgeAI · Cloud CPU demo. "
    "See the repository README for local CUDA inference "
    "and benchmarks."
)
