import ast
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL = Path(__file__).parent / "models" / "yolo11n.onnx"
SIZE = 640

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

st.set_page_config(page_title="EdgeAI · Cloud Demo", page_icon="◈", layout="wide")

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
        path, sess_options=opts, providers=["CPUExecutionProvider"]
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

    return (sess, inp.name, [o.name for o in sess.get_outputs()], names)


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------


def preprocess(im):

    h, w = im.shape[:2]

    scale = min(SIZE / h, SIZE / w)

    rw = round(w * scale)
    rh = round(h * scale)

    resized = cv2.resize(im, (rw, rh), interpolation=cv2.INTER_LINEAR)

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

    inter = np.prod(np.maximum(rb - lt, 0), axis=1)

    a = np.prod(np.maximum(one[2:] - one[:2], 0))

    b = np.prod(np.maximum(many[:, 2:] - many[:, :2], 0), axis=1)

    return inter / np.maximum(a + b - inter, 1e-9)


# --------------------------------------------------
# YOLO output decoding and Non-Maximum Suppression
# --------------------------------------------------


def decode(out, scale, left, top, w, h, names, conf, threshold):

    p = np.asarray(out, dtype=np.float32)

    if p.ndim == 3:
        p = p[0]

    if p.shape[0] == 4 + len(names):
        p = p.T

    elif p.shape[0] < p.shape[1]:
        p = p.T

    scores0 = p[:, 4:]

    ids = np.argmax(scores0, axis=1)

    scores = scores0[np.arange(len(ids)), ids]

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

    for i in sorted(selected, key=lambda j: float(scores[j]), reverse=True)[:100]:
        x1, y1, x2, y2 = boxes[i]

        cid = int(ids[i])

        result.append(
            {
                "class": names.get(cid, str(cid)),
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
# Draw bounding boxes and labels
# --------------------------------------------------


def annotate(im, dets):

    out = im.copy()

    for d in dets:
        x1, y1, x2, y2 = map(lambda x: int(round(x)), d["box"])

        color = (214, 130, 54)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{d['class']} {d['confidence']:.2f}"

        (_, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

        y = max(th + base + 6, y1)

        cv2.rectangle(
            out,
            (x1, max(0, y - th - base - 6)),
            (x1 + min(320, len(label) * 11 + 8), y),
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
# Main application UI
# --------------------------------------------------

st.markdown(
    '<div class="kicker">EDGEAI · PUBLIC CLOUD DEMO</div>', unsafe_allow_html=True
)

st.title("Vision inference, running on CPU.")

st.caption("YOLO11n · ONNX Runtime · Image detection")

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

    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

    iou_threshold = st.slider("NMS IoU threshold", 0.1, 0.9, 0.7, 0.05)

    st.caption("YOLO11n FP32 · 640×640 · ONNX Runtime CPU")


# --------------------------------------------------
# Image upload
# --------------------------------------------------

upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


if upload:
    if upload.size > 10 * 1024 * 1024:
        st.error("Image must be under 10 MB.")

        st.stop()

    image = cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not decode image.")

        st.stop()

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Input image", width="stretch"
    )

    # --------------------------------------------------
    # Run inference
    # --------------------------------------------------

    if st.button("Run detection", type="primary", width="stretch"):
        try:
            with st.spinner("Running ONNX inference on CPU…"):
                session, input_name, outputs, names = load_model(str(MODEL))

                tensor, scale, left, top = preprocess(image)

                start = time.perf_counter()

                pred = session.run(outputs, {input_name: tensor})

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

                result = annotate(image, detections)

            # --------------------------------------------------
            # Results
            # --------------------------------------------------

            st.success("Inference completed")

            a, b, c = st.columns(3)

            a.metric("Objects detected", len(detections))

            b.metric("ONNX Runtime", f"{runtime:.1f} ms")

            c.metric("Provider", "CPU")

            # --------------------------------------------------
            # Annotated image and detection details
            # --------------------------------------------------

            left_col, right_col = st.columns([1.3, 1])

            with left_col:
                st.image(
                    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                    caption="Annotated result",
                    width="stretch",
                )

                ok, encoded = cv2.imencode(".jpg", result)

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
                    for index, detection in enumerate(detections, start=1):
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
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "EdgeAI · Cloud CPU demo. "
    "See the repository README for local CUDA inference "
    "and benchmarks."
)
