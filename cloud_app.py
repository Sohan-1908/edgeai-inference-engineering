import ast
import html
import json
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import supervision as sv


# ==================================================
# CONFIGURATION
# ==================================================

MODEL = Path(__file__).parent / "models" / "yolo11n.onnx"
SIZE = 640

MAX_IMAGE_SIZE = 10 * 1024 * 1024
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

COCO = (
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,"
    "fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,"
    "elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,"
    "skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,"
    "surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,"
    "apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,"
    "potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,"
    "cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,"
    "scissors,teddy bear,hair drier,toothbrush"
).split(",")

# Annotation colours (RGB). Each object class keeps the same colour.
CLASS_PALETTE_RGB = [
    (13, 148, 136),
    (234, 88, 12),
    (37, 99, 235),
    (147, 51, 234),
    (219, 39, 119),
    (202, 138, 4),
    (22, 163, 74),
    (220, 38, 38),
    (8, 145, 178),
    (79, 70, 229),
]

# Confidence ranges, from lowest to highest (red to green).
CONFIDENCE_COLORS = ["#F87171", "#FB923C", "#FBBF24", "#A3E635", "#34D399"]


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="EdgeAI | Vision Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# DESIGN SYSTEM
# ==================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        color-scheme: dark;
        --bg: #0D1217;
        --surface: #141B22;
        --surface-2: #1A232C;
        --ink: #EDF2F7;
        --body: #B7C3CE;
        --muted: #8696A5;
        --line: #27323D;
        --line-soft: #1F2932;
        --accent: #2DD4BF;
        --accent-strong: #5EEAD4;
        --accent-deep: #0F766E;
        --accent-soft: rgba(45, 212, 191, 0.14);
        --radius: 10px;
        --font: "Plus Jakarta Sans", "Inter", "Segoe UI", -apple-system,
                BlinkMacSystemFont, Roboto, sans-serif;
    }

    html, body, [class*="css"], .stApp {
        font-family: var(--font);
    }

    .stApp :where(p, span, div, label, li, a, button, input, textarea,
                  select, h1, h2, h3, h4, h5, h6, td, th, summary):not([data-testid="stIconMaterial"]) {
        font-family: var(--font) !important;
    }

    /* ---------- Base ---------- */

    .stApp {
        background:
            radial-gradient(1100px 460px at 10% -10%, rgba(45, 212, 191, 0.13), transparent 70%),
            radial-gradient(900px 380px at 100% 0%, rgba(99, 102, 241, 0.09), transparent 70%),
            #0D1217;
        color: var(--body);
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: transparent;
    }

    .main .block-container {
        max-width: 1280px;
        padding: 1.4rem 2.4rem 3rem;
    }

    #MainMenu, footer { visibility: hidden; }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stDeployButton"],
    .stAppDeployButton,
    .stDeployButton {
        display: none !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: var(--body);
        line-height: 1.6;
    }

    [data-testid="stWidgetLabel"] p {
        color: var(--ink) !important;
        font-weight: 600;
        font-size: 0.86rem;
    }

    [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
        font-size: 0.8rem;
    }

    hr {
        border-color: var(--line) !important;
        margin: 2rem 0 !important;
    }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: #2E3A46;
        border-radius: 8px;
        border: 2px solid transparent;
        background-clip: content-box;
    }

    /* ---------- Top bar ---------- */

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.4rem 3.5rem 1rem 0;
        border-bottom: 1px solid var(--line);
        margin-bottom: 2.4rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .brand-mark {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: linear-gradient(135deg, #5EEAD4, #2DD4BF);
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .brand-name {
        color: var(--ink);
        font-weight: 800;
        font-size: 1.05rem;
        letter-spacing: -0.01em;
    }

    .brand-sub {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 500;
        margin-left: 4px;
        padding-left: 12px;
        border-left: 1px solid var(--line);
    }

    .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 999px;
        color: var(--body);
        font-size: 0.78rem;
        font-weight: 600;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #34D399;
        box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.2);
    }

    /* ---------- Hero ---------- */

    .hero-title {
        color: var(--ink);
        font-size: 2.9rem;
        font-weight: 800;
        line-height: 1.12;
        letter-spacing: -0.02em;
        word-spacing: 0.04em;
        max-width: 780px;
        margin: 0 0 1rem;
    }

    .hero-text {
        color: var(--body);
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 640px;
        margin: 0 0 2rem;
    }

    /* ---------- Stat strip ---------- */

    .stat-strip {
        display: flex;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        overflow: hidden;
        margin-bottom: 0.9rem;
    }

    .stat-cell {
        flex: 1 1 0;
        min-width: 0;
        padding: 16px 20px;
        border-right: 1px solid var(--line);
    }

    .stat-cell:last-child {
        border-right: none;
    }

    .stat-label {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .stat-value {
        color: var(--ink);
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.2;
        font-variant-numeric: tabular-nums;
        overflow-wrap: anywhere;
    }

    .stat-caption {
        color: var(--muted);
        font-size: 0.74rem;
        margin-top: 4px;
    }

    /* ---------- Sections ---------- */

    .section-heading {
        color: var(--ink);
        font-size: 1.2rem;
        font-weight: 750;
        letter-spacing: -0.01em;
        margin: 0.4rem 0 0.25rem;
    }

    .section-description {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.6;
        margin-bottom: 1rem;
        max-width: 720px;
    }

    /* ---------- Panels ---------- */

    .panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 18px 20px;
        height: 100%;
    }

    .panel-title {
        color: var(--ink);
        font-size: 0.92rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .panel-text {
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.6;
    }

    .file-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 12px 16px;
        margin-bottom: 0.9rem;
    }

    .file-name {
        color: var(--ink);
        font-weight: 650;
        font-size: 0.9rem;
        overflow-wrap: anywhere;
    }

    .file-meta {
        color: var(--muted);
        font-size: 0.8rem;
        white-space: nowrap;
        font-variant-numeric: tabular-nums;
    }

    /* ---------- Empty state ---------- */

    .empty {
        background: var(--surface);
        border: 1px dashed #36444F;
        border-radius: var(--radius);
        padding: 28px 28px 24px;
        margin-top: 0.5rem;
    }

    .empty-title {
        color: var(--ink);
        font-weight: 750;
        font-size: 1.02rem;
        margin-bottom: 4px;
    }

    .empty-text {
        color: var(--muted);
        font-size: 0.86rem;
        margin-bottom: 18px;
    }

    .steps {
        display: flex;
        gap: 28px;
        flex-wrap: wrap;
    }

    .step {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        font-size: 0.84rem;
        color: var(--body);
        max-width: 230px;
        line-height: 1.45;
    }

    .step-num {
        flex: 0 0 22px;
        height: 22px;
        border-radius: 50%;
        background: var(--accent-soft);
        color: var(--accent-strong);
        font-weight: 800;
        font-size: 0.72rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* ---------- Pipeline ---------- */

    .pipeline {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
    }

    .pipe-step {
        background: var(--surface-2);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 7px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--ink);
    }

    .pipe-arrow {
        color: #5B6B79;
        font-size: 0.9rem;
    }

    /* ---------- Key / value rows ---------- */

    .kv {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        padding: 8px 0;
        border-bottom: 1px solid var(--line-soft);
        font-size: 0.83rem;
    }

    .kv:last-child {
        border-bottom: none;
    }

    .kv-key { color: var(--muted); }

    .kv-val {
        color: var(--ink);
        font-weight: 650;
        text-align: right;
        font-variant-numeric: tabular-nums;
    }

    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        background: #10161C;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    .side-heading {
        color: var(--ink);
        font-weight: 750;
        font-size: 0.95rem;
        margin: 0.2rem 0 0.2rem;
    }

    .side-text {
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.5;
        margin-bottom: 0.8rem;
    }

    [data-testid="stSlider"] [role="slider"] {
        background: var(--accent);
        border-color: var(--accent);
    }

    [data-testid="stSliderThumbValue"],
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"] {
        color: var(--body) !important;
    }

    /* ---------- Tabs ---------- */

    [data-testid="stTabs"] [role="tablist"] {
        gap: 10px;
        border-bottom: 1px solid var(--line);
    }

    [data-testid="stTabs"] button[role="tab"] {
        padding: 10px 18px;
        font-size: 0.92rem;
        font-weight: 650;
    }

    [data-testid="stTabs"] button[role="tab"] p {
        color: var(--muted) !important;
        font-weight: 650;
    }

    [data-testid="stTabs"] button[role="tab"]:hover p {
        color: var(--ink) !important;
    }

    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
        color: var(--accent-strong) !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: var(--accent) !important;
        height: 3px;
    }

    [data-testid="stTabs"] [data-baseweb="tab-border"] {
        background-color: var(--line) !important;
    }

    /* ---------- Buttons ---------- */

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        min-height: 44px;
        font-weight: 650;
        transition: background 0.15s ease, border-color 0.15s ease;
    }

    .stButton > button p,
    .stDownloadButton > button p {
        color: inherit !important;
        font-weight: 650;
    }

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        background: var(--accent);
        color: #04201C;
        border: 1px solid var(--accent);
    }

    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button[kind="primary"]:hover {
        background: var(--accent-strong);
        border-color: var(--accent-strong);
        color: #04201C;
    }

    .stButton > button[kind="secondary"],
    .stDownloadButton > button[kind="secondary"] {
        background: var(--surface-2);
        color: var(--ink);
        border: 1px solid #36444F;
    }

    .stButton > button[kind="secondary"]:hover,
    .stDownloadButton > button[kind="secondary"]:hover {
        border-color: var(--accent);
        color: var(--accent-strong);
    }

    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible {
        outline: 3px solid rgba(45, 212, 191, 0.4);
        outline-offset: 2px;
    }

    /* ---------- Uploader ---------- */

    [data-testid="stFileUploader"] section {
        background: var(--surface);
        border: 1.5px dashed #36444F;
        border-radius: var(--radius);
        padding: 1.2rem;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: var(--accent);
        background: #17212A;
    }

    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] small {
        color: var(--muted);
    }

    [data-testid="stFileUploader"] section button {
        background: var(--surface-2);
        color: var(--ink);
        border: 1px solid #36444F;
        border-radius: 8px;
    }

    [data-testid="stFileUploaderFile"] {
        background: var(--surface-2);
        border-radius: 8px;
    }

    [data-testid="stFileUploaderFileName"] {
        color: var(--ink);
    }

    /* ---------- Media and alerts ---------- */

    [data-testid="stImage"] img {
        border: 1px solid var(--line);
        border-radius: var(--radius);
    }

    [data-testid="stVideo"] {
        border: 1px solid var(--line);
        border-radius: var(--radius);
        overflow: hidden;
        background: #05080B;
    }

    [data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--radius);
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: var(--ink) !important;
    }

    [data-testid="stAlert"] {
        background: var(--surface-2);
        border: 1px solid var(--line);
        border-radius: var(--radius);
    }

    [data-testid="stAlert"] p {
        color: var(--body);
    }

    [data-testid="stProgress"] > div > div {
        background-color: var(--line) !important;
    }

    [data-testid="stProgress"] > div > div > div {
        background-color: var(--accent) !important;
    }

    [data-testid="stSpinner"] p,
    [data-testid="stProgress"] p {
        color: var(--body) !important;
    }

    .chart-title {
        color: var(--ink);
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }

    /* ---------- Custom charts ---------- */

    .chart-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 18px 20px 16px;
        height: 100%;
    }

    .chart-card-title {
        color: var(--ink);
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 2px;
    }

    .chart-card-sub {
        color: var(--muted);
        font-size: 0.78rem;
        margin-bottom: 18px;
    }

    .hbar {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 10px 0;
    }

    .hbar-label {
        flex: 0 0 110px;
        font-size: 0.83rem;
        color: var(--ink);
        font-weight: 600;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .hbar-track {
        flex: 1;
        height: 12px;
        background: var(--line-soft);
        border-radius: 999px;
        overflow: hidden;
    }

    .hbar-fill {
        height: 100%;
        background: linear-gradient(90deg, #14B8A6, #5EEAD4);
        border-radius: 999px;
    }

    .hbar-value {
        flex: 0 0 38px;
        text-align: right;
        font-size: 0.83rem;
        font-weight: 700;
        color: var(--ink);
        font-variant-numeric: tabular-nums;
    }

    .vbars {
        display: flex;
        align-items: flex-end;
        gap: 14px;
        height: 190px;
        border-bottom: 1px solid var(--line);
    }

    .vbar-col {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-end;
    }

    .vbar-value {
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 5px;
        font-variant-numeric: tabular-nums;
    }

    .vbar {
        width: 100%;
        max-width: 60px;
        background: linear-gradient(180deg, #5EEAD4, #14B8A6);
        border-radius: 7px 7px 0 0;
    }

    .vbar.zero {
        background: #2A3640;
    }

    .vlabels {
        display: flex;
        gap: 14px;
        margin-top: 9px;
    }

    .vlabel {
        flex: 1;
        text-align: center;
        font-size: 0.76rem;
        color: var(--muted);
        font-weight: 500;
    }

    /* ---------- Custom tables ---------- */

    .table-wrap {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        overflow: auto;
        max-height: 420px;
    }

    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }

    .data-table th {
        position: sticky;
        top: 0;
        z-index: 1;
        background: var(--surface-2);
        color: var(--muted);
        font-weight: 700;
        font-size: 0.78rem;
        text-align: left;
        padding: 11px 16px;
        border-bottom: 1px solid var(--line);
        white-space: nowrap;
    }

    .data-table td {
        padding: 11px 16px;
        border-bottom: 1px solid var(--line-soft);
        color: var(--body);
        white-space: nowrap;
        font-variant-numeric: tabular-nums;
    }

    .data-table tbody tr:last-child td {
        border-bottom: none;
    }

    .data-table tbody tr:nth-child(even) td {
        background: rgba(255, 255, 255, 0.022);
    }

    .data-table tbody tr:hover td {
        background: rgba(45, 212, 191, 0.09);
    }

    .data-table .num {
        text-align: right;
    }

    .cell-strong {
        color: var(--ink);
        font-weight: 650;
    }

    .conf {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 10px;
    }

    .conf-bar {
        width: 60px;
        height: 6px;
        background: var(--line);
        border-radius: 999px;
        overflow: hidden;
    }

    .conf-fill {
        height: 100%;
        background: var(--accent);
        border-radius: 999px;
    }

    /* ---------- Colour accents ---------- */

    .dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        margin-right: 9px;
        vertical-align: middle;
    }

    .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.78rem;
    }

    .stat-cell {
        position: relative;
        padding-top: 19px;
    }

    .stat-cell::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: #2DD4BF;
    }

    .stat-cell:nth-child(2)::before { background: #A78BFA; }
    .stat-cell:nth-child(3)::before { background: #FBBF24; }
    .stat-cell:nth-child(4)::before { background: #F472B6; }

    .vbar.zero { background: #2A3640 !important; }

    /* ---------- Footer ---------- */

    .footer {
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.8;
        text-align: left;
    }

    .footer strong {
        color: var(--ink);
        font-weight: 700;
    }

    /* ---------- Responsive ---------- */

    @media (max-width: 900px) {
        .main .block-container {
            padding: 1.1rem 1.2rem 2rem;
        }
        .hero-title { font-size: 2.2rem; }
        .topbar { padding-right: 3rem; }
        .stat-strip { flex-wrap: wrap; }
        .stat-cell {
            flex: 1 1 50%;
            border-bottom: 1px solid var(--line);
        }
        .brand-sub { display: none; }
    }

    @media (max-width: 560px) {
        .hero-title { font-size: 1.8rem; }
        .stat-cell { flex: 1 1 100%; border-right: none; }
        .file-row { flex-direction: column; align-items: flex-start; gap: 4px; }
        .status { display: none; }
    }
        </style>
    """,
    unsafe_allow_html=True,
)


# Load the web font through the parent page as well; a bare @import is not
# always applied by Streamlit.
components.html(
    """
    <script>
    const doc = window.parent.document;
    if (!doc.getElementById("edgeai-font")) {
        const link = doc.createElement("link");
        link.id = "edgeai-font";
        link.rel = "stylesheet";
        link.href = "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap";
        doc.head.appendChild(link);
    }
    </script>
    """,
    height=0,
)


# ==================================================
# COLOUR HELPERS
# ==================================================


def class_color_rgb(name):
    """Stable colour per object class, matching the boxes drawn on images."""
    if name in COCO:
        index = COCO.index(name)
    else:
        index = sum(ord(ch) for ch in str(name))
    return CLASS_PALETTE_RGB[index % len(CLASS_PALETTE_RGB)]


def rgb_hex(rgb):
    return "#%02X%02X%02X" % tuple(int(v) for v in rgb)


def tint(rgb, amount=0.4):
    """Mix a colour towards white so it reads well on the dark background."""
    return tuple(v + (255 - v) * amount for v in rgb)


def class_colors(name):
    base = class_color_rgb(name)
    return rgb_hex(base), rgb_hex(tint(base, 0.45))


def confidence_color(value):
    if value >= 0.80:
        return "#34D399"
    if value >= 0.60:
        return "#FBBF24"
    return "#F87171"


# ==================================================
# REUSABLE UI HELPERS
# ==================================================

BRAND_ICON = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
    'stroke="#04201C" stroke-width="2.2" stroke-linecap="round" '
    'stroke-linejoin="round">'
    '<path d="M4 9V5a1 1 0 0 1 1-1h4"/><path d="M15 4h4a1 1 0 0 1 1 1v4"/>'
    '<path d="M20 15v4a1 1 0 0 1-1 1h-4"/><path d="M9 20H5a1 1 0 0 1-1-1v-4"/>'
    '<circle cx="12" cy="12" r="2.2" fill="#04201C" stroke="none"/></svg>'
)


def section(title, subtitle=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f'<div class="section-description">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def stat_strip(items):
    """Render a single joined row of metrics. items = [(label, value, caption)]."""
    cells = "".join(
        '<div class="stat-cell">'
        f'<div class="stat-label">{html.escape(str(label))}</div>'
        f'<div class="stat-value">{html.escape(str(value))}</div>'
        f'<div class="stat-caption">{html.escape(str(caption))}</div>'
        "</div>"
        for label, value, caption in items
    )
    st.markdown(f'<div class="stat-strip">{cells}</div>', unsafe_allow_html=True)


def key_values(rows):
    body = "".join(
        f'<div class="kv"><span class="kv-key">{html.escape(k)}</span>'
        f'<span class="kv-val">{html.escape(v)}</span></div>'
        for k, v in rows
    )
    st.markdown(body, unsafe_allow_html=True)


def file_row(name, meta):
    st.markdown(
        '<div class="file-row">'
        f'<div class="file-name">{html.escape(name)}</div>'
        f'<div class="file-meta">{html.escape(meta)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def empty_state(title, text, steps):
    items = "".join(
        f'<div class="step"><div class="step-num">{i}</div><div>{s}</div></div>'
        for i, s in enumerate(steps, start=1)
    )
    st.markdown(
        '<div class="empty">'
        f'<div class="empty-title">{title}</div>'
        f'<div class="empty-text">{text}</div>'
        f'<div class="steps">{items}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def hbar_chart(title, subtitle, rows, limit=10):
    """Horizontal bar chart drawn in HTML, one colour per class."""
    rows = list(rows)[:limit]
    top = max((v for _, v in rows), default=1) or 1

    bars = []

    for label, value in rows:
        base, light = class_colors(label)
        width = max(value / top * 100, 2)

        bars.append(
            '<div class="hbar">'
            f'<div class="hbar-label" title="{html.escape(str(label))}">'
            f'<span class="dot" style="background:{light}"></span>'
            f"{html.escape(str(label))}</div>"
            '<div class="hbar-track">'
            f'<div class="hbar-fill" style="width:{width:.1f}%;'
            f'background:linear-gradient(90deg,{base},{light})"></div>'
            "</div>"
            f'<div class="hbar-value">{value}</div>'
            "</div>"
        )

    st.markdown(
        '<div class="chart-card">'
        f'<div class="chart-card-title">{title}</div>'
        f'<div class="chart-card-sub">{subtitle}</div>'
        f"{''.join(bars)}</div>",
        unsafe_allow_html=True,
    )


def column_chart(title, subtitle, rows, colors=None, max_px=140):
    """Vertical bar chart drawn in HTML. rows = [(label, value), ...]."""
    rows = list(rows)
    top = max((v for _, v in rows), default=1) or 1
    colors = colors or ["#2DD4BF"] * len(rows)

    cols = "".join(
        '<div class="vbar-col">'
        f'<div class="vbar-value" style="color:{colors[i]}">{value}</div>'
        f'<div class="vbar{" zero" if value == 0 else ""}" '
        f'style="height:{max(round(value / top * max_px), 3)}px;'
        + (
            ""
            if value == 0
            else f"background:linear-gradient(180deg,{colors[i]},{colors[i]}99)"
        )
        + '"></div>'
        "</div>"
        for i, (_, value) in enumerate(rows)
    )

    labels = "".join(
        f'<div class="vlabel">{html.escape(str(label))}</div>' for label, _ in rows
    )

    st.markdown(
        '<div class="chart-card">'
        f'<div class="chart-card-title">{title}</div>'
        f'<div class="chart-card-sub">{subtitle}</div>'
        f'<div class="vbars">{cols}</div>'
        f'<div class="vlabels">{labels}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def html_table(df, max_height=420):
    """Styled, scrollable HTML table with colour-coded cells."""
    numeric = {c: pd.api.types.is_numeric_dtype(df[c]) for c in df.columns}

    head = "".join(
        f'<th class="{"num" if numeric[c] else ""}">{html.escape(str(c))}</th>'
        for c in df.columns
    )

    body_rows = []

    for _, row in df.iterrows():
        cells = []

        for c in df.columns:
            value = row[c]
            name = str(c).lower()

            if "confidence" in name:
                number = max(0.0, min(float(value), 1.0))
                color = confidence_color(number)
                cells.append(
                    '<td class="num"><div class="conf">'
                    f'<span style="color:{color};font-weight:700">{number:.3f}</span>'
                    '<div class="conf-bar">'
                    f'<div class="conf-fill" style="width:{number * 100:.0f}%;'
                    f'background:{color}"></div>'
                    "</div></div></td>"
                )
            elif name in ("object", "object class"):
                base, light = class_colors(value)
                cells.append(
                    '<td class="cell-strong">'
                    f'<span class="dot" style="background:{light}"></span>'
                    f"{html.escape(str(value))}</td>"
                )
            elif name == "track id":
                track = int(value)
                rgb = CLASS_PALETTE_RGB[track % len(CLASS_PALETTE_RGB)]
                base, light = rgb_hex(rgb), rgb_hex(tint(rgb, 0.45))
                cells.append(
                    '<td class="num">'
                    f'<span class="pill" style="color:{light};background:{base}33;'
                    f'border:1px solid {base}88">#{track}</span></td>'
                )
            elif numeric[c]:
                cells.append(f'<td class="num">{html.escape(str(value))}</td>')
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")

        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    st.markdown(
        f'<div class="table-wrap" style="max-height:{max_height}px">'
        f'<table class="data-table"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body_rows)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def detections_dataframe(detections):
    rows = []

    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["box"]

        row = {
            "No.": index,
            "Object": detection["class"],
            "Confidence": round(detection["confidence"], 3),
            "X1": round(x1, 1),
            "Y1": round(y1, 1),
            "X2": round(x2, 1),
            "Y2": round(y2, 1),
        }

        if detection.get("tracker_id") is not None:
            row["Track ID"] = detection["tracker_id"]

        rows.append(row)

    return pd.DataFrame(rows)


def class_distribution(detections):
    counts = Counter(d["class"] for d in detections)

    return pd.DataFrame(
        {
            "Object class": list(counts.keys()),
            "Detections": list(counts.values()),
        }
    ).sort_values("Detections", ascending=False)


def confidence_distribution(detections):
    buckets = {
        "Under 60%": 0,
        "60–70%": 0,
        "70–80%": 0,
        "80–90%": 0,
        "90–100%": 0,
    }

    for detection in detections:
        value = detection["confidence"]

        if value < 0.60:
            buckets["Under 60%"] += 1
        elif value < 0.70:
            buckets["60–70%"] += 1
        elif value < 0.80:
            buckets["70–80%"] += 1
        elif value < 0.90:
            buckets["80–90%"] += 1
        else:
            buckets["90–100%"] += 1

    return pd.DataFrame(
        {
            "Confidence range": list(buckets.keys()),
            "Detections": list(buckets.values()),
        }
    )


def render_detection_analytics(detections):
    if not detections:
        st.info("No detections to chart.")
        return

    class_df = class_distribution(detections)
    confidence_df = confidence_distribution(detections)

    section(
        "Detection breakdown",
        "How many objects of each type were found, and how sure the model was.",
    )

    left, right = st.columns(2, gap="large")

    with left:
        hbar_chart(
            "Objects by class",
            "How many of each object type were found.",
            zip(class_df["Object class"], class_df["Detections"]),
        )

    with right:
        column_chart(
            "Confidence levels",
            "How sure the model was about each detection.",
            zip(confidence_df["Confidence range"], confidence_df["Detections"]),
            colors=CONFIDENCE_COLORS,
        )


# ==================================================
# MODEL LOADING
# ==================================================


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

    return sess, inp.name, [o.name for o in sess.get_outputs()], names


# ==================================================
# IMAGE PREPROCESSING
# ==================================================


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


# ==================================================
# IOU AND NMS
# ==================================================


def box_iou(one, many):
    lt = np.maximum(one[:2], many[:, :2])
    rb = np.minimum(one[2:], many[:, 2:])

    inter = np.prod(np.maximum(rb - lt, 0), axis=1)
    a = np.prod(np.maximum(one[2:] - one[:2], 0))
    b = np.prod(np.maximum(many[:, 2:] - many[:, :2], 0), axis=1)

    return inter / np.maximum(a + b - inter, 1e-9)


# ==================================================
# YOLO OUTPUT DECODING
# ==================================================


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


# ==================================================
# SUPERVISION AND BYTETRACK
# ==================================================


def to_supervision_detections(detections):
    if not detections:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.asarray([d["box"] for d in detections], dtype=np.float32),
        confidence=np.asarray(
            [d["confidence"] for d in detections],
            dtype=np.float32,
        ),
        class_id=np.asarray(
            [d["class_id"] for d in detections],
            dtype=int,
        ),
    )


def create_tracker(fps):
    tracker = sv.ByteTrack(
        track_activation_threshold=BYTE_TRACKER_HIGH_THRESHOLD,
        lost_track_buffer=BYTE_TRACKER_BUFFER,
        minimum_matching_threshold=BYTE_TRACKER_MATCH_THRESHOLD,
        frame_rate=max(1, round(fps)),
        minimum_consecutive_frames=1,
    )

    tracker.det_thresh = BYTE_TRACKER_NEW_TRACK_THRESHOLD
    return tracker


# ==================================================
# ANNOTATION
# ==================================================


def class_color_bgr(class_id):
    r, g, b = CLASS_PALETTE_RGB[int(class_id) % len(CLASS_PALETTE_RGB)]
    return (b, g, r)


def annotate(im, dets):
    out = im.copy()
    frame_h, frame_w = out.shape[:2]

    # Scale line and text size with the image so labels stay readable,
    # and keep labels short when the scene is crowded.
    short_side = min(frame_h, frame_w)
    box_thickness = max(1, round(short_side / 400))
    font_scale = max(0.4, min(1.2, short_side / 900))
    text_thickness = max(1, round(font_scale * 2))
    crowded = len(dets) > 6

    for d in dets:
        x1, y1, x2, y2 = map(lambda x: int(round(x)), d["box"])

        # Tracked objects are coloured by track ID, plain detections by class.
        if d.get("tracker_id") is not None:
            color = class_color_bgr(d["tracker_id"])
            if crowded:
                label = f"#{d['tracker_id']} {d['confidence']:.2f}"
            else:
                label = f"#{d['tracker_id']} {d['class']} {d['confidence']:.2f}"
        else:
            color = class_color_bgr(d.get("class_id", 0))
            label = f"{d['class']} {d['confidence']:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thickness)

        (tw, th), base = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )

        # Keep the label fully inside the frame.
        label_w = tw + 10
        lx = max(0, min(x1, frame_w - label_w))
        y = max(th + base + 6, y1)

        cv2.rectangle(
            out,
            (lx, max(0, y - th - base - 6)),
            (lx + label_w, y),
            color,
            -1,
        )

        cv2.putText(
            out,
            label,
            (lx + 5, y - base - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return out


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


# ==================================================
# VIDEO INFERENCE
# ==================================================


def process_video(
    uploaded_file,
    model_bundle,
    confidence,
    iou_threshold,
    progress_callback=None,
):
    """Run ONNX CPU inference, ByteTrack and H.264 encoding."""

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
        class_counts = Counter()
        track_stats = defaultdict(
            lambda: {
                "class": "unknown",
                "frames_seen": 0,
                "confidence_total": 0.0,
                "peak_confidence": 0.0,
            }
        )

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

            warmup_tensor, _, _, _ = preprocess(first_frame)

            for _ in range(VIDEO_WARMUP_FRAMES):
                session.run(outputs, {input_name: warmup_tensor})

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

                    pred = session.run(outputs, {input_name: tensor})

                    inference_time_total += time.perf_counter() - inference_start

                    raw_detections = decode(
                        pred[0],
                        scale,
                        left,
                        top,
                        width,
                        height,
                        names,
                        min(confidence, BYTE_TRACKER_INPUT_CONFIDENCE),
                        iou_threshold,
                    )

                    total_detections += len(raw_detections)

                    for detection in raw_detections:
                        class_counts[detection["class"]] += 1

                    tracked_detections = track_frame(
                        tracker,
                        raw_detections,
                        names,
                    )

                    # The tracker needs low-score boxes to keep IDs alive, but
                    # only show objects at or above the user's threshold.
                    tracked_detections = [
                        d for d in tracked_detections if d["confidence"] >= confidence
                    ]

                    total_tracked_detections += len(tracked_detections)

                    for detection in tracked_detections:
                        track_id = detection["tracker_id"]
                        unique_track_ids.add(track_id)

                        stats = track_stats[track_id]
                        stats["class"] = detection["class"]
                        stats["frames_seen"] += 1
                        stats["confidence_total"] += detection["confidence"]
                        stats["peak_confidence"] = max(
                            stats["peak_confidence"],
                            detection["confidence"],
                        )

                    annotated = annotate(frame, tracked_detections)

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
                        progress_callback(frames_processed, reported_frames)

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

        track_rows = []

        for track_id, stats in track_stats.items():
            track_rows.append(
                {
                    "Track ID": track_id,
                    "Object": stats["class"],
                    "Frames seen": stats["frames_seen"],
                    "Average confidence": round(
                        stats["confidence_total"] / stats["frames_seen"],
                        3,
                    ),
                    "Peak confidence": round(stats["peak_confidence"], 3),
                }
            )

        metrics = {
            "reported_frames": reported_frames,
            "processed_frames": frames_processed,
            "width": width,
            "height": height,
            "fps": fps,
            "duration": frames_processed / fps,
            "processing_seconds": processing_seconds,
            "avg_inference_ms": (inference_time_total / frames_processed) * 1000,
            "processing_fps": frames_processed / processing_seconds,
            "total_detections": total_detections,
            "tracked_detections": total_tracked_detections,
            "unique_track_ids": len(unique_track_ids),
            "provider": "CPUExecutionProvider",
            "class_counts": dict(class_counts),
            "track_rows": track_rows,
        }

        return output_bytes, metrics


# ==================================================
# APPLICATION HEADER
# ==================================================

st.markdown(
    f"""
    <div class="topbar">
        <div class="brand">
            <div class="brand-mark">{BRAND_ICON}</div>
            <span class="brand-name">EdgeAI</span>
            <span class="brand-sub">Vision Intelligence</span>
        </div>
        <div class="status"><span class="status-dot"></span>Cloud CPU runtime</div>
    </div>
    <div class="hero-title">Detect and track objects in images and video.</div>
    <div class="hero-text">
        Upload a photo or a short clip. EdgeAI finds the objects in it, labels
        each one with a confidence score, and follows them across video frames
        with persistent track IDs.
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# MODEL AVAILABILITY
# ==================================================

if not MODEL.is_file():
    st.error(
        "Model file not found. Place yolo11n.onnx in the models folder "
        "next to this app and reload the page."
    )
    st.stop()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown(
        f"""
        <div class="brand" style="margin-bottom:1.4rem;">
            <div class="brand-mark">{BRAND_ICON}</div>
            <span class="brand-name">EdgeAI</span>
        </div>
        <div class="side-heading">Detection settings</div>
        <div class="side-text">These apply to both images and video.</div>
        """,
        unsafe_allow_html=True,
    )

    confidence = st.slider(
        "Confidence threshold",
        0.05,
        0.95,
        0.25,
        0.05,
        help="Higher values keep only the detections the model is most sure about.",
    )

    iou_threshold = st.slider(
        "Overlap threshold (NMS IoU)",
        0.1,
        0.9,
        0.7,
        0.05,
        help="Lower values remove more overlapping boxes around the same object.",
    )

    st.divider()

    st.markdown('<div class="side-heading">Model</div>', unsafe_allow_html=True)
    key_values(
        [
            ("Name", "YOLO11n"),
            ("Format", "ONNX FP32"),
            ("Input size", "640 × 640"),
            ("Runs on", "CPU"),
        ]
    )

    st.divider()

    st.markdown('<div class="side-heading">Upload limits</div>', unsafe_allow_html=True)
    key_values(
        [
            ("Image", "10 MB"),
            ("Video", "50 MB"),
            ("Duration", "60 sec"),
            ("Frames", "1,800"),
        ]
    )

    st.divider()

    st.markdown(
        """
        <div class="side-text">
            This hosted demo runs on CPU. GPU (CUDA) inference is available
            when you run the project locally.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# WORKSPACE TABS
# ==================================================

image_tab, video_tab = st.tabs(["Image detection", "Video tracking"])


# ==================================================
# IMAGE ANALYSIS
# ==================================================

with image_tab:
    st.write("")
    section(
        "Analyze an image",
        "Supported formats: JPG, JPEG and PNG, up to 10 MB.",
    )

    upload = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
        label_visibility="collapsed",
    )

    if upload:
        if upload.size > MAX_IMAGE_SIZE:
            st.error("This image is larger than 10 MB. Choose a smaller file.")

        else:
            image = cv2.imdecode(
                np.frombuffer(upload.getvalue(), np.uint8),
                cv2.IMREAD_COLOR,
            )

            if image is None:
                st.error(
                    "This file could not be read as an image. "
                    "Try a different JPG or PNG."
                )

            else:
                file_row(
                    upload.name,
                    f"{image.shape[1]} × {image.shape[0]} px",
                )

                if st.button(
                    "Run detection",
                    type="primary",
                    width="stretch",
                    key="image_detect",
                ):
                    try:
                        with st.spinner("Detecting objects..."):
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

                            result = annotate(image, detections)

                        st.session_state["image_result"] = {
                            "result": result,
                            "detections": detections,
                            "runtime": runtime,
                            "filename": upload.name,
                            "original": image.copy(),
                        }

                    except Exception as exc:
                        st.error(f"Detection failed: {exc}")

    # Persist results across Streamlit reruns.
    image_result = st.session_state.get("image_result")

    if image_result:
        detections = image_result["detections"]
        runtime = image_result["runtime"]

        st.divider()
        section(
            "Results",
            f"Detections for {html.escape(image_result['filename'])}.",
        )

        avg_confidence = (
            np.mean([d["confidence"] for d in detections]) if detections else 0
        )

        stat_strip(
            [
                ("Objects found", len(detections), "Total detections"),
                ("Inference time", f"{runtime:.1f} ms", "ONNX Runtime"),
                (
                    "Average confidence",
                    f"{avg_confidence * 100:.1f}%",
                    "Across all detections",
                ),
                ("Runs on", "CPU", "CPUExecutionProvider"),
            ]
        )

        st.write("")

        original_col, output_col = st.columns([1, 1.15], gap="large")

        with original_col:
            st.markdown(
                '<div class="chart-title">Original</div>',
                unsafe_allow_html=True,
            )
            st.image(
                cv2.cvtColor(
                    image_result["original"],
                    cv2.COLOR_BGR2RGB,
                ),
                width="stretch",
            )

        with output_col:
            st.markdown(
                '<div class="chart-title">Detected objects</div>',
                unsafe_allow_html=True,
            )
            st.image(
                cv2.cvtColor(
                    image_result["result"],
                    cv2.COLOR_BGR2RGB,
                ),
                width="stretch",
            )

        ok, encoded = cv2.imencode(".jpg", image_result["result"])

        if ok:
            st.download_button(
                "Download annotated image",
                encoded.tobytes(),
                "edgeai_annotated.jpg",
                "image/jpeg",
            )

        st.divider()

        render_detection_analytics(detections)

        st.divider()

        section(
            "Detection records",
            "Every detected object with its bounding-box coordinates in pixels.",
        )

        if detections:
            df = detections_dataframe(detections)

            html_table(df)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download CSV",
                csv,
                "edgeai_image_detections.csv",
                "text/csv",
            )

        else:
            st.info(
                "No objects were found at this confidence level. "
                "Lower the confidence threshold in the sidebar and run again."
            )

    else:
        empty_state(
            "No results yet",
            "Your annotated image, charts and detection table will appear here.",
            [
                "Upload an image above.",
                "Adjust the thresholds in the sidebar if needed.",
                "Select Run detection.",
            ],
        )


# ==================================================
# VIDEO ANALYSIS AND TRACKING
# ==================================================

with video_tab:
    st.write("")
    section(
        "Track objects in a video",
        "Every object gets an ID that stays with it from frame to frame. "
        "Supported formats: MP4, MOV, AVI, MKV and WebM, up to 50 MB and 60 seconds.",
    )

    stages = [
        "Detect objects",
        "Filter overlaps",
        "Assign track IDs",
        "Draw boxes",
        "Encode H.264 MP4",
    ]

    pipeline_html = '<span class="pipe-arrow">›</span>'.join(
        f'<span class="pipe-step">{s}</span>' for s in stages
    )

    st.markdown(
        f"""
        <div class="panel" style="margin-bottom:1rem;">
            <div class="panel-title">How each frame is processed</div>
            <div class="pipeline">{pipeline_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    video_upload = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="video_upload",
        label_visibility="collapsed",
    )

    if video_upload:
        if video_upload.size > MAX_VIDEO_SIZE:
            st.error("This video is larger than 50 MB. Choose a smaller file.")

        else:
            file_row(
                video_upload.name,
                f"{video_upload.size / (1024 * 1024):.1f} MB",
            )

            with st.expander("Preview uploaded video", expanded=False):
                st.video(video_upload.getvalue())

            if st.button(
                "Run detection and tracking",
                type="primary",
                width="stretch",
                key="video_detect",
            ):
                progress = st.progress(
                    0.0,
                    text="Starting...",
                )

                try:
                    model_bundle = load_model(str(MODEL))

                    def update_progress(done, total):
                        if total > 0:
                            ratio = min(done / total, 1.0)
                            progress.progress(
                                ratio,
                                text=f"Processing frame {done} of {total}",
                            )
                        else:
                            progress.progress(
                                0.0,
                                text=f"Processed {done} frames",
                            )

                    with st.spinner(
                        "Processing video on CPU. This can take a few minutes."
                    ):
                        video_bytes, metrics = process_video(
                            video_upload,
                            model_bundle,
                            confidence,
                            iou_threshold,
                            update_progress,
                        )

                    progress.progress(1.0, text="Done.")

                    st.session_state["video_result"] = {
                        "video_bytes": video_bytes,
                        "metrics": metrics,
                        "filename": video_upload.name,
                    }

                except Exception as exc:
                    progress.empty()
                    st.error(f"Video processing failed: {exc}")

    # Persist completed video results across Streamlit reruns.
    video_result = st.session_state.get("video_result")

    if video_result:
        metrics = video_result["metrics"]

        st.divider()

        section(
            "Annotated video",
            f"Result for {html.escape(video_result['filename'])}. "
            "Labels show the track ID, object class and confidence.",
        )

        video_col, side_col = st.columns([2, 1], gap="large")

        with video_col:
            st.video(video_result["video_bytes"])

        with side_col:
            st.markdown(
                f"""
                <div class="panel">
                    <div class="panel-title">Source video</div>
                    <div class="kv"><span class="kv-key">Resolution</span>
                        <span class="kv-val">{metrics["width"]} × {metrics["height"]}</span></div>
                    <div class="kv"><span class="kv-key">Duration</span>
                        <span class="kv-val">{metrics["duration"]:.2f} s</span></div>
                    <div class="kv"><span class="kv-key">Frame rate</span>
                        <span class="kv-val">{metrics["fps"]:.2f} fps</span></div>
                    <div class="kv"><span class="kv-key">Frames processed</span>
                        <span class="kv-val">{metrics["processed_frames"]}</span></div>
                    <div class="kv"><span class="kv-key">Runs on</span>
                        <span class="kv-val">CPU</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            st.download_button(
                "Download annotated MP4",
                data=video_result["video_bytes"],
                file_name="edgeai_annotated_video.mp4",
                mime="video/mp4",
                type="primary",
                width="stretch",
            )

        st.divider()

        section(
            "Performance",
            "How quickly the video was processed on this server.",
        )

        stat_strip(
            [
                (
                    "Frames processed",
                    metrics["processed_frames"],
                    f"of {metrics['reported_frames']} reported",
                ),
                (
                    "Inference time",
                    f"{metrics['avg_inference_ms']:.1f} ms",
                    "Average per frame",
                ),
                (
                    "Throughput",
                    f"{metrics['processing_fps']:.2f} fps",
                    "End to end",
                ),
                (
                    "Total time",
                    f"{metrics['processing_seconds']:.2f} s",
                    "Including encoding",
                ),
            ]
        )

        stat_strip(
            [
                ("Raw detections", metrics["total_detections"], "Before tracking"),
                (
                    "Tracked detections",
                    metrics["tracked_detections"],
                    "After ByteTrack",
                ),
                (
                    "Unique tracks",
                    metrics["unique_track_ids"],
                    "Distinct objects followed",
                ),
                ("Source frame rate", f"{metrics['fps']:.2f} fps", "Input video"),
            ]
        )

        st.divider()

        section(
            "Object classes",
            "How often each type of object was detected across all frames.",
        )

        class_counts = metrics["class_counts"]

        if class_counts:
            class_df = pd.DataFrame(
                {
                    "Object class": list(class_counts.keys()),
                    "Detections": list(class_counts.values()),
                }
            ).sort_values("Detections", ascending=False)

            chart_col, summary_col = st.columns([1.5, 1], gap="large")

            with chart_col:
                hbar_chart(
                    "Detections by class",
                    "Across all processed frames.",
                    zip(class_df["Object class"], class_df["Detections"]),
                )

            with summary_col:
                html_table(class_df, max_height=300)

        else:
            st.info("No objects were detected in this video.")

        st.divider()

        section(
            "Tracks",
            "Each row is one object that ByteTrack followed through the video.",
        )

        track_rows = metrics["track_rows"]

        if track_rows:
            track_df = pd.DataFrame(track_rows)

            html_table(track_df)

            track_csv = track_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download tracks CSV",
                track_csv,
                "edgeai_tracking_summary.csv",
                "text/csv",
            )

        else:
            st.info(
                "No objects were tracked. Lower the confidence threshold "
                "in the sidebar and run again."
            )

    else:
        empty_state(
            "No results yet",
            "Your annotated video, performance numbers and track table will appear here.",
            [
                "Upload a video above.",
                "Adjust the thresholds in the sidebar if needed.",
                "Select Run detection and tracking.",
            ],
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <strong>EdgeAI Vision Intelligence</strong><br>
        Built with YOLO11n, ONNX Runtime, ByteTrack, OpenCV and FFmpeg.<br>
        This public demo runs on CPU. Local CUDA inference, optimization
        experiments and benchmarks are in the project repository.
    </div>
    """,
    unsafe_allow_html=True,
)
