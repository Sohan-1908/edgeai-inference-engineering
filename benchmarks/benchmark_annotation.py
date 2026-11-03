"""Compare production annotation with a simple OpenCV baseline.

Inference, detection generation, and video decoding are deliberately outside
the timed region. Run from the project root:

    python benchmarks/benchmark_annotation.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edgeai.detector_types import Detection
from edgeai.postprocess import annotate_image


DEFAULT_VIDEO = ROOT / "outputs" / "cuda_video_smoke.mp4"
DEFAULT_CSV = ROOT / "benchmarks" / "results" / "annotation_benchmark.csv"
DEFAULT_REPORT = ROOT / "benchmarks" / "results" / "annotation_benchmark.md"
DENSITIES = (5, 20, 50)


def baseline_annotate_image(
    image: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    """Draw boxes and labels using a simple, fixed-above-box strategy."""
    annotated = image.copy()
    height, width = annotated.shape[:2]
    for item in detections:
        x1, y1, x2, y2 = item.bbox_xyxy
        left = max(0, min(width - 1, int(round(x1))))
        top = max(0, min(height - 1, int(round(y1))))
        right = max(0, min(width - 1, int(round(x2))))
        bottom = max(0, min(height - 1, int(round(y2))))
        color = (60, 220, 60)
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
        label = f"{item.class_name} {item.confidence:.2f}"
        if item.track_id is not None:
            label += f" ID {item.track_id}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_top = max(0, top - text_height - baseline - 6)
        cv2.rectangle(
            annotated,
            (left, label_top),
            (min(width - 1, left + text_width + 8), top),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (left + 4, max(text_height, top - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    return annotated


RENDERERS: dict[str, Callable[[np.ndarray, list[Detection]], np.ndarray]] = {
    "baseline": baseline_annotate_image,
    "production": annotate_image,
}


def _read_representative_frames(
    video_path: Path,
    frame_samples: int,
) -> tuple[list[tuple[int, np.ndarray]], int, tuple[int, int, float]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open sample video: {video_path}")
    try:
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if frame_count < 1:
            raise RuntimeError(f"Sample video reports no frames: {video_path}")
        frame_indices = sorted(
            {
                round(index * (frame_count - 1) / max(frame_samples - 1, 1))
                for index in range(min(frame_samples, frame_count))
            }
        )
        selected: list[tuple[int, np.ndarray]] = []
        wanted = set(frame_indices)
        frame_index = 0
        while frame_index <= frame_indices[-1]:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                selected.append((frame_index, frame))
            frame_index += 1
        if len(selected) != len(frame_indices):
            raise RuntimeError(
                f"Decoded {len(selected)} of {len(frame_indices)} selected sample frames."
            )
        height, width = selected[0][1].shape[:2]
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        return selected, frame_count, (width, height, fps)
    finally:
        capture.release()


def _make_detections(
    width: int,
    height: int,
    count: int,
) -> list[Detection]:
    """Create reproducible boxes with crowded labels at larger densities."""
    columns = min(count, max(1, round((count * width / max(height, 1)) ** 0.5)))
    rows = (count + columns - 1) // columns
    step_x = width / columns
    step_y = height / rows
    box_width = min(width, max(20.0, step_x * 1.45))
    box_height = min(height, max(16.0, step_y * 1.5))
    names = ("person", "car", "bicycle", "dog", "truck")
    detections = []
    for index in range(count):
        row, column = divmod(index, columns)
        center_x = (column + 0.5) * step_x
        center_y = (row + 0.5) * step_y
        x1 = max(0.0, center_x - box_width / 2)
        y1 = max(0.0, center_y - box_height / 2)
        x2 = min(float(width - 1), center_x + box_width / 2)
        y2 = min(float(height - 1), center_y + box_height / 2)
        detections.append(
            Detection(
                class_id=index % len(names),
                class_name=names[index % len(names)],
                confidence=0.95 - (index % 30) * 0.01,
                bbox_xyxy=(x1, y1, x2, y2),
                track_id=index + 1,
            )
        )
    return detections


def _percentile95(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))


def _summarize(
    samples: list[float],
    *,
    renderer: str,
    detection_count: int,
    repetitions: int,
    frame_count: int,
    warmup_runs: int,
) -> dict[str, object]:
    return {
        "renderer": renderer,
        "detection_count": detection_count,
        "representative_frame_count": frame_count,
        "warmup_runs_per_frame": warmup_runs,
        "measured_repetitions_per_frame": repetitions,
        "measured_call_count": len(samples),
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": _percentile95(samples),
        "total_ms": sum(samples),
    }


def run_benchmark(
    video_path: Path,
    csv_path: Path,
    report_path: Path,
    *,
    frame_samples: int = 6,
    warmup_runs: int = 2,
    repetitions: int = 7,
) -> list[dict[str, object]]:
    if frame_samples < 1 or warmup_runs < 1 or repetitions < 5:
        raise ValueError("Use at least one frame, one warmup, and five repetitions.")
    video_path = video_path.resolve()
    frames, source_frame_count, (width, height, fps) = _read_representative_frames(
        video_path,
        frame_samples,
    )
    detections_by_count = {
        count: _make_detections(width, height, count) for count in DENSITIES
    }

    measurements: dict[tuple[str, int], list[float]] = {
        (renderer, count): []
        for renderer in RENDERERS
        for count in DENSITIES
    }
    for count, detections in detections_by_count.items():
        for renderer, annotate in RENDERERS.items():
            for _, frame in frames:
                for _ in range(warmup_runs):
                    annotate(frame, detections)

        for repetition in range(repetitions):
            for frame_index, frame in frames:
                methods = list(RENDERERS.items())
                if (repetition + frame_index) % 2:
                    methods.reverse()
                for renderer, annotate in methods:
                    started = time.perf_counter_ns()
                    result = annotate(frame, detections)
                    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                    measurements[(renderer, count)].append(elapsed_ms)
                    if result.shape != frame.shape:
                        raise RuntimeError(
                            f"{renderer} renderer changed frame shape {frame.shape} to {result.shape}."
                        )

    summaries = [
        _summarize(
            measurements[(renderer, count)],
            renderer=renderer,
            detection_count=count,
            repetitions=repetitions,
            frame_count=len(frames),
            warmup_runs=warmup_runs,
        )
        for renderer in RENDERERS
        for count in DENSITIES
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    video_hash = hashlib.sha256(video_path.read_bytes()).hexdigest()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _make_report(
            video_path=video_path,
            video_hash=video_hash,
            source_frame_count=source_frame_count,
            width=width,
            height=height,
            fps=fps,
            frame_indices=[index for index, _ in frames],
            summaries=summaries,
            repetitions=repetitions,
            warmup_runs=warmup_runs,
        ),
        encoding="utf-8",
    )
    return summaries


def _make_report(
    *,
    video_path: Path,
    video_hash: str,
    source_frame_count: int,
    width: int,
    height: int,
    fps: float,
    frame_indices: list[int],
    summaries: list[dict[str, object]],
    repetitions: int,
    warmup_runs: int,
) -> str:
    by_key = {
        (str(row["renderer"]), int(row["detection_count"])): row
        for row in summaries
    }
    lines = [
        "# Annotation renderer benchmark",
        "",
        "## Method",
        "",
        f"- Source: `{video_path.relative_to(ROOT) if video_path.is_relative_to(ROOT) else video_path}`",
        f"- SHA-256: `{video_hash}`",
        f"- Video: {source_frame_count} frames, {width}×{height}, {fps:.3f} FPS",
        f"- Representative decoded frame indices (zero-based): `{frame_indices}`",
        f"- Renderer warmups: {warmup_runs} per renderer, density, and frame",
        f"- Measured repetitions: {repetitions} per renderer, density, and frame",
        f"- Detection densities: {', '.join(map(str, DENSITIES))}; deterministic synthetic detections",
        "- Identical in-memory frames and detection records are passed to both renderers.",
        "- Frame decoding and detection generation happen outside timed calls; no model is loaded and no inference is run.",
        "- The baseline is a benchmark-only simple OpenCV renderer: green rectangles and labels above boxes.",
        "- The production renderer is `edgeai.postprocess.annotate_image`.",
        f"- Environment: Python {platform.python_version()}, OpenCV {cv2.__version__}, {platform.platform()}",
        "",
        "## Results",
        "",
        "| Detections | Renderer | Calls | Mean (ms) | Median (ms) | P95 (ms) | Total (ms) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for count in DENSITIES:
        for renderer in RENDERERS:
            row = by_key[(renderer, count)]
            lines.append(
                f"| {count} | {renderer} | {row['measured_call_count']} "
                f"| {row['mean_ms']:.3f} | {row['median_ms']:.3f} "
                f"| {row['p95_ms']:.3f} | {row['total_ms']:.3f} |"
            )
    lines.extend(["", "## Production overhead vs baseline", ""])
    for count in DENSITIES:
        baseline = by_key[("baseline", count)]
        production = by_key[("production", count)]
        mean_delta = float(production["mean_ms"]) - float(baseline["mean_ms"])
        percent_delta = (
            100 * mean_delta / float(baseline["mean_ms"])
            if float(baseline["mean_ms"]) > 0
            else float("nan")
        )
        lines.append(
            f"- **{count} detections:** {mean_delta:+.3f} ms/call "
            f"({percent_delta:+.1f}% mean latency)."
        )
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "This isolates renderer cost for the measured frames and controlled detection counts; it does not explain variance between full video inference runs. Detection locations and identities are synthetic and deterministic, while the pixels come from the real sample video. Baseline styling and label content differ from production styling, so the comparison measures complete renderer cost, not only collision-search overhead. Results are hardware/load dependent and are not a substitute for end-to-end video timing.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    summaries = run_benchmark(
        args.video,
        args.csv,
        args.report,
        frame_samples=args.frames,
        warmup_runs=args.warmups,
        repetitions=args.repetitions,
    )
    for row in summaries:
        print(
            f"{row['renderer']:>10} detections={row['detection_count']:>2} "
            f"mean={row['mean_ms']:.3f}ms median={row['median_ms']:.3f}ms "
            f"p95={row['p95_ms']:.3f}ms total={row['total_ms']:.3f}ms"
        )
    print(f"CSV: {args.csv}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
