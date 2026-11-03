"""Compare PyTorch CUDA and ONNX CUDA through the existing video API.

Run this script once for each runtime, using the same --comparison-id and
video. The script never changes or recreates Docker services.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "runtime_benchmark.csv"
DEFAULT_REPORT = ROOT / "runtime_benchmark.md"
VIDEO_METADATA_HEADER = "X-EdgeAI-Metadata"
VIDEO_METADATA_NUMERIC_FIELDS = (
    "output_fps",
    "mean_inference_latency_ms",
    "median_inference_latency_ms",
    "p95_inference_latency_ms",
    "end_to_end_processing_fps",
    "total_processing_time_ms",
)
CSV_FIELDS = (
    "comparison_id",
    "timestamp_utc",
    "video_path",
    "video_sha256",
    "runtime",
    "record_type",
    "run_index",
    "status",
    "error",
    "device",
    "execution_provider",
    "input_resolution",
    "confidence_threshold",
    "nms_iou_threshold",
    "tracking_enabled",
    "request_latency_ms",
    "api_processing_time_ms",
    "api_mean_inference_latency_ms",
    "api_median_inference_latency_ms",
    "api_p95_inference_latency_ms",
    "api_end_to_end_fps",
    "client_end_to_end_fps",
    "source_fps",
    "output_fps",
    "source_frame_count",
    "processed_frame_count",
    "video_width",
    "video_height",
    "total_detections",
    "output_bytes",
    "gpu_memory_before_mb",
    "gpu_memory_after_mb",
    "gpu_utilization_before_percent",
    "gpu_utilization_after_percent",
    "gpu_probe_status",
    "successful_runs",
    "failed_runs",
    "mean_request_latency_ms",
    "median_request_latency_ms",
    "p95_request_latency_ms",
    "mean_run_api_inference_ms",
    "median_run_api_inference_ms",
    "p95_run_api_inference_ms",
    "mean_api_end_to_end_fps",
    "mean_client_end_to_end_fps",
    "total_processed_frames",
    "mean_total_detections",
    "detection_counts",
    "mean_gpu_memory_observed_mb",
    "mean_per_run_api_p95_inference_ms",
    "median_per_run_api_p95_inference_ms",
    "p95_of_per_run_api_p95_inference_ms",
    "total_client_elapsed_ms",
    "total_api_processing_time_ms",
    "pooled_client_fps",
    "whole_video_api_fps",
)
LEGACY_CSV_FIELDS = CSV_FIELDS[:-7]
REQUIRED_METADATA_FIELDS = (
    "runtime",
    "device",
    "model_input_resolution",
    "source_frame_count",
    "processed_frame_count",
    "mean_inference_latency_ms",
    "median_inference_latency_ms",
    "p95_inference_latency_ms",
    "end_to_end_processing_fps",
    "total_processing_time_ms",
    "total_detections",
    "tracking_enabled",
)


class BenchmarkError(RuntimeError):
    """A benchmark request or response could not be measured reliably."""


def parse_video_metadata(raw_header: str | None) -> dict[str, Any]:
    if not raw_header:
        raise BenchmarkError(f"Response is missing {VIDEO_METADATA_HEADER}.")
    try:
        metadata = json.loads(raw_header)
    except (TypeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"{VIDEO_METADATA_HEADER} is not valid JSON.") from exc
    if not isinstance(metadata, dict):
        raise BenchmarkError(f"{VIDEO_METADATA_HEADER} must contain a JSON object.")

    missing = [key for key in REQUIRED_METADATA_FIELDS if key not in metadata]
    if missing:
        raise BenchmarkError(f"Video metadata is missing required fields: {', '.join(missing)}.")

    for key in VIDEO_METADATA_NUMERIC_FIELDS:
        _finite_number(metadata, key, minimum=0.0, allow_zero=key not in {"output_fps", "end_to_end_processing_fps"})
    for key in ("source_frame_count", "processed_frame_count"):
        _integer(metadata, key, minimum=1)
    _integer(metadata, "total_detections", minimum=0)
    _integer(metadata, "model_input_resolution", minimum=1)
    if metadata["runtime"] not in {"pytorch", "onnx"}:
        raise BenchmarkError(f"Unsupported runtime in video metadata: {metadata['runtime']!r}.")
    if not isinstance(metadata["device"], str) or not metadata["device"]:
        raise BenchmarkError("Video metadata device must be a non-empty string.")
    if metadata["tracking_enabled"] is not False:
        raise BenchmarkError("Video metadata indicates tracking was enabled; benchmark requires track=false.")
    if metadata["source_frame_count"] != metadata["processed_frame_count"]:
        raise BenchmarkError("Video metadata reports an incomplete frame count.")
    if metadata.get("source_fps") is not None:
        _finite_number(metadata, "source_fps", minimum=0, allow_zero=False)
    return metadata


def _finite_number(
    payload: dict[str, Any],
    key: str,
    *,
    minimum: float,
    allow_zero: bool,
) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BenchmarkError(f"Video metadata field {key!r} must be a finite number.")
    if value < minimum or (not allow_zero and value == minimum):
        comparison = "non-negative" if allow_zero else "positive"
        raise BenchmarkError(f"Video metadata field {key!r} must be {comparison}.")
    return float(value)


def _integer(payload: dict[str, Any], key: str, *, minimum: int) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchmarkError(f"Video metadata field {key!r} must be an integer >= {minimum}.")
    return value


def percentile(values: list[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile (same convention as NumPy)."""
    if not values:
        raise ValueError("At least one value is required.")
    if not 0 <= percentile_value <= 100:
        raise ValueError("Percentile must be between 0 and 100.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        row for row in rows
        if row.get("record_type") == "measured" and row.get("status") == "ok"
    ]
    failed = [
        row for row in rows
        if row.get("record_type") == "measured" and row.get("status") != "ok"
    ]
    result: dict[str, Any] = {
        "successful_runs": len(successful),
        "failed_runs": len(failed),
        "mean_request_latency_ms": "",
        "median_request_latency_ms": "",
        "p95_request_latency_ms": "",
        "mean_run_api_inference_ms": "",
        "median_run_api_inference_ms": "",
        "p95_run_api_inference_ms": "",
        "mean_api_end_to_end_fps": "",
        "mean_client_end_to_end_fps": "",
        "total_processed_frames": "",
        "mean_total_detections": "",
        "detection_counts": "",
        "mean_gpu_memory_observed_mb": "",
        "mean_per_run_api_p95_inference_ms": "",
        "median_per_run_api_p95_inference_ms": "",
        "p95_of_per_run_api_p95_inference_ms": "",
        "total_client_elapsed_ms": "",
        "total_api_processing_time_ms": "",
        "pooled_client_fps": "",
        "whole_video_api_fps": "",
    }
    if not successful:
        return result

    latencies = [float(row["request_latency_ms"]) for row in successful]
    api_inference_means = [float(row["api_mean_inference_latency_ms"]) for row in successful]
    api_run_p95s = [float(row["api_p95_inference_latency_ms"]) for row in successful]
    api_fps = [float(row["api_end_to_end_fps"]) for row in successful]
    client_fps = [
        int(row["processed_frame_count"]) * 1000 / float(row["request_latency_ms"])
        for row in successful
    ]
    total_frames = sum(int(row["processed_frame_count"]) for row in successful)
    total_client_ms = sum(float(row["request_latency_ms"]) for row in successful)
    total_api_ms = sum(float(row["api_processing_time_ms"]) for row in successful)
    counts = [int(row["total_detections"]) for row in successful]
    gpu_values = [
        float(row[key])
        for row in successful
        for key in ("gpu_memory_before_mb", "gpu_memory_after_mb")
        if row.get(key) not in ("", None)
    ]
    result.update(
        mean_request_latency_ms=statistics.mean(latencies),
        median_request_latency_ms=statistics.median(latencies),
        p95_request_latency_ms=percentile(latencies, 95),
        mean_run_api_inference_ms=statistics.mean(api_inference_means),
        median_run_api_inference_ms=statistics.median(api_inference_means),
        p95_run_api_inference_ms=percentile(api_inference_means, 95),
        mean_api_end_to_end_fps=statistics.mean(api_fps),
        mean_client_end_to_end_fps=statistics.mean(client_fps),
        total_processed_frames=total_frames,
        mean_total_detections=statistics.mean(counts),
        detection_counts=",".join(str(count) for count in counts),
        mean_gpu_memory_observed_mb=statistics.mean(gpu_values) if gpu_values else "",
        mean_per_run_api_p95_inference_ms=statistics.mean(api_run_p95s),
        median_per_run_api_p95_inference_ms=statistics.median(api_run_p95s),
        p95_of_per_run_api_p95_inference_ms=percentile(api_run_p95s, 95),
        total_client_elapsed_ms=total_client_ms,
        total_api_processing_time_ms=total_api_ms,
        pooled_client_fps=total_frames * 1000 / total_client_ms,
        whole_video_api_fps=total_frames * 1000 / total_api_ms,
    )
    return result


def gpu_snapshot() -> tuple[dict[str, float] | None, str]:
    """Sample host-visible NVIDIA GPU memory/utilization when nvidia-smi exists."""
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=5
        )
        first_line = completed.stdout.strip().splitlines()[0]
        memory_text, utilization_text = (part.strip() for part in first_line.split(",", 1))
        memory = float(memory_text)
        utilization = float(utilization_text)
        if not math.isfinite(memory) or not math.isfinite(utilization):
            raise ValueError("nvidia-smi returned non-finite values.")
        return {"memory_mb": memory, "utilization_percent": utilization}, "ok"
    except (OSError, subprocess.SubprocessError, IndexError, ValueError) as exc:
        return None, f"unavailable: {type(exc).__name__}"


class RuntimeBenchmarkClient:
    def __init__(
        self,
        base_url: str,
        *,
        session: Any | None = None,
        connect_timeout: float = 5,
        read_timeout: float = 1800,
        gpu_probe: Callable[[], tuple[dict[str, float] | None, str]] = gpu_snapshot,
        clock: Callable[[], int] = time.perf_counter_ns,
    ):
        self.base_url = base_url.rstrip("/")
        if not self.base_url:
            raise ValueError("API base URL cannot be empty.")
        self.session = session or requests.Session()
        self.timeout = (connect_timeout, read_timeout)
        self.gpu_probe = gpu_probe
        self.clock = clock

    def get_api_settings(self) -> dict[str, Any]:
        health = self._get_json("/health")
        if health.get("status") != "ready" or health.get("detector_initialized") is not True:
            raise BenchmarkError("API health check did not report an initialized detector.")
        config = self._get_json("/api/v1/config")
        if config.get("device") != "cuda":
            raise BenchmarkError(f"API is not configured for CUDA: {config.get('device')!r}.")
        if config.get("runtime") not in {"pytorch", "onnx"}:
            raise BenchmarkError(f"API reported an unsupported runtime: {config.get('runtime')!r}.")
        if health.get("active_runtime") != config.get("runtime"):
            raise BenchmarkError("API health and configuration report different runtimes.")
        if health.get("active_device") != "cuda:0":
            raise BenchmarkError(f"API detector is not on CUDA device 0: {health.get('active_device')!r}.")
        if (
            config["runtime"] == "onnx"
            and config.get("execution_provider") != "CUDAExecutionProvider"
        ):
            raise BenchmarkError("ONNX config does not report CUDAExecutionProvider.")
        for key in ("input_resolution", "confidence_threshold", "nms_iou_threshold"):
            if key not in config:
                raise BenchmarkError(f"API config is missing benchmark setting {key!r}.")
        _integer(config, "input_resolution", minimum=1)
        confidence = _finite_number(config, "confidence_threshold", minimum=0, allow_zero=True)
        iou = _finite_number(config, "nms_iou_threshold", minimum=0, allow_zero=False)
        if confidence > 1 or iou > 1:
            raise BenchmarkError("API confidence and NMS IoU settings must not exceed 1.")
        return config

    def _get_json(self, path: str) -> dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.base_url}{path}", timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise BenchmarkError(f"API request {path} failed: {type(exc).__name__}.") from exc
        try:
            if not 200 <= response.status_code < 300:
                raise BenchmarkError(f"API request {path} returned HTTP {response.status_code}.")
            value = response.json()
            if not isinstance(value, dict):
                raise BenchmarkError(f"API response {path} must be a JSON object.")
            return value
        except (ValueError, requests.RequestException) as exc:
            raise BenchmarkError(f"API response {path} was not valid JSON.") from exc
        finally:
            response.close()

    def run_video(
        self,
        *,
        video_path: Path,
        runtime: str,
        config: dict[str, Any],
        comparison_id: str,
        video_sha256: str,
        run_type: str,
        run_index: int,
        timestamp_utc: str,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "comparison_id": comparison_id,
            "timestamp_utc": timestamp_utc,
            "video_path": str(video_path),
            "video_sha256": video_sha256,
            "runtime": runtime,
            "record_type": run_type,
            "run_index": run_index,
            "status": "failed",
            "error": "",
            "device": config.get("device", ""),
            "execution_provider": config.get("execution_provider", ""),
            "input_resolution": config.get("input_resolution", ""),
            "confidence_threshold": config.get("confidence_threshold", ""),
            "nms_iou_threshold": config.get("nms_iou_threshold", ""),
            "tracking_enabled": False,
            "request_latency_ms": "",
            "api_processing_time_ms": "",
            "api_mean_inference_latency_ms": "",
            "api_median_inference_latency_ms": "",
            "api_p95_inference_latency_ms": "",
            "api_end_to_end_fps": "",
            "client_end_to_end_fps": "",
            "source_frame_count": "",
            "processed_frame_count": "",
            "total_detections": "",
            "output_bytes": 0,
            "gpu_memory_before_mb": "",
            "gpu_memory_after_mb": "",
            "gpu_utilization_before_percent": "",
            "gpu_utilization_after_percent": "",
            "gpu_probe_status": "",
        }
        before, before_status = self.gpu_probe()
        if before is not None:
            row["gpu_memory_before_mb"] = before["memory_mb"]
            row["gpu_utilization_before_percent"] = before["utilization_percent"]
        started = self.clock()
        response = None
        try:
            mime_type = "video/mp4" if video_path.suffix.lower() == ".mp4" else "video/x-msvideo"
            with video_path.open("rb") as source:
                response = self.session.post(
                    f"{self.base_url}/api/v1/detect/video",
                    files={"file": (video_path.name, source, mime_type)},
                    data={"track": "false"},
                    timeout=self.timeout,
                    stream=True,
                )
                if not 200 <= response.status_code < 300:
                    detail = _response_error(response)
                    raise BenchmarkError(
                        f"Video endpoint returned HTTP {response.status_code}: {detail}"
                    )

                content_type = response.headers.get("Content-Type", "")
                if not content_type.lower().startswith("video/"):
                    raise BenchmarkError("Video endpoint did not return an annotated video content type.")
                metadata = parse_video_metadata(response.headers.get(VIDEO_METADATA_HEADER))
                if metadata["runtime"] != runtime:
                    raise BenchmarkError(
                        f"Response runtime {metadata['runtime']!r} did not match requested runtime {runtime!r}."
                    )
                if metadata["model_input_resolution"] != config["input_resolution"]:
                    raise BenchmarkError("Response input resolution changed during the benchmark.")
                if metadata["device"] != "cuda:0":
                    raise BenchmarkError(f"Video inference was not on CUDA device 0: {metadata['device']!r}.")

                output_bytes = sum(len(chunk) for chunk in response.iter_content(chunk_size=1024 * 1024) if chunk)
                if output_bytes == 0:
                    raise BenchmarkError("Video endpoint returned an empty annotated video.")
                row["output_bytes"] = output_bytes

            elapsed_ms = (self.clock() - started) / 1_000_000
            if elapsed_ms <= 0:
                raise BenchmarkError("Measured request latency was not positive.")
            row.update(
                status="ok",
                request_latency_ms=elapsed_ms,
                api_processing_time_ms=metadata["total_processing_time_ms"],
                api_mean_inference_latency_ms=metadata["mean_inference_latency_ms"],
                api_median_inference_latency_ms=metadata["median_inference_latency_ms"],
                api_p95_inference_latency_ms=metadata["p95_inference_latency_ms"],
                api_end_to_end_fps=metadata["end_to_end_processing_fps"],
                client_end_to_end_fps=metadata["processed_frame_count"] / (elapsed_ms / 1000),
                source_fps=metadata.get("source_fps", ""),
                output_fps=metadata["output_fps"],
                source_frame_count=metadata["source_frame_count"],
                processed_frame_count=metadata["processed_frame_count"],
                video_width=metadata.get("video_width", ""),
                video_height=metadata.get("video_height", ""),
                total_detections=metadata["total_detections"],
            )
        except (BenchmarkError, OSError, requests.RequestException) as exc:
            row["error"] = str(exc)
            row["request_latency_ms"] = (self.clock() - started) / 1_000_000
        finally:
            if response is not None:
                response.close()
            after, after_status = self.gpu_probe()
            if after is not None:
                row["gpu_memory_after_mb"] = after["memory_mb"]
                row["gpu_utilization_after_percent"] = after["utilization_percent"]
            statuses = sorted({before_status, after_status})
            row["gpu_probe_status"] = "; ".join(statuses)
        return row


def _response_error(response: Any) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict) and payload.get("detail"):
            return str(payload["detail"])
    except (ValueError, requests.RequestException):
        pass
    text = getattr(response, "text", "")
    return str(text).strip()[:500] or "No error details were returned."


def _comparison_groups(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("record_type") not in {"measured", "warmup"}:
            continue
        key = (
            str(row.get("comparison_id", "")),
            str(row.get("video_sha256", "")),
            str(row.get("input_resolution", "")),
            str(row.get("confidence_threshold", "")),
            str(row.get("nms_iou_threshold", "")),
        )
        groups[key].append(row)
    return groups


def _successful_detection_counts(rows: Any) -> list[int]:
    successful = [
        row for row in rows
        if row.get("record_type") == "measured" and row.get("status") == "ok"
    ]
    successful.sort(key=lambda row: int(row.get("run_index") or 0))
    return [int(row["total_detections"]) for row in successful]


def make_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# PyTorch CUDA vs ONNX CUDA runtime benchmark",
        "",
        "Request latency is measured by this client from upload start through complete annotated-video download. API timing/FPS values are separately reported from `X-EdgeAI-Metadata`.",
        "",
        "## Results",
        "",
    ]
    groups = _comparison_groups(rows)
    if not groups:
        lines.extend(["No benchmark runs have been recorded.", ""])
        return "\n".join(lines)

    for key, group in sorted(groups.items()):
        comparison_id, video_hash, resolution, confidence, iou = key
        lines.extend(
            [
                f"### Comparison `{comparison_id}`",
                "",
                f"- Video SHA-256: `{video_hash}`",
                f"- API inference settings: {resolution}px, confidence `{confidence}`, NMS IoU `{iou}`",
                "- Tracking is disabled for all benchmark requests.",
                "- Each runtime invocation includes one unmeasured video request. Every request also uses the server's existing up-to-three-frame warm-up: those frames are excluded from inference-latency samples but remain part of end-to-end processing/request latency.",
                "- Mean/median API inference latency are the corresponding statistics of per-run means; P95 of per-run means is not a frame-level P95. The API provides per-run inference P95 summaries, so the report separately gives their mean/median/P95. A pooled P95 across individual frame-latency samples cannot be reconstructed because those samples are not returned.",
                "- Mean API/client FPS are arithmetic means of per-run FPS. Pooled client FPS is total processed frames divided by total client request time. Whole-video API FPS is total processed frames divided by summed API total-processing time; that API total includes warm-up and is distinct from API end-to-end FPS, which excludes warm-up frames/time.",
                "",
                "| Runtime | Runs: successful / measured | Failed | Client request latency: mean / median / P95 (ms) | API per-run mean inference: mean / median / P95-of-run-means (ms) | Per-run API P95: mean / median / P95 (ms) | API FPS: mean per-run | Client FPS: mean per-run | Client FPS: pooled | API FPS: whole-video pooled | Frames total | Detection counts by run | GPU memory: mean host snapshots (MiB) |",
                "|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
            ]
        )
        runtime_summaries: dict[str, dict[str, Any]] = {}
        for runtime in ("pytorch", "onnx"):
            runtime_rows = [item for item in group if item.get("runtime") == runtime]
            summary = aggregate_runs(runtime_rows)
            runtime_summaries[runtime] = summary
            def display(value: Any, precision: int = 2) -> str:
                return "n/a" if value in ("", None) else f"{float(value):.{precision}f}"

            lines.append(
                f"| {runtime} | {summary['successful_runs']} / "
                f"{summary['successful_runs'] + summary['failed_runs']} | {summary['failed_runs']} | "
                f"{display(summary['mean_request_latency_ms'])} / "
                f"{display(summary['median_request_latency_ms'])} / "
                f"{display(summary['p95_request_latency_ms'])} | "
                f"{display(summary['mean_run_api_inference_ms'])} / "
                f"{display(summary['median_run_api_inference_ms'])} / "
                f"{display(summary['p95_run_api_inference_ms'])} | "
                f"{display(summary['mean_per_run_api_p95_inference_ms'])} / "
                f"{display(summary['median_per_run_api_p95_inference_ms'])} / "
                f"{display(summary['p95_of_per_run_api_p95_inference_ms'])} | "
                f"{display(summary['mean_api_end_to_end_fps'])} | "
                f"{display(summary['mean_client_end_to_end_fps'])} | "
                f"{display(summary['pooled_client_fps'])} | "
                f"{display(summary['whole_video_api_fps'])} | "
                f"{summary['total_processed_frames'] or 'n/a'} | "
                f"`{summary['detection_counts'] or 'n/a'}` | "
                f"{display(summary['mean_gpu_memory_observed_mb'])} |"
            )

        pytorch = runtime_summaries["pytorch"]
        onnx = runtime_summaries["onnx"]
        pytorch_mean = pytorch["mean_request_latency_ms"]
        onnx_mean = onnx["mean_request_latency_ms"]
        lines.extend(["", "#### Runtime comparison", ""])
        if pytorch_mean not in ("", None) and onnx_mean not in ("", None):
            percent = (float(pytorch_mean) - float(onnx_mean)) / float(pytorch_mean) * 100
            wording = "improvement" if percent >= 0 else "regression"
            lines.append(
                f"- ONNX mean client request latency: **{abs(percent):.2f}% {wording}** "
                f"relative to PyTorch; the percentage is computed from unrounded means."
            )
        else:
            lines.append("- Latency comparison is unavailable until both runtimes have successful measured runs.")

        pytorch_pooled_fps = pytorch["pooled_client_fps"]
        onnx_pooled_fps = onnx["pooled_client_fps"]
        if pytorch_pooled_fps not in ("", None) and onnx_pooled_fps not in ("", None):
            fps_percent = (
                float(onnx_pooled_fps) - float(pytorch_pooled_fps)
            ) / float(pytorch_pooled_fps) * 100
            wording = "higher" if fps_percent >= 0 else "lower"
            lines.append(
                f"- ONNX pooled client throughput: **{abs(fps_percent):.2f}% {wording}** "
                f"than PyTorch; computed as total frames / total client time."
            )

        pytorch_counts = _successful_detection_counts(
            item for item in group if item.get("runtime") == "pytorch"
        )
        onnx_counts = _successful_detection_counts(
            item for item in group if item.get("runtime") == "onnx"
        )
        if pytorch_counts and onnx_counts:
            equal = pytorch_counts == onnx_counts
            lines.append(
                f"- Detection-count consistency by measured run index: **{'consistent' if equal else 'not consistent'}** "
                f"(PyTorch `{','.join(map(str, pytorch_counts))}`; ONNX `{','.join(map(str, onnx_counts))}`). "
                "Matching counts do not establish matching classes, boxes, confidence scores, or detection accuracy."
            )
        else:
            lines.append("- Detection-count comparison is unavailable until both runtimes have successful measured runs.")

        failures = [
            item for item in group
            if item.get("record_type") == "measured" and item.get("status") != "ok"
        ]
        if failures:
            lines.extend(["", "#### Failed measured runs", ""])
            for item in failures:
                lines.append(
                    f"- {item.get('runtime')} run {item.get('run_index')}: "
                    f"{item.get('error') or 'Unknown benchmark failure'}"
                )

        lines.extend(
            [
                "",
                "#### GPU memory and limitations",
                "",
                "GPU memory/utilization values are host-wide `nvidia-smi` snapshots immediately before/after each request, not process-specific usage or peak VRAM. Client request latency includes upload, processing, and downloading the annotated video. Runs use the same video hash and API-reported resolution/confidence/IoU settings, but runtime changes require manual API container recreation and each runtime is measured in a separate invocation.",
                "",
            ]
        )
    lines.append("## Per-run records")
    lines.append("")
    lines.append("| Comparison | Runtime | Type | Run | Status | Client latency (ms) | API inference mean / median / run P95 (ms) | API end-to-end FPS (measured phase) | Processed frames | Detections | GPU memory before / after (MiB) | Error |")
    lines.append("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        if row.get("record_type") not in {"warmup", "measured"}:
            continue
        lines.append(
            f"| {row.get('comparison_id', '')} | {row.get('runtime', '')} | {row.get('record_type', '')} | "
            f"{row.get('run_index', '')} | {row.get('status', '')} | "
            f"{_format_report_value(row.get('request_latency_ms'))} | "
            f"{_format_report_value(row.get('api_mean_inference_latency_ms'))} / "
            f"{_format_report_value(row.get('api_median_inference_latency_ms'))} / "
            f"{_format_report_value(row.get('api_p95_inference_latency_ms'))} | "
            f"{_format_report_value(row.get('api_end_to_end_fps'))} | "
            f"{_display_cell(row.get('processed_frame_count'))} | {_display_cell(row.get('total_detections'))} | "
            f"{_display_cell(row.get('gpu_memory_before_mb'))} / {_display_cell(row.get('gpu_memory_after_mb'))} | "
            f"{row.get('error', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_report_value(value: Any) -> str:
    if value in ("", None):
        return "n/a"
    return f"{float(value):.2f}"


def _display_cell(value: Any) -> str:
    return "n/a" if value in ("", None) else str(value)


def write_results(csv_path: Path, report_path: Path, new_rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    legacy_schema = False
    if exists:
        with csv_path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames not in (list(CSV_FIELDS), list(LEGACY_CSV_FIELDS)):
                raise BenchmarkError(f"Existing CSV schema does not match {csv_path}; choose another --csv path.")
            legacy_schema = reader.fieldnames == list(LEGACY_CSV_FIELDS)
            rows = list(reader)
    else:
        rows = []
    if legacy_schema:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    with csv_path.open("a" if exists else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(new_rows)
    rows.extend(new_rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(make_report(rows), encoding="utf-8")


def refresh_csv_aggregates(csv_path: Path) -> None:
    if not csv_path.is_file():
        raise BenchmarkError(f"Benchmark CSV does not exist: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames not in (list(CSV_FIELDS), list(LEGACY_CSV_FIELDS)):
            raise BenchmarkError(f"Benchmark CSV schema is not recognized: {csv_path}")
        legacy_schema = reader.fieldnames == list(LEGACY_CSV_FIELDS)
        rows = list(reader)

    measured_rows = [row for row in rows if row.get("record_type") == "measured"]
    aggregate_rows = [row for row in rows if row.get("record_type") == "aggregate"]
    aggregate_fields = (
        "mean_per_run_api_p95_inference_ms",
        "median_per_run_api_p95_inference_ms",
        "p95_of_per_run_api_p95_inference_ms",
        "total_client_elapsed_ms",
        "total_api_processing_time_ms",
        "pooled_client_fps",
        "whole_video_api_fps",
    )
    for aggregate in aggregate_rows:
        matching = [
            row for row in measured_rows
            if all(row.get(key) == aggregate.get(key) for key in (
                "comparison_id",
                "video_sha256",
                "runtime",
                "input_resolution",
                "confidence_threshold",
                "nms_iou_threshold",
            ))
        ]
        summary = aggregate_runs(matching)
        aggregate.update({key: summary[key] for key in aggregate_fields})

    if legacy_schema:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    elif aggregate_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def regenerate_report_from_csv(csv_path: Path, report_path: Path) -> None:
    refresh_csv_aggregates(csv_path)
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(make_report(rows), encoding="utf-8")


def _base_record(
    *,
    comparison_id: str,
    timestamp_utc: str,
    video_path: Path,
    video_sha256: str,
    runtime: str,
    record_type: str,
    run_index: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(
        comparison_id=comparison_id,
        timestamp_utc=timestamp_utc,
        video_path=str(video_path),
        video_sha256=video_sha256,
        runtime=runtime,
        record_type=record_type,
        run_index=run_index,
        status="not_run",
        error="",
        device=config.get("device", ""),
        execution_provider=config.get("execution_provider", ""),
        input_resolution=config.get("input_resolution", ""),
        confidence_threshold=config.get("confidence_threshold", ""),
        nms_iou_threshold=config.get("nms_iou_threshold", ""),
        tracking_enabled=False,
    )
    return row


def run_benchmark(
    *,
    video_path: Path,
    runtime: str,
    runs: int = 5,
    comparison_id: str,
    api_url: str = "http://127.0.0.1:8000",
    csv_path: Path = DEFAULT_CSV,
    report_path: Path = DEFAULT_REPORT,
    client: RuntimeBenchmarkClient | None = None,
) -> list[dict[str, Any]]:
    video_path = video_path.resolve()
    if runtime not in {"pytorch", "onnx"}:
        raise ValueError("runtime must be 'pytorch' or 'onnx'.")
    if runs < 1:
        raise ValueError("runs must be at least one.")
    if not comparison_id.strip():
        raise ValueError("comparison_id cannot be empty.")
    if video_path.suffix.lower() not in {".mp4", ".avi"} or not video_path.is_file():
        raise ValueError(f"Video must be an existing MP4 or AVI file: {video_path}")

    video_sha256 = sha256_file(video_path)
    benchmark_client = client or RuntimeBenchmarkClient(api_url)
    config = benchmark_client.get_api_settings()
    if config["runtime"] != runtime:
        raise BenchmarkError(
            f"API runtime is {config['runtime']!r}, but this run requested {runtime!r}. "
            "Switch runtime explicitly, recreate the API service, and wait for health before benchmarking."
        )
    timestamp = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []

    warmup = _base_record(
        comparison_id=comparison_id,
        timestamp_utc=timestamp,
        video_path=video_path,
        video_sha256=video_sha256,
        runtime=runtime,
        record_type="warmup",
        run_index=0,
        config=config,
    )
    warmup.update(
        benchmark_client.run_video(
            video_path=video_path,
            runtime=runtime,
            config=config,
            comparison_id=comparison_id,
            video_sha256=video_sha256,
            run_type="warmup",
            run_index=0,
            timestamp_utc=timestamp,
        )
    )
    rows.append(warmup)

    for run_index in range(1, runs + 1):
        row = _base_record(
            comparison_id=comparison_id,
            timestamp_utc=timestamp,
            video_path=video_path,
            video_sha256=video_sha256,
            runtime=runtime,
            record_type="measured",
            run_index=run_index,
            config=config,
        )
        row.update(
            benchmark_client.run_video(
                video_path=video_path,
                runtime=runtime,
                config=config,
                comparison_id=comparison_id,
                video_sha256=video_sha256,
                run_type="measured",
                run_index=run_index,
                timestamp_utc=timestamp,
            )
        )
        rows.append(row)

    aggregate = _base_record(
        comparison_id=comparison_id,
        timestamp_utc=timestamp,
        video_path=video_path,
        video_sha256=video_sha256,
        runtime=runtime,
        record_type="aggregate",
        run_index="",
        config=config,
    )
    aggregate.update(aggregate_runs(rows))
    aggregate["status"] = "ok" if aggregate["failed_runs"] == 0 else "partial"
    rows.append(aggregate)
    write_results(csv_path, report_path, rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="MP4 or AVI input video.")
    parser.add_argument("--runtime", choices=("pytorch", "onnx"))
    parser.add_argument("--runs", type=int, default=5, help="Measured requests per runtime (default: 5).")
    parser.add_argument(
        "--comparison-id",
        help="Use the same ID in both runtime invocations to pair their results.",
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--regenerate-report",
        action="store_true",
        help="Refresh aggregate-only CSV fields and Markdown without API requests or changing raw run records.",
    )
    args = parser.parse_args(argv)
    try:
        if args.regenerate_report:
            regenerate_report_from_csv(args.csv, args.report)
            print(f"Regenerated report: {args.report.resolve()}")
            return 0
        if args.video is None or args.runtime is None or not args.comparison_id:
            parser.error("--video, --runtime, and --comparison-id are required unless --regenerate-report is used.")
        rows = run_benchmark(
            video_path=args.video,
            runtime=args.runtime,
            runs=args.runs,
            comparison_id=args.comparison_id,
            api_url=args.api_url,
            csv_path=args.csv,
            report_path=args.report,
        )
    except (BenchmarkError, OSError, ValueError) as exc:
        print(f"Benchmark setup failed: {exc}", file=sys.stderr)
        return 2
    measured = [row for row in rows if row["record_type"] == "measured"]
    failures = sum(row["status"] != "ok" for row in measured)
    warmup_failures = sum(
        row["record_type"] == "warmup" and row["status"] != "ok" for row in rows
    )
    print(
        f"Recorded {len(measured)} measured {args.runtime} runs; "
        f"measured failures: {failures}; warm-up failures: {warmup_failures}."
    )
    print(f"CSV: {args.csv.resolve()}")
    print(f"Report: {args.report.resolve()}")
    return 1 if failures or warmup_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
