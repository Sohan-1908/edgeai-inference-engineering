"""Profile CUDA inference resources for PyTorch and ONNX Runtime in isolated workers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import struct
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import psutil
import torch

import benchmark_gpu_baseline as baseline

DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.5
_GPU_METRICS = (
    "gpu_utilization_percent",
    "gpu_memory_utilization_percent",
    "gpu_memory_used_mib",
    "gpu_memory_total_mib",
    "gpu_temperature_c",
    "gpu_power_w",
    "gpu_graphics_clock_mhz",
    "gpu_memory_clock_mhz",
    "gpu_performance_state",
)
_PROCESS_METRICS = (
    "worker_rss_mib",
    "worker_cpu_percent",
    "pytorch_allocator_allocated_mib",
    "pytorch_allocator_reserved_mib",
)
_NUMERIC_METRICS = (
    *(metric for metric in _GPU_METRICS if metric != "gpu_performance_state"),
    *_PROCESS_METRICS,
    "process_gpu_memory_mib",
    "sampling_overhead_ms",
    "worker_cpu_interval_ms",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _timestamped_outputs(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        json_path = output_dir / f"phase16c_gpu_profile_{stamp}.json"
        markdown_path = output_dir / f"phase16c_gpu_profile_{stamp}.md"
        if not json_path.exists() and not markdown_path.exists():
            return json_path, markdown_path


def _statistics(values: list[float]) -> dict[str, float | int] | None:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return None
    array = np.asarray(finite, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "mean": float(mean(finite)),
        "median": float(median(finite)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
        "sample_count": len(finite),
    }


class _PhaseTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase = "input_decode"
        self._trial: int | None = None
        self.intervals: list[dict[str, Any]] = []
        self.on_phase_change = None

    def set_phase(self, phase: str, trial: int | None = None) -> None:
        with self._lock:
            self._phase = phase
            self._trial = trial
        if self.on_phase_change is not None:
            self.on_phase_change()

    def current(self) -> tuple[str, int | None]:
        with self._lock:
            return self._phase, self._trial

    @contextmanager
    def interval(self, phase: str, trial: int | None = None):
        started_utc = _utc_now()
        started_monotonic = time.perf_counter()
        self.set_phase(phase, trial)
        try:
            yield
        finally:
            ended_monotonic = time.perf_counter()
            interval = {
                "phase": phase,
                "trial": trial,
                "started_utc": started_utc,
                "ended_utc": _utc_now(),
                "started_monotonic_seconds": started_monotonic,
                "ended_monotonic_seconds": ended_monotonic,
                "duration_ms": (ended_monotonic - started_monotonic) * 1000.0,
            }
            self.intervals.append(interval)


def _nvml_metric(handle: Any, nvml: Any, name: str) -> float | int:
    if name == "gpu_utilization_percent":
        return float(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    if name == "gpu_memory_utilization_percent":
        return float(nvml.nvmlDeviceGetUtilizationRates(handle).memory)
    if name in {"gpu_memory_used_mib", "gpu_memory_total_mib"}:
        memory = nvml.nvmlDeviceGetMemoryInfo(handle)
        field = "used" if name.endswith("used_mib") else "total"
        return float(getattr(memory, field) / (1024**2))
    if name == "gpu_temperature_c":
        return float(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))
    if name == "gpu_power_w":
        return float(nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
    if name == "gpu_graphics_clock_mhz":
        return float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS))
    if name == "gpu_memory_clock_mhz":
        return float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM))
    if name == "gpu_performance_state":
        return int(nvml.nvmlDeviceGetPerformanceState(handle))
    raise KeyError(f"Unknown GPU telemetry metric: {name}")


class _TelemetrySampler:
    def __init__(
        self,
        runtime: str,
        interval_seconds: float,
        phase_tracker: _PhaseTracker,
        *,
        nvml: Any | None = None,
        process: Any | None = None,
        torch_module: Any = torch,
    ) -> None:
        self.runtime = runtime
        self.interval_seconds = interval_seconds
        self.phase_tracker = phase_tracker
        self.nvml = nvml
        self.process = process
        self.torch = torch_module
        self.samples: list[dict[str, Any]] = []
        self.errors: list[dict[str, str]] = []
        self._stop = threading.Event()
        self._phase_changed = threading.Event()
        self.phase_tracker.on_phase_change = self._phase_changed.set
        self._thread: threading.Thread | None = None
        self._handle = None
        self._nvml_initialized = False
        self._last_sample_monotonic: float | None = None
        self._total_sampling_overhead_ms = 0.0
        self._cpu_percent_primed = False
        self._last_cpu_monotonic: float | None = None
        self._process_gpu_memory_status = (
            "unavailable: WDDM does not expose reliable per-process GPU memory"
        )

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"gpu-telemetry-{self.runtime}",
            daemon=True,
        )
        self._thread.start()

    def set_phase(self, phase: str, trial: int | None = None) -> None:
        self.phase_tracker.set_phase(phase, trial)

    def _record_error(self, metric: str, exc: BaseException) -> None:
        self.errors.append(
            {"metric": metric, "error": f"{type(exc).__name__}: {exc}"}
        )

    def _collect(self) -> None:
        started = time.perf_counter()
        sample: dict[str, Any] = {
            "timestamp_utc": _utc_now(),
            "monotonic_seconds": started,
            "runtime": self.runtime,
        }
        sample["phase"], sample["trial"] = self.phase_tracker.current()
        if self._last_sample_monotonic is None:
            gap_ms = None
            missed_intervals = None
        else:
            gap_seconds = started - self._last_sample_monotonic
            gap_ms = gap_seconds * 1000.0
            missed_intervals = max(
                0, int(gap_seconds / self.interval_seconds) - 1
            )
        sample["gap_since_previous_sample_ms"] = gap_ms
        sample["missed_sampling_intervals"] = missed_intervals

        for metric in _GPU_METRICS:
            try:
                if self.nvml is None or self._handle is None:
                    raise RuntimeError("NVML is unavailable")
                sample[metric] = _nvml_metric(self._handle, self.nvml, metric)
            except BaseException as exc:
                sample[metric] = None
                self._record_error(metric, exc)

        try:
            memory_info = self.process.memory_info()
            sample["worker_rss_mib"] = float(memory_info.rss / (1024**2))
        except BaseException as exc:
            sample["worker_rss_mib"] = None
            self._record_error("worker_rss_mib", exc)

        try:
            cpu_percent = float(self.process.cpu_percent(interval=None))
            cpu_sample_time = time.perf_counter()
            cpu_interval_ms = (
                None
                if self._last_cpu_monotonic is None
                else (cpu_sample_time - self._last_cpu_monotonic) * 1000.0
            )
            sample["worker_cpu_interval_ms"] = cpu_interval_ms
            if self._cpu_percent_primed:
                if cpu_interval_ms is not None and cpu_interval_ms >= self.interval_seconds * 500.0:
                    sample["worker_cpu_percent"] = cpu_percent
                else:
                    sample["worker_cpu_percent"] = None
                    sample["worker_cpu_percent_status"] = (
                        "unavailable: CPU sampling interval was shorter than half the configured interval"
                    )
            else:
                sample["worker_cpu_percent"] = None
                sample["worker_cpu_percent_status"] = (
                    "unavailable: first non-blocking psutil CPU sample is a baseline"
                )
                self._cpu_percent_primed = True
            self._last_cpu_monotonic = cpu_sample_time
        except BaseException as exc:
            sample["worker_cpu_percent"] = None
            self._record_error("worker_cpu_percent", exc)

        if self.runtime == "pytorch":
            try:
                if self.torch.cuda.is_initialized():
                    sample["pytorch_allocator_allocated_mib"] = float(
                        self.torch.cuda.memory_allocated(0) / (1024**2)
                    )
                    sample["pytorch_allocator_reserved_mib"] = float(
                        self.torch.cuda.memory_reserved(0) / (1024**2)
                    )
                else:
                    sample["pytorch_allocator_allocated_mib"] = None
                    sample["pytorch_allocator_reserved_mib"] = None
            except BaseException as exc:
                sample["pytorch_allocator_allocated_mib"] = None
                sample["pytorch_allocator_reserved_mib"] = None
                self._record_error("pytorch_allocator_memory", exc)
        else:
            sample["pytorch_allocator_allocated_mib"] = None
            sample["pytorch_allocator_reserved_mib"] = None
            sample["pytorch_allocator_status"] = "not_applicable: ONNX Runtime worker"

        sample["process_gpu_memory_mib"] = None
        sample["process_gpu_memory_status"] = self._process_gpu_memory_status
        completed = time.perf_counter()
        sample["sampling_overhead_ms"] = (completed - started) * 1000.0
        self._total_sampling_overhead_ms += sample["sampling_overhead_ms"]
        sample["sample_completed_monotonic_seconds"] = completed
        self._last_sample_monotonic = started
        self.samples.append(sample)

    def _run(self) -> None:
        try:
            if self.nvml is None:
                try:
                    import pynvml as nvml

                    self.nvml = nvml
                except ImportError as exc:
                    self._record_error("nvml_initialization", exc)
            if self.nvml is not None:
                try:
                    self.nvml.nvmlInit()
                    self._nvml_initialized = True
                    self._handle = self.nvml.nvmlDeviceGetHandleByIndex(0)
                except BaseException as exc:
                    self._record_error("nvml_initialization", exc)
                    self._handle = None
            if self.process is None:
                try:
                    self.process = psutil.Process(os.getpid())
                    self.process.cpu_percent(interval=None)
                    self._last_cpu_monotonic = time.perf_counter()
                except BaseException as exc:
                    self._record_error("worker_process_initialization", exc)
            self._collect()
            deadline = time.perf_counter() + self.interval_seconds
            while not self._stop.is_set():
                self._phase_changed.wait(max(0.0, deadline - time.perf_counter()))
                if self._stop.is_set():
                    break
                phase_changed = self._phase_changed.is_set()
                self._phase_changed.clear()
                self._collect()
                if phase_changed:
                    deadline = time.perf_counter() + self.interval_seconds
                else:
                    deadline += self.interval_seconds
                    if deadline < time.perf_counter():
                        deadline = time.perf_counter() + self.interval_seconds
        finally:
            try:
                self._collect()
            except BaseException as exc:
                self._record_error("final_sample", exc)
            if self._nvml_initialized and self.nvml is not None:
                try:
                    self.nvml.nvmlShutdown()
                except BaseException as exc:
                    self._record_error("nvml_shutdown", exc)

    def stop(self, timeout_seconds: float = 10.0) -> None:
        self._stop.set()
        self._phase_changed.set()
        if self._thread is not None:
            self._thread.join(timeout_seconds)
            if self._thread.is_alive():
                raise RuntimeError("GPU telemetry sampler did not stop within the timeout.")

    def report(self) -> dict[str, Any]:
        phase_keys = sorted(
            {
                (sample["phase"], sample.get("trial"))
                for sample in self.samples
            },
            key=lambda key: (key[0], key[1] or 0),
        )
        phase_summaries: dict[str, Any] = {}
        for phase, trial in phase_keys:
            key = phase if trial is None else f"{phase}_trial_{trial}"
            subset = [
                sample
                for sample in self.samples
                if sample["phase"] == phase and sample.get("trial") == trial
            ]
            phase_summaries[key] = self._summaries(subset)

        return {
            "scope": {
                "gpu_metrics": "DEVICE-WIDE NVML measurements",
                "worker_metrics": "WORKER-PROCESS psutil RSS and CPU utilization",
                "process_gpu_memory": "unavailable on this WDDM system; represented as null",
                "pytorch_allocator": (
                    "PYTORCH-ALLOCATOR counters; not total process GPU memory"
                    if self.runtime == "pytorch"
                    else "not applicable; ONNX Runtime CUDA memory is not measured by PyTorch"
                ),
            },
            "sampling_interval_seconds": self.interval_seconds,
            "sample_count": len(self.samples),
            "sampling_overhead_total_ms": self._total_sampling_overhead_ms,
            "sampling_overhead_note": (
                "Sampling runs concurrently and is excluded from detector latency intervals, "
                "but can perturb overall worker scheduling."
            ),
            "samples": self.samples,
            "overall_summaries": self._summaries(self.samples),
            "phase_summaries": phase_summaries,
            "performance_state_distribution": self._performance_state_distribution(),
            "availability": self._availability(),
            "errors": self.errors,
            "phase_intervals": self.phase_tracker.intervals,
            "process_gpu_memory_status": self._process_gpu_memory_status,
        }

    @staticmethod
    def _summaries(samples: list[dict[str, Any]]) -> dict[str, Any]:
        summaries = {}
        for metric in _NUMERIC_METRICS:
            values = [
                float(sample[metric])
                for sample in samples
                if sample.get(metric) is not None
                and np.isfinite(sample[metric])
            ]
            summaries[metric] = _statistics(values)
        gaps = [
            float(sample["gap_since_previous_sample_ms"])
            for sample in samples
            if sample.get("gap_since_previous_sample_ms") is not None
        ]
        missed = sum(
            int(sample["missed_sampling_intervals"])
            for sample in samples
            if sample.get("missed_sampling_intervals") is not None
        )
        summaries["sampling_gap_ms"] = _statistics(gaps)
        summaries["missed_sampling_intervals"] = missed
        summaries["sample_count"] = len(samples)
        return summaries

    def _performance_state_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sample in self.samples:
            value = sample.get("gpu_performance_state")
            if value is not None:
                key = f"P{value}"
                counts[key] = counts.get(key, 0) + 1
        return counts

    def _availability(self) -> dict[str, dict[str, Any]]:
        result = {}
        for metric in (*_GPU_METRICS, *_PROCESS_METRICS, "process_gpu_memory_mib"):
            values = [sample.get(metric) for sample in self.samples]
            count = sum(value is not None for value in values)
            result[metric] = {
                "available": count > 0,
                "sample_count": count,
                "status": "available" if count else "unavailable",
            }
        pstate_count = sum(
            sample.get("gpu_performance_state") is not None
            for sample in self.samples
        )
        result["gpu_performance_state"] = {
            "available": pstate_count > 0,
            "sample_count": pstate_count,
            "status": "available" if pstate_count else "unavailable",
        }
        return result


def _profile_worker(
    runtime: str,
    image_path: str,
    weights_path: str,
    onnx_path: str,
    sample_interval_seconds: float,
    connection: Any,
) -> None:
    started_utc = _utc_now()
    started_monotonic = time.perf_counter()
    phase_tracker = _PhaseTracker()
    sampler = _TelemetrySampler(runtime, sample_interval_seconds, phase_tracker)
    detector = None
    failure = None
    run_samples: list[list[dict[str, Any]]] = []
    model_path: Path | None = None
    model_hash: str | None = None
    benchmark_result = None
    try:
        sampler.start()
        with phase_tracker.interval("input_decode"):
            image, decode_ms = baseline.load_image(image_path)
        with phase_tracker.interval("initialization"):
            if runtime == "pytorch":
                detector = baseline.ImageDetector(model_name=weights_path, device="cuda")
                model_path = Path(weights_path).resolve()
                if detector.device != "cuda:0":
                    raise baseline.DeviceUnavailableError(
                        f"PyTorch initialized on {detector.device}, expected cuda:0."
                    )
            elif runtime == "onnx":
                detector = baseline.OnnxDetector(
                    model_path=onnx_path,
                    device="cuda",
                    precision="fp32",
                    require_cuda=True,
                )
                model_path = Path(onnx_path).resolve()
                if (
                    detector.device != "cuda:0"
                    or detector.execution_provider != "CUDAExecutionProvider"
                    or not detector.require_cuda
                ):
                    raise baseline.DeviceUnavailableError(
                        "ONNX Runtime did not initialize with strict CUDAExecutionProvider "
                        "and CPU fallback disabled."
                    )
                if detector.input_size != baseline.INPUT_SIZE:
                    raise ValueError(
                        f"Expected a {baseline.INPUT_SIZE}x{baseline.INPUT_SIZE} ONNX input, "
                        f"got {detector.input_size}."
                    )
            else:
                raise ValueError(f"Unsupported runtime: {runtime}")

        model_hash = baseline._sha256_file(model_path)
        for trial in range(1, baseline.DEFAULT_TRIALS + 1):
            with phase_tracker.interval("warmup", trial):
                for _ in range(baseline.DEFAULT_WARMUP_RUNS):
                    baseline._predict(detector, image, runtime)
            records: list[dict[str, Any]] = []
            with phase_tracker.interval("measured", trial):
                for _ in range(baseline.DEFAULT_MEASURED_RUNS):
                    full_call_ms, stages = baseline._predict(detector, image, runtime)
                    records.append({"full_call_ms": full_call_ms, "stages": stages})
            run_samples.append(records)
        benchmark_result = baseline._summarize_configuration(
            runtime,
            detector,
            model_path,
            model_hash,
            run_samples,
        )
        benchmark_result["input_decode_ms"] = decode_ms
        benchmark_result["input_dimensions"] = [int(image.shape[1]), int(image.shape[0])]
        benchmark_result["input_size"] = [baseline.INPUT_SIZE, baseline.INPUT_SIZE]
        benchmark_result["batch_size"] = 1
        benchmark_result["precision"] = "FP32"
        benchmark_result["raw_measured_iterations_per_trial"] = baseline.DEFAULT_MEASURED_RUNS
        benchmark_result["trials"] = baseline.DEFAULT_TRIALS
    except BaseException as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            phase_tracker.set_phase("shutdown")
            with phase_tracker.interval("shutdown"):
                detector = None
                gc.collect()
        except BaseException as exc:
            if failure is None:
                failure = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
        try:
            sampler.stop()
        except BaseException as exc:
            sampler.errors.append(
                {"metric": "sampler_shutdown", "error": f"{type(exc).__name__}: {exc}"}
            )
            if failure is None:
                failure = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }

    result: dict[str, Any] = {
        "runtime": runtime,
        "status": "failed" if failure else "completed",
        "worker_pid": os.getpid(),
        "worker_started_utc": started_utc,
        "worker_ended_utc": _utc_now(),
        "worker_duration_ms": (time.perf_counter() - started_monotonic) * 1000.0,
        "telemetry": sampler.report(),
        "failure": failure,
    }
    if benchmark_result is not None and failure is None:
        result["benchmark"] = benchmark_result
    try:
        connection.send(result)
    finally:
        connection.close()


def _run_worker(
    runtime: str,
    *,
    image_path: Path,
    weights_path: Path,
    onnx_path: Path,
    sample_interval_seconds: float,
    context: Any | None = None,
) -> dict[str, Any]:
    multiprocessing_context = context or multiprocessing.get_context("spawn")
    receive_connection, send_connection = multiprocessing_context.Pipe(duplex=False)
    worker = multiprocessing_context.Process(
        target=_profile_worker,
        args=(
            runtime,
            str(image_path),
            str(weights_path),
            str(onnx_path),
            sample_interval_seconds,
            send_connection,
        ),
        name=f"edgeai-profile-{runtime}",
    )
    worker.start()
    send_connection.close()
    result = None
    try:
        while worker.is_alive():
            if receive_connection.poll(0.1):
                result = receive_connection.recv()
                break
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=10)
            raise RuntimeError(f"{runtime} profiling worker did not exit after reporting.")
        if result is None and receive_connection.poll():
            result = receive_connection.recv()
        if result is None:
            return {
                "runtime": runtime,
                "status": "failed",
                "worker_pid": worker.pid,
                "failure": {
                    "type": "WorkerProcessError",
                    "message": f"Worker exited with code {worker.exitcode} without a result.",
                    "traceback": None,
                },
            }
        result["worker_exit_code"] = worker.exitcode
        if worker.exitcode != 0 and result["status"] == "completed":
            result["status"] = "failed"
            result["failure"] = {
                "type": "WorkerProcessError",
                "message": f"Worker exited with code {worker.exitcode}.",
                "traceback": None,
            }
        return result
    except BaseException:
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=10)
        raise
    finally:
        receive_connection.close()


def _hardware_software() -> dict[str, Any]:
    return {
        **baseline._software_hardware(),
        "python_bits": struct.calcsize("P") * 8,
        "psutil": psutil.__version__,
        "nvidia_ml_py": _package_version("nvidia-ml-py"),
        "pynvml_module": _package_version("pynvml"),
    }


def _summary_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "| Runtime | Status | Full-call mean / median / p90 / p95 (ms) | "
        "Model-execution mean / median / p90 / p95 (ms) | Full-call FPS |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for runtime in ("pytorch", "onnx"):
        config = report["runtimes"][runtime]
        if config["status"] != "completed":
            lines.append(f"| {runtime} | failed | unavailable | unavailable | unavailable |")
            continue
        benchmark_report = config["benchmark"]
        full = benchmark_report["full_inference_call_latency"]
        execution = benchmark_report["stage_latency"]["model_execution"]
        execution_values = (
            "unavailable"
            if execution is None
            else f"{execution['mean_ms']:.3f} / {execution['median_ms']:.3f} / "
            f"{execution['p90_ms']:.3f} / {execution['p95_ms']:.3f}"
        )
        lines.append(
            f"| {runtime} | completed | "
            f"{full['mean_ms']:.3f} / {full['median_ms']:.3f} / "
            f"{full['p90_ms']:.3f} / {full['p95_ms']:.3f} | "
            f"{execution_values} | "
            f"{benchmark_report['pooled_full_call_throughput_fps']:.3f} |"
        )
    return lines


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 16C GPU utilization and memory profile",
        "",
        f"- Created UTC: `{report['created_utc']}`",
        f"- Overall status: **{report['status']}**",
        f"- GPU: {report['hardware_software'].get('gpu_name')}",
        f"- Input image: `{report['input']['path']}` ({report['input']['width']}x{report['input']['height']})",
        f"- Model lineage verified: `{report['model_lineage']['source_weights_match']}`",
        "",
        "## Configuration",
        "",
        f"- Input tensor: 640x640, batch 1, FP32; warm-up: {report['configuration']['warmup_runs']} per trial; "
        f"measured calls: {report['configuration']['measured_runs']} per trial; trials: "
        f"{report['configuration']['trials']}.",
        f"- Telemetry interval: {report['configuration']['sampling_interval_seconds']} s.",
        "- Runtime workers execute separately. Phase 16B latency definitions and detector-call timing are preserved.",
        "",
        "## Latency and throughput",
        "",
        "Full-call latency is wall-clock time around `predict_frame`; model-execution latency is the runtime-reported model stage. "
        "Full-call FPS and model-execution FPS are distinct pooled rates.",
        "",
        *_summary_table(report),
        "",
        "## Telemetry interpretation",
        "",
        "- GPU utilization, VRAM, temperature, power, clocks and performance state are DEVICE-WIDE NVML samples.",
        "- RSS and CPU utilization are WORKER-PROCESS measurements.",
        "- Process-specific GPU memory is unavailable on this WDDM system and remains null.",
        "- PyTorch allocated/reserved values are PYTORCH-ALLOCATOR counters, not total process GPU memory. ONNX allocator values are unavailable.",
        "- Maximum values are maxima among timestamped samples, not instantaneous peaks. Sampling can perturb scheduling; sampling overhead is reported separately.",
        "- GPU utilization alone does not establish a bottleneck or causation.",
        "",
        "## Per-runtime phase and telemetry details",
        "",
    ]
    for runtime in ("pytorch", "onnx"):
        entry = report["runtimes"][runtime]
        lines.extend([f"### {runtime}", "", f"- Status: `{entry['status']}`; worker PID: `{entry.get('worker_pid')}`."])
        if entry.get("failure"):
            lines.extend(
                [
                    f"- Failure: `{entry['failure']['type']}: {entry['failure']['message']}`",
                    "",
                    "```text",
                    entry["failure"].get("traceback") or "Traceback unavailable.",
                    "```",
                ]
            )
        telemetry = entry.get("telemetry")
        if telemetry:
            lines.extend(
                [
                    "",
                    f"- Samples: {telemetry['sample_count']}; sampling overhead total: "
                    f"{telemetry['sampling_overhead_total_ms']:.3f} ms.",
                    f"- Performance state sample distribution: `{telemetry['performance_state_distribution']}`.",
                    "",
                    "Metric availability:",
                    "",
                    "| Metric | Status | Samples |",
                    "| --- | --- | ---: |",
                ]
            )
            for metric, availability in telemetry["availability"].items():
                lines.append(
                    f"| {metric} | {availability['status']} | {availability['sample_count']} |"
                )
            lines.extend(
                [
                    "",
                    "- Phase intervals:",
                    "",
                    "| Phase | Trial | Duration (ms) |",
                    "| --- | ---: | ---: |",
                ]
            )
            for phase in telemetry["phase_intervals"]:
                lines.append(
                    f"| {phase['phase']} | {phase['trial'] if phase['trial'] is not None else '—'} "
                    f"| {phase['duration_ms']:.3f} |"
                )
            lines.extend(
                [
                    "",
                    "Telemetry statistics by phase. Maxima are sampled maxima, not instantaneous peaks.",
                    "",
                    "| Phase | Metric | Min | Mean | Median | P90 | P95 | Max | Samples |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for phase, metrics in telemetry["phase_summaries"].items():
                missed = metrics["missed_sampling_intervals"]
                if missed:
                    lines.append(
                        f"| {phase} | missed_sampling_intervals | — | — | — | — | — | — | {missed} |"
                    )
            for phase, metrics in telemetry["phase_summaries"].items():
                for metric, stats in metrics.items():
                    if metric in {"sample_count", "missed_sampling_intervals"} or stats is None:
                        continue
                    lines.append(
                        f"| {phase} | {metric} | {stats['minimum']:.3f} | {stats['mean']:.3f} | "
                        f"{stats['median']:.3f} | {stats['p90']:.3f} | {stats['p95']:.3f} | "
                        f"{stats['maximum']:.3f} | {stats['sample_count']} |"
                    )
            if telemetry["errors"]:
                lines.extend(["", "Sampler errors:", ""])
                lines.extend(
                    f"- `{error['metric']}`: {error['error']}" for error in telemetry["errors"]
                )
        lines.append("")
    lines.extend(
        [
            "## Limitations",
            "",
            "This is sampled host/device telemetry, not per-kernel profiling. WDDM does not expose reliable per-process GPU memory here. "
            "Between-sample peaks may be missed. Worker RSS includes its Python interpreter, runtime, and loaded model. "
            "Profiled telemetry is correlated by phase and timestamp but does not prove a performance bottleneck.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_reports(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    report["output_paths"] = {
        "json": str(json_path.resolve()),
        "markdown": str(markdown_path.resolve()),
    }
    markdown = _markdown_report(report)
    json_created = False
    markdown_created = False
    try:
        with json_path.open("x", encoding="utf-8") as stream:
            json_created = True
            json.dump(report, stream, indent=2)
        with markdown_path.open("x", encoding="utf-8") as stream:
            markdown_created = True
            stream.write(markdown)
    except OSError:
        if markdown_created:
            markdown_path.unlink(missing_ok=True)
        if json_created:
            json_path.unlink(missing_ok=True)
        raise


def run_gpu_profile(
    *,
    image_path: str | Path = "outputs/smoke_input.jpg",
    weights_path: str | Path = baseline.DEFAULT_MODEL,
    onnx_path: str | Path = baseline.DEFAULT_ONNX_PATH,
    output_dir: str | Path = "outputs",
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
) -> dict[str, Any]:
    if not np.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be finite and greater than zero.")
    image_file = Path(image_path).expanduser().resolve()
    weights_file = Path(weights_path).expanduser().resolve()
    onnx_file = Path(onnx_path).expanduser().resolve()
    for path in (image_file, weights_file, onnx_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required benchmark input does not exist: {path}")
    model_lineage = baseline._verify_source_model(weights_file, onnx_file)
    image, _ = baseline.load_image(image_file)
    runtimes: dict[str, Any] = {}
    for runtime in ("pytorch", "onnx"):
        runtimes[runtime] = _run_worker(
            runtime,
            image_path=image_file,
            weights_path=weights_file,
            onnx_path=onnx_file,
            sample_interval_seconds=sample_interval_seconds,
        )
    complete = all(item["status"] == "completed" for item in runtimes.values())
    report = {
        "benchmark_type": "phase16c_isolated_cuda_resource_profile",
        "created_utc": _utc_now(),
        "status": "completed" if complete else "incomplete",
        "configuration": {
            "image_path": str(image_file),
            "weights_path": str(weights_file),
            "onnx_path": str(onnx_file),
            "input_size": [baseline.INPUT_SIZE, baseline.INPUT_SIZE],
            "batch_size": 1,
            "precision": "FP32",
            "confidence_threshold": baseline.CONFIDENCE,
            "iou_threshold": baseline.IOU_THRESHOLD,
            "warmup_runs": baseline.DEFAULT_WARMUP_RUNS,
            "measured_runs": baseline.DEFAULT_MEASURED_RUNS,
            "trials": baseline.DEFAULT_TRIALS,
            "sampling_interval_seconds": sample_interval_seconds,
            "runtime_execution_order": ["pytorch", "onnx"],
            "latency_method": (
                "Reuses benchmark_gpu_baseline._predict and detector last_timings. "
                "PyTorch synchronizes CUDA around predict_frame; ONNX session.run is synchronous."
            ),
            "percentile_method": "NumPy percentile with its default linear interpolation.",
            "monotonic_clock": "time.perf_counter; monotonic and high resolution.",
        },
        "input": {
            "path": str(image_file),
            "sha256": baseline._sha256_file(image_file),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        },
        "model_lineage": model_lineage,
        "hardware_software": _hardware_software(),
        "telemetry_limitations": [
            "NVML GPU measurements are device-wide.",
            "WDDM process-specific GPU memory is unavailable and is reported as null.",
            "PyTorch allocated/reserved metrics are allocator counters, not total process GPU memory.",
            "ONNX Runtime process-specific GPU memory is unavailable.",
            "Sampled maxima are maxima observed at sample times, not instantaneous peaks.",
            "GPU utilization alone does not prove a bottleneck.",
        ],
        "runtimes": runtimes,
    }
    json_path, markdown_path = _timestamped_outputs(Path(output_dir))
    _write_reports(report, json_path, markdown_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("outputs/smoke_input.jpg"))
    parser.add_argument("--weights", type=Path, default=Path(baseline.DEFAULT_MODEL))
    parser.add_argument("--onnx-model", type=Path, default=Path(baseline.DEFAULT_ONNX_PATH))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--sample-interval-seconds", type=float, default=DEFAULT_SAMPLE_INTERVAL_SECONDS)
    args = parser.parse_args(argv)
    try:
        report = run_gpu_profile(
            image_path=args.image,
            weights_path=args.weights,
            onnx_path=args.onnx_model,
            output_dir=args.output_dir,
            sample_interval_seconds=args.sample_interval_seconds,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"GPU profiling failed: {exc}\n")
    print(_markdown_report(report))
    print(f"\nJSON: {report['output_paths']['json']}\nMarkdown: {report['output_paths']['markdown']}")
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
