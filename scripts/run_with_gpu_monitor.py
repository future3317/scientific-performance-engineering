#!/usr/bin/env python3
"""Run a fixed command while sampling nvidia-smi; pair with project throughput metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # Host telemetry is optional; GPU monitoring remains usable.
    psutil = None

QUERY_FIELDS = (
    "timestamp,index,name,utilization.gpu,utilization.memory,memory.used,"
    "power.draw,clocks.sm,temperature.gpu"
)


def parse_number(value: str) -> float | None:
    value = value.strip()
    if value in {"", "N/A", "[N/A]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def host_sample() -> dict[str, Any]:
    """Capture host contention at the same cadence as GPU telemetry."""
    captured = datetime.now(timezone.utc).isoformat()
    sample: dict[str, Any] = {"captured_utc": captured, "available": psutil is not None}
    if psutil is None:
        return sample

    load_average: tuple[float, float, float] | None = None
    try:
        load_average = tuple(float(value) for value in psutil.getloadavg())
    except (AttributeError, NotImplementedError, OSError):
        try:
            load_average = tuple(float(value) for value in os.getloadavg())
        except (AttributeError, NotImplementedError, OSError):
            pass
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    sample.update(
        {
            "cpu_percent": float(psutil.cpu_percent(interval=None)),
            "load_average": load_average,
            "available_memory_mb": float(memory.available) / (1024.0 * 1024.0),
            "swap_percent": float(swap.percent),
            "process_rss_mb": float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0),
        }
    )
    return sample


def sample_gpu(
    samples: list[dict[str, Any]],
    host_samples: list[dict[str, Any]],
    stop: threading.Event,
    interval: float,
) -> None:
    while not stop.is_set():
        start = time.monotonic()
        host_samples.append(host_sample())
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={QUERY_FIELDS}", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        captured = datetime.now(timezone.utc).isoformat()
        if result.returncode == 0:
            for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
                if len(row) != 9:
                    continue
                samples.append(
                    {
                        "captured_utc": captured,
                        "nvidia_timestamp": row[0].strip(),
                        "index": int(row[1]),
                        "name": row[2].strip(),
                        "gpu_util_percent": parse_number(row[3]),
                        "memory_util_percent": parse_number(row[4]),
                        "memory_used_mb": parse_number(row[5]),
                        "power_w": parse_number(row[6]),
                        "sm_clock_mhz": parse_number(row[7]),
                        "temperature_c": parse_number(row[8]),
                    }
                )
        elapsed = time.monotonic() - start
        stop.wait(max(0.0, interval - elapsed))


def summarize_host(samples: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("cpu_percent", "available_memory_mb", "swap_percent", "process_rss_mb")
    summary: dict[str, Any] = {
        "samples": len(samples),
        "telemetry_available": any(sample.get("available") for sample in samples),
    }
    for field in fields:
        values = [float(sample[field]) for sample in samples if sample.get(field) is not None]
        if values:
            summary[field] = {
                "mean": statistics.fmean(values),
                "p50": percentile(values, 0.50),
                "p95": percentile(values, 0.95),
                "max": max(values),
            }
    load_values = [
        float(sample["load_average"][0])
        for sample in samples
        if sample.get("load_average")
    ]
    if load_values:
        summary["load_average_1m"] = {
            "mean": statistics.fmean(load_values),
            "p50": percentile(load_values, 0.50),
            "p95": percentile(load_values, 0.95),
            "max": max(load_values),
        }
    return summary


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    indices = sorted({int(sample["index"]) for sample in samples})
    for index in indices:
        current = [sample for sample in samples if int(sample["index"]) == index]
        fields = (
            "gpu_util_percent",
            "memory_util_percent",
            "memory_used_mb",
            "power_w",
            "sm_clock_mhz",
            "temperature_c",
        )
        stats: dict[str, Any] = {"samples": len(current), "name": current[0]["name"]}
        for field in fields:
            values = [float(sample[field]) for sample in current if sample[field] is not None]
            if values:
                stats[field] = {
                    "mean": statistics.fmean(values),
                    "p50": percentile(values, 0.50),
                    "p95": percentile(values, 0.95),
                    "max": max(values),
                }
        summary[str(index)] = stats
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("provide a command after --")
    if args.interval < 0.1:
        parser.error("--interval must be at least 0.1 seconds")
    if not shutil.which("nvidia-smi"):
        raise SystemExit("nvidia-smi is not available")

    samples: list[dict[str, Any]] = []
    host_samples: list[dict[str, Any]] = []
    stop = threading.Event()
    monitor = threading.Thread(
        target=sample_gpu,
        args=(samples, host_samples, stop, args.interval),
        daemon=True,
    )
    start_utc = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    monitor.start()
    try:
        completed = subprocess.run(command, check=False)
        returncode = completed.returncode
    finally:
        stop.set()
        monitor.join(timeout=max(2.0, args.interval * 3))
    duration = time.monotonic() - start

    record = {
        "command": command,
        "start_utc": start_utc,
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "returncode": returncode,
        "warning": "GPU utilization is supporting evidence, not a throughput or optimization claim.",
        "summary": summarize(samples),
        "host_summary": summarize_host(host_samples),
        "samples": samples,
        "host_samples": host_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
