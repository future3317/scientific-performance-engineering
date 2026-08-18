#!/usr/bin/env python3
"""Run a command while sampling selected NVIDIA GPUs and the command process tree."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
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
    "timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,"
    "power.draw,power.limit,clocks.sm,clocks.mem,pstate,temperature.gpu,"
    "clocks_throttle_reasons.active,mig.mode.current,persistence_mode"
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


def visible_devices() -> list[str]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    return [] if value is None else [item.strip() for item in value.split(",") if item.strip()]


def logical_index_for(physical_index: int, uuid: str, visible: list[str]) -> int | None:
    if not visible:
        return physical_index
    for logical, token in enumerate(visible):
        if token == str(physical_index) or token == uuid:
            return logical
    return None


def process_tree_memory(target_pid: int | None) -> dict[str, Any]:
    """Report trainer, descendants (usually DataLoader workers), and total RSS/PSS."""
    result: dict[str, Any] = {
        "target_pid": target_pid,
        "process_count": 0,
        "trainer_rss_mb": None,
        "worker_rss_mb": None,
        "process_tree_rss_mb": None,
        "trainer_pss_mb": None,
        "worker_pss_mb": None,
        "process_tree_pss_mb": None,
    }
    if psutil is None or target_pid is None:
        return result
    try:
        root = psutil.Process(target_pid)
        descendants = root.children(recursive=True)
        processes = [root, *descendants]
        rss_values = [float(process.memory_info().rss) / (1024.0 * 1024.0) for process in processes]
        result.update(
            {
                "process_count": len(processes),
                "trainer_rss_mb": rss_values[0],
                "worker_rss_mb": sum(rss_values[1:]),
                "process_tree_rss_mb": sum(rss_values),
            }
        )
        pss_values: list[float] = []
        for process in processes:
            try:
                pss_values.append(float(process.memory_full_info().pss) / (1024.0 * 1024.0))
            except (AttributeError, psutil.Error):
                pss_values = []
                break
        if pss_values:
            result.update(
                {
                    "trainer_pss_mb": pss_values[0],
                    "worker_pss_mb": sum(pss_values[1:]),
                    "process_tree_pss_mb": sum(pss_values),
                }
            )
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass
    return result


def host_sample(target_pid: int | None = None) -> dict[str, Any]:
    """Capture host contention and the monitored command's process tree."""
    captured = datetime.now(timezone.utc).isoformat()
    sample: dict[str, Any] = {
        "captured_utc": captured,
        "available": psutil is not None,
        "process_tree": process_tree_memory(target_pid),
    }
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
            "monitor_rss_mb": float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0),
        }
    )
    return sample


def sample_gpu(
    samples: list[dict[str, Any]],
    host_samples: list[dict[str, Any]],
    stop: threading.Event,
    interval: float,
    gpu_specs: tuple[str, ...],
    target_pid: int,
    visible: list[str],
) -> None:
    while not stop.is_set():
        start = time.monotonic()
        host_samples.append(host_sample(target_pid))
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={','.join(gpu_specs)}",
                f"--query-gpu={QUERY_FIELDS}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        captured = datetime.now(timezone.utc).isoformat()
        if result.returncode == 0:
            for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
                if len(row) != 16:
                    continue
                physical_index = int(row[1])
                uuid = row[2].strip()
                samples.append(
                    {
                        "captured_utc": captured,
                        "nvidia_timestamp": row[0].strip(),
                        "physical_index": physical_index,
                        "logical_index": logical_index_for(physical_index, uuid, visible),
                        "uuid": uuid,
                        "name": row[3].strip(),
                        "gpu_util_percent": parse_number(row[4]),
                        "memory_util_percent": parse_number(row[5]),
                        "memory_used_mb": parse_number(row[6]),
                        "power_w": parse_number(row[7]),
                        "power_limit_w": parse_number(row[8]),
                        "sm_clock_mhz": parse_number(row[9]),
                        "mem_clock_mhz": parse_number(row[10]),
                        "pstate": row[11].strip(),
                        "temperature_c": parse_number(row[12]),
                        "throttle_reason": row[13].strip(),
                        "mig_mode": row[14].strip(),
                        "persistence_mode": row[15].strip(),
                    }
                )
        elapsed = time.monotonic() - start
        stop.wait(max(0.0, interval - elapsed))


def summarize_host(samples: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "cpu_percent",
        "available_memory_mb",
        "swap_percent",
        "monitor_rss_mb",
    )
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
    tree_fields = ("trainer_rss_mb", "worker_rss_mb", "process_tree_rss_mb", "process_tree_pss_mb")
    for field in tree_fields:
        values = [
            float(sample["process_tree"][field])
            for sample in samples
            if sample.get("process_tree", {}).get(field) is not None
        ]
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
    indices = sorted({int(sample["physical_index"]) for sample in samples})
    for index in indices:
        current = [sample for sample in samples if int(sample["physical_index"]) == index]
        fields = (
            "gpu_util_percent",
            "memory_util_percent",
            "memory_used_mb",
            "power_w",
            "power_limit_w",
            "sm_clock_mhz",
            "mem_clock_mhz",
            "temperature_c",
        )
        stats: dict[str, Any] = {
            "samples": len(current),
            "name": current[0]["name"],
            "uuid": current[0]["uuid"],
            "logical_indices": sorted({sample["logical_index"] for sample in current}),
        }
        for field in fields:
            values = [float(sample[field]) for sample in current if sample[field] is not None]
            if values:
                output_field = "avg_power_w" if field == "power_w" else field
                stats[output_field] = {
                    "mean": statistics.fmean(values),
                    "p50": percentile(values, 0.50),
                    "p95": percentile(values, 0.95),
                    "max": max(values),
                }
        for field in ("pstate", "throttle_reason", "mig_mode", "persistence_mode"):
            values = sorted({str(sample[field]) for sample in current if sample.get(field) not in {None, "", "[N/A]"}})
            if values:
                stats[field] = values if len(values) > 1 else values[0]
        summary[str(index)] = stats
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--gpu", action="append", required=True, help="NVIDIA index or UUID; repeat for multiple GPUs")
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
    visible = visible_devices()
    stop = threading.Event()
    start_utc = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    completed = subprocess.Popen(command)
    monitor = threading.Thread(
        target=sample_gpu,
        args=(samples, host_samples, stop, args.interval, tuple(args.gpu), completed.pid, visible),
        daemon=True,
    )
    monitor.start()
    try:
        returncode = completed.wait()
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
        "gpu_selection": {
            "requested": args.gpu,
            "cuda_visible_devices": visible,
            "logical_to_physical_uuid": {
                str(sample["logical_index"]): sample["uuid"]
                for sample in samples
                if sample["logical_index"] is not None
            },
        },
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
