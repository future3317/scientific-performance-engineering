#!/usr/bin/env python3
"""Hardware/software fingerprint capture (BENCHMARK_DESIGN.md sections 6.1, 13).

Every result.json carries a fingerprint; results with mismatched fingerprints are
never compared (section 8: fingerprints gate comparability). Stdlib + torch only;
psutil extras are optional and degrade gracefully.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

_ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")

# Fields whose mismatch blocks comparability, and the reason template used.
_COMPARABLE_FIELDS = (
    "python_version",
    "platform",
    "torch_version",
    "cuda_version",
    "gpu_name",
    "gpu_count",
    "torch_geometric_version",
    "cpu_affinity",
    "driver_version",
    "cuda_visible_devices",
    "gpu_uuid",
    "thread_topology",
)


def _nvidia_smi_driver() -> str | None:
    """Best-effort driver version via nvidia-smi; None when unavailable."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return output.strip().splitlines()[0].strip() or None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError):
        return None


def _nvidia_smi_gpu_uuids() -> list[str]:
    """Return physical GPU UUIDs in nvidia-smi order when available."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return [line.strip() for line in output.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []


def _selected_gpu_uuid() -> str | None:
    uuids = _nvidia_smi_gpu_uuids()
    if not uuids:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        first = visible.split(",", 1)[0].strip()
        if first.startswith("GPU-"):
            return first
        try:
            index = int(first)
        except ValueError:
            index = 0
        return uuids[index] if 0 <= index < len(uuids) else None
    return uuids[0]


def _number(value: str) -> int | float | str:
    value = value.strip()
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def _optional_pyg_version() -> str | None:
    try:
        import torch_geometric  # noqa: PLC0415

        return str(torch_geometric.__version__)
    except Exception:
        return None


def _psutil_extras() -> dict[str, Any]:
    try:
        import psutil  # noqa: PLC0415

        memory = psutil.virtual_memory()
        return {
            "total_memory_mb": round(memory.total / (1024 * 1024), 1),
            "cpu_freq_mhz": (psutil.cpu_freq().current if psutil.cpu_freq() else None),
            "load_average": list(psutil.getloadavg()),
        }
    except Exception:
        return {}


def selected_gpu_preflight() -> dict[str, Any]:
    """Capture selected-GPU admission and observed co-tenant state."""
    selected = _selected_gpu_uuid()
    if not selected:
        return {"status": "unavailable", "foreign_pids": [], "reason": "selected GPU UUID is unavailable"}
    try:
        compute_output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
        gpu_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.used,memory.free,utilization.gpu,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "foreign_pids": [], "reason": f"nvidia-smi query failed: {exc}"}
    pids: list[int] = []
    foreign_processes: list[dict[str, Any]] = []
    for line in compute_output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3 or fields[1] != selected:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        pids.append(pid)
        foreign_processes.append({"pid": pid, "used_memory": fields[2]})
    gpu_state = None
    for line in gpu_output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 9 and fields[0] == selected:
            gpu_state = {
                "uuid": fields[0],
                "memory_used_mib": _number(fields[1]),
                "memory_free_mib": _number(fields[2]),
                "utilization_gpu_percent": _number(fields[3]),
                "clock_sm_mhz": _number(fields[4]),
                "clock_memory_mhz": _number(fields[5]),
                "pstate": fields[6],
                "power_w": _number(fields[7]),
                "temperature_c": _number(fields[8]),
            }
            break
    if gpu_state is None:
        return {"status": "unavailable", "foreign_pids": pids, "foreign_processes": foreign_processes, "gpu_uuid": selected, "reason": "selected GPU telemetry unavailable"}
    pids = sorted(set(pids))
    return {
        "status": "busy" if pids else "clean",
        "foreign_pids": pids,
        "foreign_processes": foreign_processes,
        "gpu_uuid": selected,
        "gpu_state": gpu_state,
        "reason": "selected GPU has compute processes" if pids else "no compute processes reported",
    }


def wait_for_selected_gpu_admission(*, timeout_s: float, poll_interval_s: float) -> dict[str, Any]:
    """Check selected-GPU identity; co-tenants are recorded, not rejected."""
    started = time.monotonic()
    deadline = started + float(timeout_s)
    latest = selected_gpu_preflight()
    while latest.get("status") not in {"clean", "busy"} and time.monotonic() < deadline:
        time.sleep(min(float(poll_interval_s), max(0.0, deadline - time.monotonic())))
        latest = selected_gpu_preflight()
    admitted = latest.get("status") in {"clean", "busy"}
    result = dict(latest)
    result["admission_status"] = "admitted" if admitted else "timeout"
    result["admission_waited_s"] = round(time.monotonic() - started, 3)
    return result


def capture_runtime_provenance() -> dict[str, Any]:
    """Capture host/GPU state alongside a timing control artifact."""
    provenance: dict[str, Any] = {
        "pid": os.getpid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    provenance["host"] = _psutil_extras()
    shared_state = selected_gpu_preflight()
    provenance["gpu_shared_state"] = shared_state
    gpu_state = shared_state.get("gpu_state") if isinstance(shared_state, dict) else None
    provenance["gpu"] = {
        "uuid": gpu_state.get("uuid"),
        "pstate": gpu_state.get("pstate"),
        "clock_sm_mhz": gpu_state.get("clock_sm_mhz"),
        "power_w": gpu_state.get("power_w"),
        "temperature_c": gpu_state.get("temperature_c"),
    } if isinstance(gpu_state, dict) else None
    return provenance

def capture_fingerprint() -> dict[str, Any]:
    """Capture the current hw/sw fingerprint as a JSON-serializable dict."""
    fingerprint: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "gpu_count": 0,
        "driver_version": None,
        "gpu_uuid": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "torch_geometric_version": _optional_pyg_version(),
        "env": {key: os.environ.get(key) for key in _ENV_KEYS},
        "thread_topology": {
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
            "torch_num_threads": None,
            "torch_num_interop_threads": None,
            "compiler_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS") or os.environ.get("MAX_JOBS"),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch  # noqa: PLC0415

        fingerprint["torch_version"] = torch.__version__
        fingerprint["cuda_available"] = bool(torch.cuda.is_available())
        fingerprint["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            fingerprint["gpu_count"] = torch.cuda.device_count()
            fingerprint["gpu_name"] = torch.cuda.get_device_name(0)
            fingerprint["driver_version"] = _nvidia_smi_driver()
            fingerprint["gpu_uuid"] = _selected_gpu_uuid()
        fingerprint["thread_topology"]["torch_num_threads"] = torch.get_num_threads()
        fingerprint["thread_topology"]["torch_num_interop_threads"] = torch.get_num_interop_threads()
    except Exception as exc:  # torch missing/broken must not kill the harness
        fingerprint["torch_error"] = repr(exc)
    fingerprint["psutil_extras"] = _psutil_extras()
    return fingerprint


def fingerprints_compatible(a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, list[str]]:
    """Check two fingerprints for comparability; return (compatible, reasons).

    Mismatching python/platform/torch/CUDA/GPU/PyG versions block comparison.
    ``cuda_available`` disagreements block as well. CPU count and env vars are
    reported as advisory warnings but do not block (recorded, not fatal).
    """
    reasons: list[str] = []
    for field in _COMPARABLE_FIELDS:
        if a.get(field) != b.get(field):
            reasons.append(f"{field} differs: {a.get(field)!r} vs {b.get(field)!r}")
    if bool(a.get("cuda_available")) != bool(b.get("cuda_available")):
        reasons.append(
            f"cuda_available differs: {a.get('cuda_available')!r} vs {b.get('cuda_available')!r}"
        )
    return (not reasons, reasons)
