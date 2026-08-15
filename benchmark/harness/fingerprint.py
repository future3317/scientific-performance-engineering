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
        }
    except Exception:
        return {}


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
        "torch_geometric_version": _optional_pyg_version(),
        "env": {key: os.environ.get(key) for key in _ENV_KEYS},
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
