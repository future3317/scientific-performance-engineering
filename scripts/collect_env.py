#!/usr/bin/env python3
"""Collect a non-secret, reproducible GPU/PyTorch performance environment record."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SAFE_ENV_KEYS = (
    "PYTHONHASHSEED",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_MODULE_LOADING",
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "MKL_NUM_THREADS",
    "MKL_DYNAMIC",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "KMP_AFFINITY",
    "KMP_BLOCKTIME",
    "GOMP_CPU_AFFINITY",
    "NCCL_DEBUG",
    "NCCL_P2P_DISABLE",
    "NCCL_IB_DISABLE",
    "TORCH_COMPILE_DEBUG",
    "TORCH_LOGS",
    "TORCH_DISTRIBUTED_DEBUG",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_ENABLE_TIMING",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCHINDUCTOR_CPP_WRAPPER",
    "TORCHINDUCTOR_FREEZING",
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUBLAS_WORKSPACE_CONFIG",
)


def cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "logical_cpus": os.cpu_count(),
        "load_average": None,
        "cpu_percent": None,
    }
    try:
        info["load_average"] = list(os.getloadavg())
    except (AttributeError, OSError):
        pass
    try:
        info["affinity"] = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    try:
        import psutil

        info["physical_cpus"] = psutil.cpu_count(logical=False)
        info["logical_cpus_psutil"] = psutil.cpu_count(logical=True)
        info["process_affinity"] = psutil.Process().cpu_affinity()
        try:
            info["load_average"] = list(psutil.getloadavg())
        except (AttributeError, OSError):
            pass
        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        info["available_memory_bytes"] = memory.available
        info["swap_used_bytes"] = swap.used
        info["swap_percent"] = swap.percent
    except (ImportError, AttributeError, OSError, RuntimeError):
        pass
    numa_root = Path("/sys/devices/system/node")
    if numa_root.is_dir():
        info["numa_nodes"] = len(list(numa_root.glob("node[0-9]*")))
    return info


def run(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "error": repr(exc)}

    info: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "debug_build": bool(getattr(torch.version, "debug", False)),
        "cuda_runtime": torch.version.cuda,
        "hip_runtime": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "capabilities": {
            "compile": callable(getattr(torch, "compile", None)),
            "export": hasattr(torch, "export"),
        },
    }
    try:
        info["build_config"] = torch.__config__.show()
    except Exception as exc:
        info["build_config_error"] = repr(exc)
    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        info["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    except Exception as exc:
        info["cudnn_error"] = repr(exc)

    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "multiprocessor_count": props.multi_processor_count,
                    "compute_capability": [props.major, props.minor],
                }
            )
        info["devices"] = devices
        try:
            info["allow_tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
            info["allow_tf32_cudnn"] = torch.backends.cudnn.allow_tf32
        except Exception as exc:
            info["tf32_error"] = repr(exc)
        try:
            info["nccl_version"] = torch.cuda.nccl.version()
        except Exception as exc:
            info["nccl_error"] = repr(exc)
        try:
            info["cuda_capabilities"] = {
                "flash_attention_available": torch.backends.cuda.is_flash_attention_available(),
                "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
                "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
            }
        except (AttributeError, RuntimeError) as exc:
            info["cuda_capabilities_error"] = repr(exc)
    return info


def optional_package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    packages = [
        "torch-geometric",
        "e3nn",
        "cuequivariance",
        "cuequivariance-torch",
        "triton",
        "torchao",
        "transformer-engine",
        "flash-attn",
        "xformers",
        "numpy",
        "scipy",
        "psutil",
    ]
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = str(version(name))
        except PackageNotFoundError:
            continue
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("environment.json"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo = args.repo.resolve()
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
        },
        "cpu": cpu_info(),
        "environment": {key: os.environ[key] for key in SAFE_ENV_KEYS if key in os.environ},
        "torch": torch_info(),
        "packages": optional_package_versions(),
        "git": {
            "root": run(["git", "rev-parse", "--show-toplevel"], repo),
            "commit": run(["git", "rev-parse", "HEAD"], repo),
            "status": run(["git", "status", "--short"], repo),
        },
    }
    if shutil.which("nvidia-smi"):
        record["nvidia_smi"] = run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total,power.limit,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        )
    else:
        record["nvidia_smi"] = {"available": False}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
