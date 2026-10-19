"""Observed worker runtime identity; missing physical-device proof fails closed."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from typing import Any


_CROSS_PROCESS_FIELDS = (
    "device_uuid", "device_name", "device_capability", "torch_version",
    "torch_cuda_version", "cudnn_version", "packages", "driver", "python_version",
)


def _gpu_uuid_matches(observed: Any, expected: Any) -> bool:
    if not isinstance(observed, str) or not isinstance(expected, str):
        return False
    observed_uuid = observed.removeprefix("GPU-")
    expected_uuid = expected.removeprefix("GPU-")
    return bool(observed_uuid) and observed_uuid == expected_uuid


def cross_process_runtime_differences(reference: dict[str, Any],
                                     observed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Compare stable hardware/software identity while ignoring PID and thread env."""
    differences = {}
    for name in _CROSS_PROCESS_FIELDS:
        if name not in reference or name not in observed or reference.get(name) != observed.get(name):
            differences[name] = {
                "reference_present": name in reference,
                "observed_present": name in observed,
                "reference": reference.get(name),
                "observed": observed.get(name),
            }
    return differences


def selected_gpu_identity_differences(runtime: dict[str, Any], expected_gpu_uuid: str
                                      ) -> dict[str, dict[str, Any]]:
    driver = runtime.get("driver")
    driver = driver if isinstance(driver, dict) else {}
    expected = {"device_uuid": expected_gpu_uuid, "driver.gpu_uuid": expected_gpu_uuid}
    observed = {"device_uuid": runtime.get("device_uuid"), "driver.gpu_uuid": driver.get("gpu_uuid")}
    return {
        name: {"expected": expected[name], "observed": observed[name]}
        for name in expected if not _gpu_uuid_matches(observed[name], expected[name])
    }


def capture_runtime(expected_gpu_uuid: str) -> dict[str, Any]:
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != expected_gpu_uuid:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r} does not equal selected UUID")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected worker")
    torch.cuda.init()
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    device_index = torch.cuda.current_device()
    if device_index != 0:
        raise RuntimeError(f"single-GPU worker selected logical CUDA device {device_index}, expected 0")
    properties = torch.cuda.get_device_properties(device_index)

    query = subprocess.run(
        ["nvidia-smi", f"--id={expected_gpu_uuid}",
         "--query-gpu=uuid,driver_version,name,pci.bus_id,memory.total,power.limit",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=1.0, check=True,
    )
    fields = [item.strip() for item in query.stdout.strip().split(",")]
    if len(fields) != 6 or fields[0] != expected_gpu_uuid:
        raise RuntimeError("nvidia-smi did not confirm the selected physical GPU UUID")

    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=1.0, check=True,
    )
    current_pid_devices: set[str] = set()
    for line in processes.stdout.splitlines():
        values = [item.strip() for item in line.split(",", 3)]
        if len(values) == 4 and values[1] == str(os.getpid()):
            current_pid_devices.add(values[0])
    if current_pid_devices != {expected_gpu_uuid}:
        raise RuntimeError(
            f"nvidia-smi process map for worker PID {os.getpid()} is "
            f"{sorted(current_pid_devices)}, expected only {expected_gpu_uuid}"
        )

    property_uuid = getattr(properties, "uuid", None)
    if property_uuid is not None and not _gpu_uuid_matches(str(property_uuid), expected_gpu_uuid):
        raise RuntimeError(f"PyTorch device UUID {property_uuid} differs from selected UUID")

    packages: dict[str, str | None] = {}
    missing_packages: list[str] = []
    for package in ("torch", "torchvision", "triton", "timm", "numpy", "PyYAML"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
            missing_packages.append(package)
    driver = {
        "gpu_uuid": fields[0], "driver_version": fields[1], "device_name": fields[2],
        "pci_bus_id": fields[3], "memory_total_mb": fields[4], "power_limit_w": fields[5],
    }
    return {
        "device_uuid": expected_gpu_uuid,
        "device_name": torch.cuda.get_device_name(device_index),
        "device_capability": list(torch.cuda.get_device_capability(device_index)),
        "device_index": device_index,
        "cuda_visible_devices": visible,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "packages": packages,
        "missing_packages": missing_packages,
        "driver": driver,
        "python_version": sys.version,
        "pid": os.getpid(),
        "thread_settings": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS")
        },
    }
