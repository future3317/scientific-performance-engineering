"""Full-device inventory and conservative paired exposure comparison."""

from __future__ import annotations

import subprocess
from typing import Any, Mapping, Sequence


def scan_all_gpus() -> list[dict[str, Any]]:
    query = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,pstate,power.draw,temperature.gpu,driver_version",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=2.0, check=True,
    )
    process_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=2.0, check=True,
    )
    processes: dict[str, list[dict[str, Any]]] = {}
    for line in process_query.stdout.splitlines():
        fields = [item.strip() for item in line.split(",", 3)]
        if len(fields) != 4:
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            pid = None
        processes.setdefault(fields[0], []).append({
            "pid": pid, "name": fields[2], "memory_mb": fields[3],
        })

    rows = []
    for line in query.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 11:
            raise RuntimeError(f"nvidia-smi GPU inventory row has {len(fields)} fields")
        try:
            index = int(fields[0])
        except ValueError as exc:
            raise RuntimeError(f"nvidia-smi returned invalid GPU index {fields[0]!r}") from exc
        rows.append({
            "index": index, "uuid": fields[1], "name": fields[2],
            "memory_total_mb": fields[3], "memory_used_mb": fields[4],
            "memory_free_mb": fields[5], "utilization_percent": fields[6],
            "pstate": fields[7], "power_w": fields[8], "temperature_c": fields[9],
            "driver_version": fields[10],
            "compute_processes": processes.get(fields[1], []),
        })
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU inventory rows")
    return sorted(rows, key=lambda row: row["index"])


def choose_idle_gpu(scan: Sequence[Mapping[str, Any]], *,
                    preferred_uuids: Sequence[str] = ()) -> Mapping[str, Any] | None:
    by_uuid = {str(row["uuid"]): row for row in scan}
    order = [by_uuid[uuid] for uuid in preferred_uuids if uuid in by_uuid]
    order.extend(row for row in scan if row not in order)
    for row in order:
        try:
            used = float(str(row["memory_used_mb"]).replace("MiB", "").strip())
            utilization = float(str(row["utilization_percent"]).replace("%", "").strip())
        except (TypeError, ValueError):
            continue
        if not row.get("compute_processes") and used <= 512 and utilization == 0:
            return row
    return None


def compare_arm_exposure(s0: Mapping[str, Any], s1: Mapping[str, Any]) -> dict[str, Any]:
    """Require complete same-device, uncontended windows for matched inference."""
    arm_rows = {}
    errors = []
    foreign = {}
    unattributed_activity = {}
    devices = {}
    for name, arm in (("S0", s0), ("S1", s1)):
        rows = arm.get("telemetry")
        rows = rows if isinstance(rows, list) else []
        snapshots = []
        for item in rows:
            if not isinstance(item, Mapping):
                errors.append(f"{name}:invalid_sample")
                continue
            if item.get("error"):
                errors.append(f"{name}:{item['error']}")
                continue
            if not item.get("gpu_uuid") or not isinstance(item.get("compute_processes"), list):
                errors.append(f"{name}:missing_device_or_process_telemetry")
                continue
            snapshots.append(item)
        if not snapshots:
            errors.append(f"{name}:no_complete_samples")
        arm_rows[name] = snapshots
        devices[name] = sorted({str(item["gpu_uuid"]) for item in snapshots})
        foreign[name] = [
            process
            for item in snapshots
            for process in item["compute_processes"]
            if process.get("origin") != "STUDY_PROCESS_GROUP"
        ]
        unattributed_activity[name] = [
            item for item in snapshots
            if _positive_utilization(item.get("utilization_percent"))
            and not item["compute_processes"]
        ]

    if errors:
        return {"matched": None, "reason": "TELEMETRY_INCOMPLETE",
                "errors": errors, "devices": devices,
                "foreign_process_count": {key: len(value) for key, value in foreign.items()}}
    if devices["S0"] != devices["S1"] or len(devices["S0"]) != 1:
        return {"matched": False, "reason": "PHYSICAL_GPU_UUID_MISMATCH",
                "devices": devices,
                "foreign_process_count": {key: len(value) for key, value in foreign.items()}}
    if any(foreign.values()):
        return {
            "matched": False,
            "reason": "CO_TENANT_OBSERVED_NOT_MATCHED",
            "devices": devices,
            "foreign_process_count": {key: len(value) for key, value in foreign.items()},
            "foreign_processes": foreign,
            "co_tenant_observed": {key: bool(value) for key, value in foreign.items()},
            "arm_sample_count": {key: len(value) for key, value in arm_rows.items()},
        }
    if any(unattributed_activity.values()):
        return {
            "matched": None,
            "reason": "GPU_ACTIVITY_WITHOUT_PROCESS_ATTRIBUTION",
            "devices": devices,
            "foreign_process_count": {key: len(value) for key, value in foreign.items()},
            "unattributed_activity_sample_count": {
                key: len(value) for key, value in unattributed_activity.items()
            },
        }
    return {
        "matched": True,
        "reason": "SAME_UUID_COMPLETE_UNCONTENDED_TELEMETRY",
        "devices": devices,
        "foreign_process_count": {key: len(value) for key, value in foreign.items()},
        "foreign_processes": foreign,
        "co_tenant_observed": {key: bool(value) for key, value in foreign.items()},
        "arm_sample_count": {key: len(value) for key, value in arm_rows.items()},
    }


def _positive_utilization(value: Any) -> bool:
    if value is None:
        return False
    try:
        return float(str(value).replace("%", "").strip()) > 0.0
    except ValueError:
        return False
