"""Pair-scoped GPU reservation, attribution, and worker lifecycle supervision."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

from ..real_intervention_v4.outcomes import atomic_json, canonical_digest
from ..real_intervention_v4.process_supervisor import (
    _enable_child_subreaper,
    _process_group_pids,
    _signal_group,
)
from .validity import classify_pair


SAMPLE_INTERVAL_S = 0.2
IDLE_MEMORY_LIMIT_MIB = 512
TERM_GRACE_S = 1.0
KILL_REAP_S = 1.0
# NVML's per-process utilization samples are emitted on a one-second window.
NVML_PROCESS_ATTRIBUTION_WINDOW_S = 1.0


def _read_proc() -> dict[int, dict[str, int | str]]:
    rows: dict[int, dict[str, int | str]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="ascii")
            tail = stat[stat.rfind(")") + 2 :].split()
            rows[int(entry.name)] = {
                "state": tail[0], "ppid": int(tail[1]), "pgid": int(tail[2]),
                "session_id": int(tail[3]), "start_ticks": int(tail[19]),
            }
        except (OSError, ValueError, IndexError):
            continue
    return rows


def _pid_tree(root_pid: int, rows: Mapping[int, Mapping[str, Any]]) -> list[dict[str, int | str]]:
    parents = {pid: int(row["ppid"]) for pid, row in rows.items()}
    owned = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in owned and pid not in owned:
                owned.add(pid)
                changed = True
    return [
        {"pid": pid, **dict(rows[pid])}
        for pid in sorted(owned) if pid in rows
    ]


def _nvml_value(nvml: Any, handle: Any, function: str, *args: Any) -> Any:
    try:
        return getattr(nvml, function)(handle, *args)
    except Exception as exc:
        unsupported = getattr(nvml, "NVMLError_NotSupported", ())
        if unsupported and isinstance(exc, unsupported):
            return "NOT_SUPPORTED"
        return f"UNAVAILABLE:{type(exc).__name__}:{exc}"


class _GpuMonitor:
    def __init__(self, gpu_uuid: str) -> None:
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid.encode("ascii"))
        self.last_process_sample_us = 0
        self.process_utilization_status = "AVAILABLE"

    def close(self) -> None:
        self.nvml.nvmlShutdown()

    def sample(self, owner_pid: int, pgid: int | None) -> dict[str, Any]:
        now_wall = time.time()
        now_mono = time.monotonic()
        nvml, handle = self.nvml, self.handle
        try:
            uuid_value = nvml.nvmlDeviceGetUUID(handle)
            uuid = uuid_value.decode("ascii") if isinstance(uuid_value, bytes) else str(uuid_value)
            processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
            process_rows = [
                {"pid": int(item.pid), "memory_bytes": item.usedGpuMemory}
                for item in processes
            ]
            process_query_error = None
        except Exception as exc:
            uuid = None
            process_rows = []
            process_query_error = f"{type(exc).__name__}:{exc}"

        memory = _nvml_value(nvml, handle, "nvmlDeviceGetMemoryInfo")
        utilization = _nvml_value(nvml, handle, "nvmlDeviceGetUtilizationRates")
        if memory in ("NOT_SUPPORTED",) or isinstance(memory, str):
            memory_used_mib = None
            memory_free_mib = None
        else:
            memory_used_mib = memory.used / (1024 * 1024)
            memory_free_mib = memory.free / (1024 * 1024)
        if utilization == "NOT_SUPPORTED" or isinstance(utilization, str):
            utilization_percent = None
        else:
            utilization_percent = int(utilization.gpu)

        process_utilization = "NOT_SUPPORTED"
        if self.process_utilization_status != "NOT_SUPPORTED":
            try:
                samples = nvml.nvmlDeviceGetProcessUtilization(
                    handle, self.last_process_sample_us,
                )
                process_utilization = [
                    {"pid": int(item.pid), "timestamp_us": int(item.timeStamp),
                     "sm_util_percent": int(item.smUtil), "memory_util_percent": int(item.memUtil)}
                    for item in samples
                ]
                if samples:
                    self.last_process_sample_us = max(int(item.timeStamp) for item in samples)
            except Exception as exc:
                unsupported = getattr(nvml, "NVMLError_NotSupported", ())
                no_samples = getattr(nvml, "NVMLError_NotFound", ())
                if unsupported and isinstance(exc, unsupported):
                    self.process_utilization_status = "NOT_SUPPORTED"
                    process_utilization = "NOT_SUPPORTED"
                elif no_samples and isinstance(exc, no_samples):
                    process_utilization = []
                else:
                    self.process_utilization_status = f"UNAVAILABLE:{type(exc).__name__}:{exc}"
                    process_utilization = self.process_utilization_status
        elif self.process_utilization_status.startswith("UNAVAILABLE:"):
            process_utilization = self.process_utilization_status

        optional = {
            "pstate": _nvml_value(nvml, handle, "nvmlDeviceGetPowerState"),
            "temperature_c": _nvml_value(
                nvml, handle, "nvmlDeviceGetTemperature", nvml.NVML_TEMPERATURE_GPU,
            ),
            "sm_clock_mhz": _nvml_value(
                nvml, handle, "nvmlDeviceGetClockInfo", nvml.NVML_CLOCK_SM,
            ),
            "memory_clock_mhz": _nvml_value(
                nvml, handle, "nvmlDeviceGetClockInfo", nvml.NVML_CLOCK_MEM,
            ),
            "power_w": _nvml_value(nvml, handle, "nvmlDeviceGetPowerUsage"),
        }
        proc_rows = _read_proc()
        owned = set(_process_group_pids(pgid)) if pgid is not None else set()
        if pgid is not None:
            owned.update(row["pid"] for row in _pid_tree(pgid, proc_rows))
        for process in process_rows:
            pid = process["pid"]
            process["origin"] = "STUDY_PROCESS_TREE" if pid in owned else (
                "EXTERNAL" if pid is not None else "UNRESOLVED"
            )
        if isinstance(process_utilization, list):
            for process in process_utilization:
                pid = process["pid"]
                process["origin"] = "STUDY_PROCESS_TREE" if pid in owned else (
                    "EXTERNAL" if pid is not None else "UNRESOLVED"
                )
        required_complete = (
            process_query_error is None
            and uuid is not None
            and memory_used_mib is not None
            and utilization_percent is not None
            and not self.process_utilization_status.startswith("UNAVAILABLE:")
        )
        return {
            "timestamp_unix": now_wall, "timestamp_monotonic": now_mono,
            "gpu_uuid": uuid, "memory_used_mib": memory_used_mib,
            "memory_free_mib": memory_free_mib,
            "utilization_percent": utilization_percent,
            "compute_processes": process_rows,
            "process_query_error": process_query_error,
            "process_utilization": process_utilization,
            "process_utilization_poll_interval_s": SAMPLE_INTERVAL_S,
            "process_utilization_attribution_window_s": NVML_PROCESS_ATTRIBUTION_WINDOW_S,
            "process_utilization_status": self.process_utilization_status,
            "optional_fields": optional,
            "owner_pid": owner_pid, "worker_pgid": pgid,
            "owner_pid_tree": _pid_tree(owner_pid, proc_rows),
            "worker_process_group_pids": sorted(_process_group_pids(pgid)) if pgid is not None else [],
            "telemetry_complete": required_complete,
        }


def _reservation_path(gpu_uuid: str) -> Path:
    safe_uuid = "".join(char if char.isalnum() or char in "-_" else "_" for char in gpu_uuid)
    return Path(tempfile.gettempdir()) / "evospe-v5-gpu-reservations" / f"{safe_uuid}.lock"


def sample_gpu(gpu_uuid: str) -> dict[str, Any]:
    """Return one NVML resource sample for the parent’s GPU selector."""
    monitor = _GpuMonitor(gpu_uuid)
    try:
        sample = monitor.sample(os.getpid(), None)
        sample["phase"] = "RESOURCE_SELECTION"
        return sample
    finally:
        monitor.close()


def _atomic_marker(path: Path, value: Mapping[str, Any]) -> None:
    atomic_json(path, dict(value), exclusive=True)


def _worker_execution(payload: Mapping[str, Any], returncode: int) -> str:
    value = str(payload.get("execution_status", payload.get("arm_execution", ""))).upper()
    if value in {"NOT_STARTED", "STARTED", "COMPLETE", "ERROR", "UNKNOWN"}:
        return value
    return "COMPLETE" if returncode == 0 else "ERROR"


def _worker_scientific(payload: Mapping[str, Any]) -> str:
    value = str(payload.get("scientific_status", payload.get("scientific_correctness", ""))).upper()
    if value in {"PASS", "NUMERICAL_UNSAFE", "UNVERIFIED"}:
        return value
    correctness = payload.get("correctness")
    if isinstance(correctness, Mapping):
        if correctness.get("passed") is True:
            return "PASS"
        if correctness.get("passed") is False:
            return "NUMERICAL_UNSAFE"
    return "UNVERIFIED"


def _session_result(*, arms: list[dict[str, Any]], samples: list[dict[str, Any]],
                    measurement_samples: list[dict[str, Any]],
                    gpu_uuid: str, started_wall: float, started_mono: float,
                    ended_wall: float, ended_mono: float, error_code: str | None = None,
                    first_failing_gate: str | None = None) -> dict[str, Any]:
    exposure = _exposure(measurement_samples, gpu_uuid)
    validity = classify_pair(arms, exposure)
    result = {
        **validity,
        "timestamp": ended_wall,
        "gpu_uuid": gpu_uuid,
        "arms": arms,
        "monitor": {
            "samples": samples,
            "measurement_sample_indexes": [
                index for index, sample in enumerate(samples)
                if sample.get("measurement_active") is True
            ],
            "measurement_active_samples": measurement_samples,
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "telemetry_complete": exposure["telemetry_complete"],
            "co_tenant_confirmed": exposure["co_tenant_confirmed"],
            "co_tenant_pids": exposure["co_tenant_pids"],
            "premeasurement_co_tenant_pids": sorted({
                int(process["pid"])
                for sample in samples if not sample.get("measurement_active")
                for process in sample.get("compute_processes", [])
                if process.get("origin") == "EXTERNAL" and process.get("pid") is not None
            }),
            "unattributed_activity": exposure["unattributed_activity"],
            "process_utilization_status": sorted({
                str(sample.get("process_utilization_status", "UNAVAILABLE")) for sample in samples
            }),
        },
        "exposure": exposure,
        "timings": {
            "session_wall_seconds": ended_mono - started_mono,
            "started_unix": started_wall, "ended_unix": ended_wall,
            "started_monotonic": started_mono, "ended_monotonic": ended_mono,
            "arm_parent_wall_seconds": [arm.get("parent_wall_seconds") for arm in arms],
        },
        "error_code": error_code or validity["error_code"],
        "raw_digest": canonical_digest({"arms": arms, "telemetry": samples}),
    }
    if first_failing_gate and result.get("first_failing_gate") is None:
        result["first_failing_gate"] = first_failing_gate
    return result


def _exposure(samples: Sequence[Mapping[str, Any]], gpu_uuid: str) -> dict[str, Any]:
    complete = bool(samples) and all(sample.get("telemetry_complete") is True for sample in samples)
    co_tenant_pids = sorted({
        int(process["pid"])
        for sample in samples
        for process in sample.get("compute_processes", [])
        if process.get("origin") == "EXTERNAL" and process.get("pid") is not None
    })
    co_tenant_pids = sorted(set(co_tenant_pids) | {
        int(process["pid"])
        for sample in samples
        for process in sample.get("process_utilization", [])
        if isinstance(process, Mapping) and process.get("origin") == "EXTERNAL"
        and process.get("pid") is not None
    })
    unknown_process = any(
        process.get("origin") == "UNRESOLVED"
        for sample in samples
        for process in [*sample.get("compute_processes", []),
                        *sample.get("process_utilization", [])]
        if isinstance(process, Mapping)
    )
    unattributed_activity = any(
        (sample.get("utilization_percent") or 0) > 0
        and not sample.get("compute_processes")
        and sample.get("activity_attribution") != "WORKER_LIFECYCLE_WITHIN_NVML_WINDOW"
        and not any(
            isinstance(process, Mapping)
            and process.get("origin") == "STUDY_PROCESS_TREE"
            and (process.get("sm_util_percent") or 0) > 0
            for process in sample.get("process_utilization", [])
        )
        for sample in samples
    )
    devices = {sample.get("gpu_uuid") for sample in samples}
    complete = complete and not unknown_process
    if co_tenant_pids:
        status = "CONTENDED"
    elif not complete or unattributed_activity:
        status = "INCOMPLETE"
    elif devices != {gpu_uuid}:
        status = "UNKNOWN"
    else:
        status = "MATCHED"
    return {
        "status": status,
        "telemetry_complete": complete,
        "co_tenant_confirmed": bool(co_tenant_pids),
        "co_tenant_pids": co_tenant_pids,
        "unattributed_activity": unattributed_activity,
        "gpu_uuids": sorted(str(value) for value in devices if value is not None),
    }


def _substitute(command: Sequence[str], output: Path, ack: Path) -> list[str]:
    if not any("{output}" in item for item in command) or not any("{ack}" in item for item in command):
        raise ValueError("each arm command must include {output} and {ack} placeholders")
    return [item.replace("{output}", str(output)).replace("{ack}", str(ack)) for item in command]


def run_session(commands: list[list[str]], session_dir: Path, gpu_uuid: str,
                timeout_s: float, env: dict[str, str]) -> dict[str, Any]:
    """Run one reserved session of one to four arms; identities are never replayed.

    Each command receives per-arm result and ack paths through `{output}` and
    `{ack}` substitutions. The worker must write `<output stem>.ready.json`
    containing its PID and physical GPU UUID, then wait for `<output stem>.go`.
    The parent only opens that gate after NVML attributes the worker PID to the
    reserved UUID. The worker writes its result and remains alive until ACK.
    """
    if not 1 <= len(commands) <= 4 or any(not command for command in commands):
        raise ValueError("a session requires 1 to 4 non-empty worker commands")
    if not gpu_uuid or timeout_s <= 0:
        raise ValueError("gpu_uuid and a positive timeout_s are required")
    session_dir = Path(session_dir).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    started_wall, started_mono = time.time(), time.monotonic()
    deadline = started_mono + timeout_s
    reservation = _reservation_path(gpu_uuid)
    reservation.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(reservation, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"GPU {gpu_uuid} already has a V5 session reservation") from exc
        intent_path = session_dir / "intent.json"
        if any(path.exists() for path in (intent_path, session_dir / "terminal.json")):
            raise RuntimeError("session identity already has lifecycle records; refusing replay")
        expected_files = [session_dir / f"arm-{index}.result.json" for index in range(len(commands))]
        for index, output in enumerate(expected_files):
            for path in (output, output.with_suffix(".ready.json"), output.with_suffix(".go"),
                         session_dir / f"arm-{index}.ack",
                         output.with_name(output.stem + ".stdout.log"),
                         output.with_name(output.stem + ".stderr.log")):
                if path.exists():
                    raise RuntimeError(f"session worker artifact already exists: {path}")

        monitor: _GpuMonitor | None = None
        try:
            monitor = _GpuMonitor(gpu_uuid)
        except Exception as exc:
            result = _session_result(
                arms=[{"execution_status": "NOT_STARTED", "scientific_status": "UNVERIFIED"}
                      for _ in commands], samples=[], measurement_samples=[], gpu_uuid=gpu_uuid,
                started_wall=started_wall, started_mono=started_mono,
                ended_wall=time.time(), ended_mono=time.monotonic(),
                error_code=f"NVML_INITIALIZATION:{type(exc).__name__}",
            )
            result["execution_status"] = "NOT_STARTED"
            result["platform_status"] = "UNCERTAIN"
            atomic_json(session_dir / "terminal.json", result, exclusive=True)
            return result

        samples: list[dict[str, Any]] = []
        measurement_samples: list[dict[str, Any]] = []
        arms: list[dict[str, Any]] = []
        process: subprocess.Popen[bytes] | None = None
        log_handles = []
        pgid: int | None = None
        error_code: str | None = None
        first_failing_gate: str | None = None
        last_worker_attribution_mono: float | None = None

        def record_sample(sample: dict[str, Any], phase: str, *, worker_pid: int | None = None,
                          measurement_active: bool = False) -> None:
            nonlocal last_worker_attribution_mono
            sample["phase"] = phase
            sample["measurement_active"] = measurement_active
            owned_process = any(
                row.get("pid") == worker_pid and row.get("origin") == "STUDY_PROCESS_TREE"
                for row in sample.get("compute_processes", [])
            ) if worker_pid is not None else False
            owned_utilization = any(
                row.get("pid") == worker_pid and row.get("origin") == "STUDY_PROCESS_TREE"
                for row in sample.get("process_utilization", [])
                if isinstance(row, Mapping)
            ) if worker_pid is not None else False
            if owned_process or owned_utilization:
                last_worker_attribution_mono = float(sample["timestamp_monotonic"])
            if (measurement_active and worker_pid is not None
                    and process is not None and process.pid == worker_pid
                    and process.poll() is None
                    and (sample.get("utilization_percent") or 0) > 0
                    and not sample.get("compute_processes")
                    and last_worker_attribution_mono is not None
                    and float(sample["timestamp_monotonic"]) - last_worker_attribution_mono
                    <= NVML_PROCESS_ATTRIBUTION_WINDOW_S):
                sample["activity_attribution"] = "WORKER_LIFECYCLE_WITHIN_NVML_WINDOW"
                sample["attribution_window_s"] = NVML_PROCESS_ATTRIBUTION_WINDOW_S
            samples.append(sample)
            if measurement_active:
                measurement_samples.append(sample)

        def take_sample(phase: str, pgid: int | None, worker_pid: int | None = None,
                        measurement_active: bool = False) -> dict[str, Any]:
            try:
                sample = monitor.sample(os.getpid(), pgid)
            except Exception as exc:
                sample = {
                    "timestamp_unix": time.time(), "timestamp_monotonic": time.monotonic(),
                    "gpu_uuid": gpu_uuid, "compute_processes": [],
                    "process_utilization_status": "UNAVAILABLE",
                    "telemetry_complete": False,
                    "error": f"{type(exc).__name__}:{exc}",
                }
            record_sample(sample, phase, worker_pid=worker_pid,
                          measurement_active=measurement_active)
            return sample

        try:
            _enable_child_subreaper()
            while time.monotonic() < deadline:
                sample = take_sample("WAITING_FOR_IDLE", None)
                if not sample["telemetry_complete"]:
                    error_code = "GPU_TELEMETRY_INCOMPLETE_BEFORE_START"
                    break
                processes = sample["compute_processes"]
                used = sample["memory_used_mib"]
                idle = not processes and used <= IDLE_MEMORY_LIMIT_MIB and sample["utilization_percent"] == 0
                if idle:
                    break
                time.sleep(min(SAMPLE_INTERVAL_S, max(0.0, deadline - time.monotonic())))
            else:
                error_code = "GPU_IDLE_WAIT_TIMEOUT"

            if error_code is None:
                initial = samples[-1]
                if initial.get("compute_processes") or initial.get("memory_used_mib") > IDLE_MEMORY_LIMIT_MIB:
                    error_code = "GPU_NO_LONGER_IDLE_AT_SESSION_START"

            if error_code is None:
                _atomic_marker(intent_path, {
                    "gpu_uuid": gpu_uuid, "commands_digest": canonical_digest(commands),
                    "created_unix": time.time(), "created_monotonic": time.monotonic(),
                    "timeout_s": timeout_s,
                })
                child_env = os.environ.copy()
                child_env.update(env)
                child_env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
                for index, command in enumerate(commands):
                    last_worker_attribution_mono = None
                    if time.monotonic() >= deadline:
                        error_code = "SESSION_DEADLINE_BEFORE_ARM"
                        arms.append({"execution_status": "NOT_STARTED", "scientific_status": "UNVERIFIED"})
                        break
                    output = session_dir / f"arm-{index}.result.json"
                    ack = session_dir / f"arm-{index}.ack"
                    ready = output.with_suffix(".ready.json")
                    go = output.with_suffix(".go")
                    stdout_path = output.with_name(output.stem + ".stdout.log")
                    stderr_path = output.with_name(output.stem + ".stderr.log")
                    arm_started = time.monotonic()
                    stdout_log = stdout_path.open("xb")
                    log_handles.append(stdout_log)
                    stderr_log = stderr_path.open("xb")
                    log_handles.append(stderr_log)
                    try:
                        process = subprocess.Popen(
                            _substitute(command, output, ack), env=child_env,
                            start_new_session=True, stdout=stdout_log, stderr=stderr_log,
                        )
                    except OSError as exc:
                        error_code = f"WORKER_LAUNCH:{type(exc).__name__}"
                        arms.append({
                            "execution_status": "ERROR", "scientific_status": "UNVERIFIED",
                            "arm_identity": {"session_id": session_dir.name, "arm_index": index,
                                             "gpu_uuid": gpu_uuid},
                            "parent_wall_seconds": time.monotonic() - arm_started,
                            "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
                            "error_code": error_code,
                        })
                        break
                    pgid = process.pid
                    ready_payload: dict[str, Any] | None = None
                    arm_samples_start = len(samples)
                    while time.monotonic() < deadline:
                        sample = take_sample("WAITING_FOR_WORKER_ATTRIBUTION", pgid, process.pid)
                        if not sample["telemetry_complete"]:
                            error_code = "GPU_TELEMETRY_INCOMPLETE_DURING_ARM"
                            break
                        if ready.is_file():
                            try:
                                ready_payload = json.loads(ready.read_text(encoding="utf-8"))
                            except (OSError, json.JSONDecodeError):
                                error_code = "WORKER_READY_INVALID"
                                break
                            own_pids = {item.get("pid") for item in sample["compute_processes"]
                                        if item.get("origin") == "STUDY_PROCESS_TREE"}
                            foreign = [item for item in sample["compute_processes"]
                                       if item.get("origin") == "EXTERNAL"]
                            if foreign:
                                error_code = "GPU_CONTENDED_BEFORE_ARM_GO"
                                break
                            if (ready_payload.get("pid") != process.pid
                                    or ready_payload.get("gpu_uuid") != gpu_uuid
                                    or process.pid not in own_pids):
                                error_code = "WORKER_GPU_ATTRIBUTION_FAILED"
                                break
                            _atomic_marker(go, {"created_unix": time.time(), "pid": process.pid})
                            break
                        if process.poll() is not None:
                            error_code = "WORKER_EXITED_BEFORE_READY"
                            break
                        time.sleep(min(SAMPLE_INTERVAL_S, max(0.0, deadline - time.monotonic())))
                    if error_code is not None or ready_payload is None:
                        if error_code is None:
                            error_code = "WORKER_READY_TIMEOUT"
                        _terminate_group(process, pgid)
                        arms.append({
                            "execution_status": "UNKNOWN",
                            "scientific_status": "UNVERIFIED",
                            "arm_identity": {"session_id": session_dir.name, "arm_index": index,
                                             "gpu_uuid": gpu_uuid},
                            "parent_wall_seconds": time.monotonic() - arm_started,
                            "worker_pid": process.pid, "worker_pgid": pgid,
                            "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
                            "error_code": error_code,
                        })
                        break

                    payload: dict[str, Any] | None = None
                    while time.monotonic() < deadline:
                        sample = take_sample("MEASUREMENT_ACTIVE", pgid, process.pid, True)
                        if not sample["telemetry_complete"]:
                            error_code = "GPU_TELEMETRY_INCOMPLETE_DURING_ARM"
                        if output.is_file():
                            try:
                                payload = json.loads(output.read_text(encoding="utf-8"))
                                if not isinstance(payload, dict):
                                    raise ValueError("worker result must be a JSON object")
                            except (OSError, ValueError, json.JSONDecodeError):
                                error_code = "WORKER_RESULT_INVALID"
                            break
                        if process.poll() is not None:
                            error_code = "WORKER_EXITED_WITHOUT_RESULT"
                            break
                        time.sleep(min(SAMPLE_INTERVAL_S, max(0.0, deadline - time.monotonic())))

                    if payload is None:
                        error_code = error_code or "WORKER_TIMEOUT"
                        _terminate_group(process, pgid)
                        arms.append({
                            "execution_status": "UNKNOWN",
                            "scientific_status": "UNVERIFIED",
                            "arm_identity": {"session_id": session_dir.name, "arm_index": index,
                                             "gpu_uuid": gpu_uuid},
                            "parent_wall_seconds": time.monotonic() - arm_started,
                            "worker_pid": process.pid, "worker_pgid": pgid,
                            "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
                            "error_code": error_code,
                        })
                        break

                    # Keep the CUDA worker alive and attributable until this ACK.
                    sample = take_sample("MEASUREMENT_ACTIVE_RESULT_BOUNDARY", pgid,
                                         process.pid, True)
                    ack_sent = process.poll() is None
                    if not ack_sent:
                        error_code = error_code or "WORKER_EXITED_BEFORE_ACK"
                    else:
                        _atomic_marker(ack, {"created_unix": time.time(), "pid": process.pid})
                        try:
                            process.wait(timeout=max(0.0, deadline - time.monotonic()))
                        except subprocess.TimeoutExpired:
                            error_code = "WORKER_ACK_REAP_TIMEOUT"
                            _terminate_group(process, pgid)
                    ended = time.monotonic()
                    if not ack_sent or error_code == "WORKER_ACK_REAP_TIMEOUT":
                        execution_status = "UNKNOWN"
                    elif process.returncode != 0:
                        execution_status = "ERROR"
                    else:
                        execution_status = _worker_execution(payload, process.returncode)
                    arm = {
                        "execution_status": execution_status,
                        "scientific_status": _worker_scientific(payload),
                        "platform_status": "HEALTHY" if process.returncode == 0 else "PLATFORM_FAILURE",
                        "arm_identity": {"session_id": session_dir.name, "arm_index": index,
                                         "gpu_uuid": gpu_uuid},
                        "worker_result": payload,
                        "worker_pid": process.pid, "worker_pgid": pgid,
                        "worker_ready": ready_payload,
                        "worker_started_monotonic": arm_started,
                        "worker_ended_monotonic": ended,
                        "parent_wall_seconds": ended - arm_started,
                        "result_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                        "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
                        "monitor_sample_count": len(samples) - arm_samples_start,
                    }
                    arms.append(arm)
                    process = None
                    pgid = None
                    if error_code is not None:
                        break
            if len(arms) < len(commands):
                arms.extend({"execution_status": "NOT_STARTED", "scientific_status": "UNVERIFIED"}
                            for _ in range(len(commands) - len(arms)))
        finally:
            if process is not None and process.poll() is None and pgid is not None:
                _terminate_group(process, pgid)
            for stream in log_handles:
                if not stream.closed:
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
            monitor.close()

        result = _session_result(
            arms=arms, samples=samples, measurement_samples=measurement_samples,
            gpu_uuid=gpu_uuid,
            started_wall=started_wall, started_mono=started_mono,
            ended_wall=time.time(), ended_mono=time.monotonic(),
            error_code=error_code, first_failing_gate=first_failing_gate,
        )
        atomic_json(session_dir / "terminal.json", result, exclusive=True)
        return result
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _terminate_group(process: subprocess.Popen[bytes], pgid: int) -> None:
    _signal_group(pgid, signal.SIGTERM)
    term_deadline = time.monotonic() + TERM_GRACE_S
    while time.monotonic() < term_deadline:
        if not _process_group_pids(pgid):
            break
        try:
            process.wait(timeout=min(0.05, max(0.0, term_deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            pass
    if _process_group_pids(pgid):
        _signal_group(pgid, signal.SIGKILL)
    try:
        process.wait(timeout=KILL_REAP_S)
    except subprocess.TimeoutExpired:
        pass
