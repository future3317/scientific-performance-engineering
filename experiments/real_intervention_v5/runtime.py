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
import threading
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
# Sampling granularity used when interpreting NVML's timestamped process data;
# this is an assumption, not a guarantee of the NVML API.
PROCESS_UTILIZATION_GRANULARITY_ASSUMPTION_S = 1.0


def _read_proc() -> dict[int, dict[str, int | str]]:
    rows: dict[int, dict[str, int | str]] = {}
    page_size = os.sysconf("SC_PAGE_SIZE")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="ascii")
            tail = stat[stat.rfind(")") + 2 :].split()
            rows[int(entry.name)] = {
                "state": tail[0], "ppid": int(tail[1]), "pgid": int(tail[2]),
                "session_id": int(tail[3]), "start_ticks": int(tail[19]),
                "utime_ticks": int(tail[11]), "stime_ticks": int(tail[12]),
                "rss_bytes": int(tail[21]) * page_size,
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


def _attribute_process_rows(rows: list[dict[str, Any]], active_pids: set[int],
                            study_pid_history: set[int] | None = None) -> None:
    for process in rows:
        pid = process["pid"]
        if pid in active_pids:
            process["origin"] = "STUDY_PROCESS_TREE"
        elif study_pid_history is not None and pid in study_pid_history:
            process["origin"] = "STUDY_PROCESS_HISTORY"
        else:
            process["origin"] = "EXTERNAL" if pid is not None else "UNRESOLVED"


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
        self.study_pid_history: set[int] = set()

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

        power_mw = _nvml_value(nvml, handle, "nvmlDeviceGetPowerUsage")
        power_limit_mw = _nvml_value(nvml, handle, "nvmlDeviceGetEnforcedPowerLimit")
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
            "power_w": power_mw / 1000 if isinstance(power_mw, (int, float)) else power_mw,
            "power_limit_w": power_limit_mw / 1000
            if isinstance(power_limit_mw, (int, float)) else power_limit_mw,
        }
        proc_rows = _read_proc()
        owned = set(_process_group_pids(pgid)) if pgid is not None else set()
        if pgid is not None:
            group_roots = {
                pid for pid, row in proc_rows.items()
                if row["pgid"] == pgid and row["state"] != "Z"
            }
            for root_pid in group_roots:
                owned.update(row["pid"] for row in _pid_tree(root_pid, proc_rows))
        _attribute_process_rows(process_rows, owned)
        self.study_pid_history.update(owned)
        if isinstance(process_utilization, list):
            _attribute_process_rows(process_utilization, owned, self.study_pid_history)
        owner_tree = _pid_tree(owner_pid, proc_rows)
        worker_tree = [
            {"pid": pid, **dict(proc_rows[pid])}
            for pid in sorted(owned) if pid in proc_rows
        ]
        clock_ticks = os.sysconf("SC_CLK_TCK")
        worker_cpu_ticks = sum(
            int(row["utime_ticks"]) + int(row["stime_ticks"]) for row in worker_tree
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
            "process_utilization_granularity_assumption_s": PROCESS_UTILIZATION_GRANULARITY_ASSUMPTION_S,
            "process_utilization_status": self.process_utilization_status,
            "optional_fields": optional,
            "owner_pid": owner_pid, "worker_pgid": pgid,
            "owner_pid_tree": owner_tree,
            "owner_tree_cpu_seconds_lower_bound": sum(
                int(row["utime_ticks"]) + int(row["stime_ticks"]) for row in owner_tree
            ) / clock_ticks,
            "worker_process_tree": worker_tree,
            "worker_tree_cpu_seconds_lower_bound": worker_cpu_ticks / clock_ticks,
            "worker_tree_rss_bytes": sum(int(row["rss_bytes"]) for row in worker_tree),
            "cpu_clock_ticks_per_second": clock_ticks,
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
    # Resource-selection and pre-GO samples are not session exposure.
    if any("measurement_active" in sample for sample in samples):
        samples = [sample for sample in samples if sample.get("measurement_active") is True]
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
    previous_sigterm_handler = None
    sigterm_handler_installed = False
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"GPU {gpu_uuid} already has a V5 session reservation") from exc
        intent_path = session_dir / "intent.json"
        if any(path.exists() for path in (intent_path, session_dir / "terminal.json")):
            raise RuntimeError("session identity already has lifecycle records; refusing replay")
        expected_files = [session_dir / f"arm-{index}.result.json" for index in range(len(commands))]
        for path in (session_dir / "heartbeat.json", session_dir / "telemetry.jsonl"):
            if path.exists():
                raise RuntimeError(f"session lifecycle artifact already exists: {path}")
        for index, output in enumerate(expected_files):
            for path in (output, output.with_suffix(".ready.json"), output.with_suffix(".go"),
                         session_dir / f"arm-{index}.ack",
                         session_dir / f"arm-{index}.started.json",
                         session_dir / f"arm-{index}.exit.json",
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
        active_arm_index: int | None = None
        log_handles = []
        telemetry_stream = None
        last_telemetry_sync_mono: float | None = None
        last_heartbeat_mono: float | None = None
        pgid: int | None = None
        error_code: str | None = None
        first_failing_gate: str | None = None

        def write_telemetry(sample: Mapping[str, Any], sample_index: int) -> str:
            nonlocal last_telemetry_sync_mono
            record = {**sample, "sample_index": sample_index}
            record["raw_digest"] = canonical_digest(record)
            telemetry_stream.write(
                (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            )
            now = float(sample["timestamp_monotonic"])
            if last_telemetry_sync_mono is None or now - last_telemetry_sync_mono >= 1.0:
                os.fsync(telemetry_stream.fileno())
                last_telemetry_sync_mono = now
            return str(record["raw_digest"])

        def write_heartbeat(sample: Mapping[str, Any], sample_index: int,
                            telemetry_digest: str) -> None:
            nonlocal last_heartbeat_mono
            heartbeat = {
                "timestamp_unix": sample["timestamp_unix"],
                "timestamp_monotonic": sample["timestamp_monotonic"],
                "sample_index": sample_index,
                "phase": sample["phase"],
                "worker_pid": process.pid if process is not None else None,
                "worker_pgid": pgid,
                "telemetry_digest": telemetry_digest,
            }
            heartbeat["raw_digest"] = canonical_digest(heartbeat)
            atomic_json(session_dir / "heartbeat.json", heartbeat)
            last_heartbeat_mono = float(sample["timestamp_monotonic"])

        def record_sample(sample: dict[str, Any], phase: str,
                          measurement_active: bool = False) -> None:
            nonlocal last_heartbeat_mono
            sample["phase"] = phase
            sample["measurement_active"] = measurement_active
            samples.append(sample)
            if measurement_active:
                measurement_samples.append(sample)
            if telemetry_stream is not None:
                sample_index = len(samples) - 1
                telemetry_digest = write_telemetry(sample, sample_index)
                if (last_heartbeat_mono is None
                        or float(sample["timestamp_monotonic"]) - last_heartbeat_mono >= 1.0):
                    write_heartbeat(sample, sample_index, telemetry_digest)

        def write_exit(arm_index: int, worker: subprocess.Popen[bytes] | None,
                       worker_pgid: int | None, phase: str,
                       launch_error: str | None = None) -> None:
            remaining = sorted(_process_group_pids(worker_pgid)) if worker_pgid is not None else []
            event = {
                "arm_index": arm_index,
                "pid": worker.pid if worker is not None else None,
                "pgid": worker_pgid,
                "returncode": worker.poll() if worker is not None else None,
                "owned_group_remaining": remaining,
                "phase": phase,
                "launch_error": launch_error,
                "timestamp_unix": time.time(),
                "timestamp_monotonic": time.monotonic(),
            }
            event["raw_digest"] = canonical_digest(event)
            _atomic_marker(session_dir / f"arm-{arm_index}.exit.json", event)

        def take_sample(phase: str, pgid: int | None,
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
            record_sample(sample, phase, measurement_active)
            return sample

        try:
            if threading.current_thread() is threading.main_thread():
                previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
                first_sigterm = [True]

                def interrupt_session(signum: int, frame: Any) -> None:
                    if first_sigterm[0]:
                        first_sigterm[0] = False
                        raise InterruptedError("V5 session interrupted by SIGTERM")

                sigterm_handler_installed = True
                signal.signal(signal.SIGTERM, interrupt_session)
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
                intent = {
                    "gpu_uuid": gpu_uuid, "commands_digest": canonical_digest(commands),
                    "created_unix": time.time(), "created_monotonic": time.monotonic(),
                    "timeout_s": timeout_s,
                }
                intent["raw_digest"] = canonical_digest(intent)
                _atomic_marker(intent_path, intent)
                telemetry_fd = os.open(session_dir / "telemetry.jsonl",
                                       os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                telemetry_stream = os.fdopen(telemetry_fd, "wb", buffering=0)
                for sample_index, sample in enumerate(samples):
                    telemetry_digest = write_telemetry(sample, sample_index)
                if samples:
                    write_heartbeat(samples[-1], len(samples) - 1, telemetry_digest)
                child_env = os.environ.copy()
                child_env.update(env)
                child_env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
                for index, command in enumerate(commands):
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
                        write_exit(index, None, None, "LAUNCH_FAILED", error_code)
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
                    active_arm_index = index
                    started = {
                        "arm_index": index,
                        "pid": process.pid,
                        "pgid": pgid,
                        "timestamp_unix": time.time(),
                        "timestamp_monotonic": time.monotonic(),
                    }
                    started["raw_digest"] = canonical_digest(started)
                    _atomic_marker(session_dir / f"arm-{index}.started.json", started)
                    ready_payload: dict[str, Any] | None = None
                    arm_samples_start = len(samples)
                    while time.monotonic() < deadline:
                        sample = take_sample("WAITING_FOR_WORKER_ATTRIBUTION", pgid)
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
                        sample = take_sample("MEASUREMENT_ACTIVE", pgid, True)
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
                    sample = take_sample("MEASUREMENT_ACTIVE_RESULT_BOUNDARY", pgid, True)
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
                    if process.poll() is not None and _process_group_pids(pgid):
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
                    write_exit(index, process, pgid, "ACKED" if ack_sent else "EXITED_BEFORE_ACK")
                    process = None
                    pgid = None
                    active_arm_index = None
                    if error_code is not None:
                        break
            if len(arms) < len(commands):
                arms.extend({"execution_status": "NOT_STARTED", "scientific_status": "UNVERIFIED"}
                            for _ in range(len(commands) - len(arms)))
        finally:
            try:
                if process is not None:
                    if process.poll() is None and pgid is not None:
                        _terminate_group(process, pgid)
                    if active_arm_index is not None:
                        write_exit(active_arm_index, process, pgid, "SESSION_FINALLY")
            finally:
                for stream in log_handles:
                    if not stream.closed:
                        stream.flush()
                        os.fsync(stream.fileno())
                        stream.close()
                if telemetry_stream is not None and not telemetry_stream.closed:
                    telemetry_stream.flush()
                    os.fsync(telemetry_stream.fileno())
                    telemetry_stream.close()
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
        if sigterm_handler_installed:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
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
