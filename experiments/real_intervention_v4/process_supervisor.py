"""Deadline-bounded worker execution with process-group cleanup and receipts."""

from __future__ import annotations

import hashlib
import json
import ctypes
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from fcntl import LOCK_EX, LOCK_NB, LOCK_UN, flock
from typing import Any, Mapping, Sequence

from .outcomes import ArmExecution, atomic_json, canonical_digest, record_event, recover_attempt


def _process_group_pids(group_id: int) -> set[int]:
    pids: set[int] = set()
    proc = Path("/proc")
    try:
        entries = list(proc.iterdir())
    except OSError:
        return pids
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text(encoding="ascii")
            tail = text[text.rfind(")") + 2:].split()
            if tail[0] != "Z" and int(tail[2]) == group_id:
                pids.add(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return pids


def _gpu_snapshot(gpu_uuid: str, group_id: int) -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", f"--id={gpu_uuid}",
         "--query-gpu=uuid,memory.used,memory.free,utilization.gpu,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=0.5,
    )
    if query.returncode:
        return {"error": query.stderr.strip() or "gpu_query_failed", "gpu_uuid": gpu_uuid,
                "sample_unix": time.time()}
    fields = [item.strip() for item in query.stdout.strip().split(",")]
    if len(fields) != 9:
        return {"error": "invalid_gpu_query_fields", "gpu_uuid": gpu_uuid,
                "sample_unix": time.time()}
    process_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=0.5,
    )
    pids = _process_group_pids(group_id)
    processes = []
    if process_query.returncode == 0:
        for line in process_query.stdout.splitlines():
            values = [item.strip() for item in line.split(",", 3)]
            if len(values) == 4 and values[0] == fields[0]:
                try:
                    pid = int(values[1])
                except ValueError:
                    pid = None
                processes.append({"gpu_uuid": values[0], "pid": pid, "name": values[2],
                                  "memory_mb": values[3],
                                  "origin": "STUDY_PROCESS_GROUP" if pid in pids else "EXTERNAL_OR_UNRESOLVED"})
    else:
        return {"error": "compute_process_query_failed", "process_query_error": process_query.stderr.strip(),
                "gpu_uuid": fields[0], "sample_unix": time.time()}
    names = ("gpu_uuid", "memory_used_mb", "memory_free_mb", "utilization_percent", "sm_clock_mhz",
             "memory_clock_mhz", "pstate", "power_w", "temperature_c")
    return {**dict(zip(names, fields)), "compute_processes": processes,
            "study_process_group_pids": sorted(pids), "sample_unix": time.time()}


def _signal_group(group_id: int, sig: int) -> bool:
    try:
        os.killpg(group_id, sig)
    except ProcessLookupError:
        return False
    return True


def _enable_child_subreaper() -> None:
    """Adopt orphaned worker descendants so their process groups stay attributable."""
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    if prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _try_cache_scope_lock(folder: Path) -> int | None:
    """Coordinate cache materialization with starting this attempt's worker."""
    fd = os.open(folder / "CACHE_SCOPE.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        flock(fd, LOCK_EX | LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise
    return fd


def _release_cache_scope_lock(fd: int) -> None:
    try:
        flock(fd, LOCK_UN)
    finally:
        os.close(fd)


def _prelaunch_artifacts_allowed(folder: Path, spec: Mapping[str, Any]) -> bool:
    entries = list(folder.iterdir())
    if not entries:
        return True
    if {path.name for path in entries} == {"CACHE_SCOPE.lock"}:
        return True
    if {path.name for path in entries} - {"cache", "CACHE_SCOPE.json", "CACHE_SCOPE.lock"}:
        return False
    record_path = folder / "CACHE_SCOPE.json"
    cache_root = folder / "cache"
    expected = spec.get("cache_scope_path")
    if not record_path.is_file() or not cache_root.is_dir() or not expected:
        return False
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        return (Path(record["root"]).resolve(strict=True) == Path(expected).resolve(strict=True)
                and Path(record["root"]).resolve(strict=True).is_relative_to(cache_root.resolve(strict=True)))
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


class _LogCapture:
    def __init__(self, folder: Path, stream_name: str, limit_bytes: int) -> None:
        self.path = folder / f"{stream_name}.log"
        self.stream = self.path.open("wb")
        self.limit_bytes = limit_bytes
        self.bytes_seen = 0
        self.bytes_saved = 0
        self.digest = hashlib.sha256()

    def append(self, chunk: bytes) -> None:
        self.bytes_seen += len(chunk)
        self.digest.update(chunk)
        keep = min(len(chunk), max(0, self.limit_bytes - self.bytes_saved))
        if keep:
            self.stream.write(chunk[:keep])
            self.bytes_saved += keep

    def close(self) -> dict[str, Any]:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        return {"path": str(self.path), "bytes_seen": self.bytes_seen,
                "bytes_saved": self.bytes_saved, "truncated": self.bytes_seen > self.bytes_saved,
                "sha256_full_stream": self.digest.hexdigest()}


def _drain(selector: selectors.BaseSelector, captures: dict[int, _LogCapture]) -> set[int]:
    closed: set[int] = set()
    for key, _ in selector.select(timeout=0):
        fd = key.fd
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            continue
        except OSError:
            closed.add(fd)
            continue
        if chunk:
            captures[fd].append(chunk)
        else:
            closed.add(fd)
    for fd in closed:
        try:
            selector.unregister(fd)
        except (KeyError, ValueError):
            pass
    return closed


def _execute_arm_inner(*, results_root: Path, attempt_id: str, command: Sequence[str],
                       spec: Mapping[str, Any], identity: Mapping[str, Any], gpu_uuid: str,
                       timeout_seconds: float, cleanup_grace_seconds: float = 15.0,
                       term_grace_seconds: float = 5.0, heartbeat_seconds: float = 30.0,
                       sample_seconds: float = 5.0, max_log_bytes: int = 8 * 1024 * 1024,
                       environment: Mapping[str, str] | None = None,
                       cache_setup: Mapping[str, Any] | None = None,
                       cache_scope_lock_fd: int | None = None,
                       monitor_fixture: Sequence[Mapping[str, Any]] | None = None,
                       started_monotonic: float | None = None) -> dict[str, Any]:
    """Run one arm in the host-supervised process; never replay UNKNOWN."""
    if timeout_seconds <= 0 or cleanup_grace_seconds < 0 or term_grace_seconds < 0:
        raise ValueError("timeouts and cleanup grace must be non-negative, with a positive deadline")
    if term_grace_seconds > cleanup_grace_seconds:
        raise ValueError("term grace cannot exceed total cleanup grace")
    if not attempt_id or Path(attempt_id).name != attempt_id:
        raise ValueError("attempt_id must be one path component")
    if not command:
        raise ValueError("worker command is required")

    started = started_monotonic if started_monotonic is not None else time.monotonic()
    deadline = started + timeout_seconds
    hard_deadline = deadline + cleanup_grace_seconds
    folder = results_root / "raw" / attempt_id
    folder.mkdir(parents=True, exist_ok=True)
    frozen_identity = {**dict(identity), "attempt_id": attempt_id,
                       "gpu_uuid": gpu_uuid, "spec_digest": canonical_digest(spec)}
    existing = recover_attempt(folder, frozen_identity)
    if existing.get("arm_execution") == ArmExecution.UNKNOWN.value:
        return existing
    if existing.get("arm_execution") != ArmExecution.NOT_STARTED.value:
        return existing
    if not _prelaunch_artifacts_allowed(folder, spec):
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "reason": "attempt has artifacts but no lifecycle intent; replay is forbidden",
                "identity": frozen_identity}
    try:
        lock_fd = os.open(folder / "supervisor.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "reason": "attempt is already owned or has an incomplete prior supervisor",
                "identity": frozen_identity}
    try:
        os.write(lock_fd, json.dumps(frozen_identity, sort_keys=True).encode("utf-8"))
        os.fsync(lock_fd)
    finally:
        os.close(lock_fd)

    record_event(folder, "intent", {"identity": frozen_identity, "command": list(command),
                                     "created_unix": time.time(),
                                     "deadline_monotonic": deadline,
                                     "hard_deadline_monotonic": hard_deadline,
                                     "cache_setup_required": cache_setup is not None})
    atomic_json(folder / "spec.json", dict(spec))
    child_env = os.environ.copy()
    child_env.update(dict(environment or {}))
    if cache_setup is not None:
        try:
            from .cache_lifecycle import create_cache_scope

            cache = create_cache_scope(
                folder, mode=str(cache_setup["mode"]),
                lifecycle_lock_fd=cache_scope_lock_fd, intent_recorded=True,
            )
            child_env.update({key: str(value) for key, value in cache["environment"].items()})
        except Exception as exc:
            if time.monotonic() >= deadline:
                result_payload = {
                    "arm_execution": ArmExecution.TIMEOUT.value,
                    "scientific_correctness": "UNASSESSED",
                    "first_failure_stage": "CACHE_INITIALIZATION_DEADLINE",
                    "root_cause": "ARM_HARD_DEADLINE_DURING_CACHE_INITIALIZATION",
                    "infrastructure_classification": "INFRA",
                    "identity": frozen_identity, "process_returncode": None,
                    "parent_wall_seconds": time.monotonic() - started,
                    "elapsed_ceiling_seconds": timeout_seconds + cleanup_grace_seconds,
                    "within_deadline_plus_grace": (
                        time.monotonic() - started <= timeout_seconds + cleanup_grace_seconds
                    ),
                }
                atomic_json(folder / "receipt.json", result_payload)
                receipt_sha = hashlib.sha256((folder / "receipt.json").read_bytes()).hexdigest()
                record_event(folder, "terminal", {
                    "identity": frozen_identity, "arm_execution": result_payload["arm_execution"],
                    "receipt_sha256": receipt_sha, "process_returncode": None,
                    "ended_unix": time.time(), "elapsed_seconds": time.monotonic() - started,
                })
                return {**result_payload, "raw_receipt_sha256": receipt_sha}
            result_payload = {
                "arm_execution": ArmExecution.FAILED.value,
                "scientific_correctness": "UNASSESSED",
                "first_failure_stage": "CACHE_INITIALIZATION",
                "root_cause": f"CACHE_INITIALIZATION:{type(exc).__name__}:{exc}",
                "infrastructure_classification": (
                    "INFRA" if isinstance(exc, (PermissionError, FileNotFoundError)) else None
                ),
                "cache_error_type": type(exc).__name__, "cache_error": str(exc),
                "traceback": traceback.format_exc(), "identity": frozen_identity,
                "process_returncode": None,
                "parent_wall_seconds": time.monotonic() - started,
                "elapsed_ceiling_seconds": timeout_seconds + cleanup_grace_seconds,
                "within_deadline_plus_grace": time.monotonic() - started <= timeout_seconds + cleanup_grace_seconds,
            }
            atomic_json(folder / "receipt.json", result_payload)
            receipt_sha = hashlib.sha256((folder / "receipt.json").read_bytes()).hexdigest()
            record_event(folder, "terminal", {
                "identity": frozen_identity, "arm_execution": result_payload["arm_execution"],
                "receipt_sha256": receipt_sha, "process_returncode": None,
                "ended_unix": time.time(), "elapsed_seconds": time.monotonic() - started,
            })
            return {**result_payload, "raw_receipt_sha256": receipt_sha}
        if time.monotonic() >= deadline:
            result_payload = {
                "arm_execution": ArmExecution.TIMEOUT.value,
                "scientific_correctness": "UNASSESSED",
                "first_failure_stage": "CACHE_INITIALIZATION_DEADLINE",
                "root_cause": "ARM_HARD_DEADLINE_DURING_CACHE_INITIALIZATION",
                "infrastructure_classification": "INFRA",
                "identity": frozen_identity, "process_returncode": None,
                "parent_wall_seconds": time.monotonic() - started,
                "elapsed_ceiling_seconds": timeout_seconds + cleanup_grace_seconds,
                "within_deadline_plus_grace": (
                    time.monotonic() - started <= timeout_seconds + cleanup_grace_seconds
                ),
            }
            atomic_json(folder / "receipt.json", result_payload)
            receipt_sha = hashlib.sha256((folder / "receipt.json").read_bytes()).hexdigest()
            record_event(folder, "terminal", {
                "identity": frozen_identity, "arm_execution": result_payload["arm_execution"],
                "receipt_sha256": receipt_sha, "process_returncode": None,
                "ended_unix": time.time(), "elapsed_seconds": time.monotonic() - started,
            })
            return {**result_payload, "raw_receipt_sha256": receipt_sha}
    child_env.update({"CUDA_VISIBLE_DEVICES": gpu_uuid, "OMP_NUM_THREADS": "1",
                      "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                      "NUMEXPR_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    capture_stdout = capture_stderr = None
    process = None
    telemetry: list[dict[str, Any]] = []
    timed_out = False
    orphan_cleanup = False
    failure_stage: str | None = None
    cleanup_deadline: float | None = None
    term_deadline: float | None = None
    kill_sent = False
    next_heartbeat = started + heartbeat_seconds
    next_sample = started
    selector = selectors.DefaultSelector()
    captures: dict[int, _LogCapture] = {}
    open_fds: set[int] = set()
    result_payload: dict[str, Any] = {}
    direct_returncode: int | None = None
    observed_exit_after_deadline = False
    gate_read_fd: int | None = None
    gate_write_fd: int | None = None
    worker_gate_released = False
    try:
        if time.monotonic() >= deadline:
            result_payload = {
                "arm_execution": ArmExecution.TIMEOUT.value,
                "scientific_correctness": "UNASSESSED",
                "first_failure_stage": "SUPERVISOR_START_DEADLINE",
                "root_cause": "ARM_HARD_DEADLINE_EXPIRED_BEFORE_WORKER_START",
                "identity": frozen_identity,
                "process_returncode": None,
                "parent_wall_seconds": time.monotonic() - started,
                "elapsed_ceiling_seconds": timeout_seconds + cleanup_grace_seconds,
                "within_deadline_plus_grace": (
                    time.monotonic() - started <= timeout_seconds + cleanup_grace_seconds
                ),
            }
            atomic_json(folder / "receipt.json", result_payload)
            receipt_sha = hashlib.sha256((folder / "receipt.json").read_bytes()).hexdigest()
            record_event(folder, "terminal", {
                "identity": frozen_identity,
                "arm_execution": result_payload["arm_execution"],
                "receipt_sha256": receipt_sha,
                "process_returncode": None,
                "ended_unix": time.time(),
                "elapsed_seconds": time.monotonic() - started,
            })
            return {**result_payload, "raw_receipt_sha256": receipt_sha}

        gate_read_fd, gate_write_fd = os.pipe()
        _enable_child_subreaper()
        worker_launcher = [
            sys.executable, "-m", "experiments.real_intervention_v4.process_supervisor",
            "--gated-worker-fd", str(gate_read_fd), "--", *command,
        ]
        process = subprocess.Popen(
            worker_launcher, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            close_fds=True, pass_fds=(gate_read_fd,), bufsize=0,
        )
        os.close(gate_read_fd)
        gate_read_fd = None
        assert process.stdout is not None and process.stderr is not None
        group_id = process.pid
        owned_groups = {group_id}

        def refresh_owned_groups() -> None:
            owned_groups.update(_descendant_process_groups(os.getpid()))

        record_event(folder, "started", {"identity": frozen_identity, "pid": process.pid,
                                         "process_group_id": group_id,
                                         "deadline_monotonic": deadline,
                                         "hard_deadline_monotonic": hard_deadline,
                                         "started_unix": time.time(),
                                         "start_gate": "durable-before-release"})
        pipe_fds = {"stdout": process.stdout.fileno(), "stderr": process.stderr.fileno()}
        for pipe, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            fd = pipe.fileno()
            os.set_blocking(fd, False)
            selector.register(fd, selectors.EVENT_READ)
            captures[fd] = _LogCapture(folder, name, max_log_bytes)
            open_fds.add(fd)
        os.write(gate_write_fd, b"1")
        os.close(gate_write_fd)
        gate_write_fd = None
        worker_gate_released = True
        if monitor_fixture is None:
            sampler = _gpu_snapshot
        else:
            snapshots = list(monitor_fixture)
            sample_index = 0

            def sampler(_gpu_uuid: str, _group_id: int) -> dict[str, Any]:
                nonlocal sample_index
                if not snapshots:
                    return {"error": "injected_telemetry_unavailable", "gpu_uuid": gpu_uuid}
                result = dict(snapshots[min(sample_index, len(snapshots) - 1)])
                sample_index += 1
                return result

        if time.monotonic() >= deadline:
            timed_out = True
            failure_stage = "WORKER_START_DEADLINE"
            cleanup_deadline = hard_deadline
            term_deadline = min(time.monotonic() + term_grace_seconds, cleanup_deadline)
            _signal_group(group_id, signal.SIGTERM)
        while True:
            now = time.monotonic()
            parent_done = process.poll() is not None
            if parent_done and now >= deadline:
                # The child can finish during a bounded telemetry operation.
                # Its result must carry its own monotonic completion timestamp
                # or the supervisor cannot prove it met the deadline.
                observed_exit_after_deadline = True
            if parent_done or timed_out or now >= deadline:
                refresh_owned_groups()
            alive_groups = _poll_groups(owned_groups)
            group_pids = set().union(*(_process_group_pids(gid) for gid in alive_groups)) if alive_groups else set()
            group_alive = bool(alive_groups)
            if parent_done and group_alive and not orphan_cleanup:
                orphan_cleanup = True
                failure_stage = "WORKER_DESCENDANT_CLEANUP"
                cleanup_deadline = min(now + cleanup_grace_seconds, deadline + cleanup_grace_seconds)
                term_deadline = min(now + term_grace_seconds, cleanup_deadline)
                for worker_group in alive_groups:
                    _signal_group(worker_group, signal.SIGTERM)
            if now >= deadline and not timed_out and not parent_done:
                timed_out = True
                failure_stage = "WORKER_DEADLINE"
                cleanup_deadline = deadline + cleanup_grace_seconds
                term_deadline = min(now + term_grace_seconds, cleanup_deadline)
                for worker_group in alive_groups:
                    _signal_group(worker_group, signal.SIGTERM)
                record_event(folder, "heartbeat", {"identity": frozen_identity, "action": "SIGTERM_GROUP",
                                                    "elapsed_seconds": now - started,
                                                    "remaining_cleanup_seconds": cleanup_grace_seconds})
            if (timed_out or orphan_cleanup) and not kill_sent and term_deadline is not None and now >= term_deadline:
                refresh_owned_groups()
                for worker_group in _poll_groups(owned_groups):
                    _signal_group(worker_group, signal.SIGKILL)
                kill_sent = True
                record_event(folder, "heartbeat", {"identity": frozen_identity, "action": "SIGKILL_GROUP",
                                                    "elapsed_seconds": now - started})
            if now >= next_sample and time.monotonic() < deadline - 1.25:
                try:
                    telemetry.append(sampler(gpu_uuid, group_id))
                except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
                    telemetry.append({"error": f"{type(exc).__name__}: {exc}", "sample_unix": time.time()})
                next_sample = now + sample_seconds
            if now >= next_heartbeat and not (timed_out or orphan_cleanup):
                record_event(folder, "heartbeat", {"identity": frozen_identity,
                                                    "elapsed_seconds": now - started,
                                                    "process_group_pids": sorted(group_pids)})
                next_heartbeat = now + heartbeat_seconds

            open_fds.difference_update(_drain(selector, captures))
            parent_done = process.poll() is not None
            if parent_done or timed_out or time.monotonic() >= deadline:
                refresh_owned_groups()
            alive_groups = _poll_groups(owned_groups)
            group_alive = bool(alive_groups)
            if parent_done and open_fds and not group_alive and not orphan_cleanup:
                orphan_cleanup = True
                failure_stage = "PIPE_HELD_OUTSIDE_WORKER_GROUP"
                cleanup_deadline = min(now + cleanup_grace_seconds, deadline + cleanup_grace_seconds)
            if parent_done and not group_alive and not open_fds:
                direct_returncode = process.returncode
                break
            if cleanup_deadline is not None and now >= cleanup_deadline:
                direct_returncode = process.poll()
                failure_stage = failure_stage or "CLEANUP_GRACE_EXHAUSTED"
                break
            wait_until = min(now + 0.2, deadline if not parent_done and not timed_out else now + 0.2,
                             next_sample, next_heartbeat if not (timed_out or orphan_cleanup) else now + 0.2)
            selector.select(timeout=max(0.0, wait_until - time.monotonic()))

        # One final non-blocking drain, then close held descriptors at the hard ceiling.
        while open_fds and time.monotonic() < (cleanup_deadline or deadline):
            open_fds.difference_update(_drain(selector, captures))
            if open_fds:
                selector.select(timeout=min(0.05, max(0.0, (cleanup_deadline or deadline) - time.monotonic())))
        if open_fds:
            failure_stage = failure_stage or "PIPE_DRAIN_DEADLINE"
            for fd in tuple(open_fds):
                try:
                    selector.unregister(fd)
                except (KeyError, ValueError):
                    pass
                open_fds.discard(fd)
        direct_returncode = process.poll()
        process.stdout.close()
        process.stderr.close()
        log_info = {name: capture.close() for name, capture in (("stdout", captures.get(pipe_fds["stdout"])),
                                                                 ("stderr", captures.get(pipe_fds["stderr"])))
                    if capture is not None and not capture.stream.closed}
        result_path = folder / "result.json"
        worker_result: dict[str, Any] | None = None
        if not timed_out and direct_returncode == 0:
            try:
                worker_result = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(worker_result, dict):
                    raise ValueError("worker result must be a JSON object")
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                failure_stage = failure_stage or "WORKER_RESULT_RECEIPT"
                result_payload = {"arm_execution": ArmExecution.FAILED.value,
                                  "first_failure_stage": failure_stage,
                                  "root_cause": f"WORKER_RESULT_INVALID:{type(exc).__name__}",
                                  "worker_result_error": str(exc)}
            else:
                try:
                    worker_finished = float(worker_result["finished_monotonic"])
                except (KeyError, TypeError, ValueError):
                    worker_finished = None
                if observed_exit_after_deadline and (worker_finished is None or worker_finished > deadline):
                    timed_out = True
                if timed_out:
                    result_payload = {"arm_execution": ArmExecution.TIMEOUT.value,
                                      "scientific_correctness": "UNASSESSED",
                                      "first_failure_stage": "WORKER_DEADLINE",
                                      "root_cause": "ARM_HARD_DEADLINE"}
                elif worker_result.get("arm_execution") in {value.value for value in ArmExecution}:
                    result_payload = {
                        "arm_execution": worker_result["arm_execution"],
                        "scientific_correctness": worker_result.get(
                            "scientific_correctness", "UNASSESSED",
                        ),
                        "first_failure_stage": worker_result.get("first_failure_stage"),
                        "root_cause": worker_result.get("root_cause") or (
                            f"{worker_result.get('error_type')}:{worker_result.get('error')}"
                            if worker_result.get("error_type") else None
                        ),
                        "infrastructure_classification": worker_result.get("infrastructure_classification"),
                        "error_type": worker_result.get("error_type"),
                        "error": worker_result.get("error"),
                        "traceback": worker_result.get("traceback"),
                    }
                    if result_payload["scientific_correctness"] == "FAIL":
                        failure_stage = failure_stage or worker_result.get(
                            "first_failure_stage", "NUMERICAL_COMPARISON",
                        )
                elif worker_result.get("status") == "valid":
                    result_payload = {"arm_execution": ArmExecution.COMPLETE.value,
                                      "scientific_correctness": "PASS" if (worker_result.get("correctness") or {}).get("passed") else "UNASSESSED"}
                elif worker_result.get("status") == "numeric_mismatch":
                    result_payload = {"arm_execution": ArmExecution.COMPLETE.value,
                                      "scientific_correctness": "FAIL"}
                    failure_stage = "NUMERICAL_COMPARISON"
                else:
                    result_payload = {"arm_execution": ArmExecution.FAILED.value,
                                      "scientific_correctness": "UNASSESSED",
                                      "root_cause": str(worker_result.get("error_type") or worker_result.get("status") or "WORKER_ERROR"),
                                      "first_failure_stage": "MODEL_WORKER"}
        elif timed_out:
            result_payload = {"arm_execution": ArmExecution.TIMEOUT.value,
                              "scientific_correctness": "UNASSESSED",
                              "first_failure_stage": failure_stage,
                              "root_cause": "ARM_HARD_DEADLINE"}
        else:
            result_payload = {"arm_execution": ArmExecution.FAILED.value,
                              "scientific_correctness": "UNASSESSED",
                              "first_failure_stage": failure_stage or "WORKER_PROCESS",
                              "root_cause": f"WORKER_EXIT_{direct_returncode}"}
        refresh_owned_groups()
        remaining_group_pids = set().union(*(_process_group_pids(gid) for gid in owned_groups))
        if orphan_cleanup or direct_returncode is None or remaining_group_pids:
            result_payload["arm_execution"] = ArmExecution.UNKNOWN.value if direct_returncode is None or remaining_group_pids else ArmExecution.FAILED.value
            result_payload["root_cause"] = (
                "WORKER_PROCESS_GROUP_NOT_CLEAN" if remaining_group_pids else
                "WORKER_DESCENDANT_REQUIRED_CLEANUP" if orphan_cleanup else
                "WORKER_NOT_REAPED_WITHIN_CEILING"
            )
            result_payload["remaining_process_group_pids"] = sorted(remaining_group_pids)
        gpu_identity = (worker_result or {}).get("runtime_identity") or {}
        snapshots = telemetry
        errors = [sample.get("error") for sample in snapshots if sample.get("error")]
        result_payload.update({
            "identity": frozen_identity, "worker_result": worker_result,
            "timing": (worker_result or {}).get("timing"),
            "correctness": (worker_result or {}).get("correctness"),
            "runtime_identity": (worker_result or {}).get("runtime_identity"),
            "cache_scope_path": (worker_result or {}).get("cache_scope_path"),
            "runtime_gpu_uuid": gpu_identity.get("device_uuid"),
            "inputs_sha256": (worker_result or {}).get("inputs_sha256"),
            "weights_sha256": (worker_result or {}).get("weights_sha256"),
            "source_commit": frozen_identity.get("source_commit"),
            "config_digest": frozen_identity.get("config_digest"),
            "context_digest": frozen_identity.get("context_digest"),
            "process_returncode": direct_returncode,
            "timed_out": timed_out,
            "orphan_cleanup": orphan_cleanup,
            "parent_wall_seconds": time.monotonic() - started,
            "elapsed_ceiling_seconds": timeout_seconds + cleanup_grace_seconds,
            "within_deadline_plus_grace": time.monotonic() - started <= timeout_seconds + cleanup_grace_seconds,
            "telemetry": telemetry,
            "telemetry_status": "COMPLETE" if not errors and telemetry else "UNKNOWN",
            "log_capture": log_info,
            "worker_result_sha256": (
                hashlib.sha256(result_path.read_bytes()).hexdigest()
                if worker_result is not None and result_path.is_file() else None
            ),
        })
        atomic_json(folder / "receipt.json", result_payload)
        receipt_sha = hashlib.sha256((folder / "receipt.json").read_bytes()).hexdigest()
        record_event(folder, "terminal", {"identity": frozen_identity,
                                           "arm_execution": result_payload["arm_execution"],
                                           "receipt_sha256": receipt_sha,
                                           "process_returncode": direct_returncode,
                                           "ended_unix": time.time(),
                                           "elapsed_seconds": result_payload["parent_wall_seconds"]})
        return {**result_payload, "raw_receipt_sha256": receipt_sha}
    except BaseException as exc:
        if process is not None:
            group_id = process.pid
            cleanup_deadline = hard_deadline
            term_deadline = min(time.monotonic() + term_grace_seconds, cleanup_deadline)
            owned_groups = {group_id}
            if worker_gate_released:
                refresh = lambda: owned_groups.update(_descendant_process_groups(os.getpid()))
                refresh()
                for worker_group in _poll_groups(owned_groups):
                    _signal_group(worker_group, signal.SIGTERM)
            elif process.poll() is None:
                process.terminate()
            while time.monotonic() < term_deadline:
                if worker_gate_released:
                    refresh()
                    for worker_group in _poll_groups(owned_groups):
                        _signal_group(worker_group, signal.SIGTERM)
                if process.poll() is not None and (not worker_gate_released or not _poll_groups(owned_groups)):
                    break
                time.sleep(min(0.05, max(0.0, term_deadline - time.monotonic())))
            if worker_gate_released:
                refresh()
                for worker_group in _poll_groups(owned_groups):
                    _signal_group(worker_group, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
            while time.monotonic() < cleanup_deadline:
                if worker_gate_released:
                    refresh()
                    for worker_group in _poll_groups(owned_groups):
                        _signal_group(worker_group, signal.SIGKILL)
                if process.poll() is not None and (not worker_gate_released or not _poll_groups(owned_groups)):
                    break
                time.sleep(min(0.05, max(0.0, cleanup_deadline - time.monotonic())))
            direct_returncode = process.poll()
            if worker_gate_released:
                refresh()
                remaining_pids = set().union(*(_process_group_pids(gid) for gid in owned_groups))
            else:
                remaining_pids = set()
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
        else:
            remaining_pids = set()
        elapsed = time.monotonic() - started
        result_payload = {"arm_execution": ArmExecution.UNKNOWN.value if (process and process.poll() is None) or remaining_pids else ArmExecution.FAILED.value,
                          "scientific_correctness": "UNASSESSED", "first_failure_stage": failure_stage or "SUPERVISOR_EXCEPTION",
                          "root_cause": f"SUPERVISOR_EXCEPTION:{type(exc).__name__}:{exc}",
                          "identity": frozen_identity, "process_returncode": direct_returncode,
                          "parent_wall_seconds": elapsed,
                          "elapsed_ceiling_seconds": timeout_seconds + cleanup_grace_seconds,
                          "within_deadline_plus_grace": elapsed <= timeout_seconds + cleanup_grace_seconds,
                          "telemetry": telemetry,
                          "remaining_process_group_pids": sorted(remaining_pids)}
        if not (folder / "receipt.json").exists():
            for capture in captures.values():
                if not capture.stream.closed:
                    try:
                        capture.close()
                    except OSError:
                        capture.stream.close()
            atomic_json(folder / "receipt.json", result_payload)
            receipt_sha = hashlib.sha256((folder / "receipt.json").read_bytes()).hexdigest()
            try:
                if _read_event_kind(folder) is None:
                    record_event(folder, "intent", {"identity": frozen_identity, "command": list(command)})
                if _read_event_kind(folder) == "intent":
                    record_event(folder, "terminal", {"identity": frozen_identity, "arm_execution": result_payload["arm_execution"],
                                                       "receipt_sha256": receipt_sha, "ended_unix": time.time()})
                elif _read_event_kind(folder) in {"started", "heartbeat"}:
                    record_event(folder, "terminal", {"identity": frozen_identity, "arm_execution": result_payload["arm_execution"],
                                                       "receipt_sha256": receipt_sha, "ended_unix": time.time()})
            except (OSError, ValueError, FileExistsError):
                pass
        raise
    finally:
        for fd in (gate_read_fd, gate_write_fd):
            if fd is not None:
                os.close(fd)
        try:
            selector.close()
        except Exception:
            pass
        for capture in captures.values():
            if not capture.stream.closed:
                capture.stream.close()


def _read_event_kind(attempt_dir: Path) -> str | None:
    from .outcomes import _read_events
    events = _read_events(attempt_dir)
    return events[-1]["kind"] if events else None


def _descendant_process_groups(parent_pid: int) -> set[int]:
    records: dict[int, tuple[int, int]] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return set()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            tail = (entry / "stat").read_text(encoding="ascii").split(")", 1)[1].split()
            records[int(entry.name)] = (int(tail[1]), int(tail[2]))
        except (OSError, ValueError, IndexError):
            continue
    descendants = {parent_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in records.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    parent_group = records.get(parent_pid, (0, parent_pid))[1]
    return {records[pid][1] for pid in descendants if pid != parent_pid
            and pid in records and records[pid][1] != parent_group}


def _worker_groups(attempt_dir: Path, supervisor_pid: int) -> set[int]:
    groups = _descendant_process_groups(supervisor_pid)
    try:
        from .outcomes import _read_events
        for event in _read_events(attempt_dir):
            group_id = event.get("process_group_id")
            if isinstance(group_id, int) and group_id > 1:
                groups.add(group_id)
    except (OSError, ValueError):
        pass
    return groups


def _poll_groups(group_ids: set[int]) -> set[int]:
    return {group_id for group_id in group_ids if _process_group_pids(group_id)}


def _observed_terminal(folder: Path, identity: Mapping[str, Any], recovered: dict[str, Any], *,
                      started: float, ceiling: float) -> dict[str, Any]:
    elapsed = time.monotonic() - started
    result = {**recovered, "host_supervisor_wall_seconds": elapsed,
              "elapsed_ceiling_seconds": ceiling,
              "within_deadline_plus_grace": elapsed <= ceiling}
    if elapsed > ceiling:
        atomic_json(folder / "WATCHDOG.json", {
            "identity": dict(identity), "action": "HARD_DEADLINE_EXCEEDED",
            "elapsed_seconds": elapsed, "elapsed_ceiling_seconds": ceiling,
            "raw_terminal_arm_execution": recovered.get("arm_execution"),
        })
        result.update({
            "arm_execution": ArmExecution.TIMEOUT.value,
            "scientific_correctness": "UNASSESSED",
            "first_failure_stage": "HOST_SUPERVISOR_HARD_DEADLINE",
            "root_cause": "HOST_SUPERVISOR_RETURNED_AFTER_DECLARED_CEILING",
            "raw_terminal_arm_execution": recovered.get("arm_execution"),
        })
    return result


def _cleanup_owned_processes(process: subprocess.Popen[bytes], group_ids: set[int], *,
                             deadline: float, term_grace_seconds: float) -> tuple[set[int], list[str]]:
    signals_sent: list[str] = []
    for group_id in _poll_groups(group_ids):
        if _signal_group(group_id, signal.SIGTERM):
            signals_sent.append("SIGTERM_WORKER_GROUP")
    if process.poll() is None and _signal_group(process.pid, signal.SIGTERM):
        signals_sent.append("SIGTERM_SUPERVISOR_GROUP")

    term_end = min(time.monotonic() + term_grace_seconds, deadline)
    while time.monotonic() < term_end:
        process.poll()
        remaining = _poll_groups(group_ids)
        if process.poll() is not None and not remaining:
            return set(), signals_sent
        time.sleep(min(0.025, max(0.0, term_end - time.monotonic())))

    for group_id in _poll_groups(group_ids):
        if _signal_group(group_id, signal.SIGKILL):
            signals_sent.append("SIGKILL_WORKER_GROUP")
    if process.poll() is None and _signal_group(process.pid, signal.SIGKILL):
        signals_sent.append("SIGKILL_SUPERVISOR_GROUP")
    while time.monotonic() < deadline:
        process.poll()
        remaining = _poll_groups(group_ids)
        if process.poll() is not None and not remaining:
            return set(), signals_sent
        time.sleep(min(0.025, max(0.0, deadline - time.monotonic())))
    return _poll_groups(group_ids), signals_sent


def _terminalize_watchdog_timeout(folder: Path, identity: Mapping[str, Any], *,
                                  supervisor_returncode: int | None,
                                  elapsed_seconds: float, ceiling_seconds: float) -> dict[str, Any]:
    from .outcomes import _read_events

    events = _read_events(folder)
    if not events:
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "reason": "host watchdog stopped supervisor before durable intent"}
    if events[-1]["kind"] == "terminal":
        return recover_attempt(folder, identity)

    intent = events[0]
    if not any(event["kind"] == "started" for event in events):
        cache_initialization_incomplete = (
            intent.get("cache_setup_required") is True
            and not (folder / "CACHE_SCOPE.json").is_file()
        )
        stage = ("CACHE_INITIALIZATION_DEADLINE" if cache_initialization_incomplete
                 else "HOST_WATCHDOG_BEFORE_WORKER_START")
        cause = ("ARM_HARD_DEADLINE_DURING_CACHE_INITIALIZATION"
                 if cache_initialization_incomplete
                 else "HOST_WATCHDOG_STOPPED_BEFORE_WORKER_START")
        receipt = {
            "arm_execution": ArmExecution.TIMEOUT.value,
            "scientific_correctness": "UNASSESSED",
            "first_failure_stage": stage,
            "root_cause": cause,
            "infrastructure_classification": "INFRA",
            "identity": dict(identity), "process_returncode": None,
            "supervisor_returncode": supervisor_returncode,
            "parent_wall_seconds": elapsed_seconds,
            "elapsed_ceiling_seconds": ceiling_seconds,
            "within_deadline_plus_grace": elapsed_seconds <= ceiling_seconds,
        }
        atomic_json(folder / "receipt.json", receipt)
        receipt_digest = hashlib.sha256((folder / "receipt.json").read_bytes()).hexdigest()
        record_event(folder, "terminal", {
            "identity": dict(identity), "arm_execution": receipt["arm_execution"],
            "receipt_sha256": receipt_digest, "process_returncode": None,
            "ended_unix": time.time(), "elapsed_seconds": elapsed_seconds,
            "closed_by": "HOST_WATCHDOG_BEFORE_WORKER_START",
        })
        return recover_attempt(folder, identity)
    return {"arm_execution": ArmExecution.UNKNOWN.value,
            "reason": "host watchdog could not establish a terminal worker result",
            "events": events}


def execute_arm(*, results_root: Path, attempt_id: str, command: Sequence[str],
                spec: Mapping[str, Any], identity: Mapping[str, Any], gpu_uuid: str,
                timeout_seconds: float, cleanup_grace_seconds: float = 15.0,
                term_grace_seconds: float = 5.0, heartbeat_seconds: float = 30.0,
                sample_seconds: float = 5.0, max_log_bytes: int = 8 * 1024 * 1024,
                environment: Mapping[str, str] | None = None,
                cache_setup: Mapping[str, Any] | None = None,
                monitor_fixture: Sequence[Mapping[str, Any]] | None = None,
                study_deadline_monotonic: float | None = None) -> dict[str, Any]:
    """Run under a hard deadline; cache setup and worker launch share an attempt lock."""
    if timeout_seconds <= 0 or cleanup_grace_seconds < 0:
        raise ValueError("timeouts and cleanup grace must be non-negative, with a positive deadline")
    if term_grace_seconds < 0 or term_grace_seconds > cleanup_grace_seconds:
        raise ValueError("term grace cannot exceed total cleanup grace")
    if not attempt_id or Path(attempt_id).name != attempt_id or not command:
        raise ValueError("attempt_id and worker command are required")

    started = time.monotonic()
    if study_deadline_monotonic is not None:
        remaining = study_deadline_monotonic - started
        effective_timeout = min(timeout_seconds, remaining - cleanup_grace_seconds)
        if effective_timeout <= 0:
            elapsed = time.monotonic() - started
            study_window = max(0.0, remaining)
            return {
                "arm_execution": ArmExecution.NOT_STARTED.value,
                "scientific_correctness": "UNASSESSED",
                "first_failure_stage": "STUDY_WALL_CAP",
                "reason": "insufficient study wall time remains for the arm and its cleanup grace",
                "parent_wall_seconds": elapsed,
                "elapsed_ceiling_seconds": study_window,
                "within_deadline_plus_grace": elapsed <= study_window,
            }
    else:
        effective_timeout = timeout_seconds
    folder = results_root / "raw" / attempt_id
    folder.mkdir(parents=True, exist_ok=True)
    frozen_identity = {**dict(identity), "attempt_id": attempt_id,
                       "gpu_uuid": gpu_uuid, "spec_digest": canonical_digest(spec)}
    cache_lock_fd = _try_cache_scope_lock(folder)
    if cache_lock_fd is None:
        existing = recover_attempt(folder, frozen_identity)
        if existing.get("arm_execution") != ArmExecution.NOT_STARTED.value:
            return existing
        return {"arm_execution": ArmExecution.NOT_STARTED.value,
                "reason": "attempt cache initialization is in progress",
                "identity": frozen_identity}

    def release_cache_lock() -> None:
        nonlocal cache_lock_fd
        if cache_lock_fd is not None:
            _release_cache_scope_lock(cache_lock_fd)
            cache_lock_fd = None

    try:
        existing = recover_attempt(folder, frozen_identity)
        if existing.get("arm_execution") != ArmExecution.NOT_STARTED.value:
            release_cache_lock()
            return existing
        if not _prelaunch_artifacts_allowed(folder, spec):
            release_cache_lock()
            return {"arm_execution": ArmExecution.UNKNOWN.value,
                    "reason": "attempt has artifacts but no lifecycle intent; replay is forbidden",
                    "identity": frozen_identity}

        lock_dir = results_root / ".supervisor-locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        host_lock = lock_dir / f"{attempt_id}.lock"
        try:
            lock_fd = os.open(host_lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            release_cache_lock()
            return {"arm_execution": ArmExecution.UNKNOWN.value,
                    "reason": "attempt already has a host supervisor or an incomplete prior attempt",
                    "identity": frozen_identity}
        try:
            os.write(lock_fd, json.dumps(frozen_identity, sort_keys=True).encode("utf-8"))
            os.fsync(lock_fd)
        finally:
            os.close(lock_fd)

        request_dir = results_root / ".supervisor-requests"
        request_dir.mkdir(parents=True, exist_ok=True)
        request_path = request_dir / f"{attempt_id}-{os.getpid()}-{time.monotonic_ns()}.json"
        request = {
            "results_root": str(results_root.resolve()), "attempt_id": attempt_id,
            "command": list(command), "spec": dict(spec), "identity": dict(identity),
            "gpu_uuid": gpu_uuid, "timeout_seconds": effective_timeout,
            "cleanup_grace_seconds": cleanup_grace_seconds,
            "term_grace_seconds": term_grace_seconds,
            "heartbeat_seconds": heartbeat_seconds, "sample_seconds": sample_seconds,
            "max_log_bytes": max_log_bytes, "environment": dict(environment or {}),
            "cache_setup": dict(cache_setup) if cache_setup is not None else None,
            "cache_scope_lock_fd": cache_lock_fd,
            "monitor_fixture": list(monitor_fixture) if monitor_fixture is not None else None,
            "started_monotonic": started,
        }
        atomic_json(request_path, request)
        supervisor_log_root = results_root / "supervisor-logs"
        supervisor_log_root.mkdir(parents=True, exist_ok=True)
        supervisor_stdout = supervisor_log_root / f"{attempt_id}.stdout.log"
        supervisor_stderr = supervisor_log_root / f"{attempt_id}.stderr.log"
        command_line = [sys.executable, "-m", "experiments.real_intervention_v4.process_supervisor",
                        "--supervisor-request", str(request_path)]
        ceiling = effective_timeout + cleanup_grace_seconds
        hard_deadline = started + ceiling
        work_deadline = started + effective_timeout
        process: subprocess.Popen[bytes] | None = None
    except BaseException:
        release_cache_lock()
        raise
    try:
        with supervisor_stdout.open("wb") as out, supervisor_stderr.open("wb") as err:
            process = subprocess.Popen(command_line, stdin=subprocess.DEVNULL, stdout=out,
                                       stderr=err, start_new_session=True, close_fds=True,
                                       pass_fds=(cache_lock_fd,))
            while process.poll() is None and time.monotonic() < work_deadline:
                try:
                    process.wait(timeout=min(0.1, max(0.0, work_deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is not None:
                recovered = recover_attempt(folder, frozen_identity)
                events = recovered.get("events", [])
                if events and events[-1].get("kind") == "terminal":
                    return _observed_terminal(folder, frozen_identity, recovered,
                                              started=started, ceiling=ceiling)
                if not events:
                    return {"arm_execution": ArmExecution.UNKNOWN.value,
                            "reason": "supervisor exited before durable intent; no worker was started",
                            "identity": frozen_identity}

                # A supervisor that has already exited without a terminal
                # receipt did not remain alive through the deadline.  Clean
                # any process group recorded by its durable events, but keep
                # the attempt UNKNOWN: an observed started event has no
                # trustworthy terminal outcome and must never be replayed.
                worker_groups = _worker_groups(folder, process.pid)
                cleanup_deadline = min(hard_deadline, time.monotonic() + cleanup_grace_seconds)
                remaining, signals_sent = _cleanup_owned_processes(
                    process, worker_groups, deadline=cleanup_deadline,
                    term_grace_seconds=term_grace_seconds,
                )
                exit_code = process.poll()
                root_cause = (
                    f"SUPERVISOR_EXIT_{exit_code}_WITHOUT_TERMINAL"
                    if exit_code is not None else "SUPERVISOR_NOT_REAPED_WITHIN_CEILING"
                )
                watchdog = {
                    "identity": frozen_identity,
                    "action": "SUPERVISOR_EXIT_WITHOUT_TERMINAL",
                    "first_failure_stage": "HOST_SUPERVISOR_EXIT",
                    "root_cause": root_cause,
                    "supervisor_returncode": exit_code,
                    "event_kinds": [event.get("kind") for event in events],
                    "remaining_process_groups": sorted(remaining),
                    "cleanup_confirmed": not remaining,
                    "signals_sent": signals_sent,
                    "elapsed_seconds": time.monotonic() - started,
                    "elapsed_ceiling_seconds": ceiling,
                }
                atomic_json(folder / "WATCHDOG.json", watchdog)
                return {
                    "arm_execution": ArmExecution.UNKNOWN.value,
                    "scientific_correctness": "UNKNOWN",
                    "first_failure_stage": watchdog["first_failure_stage"],
                    "root_cause": root_cause,
                    "supervisor_returncode": exit_code,
                    "cleanup_confirmed": not remaining,
                    "remaining_process_groups": sorted(remaining),
                    "parent_wall_seconds": watchdog["elapsed_seconds"],
                    "elapsed_ceiling_seconds": ceiling,
                    "within_deadline_plus_grace": watchdog["elapsed_seconds"] <= ceiling,
                    "identity": frozen_identity,
                }

            worker_groups = _worker_groups(folder, process.pid)
            signals_sent: list[str] = []
            for group_id in worker_groups:
                if _process_group_pids(group_id) and _signal_group(group_id, signal.SIGTERM):
                    signals_sent.append("SIGTERM")

            cleanup_budget = max(0.0, hard_deadline - work_deadline)
            report_reserve = min(0.1, cleanup_budget * 0.2)
            supervisor_term_grace = min(0.1, cleanup_budget * 0.2)
            supervisor_term_deadline = max(
                work_deadline, hard_deadline - report_reserve - supervisor_term_grace,
            )
            supervisor_kill_deadline = max(
                supervisor_term_deadline, hard_deadline - report_reserve,
            )
            term_end = min(work_deadline + term_grace_seconds, supervisor_term_deadline)
            term_end = max(work_deadline, term_end)
            while time.monotonic() < term_end:
                process.poll()
                if process.poll() is not None:
                    recovered = recover_attempt(folder, frozen_identity)
                    events = recovered.get("events", [])
                    if events and events[-1].get("kind") == "terminal":
                        elapsed = time.monotonic() - started
                        atomic_json(folder / "WATCHDOG.json", {
                            "identity": frozen_identity, "action": "SUPERVISOR_CLOSED_DURING_CLEANUP",
                            "remaining_process_groups": [], "signals_sent": signals_sent,
                            "supervisor_returncode": process.poll(), "elapsed_seconds": elapsed,
                            "elapsed_ceiling_seconds": ceiling,
                        })
                        return _observed_terminal(folder, frozen_identity, recovered,
                                                  started=started, ceiling=ceiling)
                alive = _poll_groups(worker_groups)
                if not alive:
                    break
                time.sleep(min(0.025, max(0.0, term_end - time.monotonic())))

            remaining = _poll_groups(worker_groups)
            for group_id in remaining:
                if _signal_group(group_id, signal.SIGKILL):
                    signals_sent.append("SIGKILL")
            while time.monotonic() < supervisor_kill_deadline:
                process.poll()
                if process.poll() is not None:
                    recovered = recover_attempt(folder, frozen_identity)
                    events = recovered.get("events", [])
                    if events and events[-1].get("kind") == "terminal":
                        elapsed = time.monotonic() - started
                        atomic_json(folder / "WATCHDOG.json", {
                            "identity": frozen_identity, "action": "SUPERVISOR_CLOSED_DURING_CLEANUP",
                            "remaining_process_groups": [], "signals_sent": signals_sent,
                            "supervisor_returncode": process.poll(), "elapsed_seconds": elapsed,
                            "elapsed_ceiling_seconds": ceiling,
                        })
                        return _observed_terminal(folder, frozen_identity, recovered,
                                                  started=started, ceiling=ceiling)
                remaining = _poll_groups(worker_groups)
                for group_id in remaining:
                    if _signal_group(group_id, signal.SIGKILL):
                        signals_sent.append("SIGKILL")
                if time.monotonic() >= supervisor_term_deadline and process.poll() is None:
                    if _signal_group(process.pid, signal.SIGTERM):
                        signals_sent.append("SIGTERM")
                if not remaining and process.poll() is not None:
                    break
                time.sleep(min(0.05, max(0.0, supervisor_kill_deadline - time.monotonic())))

            remaining = _poll_groups(worker_groups)
            for group_id in remaining:
                if _signal_group(group_id, signal.SIGKILL):
                    signals_sent.append("SIGKILL")
            if process.poll() is None and _signal_group(process.pid, signal.SIGKILL):
                signals_sent.append("SIGKILL")
            while time.monotonic() < hard_deadline:
                process.poll()
                remaining = _poll_groups(worker_groups)
                if process.poll() is not None and not remaining:
                    break
                time.sleep(min(0.025, max(0.0, hard_deadline - time.monotonic())))

        elapsed = time.monotonic() - started
        remaining = _poll_groups(worker_groups)
        supervisor_returncode = process.poll()
        if remaining or supervisor_returncode is None:
            atomic_json(folder / "WATCHDOG.json", {
                "identity": frozen_identity, "action": "CLEANUP_UNCONFIRMED",
                "remaining_process_groups": sorted(remaining),
                "signals_sent": signals_sent,
                "supervisor_returncode": supervisor_returncode,
                "elapsed_seconds": elapsed, "elapsed_ceiling_seconds": ceiling,
            })
            elapsed = time.monotonic() - started
            return {"arm_execution": ArmExecution.UNKNOWN.value,
                    "reason": "host watchdog could not confirm owned process-group cleanup",
                    "remaining_process_groups": sorted(remaining),
                    "supervisor_returncode": supervisor_returncode,
                    "parent_wall_seconds": elapsed,
                    "elapsed_ceiling_seconds": ceiling,
                    "within_deadline_plus_grace": elapsed <= ceiling}

        atomic_json(folder / "WATCHDOG.json", {
            "identity": frozen_identity, "action": "PROCESS_GROUPS_CLEANED",
            "remaining_process_groups": [], "supervisor_returncode": supervisor_returncode,
            "signals_sent": signals_sent,
            "elapsed_seconds": elapsed, "elapsed_ceiling_seconds": ceiling,
        })
        recovered = _terminalize_watchdog_timeout(
            folder, frozen_identity, supervisor_returncode=supervisor_returncode,
            elapsed_seconds=elapsed, ceiling_seconds=ceiling,
        )
        if recovered.get("arm_execution") == ArmExecution.NOT_STARTED.value:
            recovered = {"arm_execution": ArmExecution.UNKNOWN.value,
                         "reason": "host watchdog stopped before a durable arm intent",
                         "identity": frozen_identity}
        return _observed_terminal(folder, frozen_identity, recovered,
                                  started=started, ceiling=ceiling)
    except BaseException as exc:
        signals_sent: list[str] = []
        remaining: set[int] = set()
        if process is not None:
            worker_groups = _worker_groups(folder, process.pid)
            cleanup_deadline = hard_deadline - min(0.1, cleanup_grace_seconds * 0.2)
            remaining, signals_sent = _cleanup_owned_processes(
                process, worker_groups, deadline=cleanup_deadline,
                term_grace_seconds=term_grace_seconds,
            )
            recovered = recover_attempt(folder, frozen_identity)
            events = recovered.get("events", [])
            if not isinstance(exc, OSError):
                raise
            if events and events[-1].get("kind") == "terminal":
                return _observed_terminal(folder, frozen_identity, recovered,
                                          started=started, ceiling=ceiling)
        else:
            if not isinstance(exc, OSError):
                raise
            recovered = recover_attempt(folder, frozen_identity)
            if recovered.get("arm_execution") == ArmExecution.NOT_STARTED.value:
                return {**recovered, "reason": f"host supervisor launch failed before arm start: {type(exc).__name__}: {exc}"}
        elapsed = time.monotonic() - started
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "reason": f"host supervisor launch/cleanup failed: {type(exc).__name__}: {exc}",
                "identity": frozen_identity, "parent_wall_seconds": elapsed,
                "elapsed_ceiling_seconds": ceiling,
                "within_deadline_plus_grace": elapsed <= ceiling,
                "remaining_process_groups": sorted(remaining),
                "signals_sent": signals_sent}
    finally:
        request_path.unlink(missing_ok=True)
        release_cache_lock()


def _supervisor_main(request_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["results_root"] = Path(request["results_root"])
    _execute_arm_inner(**request)


def _gated_worker_main(gate_fd: int, command: Sequence[str]) -> None:
    if not command:
        raise ValueError("gated worker command is required")
    with os.fdopen(gate_fd, "rb") as gate:
        if gate.read(1) != b"1":
            raise RuntimeError("worker start gate closed before durable started event")
    os.setsid()
    os.execvpe(command[0], list(command), os.environ)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor-request", type=Path)
    parser.add_argument("--gated-worker-fd", type=int)
    parser.add_argument("worker_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.gated_worker_fd is not None:
        command = args.worker_command
        if command and command[0] == "--":
            command = command[1:]
        _gated_worker_main(args.gated_worker_fd, command)
    elif args.supervisor_request is not None:
        _supervisor_main(args.supervisor_request)
    else:
        parser.error("provide --supervisor-request or --gated-worker-fd")
