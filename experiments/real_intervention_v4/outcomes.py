"""Append-only arm lifecycle and four-layer paired validity classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
from fcntl import LOCK_EX, LOCK_UN, flock

from .runtime_identity import cross_process_runtime_differences, selected_gpu_identity_differences


class ArmExecution(str, Enum):
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"
    NOT_STARTED = "NOT_STARTED"


class Correctness(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNASSESSED = "UNASSESSED"
    UNKNOWN = "UNKNOWN"


class Comparability(str, Enum):
    MATCHED = "MATCHED"
    CONFOUNDED = "CONFOUNDED"
    INCOMPLETE = "INCOMPLETE"
    UNKNOWN = "UNKNOWN"


class PolicyEvidence(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    UNSAFE = "UNSAFE"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


_ALLOWED_EVENTS = {
    None: {"intent"},
    "intent": {"started", "terminal"},
    "started": {"heartbeat", "terminal"},
    "heartbeat": {"heartbeat", "terminal"},
    "terminal": set(),
}


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    """Publish one JSON artifact only after its full contents are durable."""
    atomic_write_text(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
                      exclusive=exclusive)


def atomic_write_text(path: Path, content: str, *, exclusive: bool = False) -> None:
    """Atomically replace a text artifact after making the new bytes durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    data = content.encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _read_events(attempt_dir: Path) -> list[dict[str, Any]]:
    event_dir = attempt_dir / "events"
    if not event_dir.exists():
        return []
    events = []
    for sequence, path in enumerate(sorted(event_dir.glob("*.json"))):
        try:
            event = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"corrupt lifecycle event: {path}") from exc
        if not isinstance(event, dict) or event.get("kind") not in _ALLOWED_EVENTS:
            raise ValueError(f"invalid lifecycle event: {path}")
        if event.get("sequence") != sequence or path.name != f"{sequence:04d}-{event['kind']}.json":
            raise ValueError(f"lifecycle event sequence/name mismatch at {path}")
        identity = event.get("identity")
        if not isinstance(identity, dict):
            raise ValueError(f"lifecycle event identity is missing or invalid at {path}")
        if events and identity != events[0]["identity"]:
            raise ValueError(f"lifecycle event identity differs at {path}")
        events.append(event)
    previous = None
    for event in events:
        kind = event["kind"]
        if kind not in _ALLOWED_EVENTS[previous]:
            raise ValueError(f"illegal lifecycle transition {previous!r} -> {kind!r}")
        previous = kind
    return events


def record_event(attempt_dir: Path, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if {"sequence", "kind", "monotonic_seconds", "unix_seconds"} & payload.keys():
        raise ValueError("lifecycle payload cannot override event metadata")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("lifecycle event requires an identity")
    event_dir = attempt_dir / "events"
    event_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(event_dir / ".append.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        flock(lock_fd, LOCK_EX)
        events = _read_events(attempt_dir)
        previous = events[-1]["kind"] if events else None
        if kind not in _ALLOWED_EVENTS.get(previous, set()):
            raise ValueError(f"illegal lifecycle transition {previous!r} -> {kind!r}")
        if events and dict(identity) != events[0]["identity"]:
            raise ValueError("lifecycle event identity differs from the attempt identity")
        event = {"sequence": len(events), "kind": kind, "monotonic_seconds": time.monotonic(),
                 "unix_seconds": time.time(), **dict(payload), "identity": dict(identity)}
        path = event_dir / f"{len(events):04d}-{kind}.json"
        if path.exists():
            raise FileExistsError(f"lifecycle event already exists: {path}")
        atomic_json(path, event)
        return event
    finally:
        flock(lock_fd, LOCK_UN)
        os.close(lock_fd)


def recover_attempt(attempt_dir: Path, expected_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return a terminal receipt, NOT_STARTED, or an unreplayable UNKNOWN."""
    try:
        events = _read_events(attempt_dir)
    except (OSError, ValueError) as exc:
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "scientific_correctness": Correctness.UNKNOWN.value,
                "events": [], "reason": f"lifecycle history is invalid: {exc}"}
    if not events:
        return {"arm_execution": ArmExecution.NOT_STARTED.value, "events": []}
    if any(event["identity"] != dict(expected_identity) for event in events):
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "scientific_correctness": Correctness.UNKNOWN.value,
                "events": events,
                "reason": "lifecycle event identity differs from frozen attempt identity"}
    terminal = events[-1] if events[-1]["kind"] == "terminal" else None
    if terminal is None:
        watchdog_path = attempt_dir / "WATCHDOG.json"
        if watchdog_path.is_file():
            try:
                watchdog = json.loads(watchdog_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                return {"arm_execution": ArmExecution.UNKNOWN.value,
                        "scientific_correctness": Correctness.UNKNOWN.value,
                        "events": events,
                        "reason": f"host watchdog record is corrupt: {exc}"}
            if not isinstance(watchdog, dict) or watchdog.get("identity") != dict(expected_identity):
                return {"arm_execution": ArmExecution.UNKNOWN.value,
                        "scientific_correctness": Correctness.UNKNOWN.value,
                        "events": events,
                        "reason": "host watchdog identity differs from frozen attempt identity"}
            if watchdog.get("action") == "SUPERVISOR_EXIT_WITHOUT_TERMINAL":
                return {
                    "arm_execution": ArmExecution.UNKNOWN.value,
                    "scientific_correctness": Correctness.UNKNOWN.value,
                    "first_failure_stage": watchdog.get("first_failure_stage"),
                    "root_cause": watchdog.get("root_cause"),
                    "supervisor_returncode": watchdog.get("supervisor_returncode"),
                    "cleanup_confirmed": watchdog.get("cleanup_confirmed"),
                    "remaining_process_groups": watchdog.get("remaining_process_groups", []),
                    "events": events,
                    "reason": "supervisor exited without a terminal receipt; replay is forbidden",
                }
        return {"arm_execution": ArmExecution.UNKNOWN.value, "events": events,
                "reason": "started attempt lacks a terminal event; replay is forbidden"}
    receipt_path = attempt_dir / "receipt.json"
    if not receipt_path.is_file():
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "scientific_correctness": Correctness.UNKNOWN.value,
                "events": events, "reason": "terminal lifecycle event has no receipt"}
    try:
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "scientific_correctness": Correctness.UNKNOWN.value,
                "events": events, "reason": f"terminal receipt is corrupt: {exc}"}
    if not isinstance(receipt, dict):
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "scientific_correctness": Correctness.UNKNOWN.value,
                "events": events, "reason": "terminal receipt is not a JSON object"}
    expected_digest = terminal.get("receipt_sha256")
    actual_digest = hashlib.sha256(receipt_bytes).hexdigest()
    if not expected_digest or expected_digest != actual_digest:
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "scientific_correctness": Correctness.UNKNOWN.value,
                "events": events, "reason": "terminal receipt digest mismatch"}
    if receipt.get("identity") != dict(expected_identity):
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "scientific_correctness": Correctness.UNKNOWN.value,
                "events": events, "reason": "receipt identity differs from frozen attempt identity"}
    if terminal.get("arm_execution") != receipt.get("arm_execution"):
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "scientific_correctness": Correctness.UNKNOWN.value,
                "events": events, "reason": "terminal event execution status differs from receipt"}
    if terminal.get("process_returncode") != receipt.get("process_returncode"):
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "scientific_correctness": Correctness.UNKNOWN.value,
                "events": events, "reason": "terminal event return code differs from receipt"}
    if receipt.get("arm_execution") not in {value.value for value in ArmExecution}:
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "scientific_correctness": Correctness.UNKNOWN.value,
                "events": events, "reason": "terminal receipt has an unrecognized execution state"}
    if receipt.get("scientific_correctness", Correctness.UNASSESSED.value) not in {
        value.value for value in Correctness
    }:
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "scientific_correctness": Correctness.UNKNOWN.value,
                "events": events, "reason": "terminal receipt has an unrecognized correctness state"}

    worker_result = receipt.get("worker_result")
    worker_digest = receipt.get("worker_result_sha256")
    result_path = attempt_dir / "result.json"
    if worker_result is not None and not isinstance(worker_result, dict):
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "scientific_correctness": Correctness.UNKNOWN.value,
                "events": events, "reason": "receipt worker result is not a JSON object"}
    if worker_digest is None:
        if worker_result is not None:
            return {"arm_execution": ArmExecution.UNKNOWN.value,
                    "scientific_correctness": Correctness.UNKNOWN.value,
                    "events": events, "reason": "receipt embeds a worker result without its digest"}
    else:
        try:
            result_bytes = result_path.read_bytes()
        except OSError as exc:
            return {"arm_execution": ArmExecution.UNKNOWN.value,
                    "scientific_correctness": Correctness.UNKNOWN.value,
                    "events": events, "reason": f"receipt-bound worker result is missing: {exc}"}
        if worker_digest != hashlib.sha256(result_bytes).hexdigest():
            return {"arm_execution": ArmExecution.UNKNOWN.value,
                    "scientific_correctness": Correctness.UNKNOWN.value,
                    "events": events, "reason": "worker result digest differs from terminal receipt"}
        try:
            disk_result = json.loads(result_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {"arm_execution": ArmExecution.UNKNOWN.value,
                    "scientific_correctness": Correctness.UNKNOWN.value,
                    "events": events, "reason": f"receipt-bound worker result is corrupt: {exc}"}
        if not isinstance(disk_result, dict):
            return {"arm_execution": ArmExecution.UNKNOWN.value,
                    "scientific_correctness": Correctness.UNKNOWN.value,
                    "events": events, "reason": "receipt-bound worker result is not a JSON object"}
        if worker_result is not None and disk_result != worker_result:
            return {"arm_execution": ArmExecution.UNKNOWN.value,
                    "scientific_correctness": Correctness.UNKNOWN.value,
                    "events": events, "reason": "receipt worker result differs from result.json"}
        worker_result = disk_result
    if receipt.get("arm_execution") == ArmExecution.COMPLETE.value and (
        receipt.get("process_returncode") != 0
        or receipt.get("timed_out") is True
        or receipt.get("orphan_cleanup") is True
        or worker_digest is None
        or not isinstance(worker_result, dict)
    ):
        return {"arm_execution": ArmExecution.UNKNOWN.value,
                "scientific_correctness": Correctness.UNKNOWN.value,
                "events": events,
                "reason": "complete receipt lacks a clean, digest-bound worker result"}
    if isinstance(worker_result, dict):
        worker_runtime = worker_result.get("runtime_identity")
        worker_runtime = worker_runtime if isinstance(worker_runtime, dict) else {}
        copied_fields = {
            "timing": worker_result.get("timing"),
            "correctness": worker_result.get("correctness"),
            "runtime_identity": worker_result.get("runtime_identity"),
            "runtime_gpu_uuid": worker_runtime.get("device_uuid"),
            "cache_scope_path": worker_result.get("cache_scope_path"),
            "inputs_sha256": worker_result.get("inputs_sha256"),
            "weights_sha256": worker_result.get("weights_sha256"),
        }
        if any(receipt.get(key) != value for key, value in copied_fields.items()):
            return {"arm_execution": ArmExecution.UNKNOWN.value,
                    "scientific_correctness": Correctness.UNKNOWN.value,
                    "events": events,
                    "reason": "terminal receipt summary differs from its worker result"}
        if (receipt.get("process_returncode") == 0 and not receipt.get("timed_out")
                and not receipt.get("orphan_cleanup")):
            worker_execution = worker_result.get("arm_execution")
            if worker_execution not in {value.value for value in ArmExecution}:
                if worker_result.get("status") in {"valid", "numeric_mismatch"}:
                    worker_execution = ArmExecution.COMPLETE.value
                else:
                    worker_execution = ArmExecution.FAILED.value
            worker_correctness = worker_result.get("scientific_correctness")
            if worker_correctness not in {value.value for value in Correctness}:
                if worker_result.get("status") == "numeric_mismatch":
                    worker_correctness = Correctness.FAIL.value
                elif worker_result.get("status") == "valid":
                    passed = (worker_result.get("correctness") or {}).get("passed")
                    worker_correctness = (Correctness.PASS.value if passed
                                          else Correctness.UNASSESSED.value)
                else:
                    worker_correctness = Correctness.UNASSESSED.value
            if (receipt.get("arm_execution") != worker_execution
                    or receipt.get("scientific_correctness", Correctness.UNASSESSED.value)
                    != worker_correctness):
                return {"arm_execution": ArmExecution.UNKNOWN.value,
                        "scientific_correctness": Correctness.UNKNOWN.value,
                        "events": events,
                        "reason": "terminal receipt outcome differs from its worker result"}
    result = {**receipt, "worker_result": worker_result,
              "raw_receipt_sha256": actual_digest, "events": events}
    result["receipt_verified"] = True
    watchdog_path = attempt_dir / "WATCHDOG.json"
    if watchdog_path.is_file():
        try:
            watchdog = json.loads(watchdog_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {"arm_execution": ArmExecution.UNKNOWN.value,
                    "scientific_correctness": Correctness.UNKNOWN.value,
                    "events": events, "reason": f"host watchdog record is corrupt: {exc}"}
        if not isinstance(watchdog, dict) or watchdog.get("identity") != dict(expected_identity):
            return {"arm_execution": ArmExecution.UNKNOWN.value,
                    "scientific_correctness": Correctness.UNKNOWN.value,
                    "events": events,
                    "reason": "host watchdog identity differs from frozen attempt identity"}
        if watchdog.get("action") == "HARD_DEADLINE_EXCEEDED":
            result.update({
                "arm_execution": ArmExecution.TIMEOUT.value,
                "scientific_correctness": Correctness.UNASSESSED.value,
                "first_failure_stage": "HOST_SUPERVISOR_HARD_DEADLINE",
                "root_cause": "HOST_SUPERVISOR_RETURNED_AFTER_DECLARED_CEILING",
                "raw_terminal_arm_execution": receipt.get("arm_execution"),
                "within_deadline_plus_grace": False,
            })
    return result


@dataclass(frozen=True)
class PairValidity:
    arm_execution: str
    scientific_correctness: str
    paired_comparability: str
    policy_evidence: str
    first_failure_stage: str | None
    root_cause: str | None
    raw_receipt_sha256: dict[str, str | None]
    arm_states: dict[str, dict[str, str]]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_pair(s0: Mapping[str, Any] | None, s1: Mapping[str, Any] | None, *,
                  intervention_applicable: bool, exposure_match: bool | None,
                  expected_gpu_uuid: str | None = None,
                  mechanism_active: bool | None = True) -> PairValidity:
    arms = {"S0": s0 or {}, "S1": s1 or {}}
    receipt_hashes = {name: arm.get("raw_receipt_sha256") for name, arm in arms.items()}
    valid_executions = {value.value for value in ArmExecution}
    valid_correctness = {value.value for value in Correctness}
    raw_executions = {
        name: str(arm.get("arm_execution", ArmExecution.NOT_STARTED.value))
        for name, arm in arms.items()
    }
    invalid_execution = any(value not in valid_executions for value in raw_executions.values())
    execution_by_arm = {
        name: value if value in valid_executions else ArmExecution.UNKNOWN.value
        for name, value in raw_executions.items()
    }
    raw_correctness = {
        name: str(arm.get("scientific_correctness", Correctness.UNASSESSED.value))
        for name, arm in arms.items()
    }
    correctness_by_arm = {}
    for name, value in raw_correctness.items():
        if value not in valid_correctness:
            correctness_by_arm[name] = Correctness.UNKNOWN.value
        elif (value in {Correctness.PASS.value, Correctness.FAIL.value}
              and arms[name].get("receipt_verified") is not True):
            correctness_by_arm[name] = Correctness.UNKNOWN.value
        else:
            correctness_by_arm[name] = value
    arm_states = {
        name: {"arm_execution": execution_by_arm[name],
               "scientific_correctness": correctness_by_arm[name]}
        for name in arms
    }
    if not intervention_applicable:
        return PairValidity(ArmExecution.NOT_STARTED.value, Correctness.UNASSESSED.value,
                            Comparability.INCOMPLETE.value, PolicyEvidence.NOT_APPLICABLE.value,
                            "PREEXECUTION_APPLICABILITY", "INTERVENTION_NOT_APPLICABLE",
                            receipt_hashes, arm_states,
                            "The frozen pre-execution context is outside intervention scope")

    executions = list(execution_by_arm.values())
    correctness = list(correctness_by_arm.values())

    def pair_execution() -> str:
        for status in (ArmExecution.UNKNOWN.value, ArmExecution.TIMEOUT.value,
                       ArmExecution.FAILED.value, ArmExecution.NOT_STARTED.value):
            if status in executions:
                return status
        return ArmExecution.COMPLETE.value

    # Only an S1 failure is evidence of candidate harm. An S0-only failure
    # invalidates the baseline contrast but must not be attributed to S1.
    if correctness_by_arm["S1"] == Correctness.FAIL.value:
        failed = arms["S1"]
        return PairValidity(pair_execution(), Correctness.FAIL.value,
                            Comparability.INCOMPLETE.value, PolicyEvidence.UNSAFE.value,
                            failed.get("first_failure_stage") or "CANDIDATE_SCIENTIFIC_CORRECTNESS",
                            failed.get("root_cause") or "KNOWN_CANDIDATE_CORRECTNESS_FAILURE",
                            receipt_hashes, arm_states,
                            "Known correctness failure is unsafe and cannot be traded for speed")
    if correctness_by_arm["S0"] == Correctness.FAIL.value:
        failed = arms["S0"]
        return PairValidity(pair_execution(), Correctness.FAIL.value,
                            Comparability.INCOMPLETE.value, PolicyEvidence.MISSING.value,
                            failed.get("first_failure_stage") or "BASELINE_SCIENTIFIC_CORRECTNESS",
                            failed.get("root_cause") or "BASELINE_CORRECTNESS_FAILURE",
                            receipt_hashes, arm_states,
                            "The baseline failed its scientific correctness gate; candidate harm is not identified")
    if invalid_execution:
        failed_name = next(name for name, status in raw_executions.items()
                           if status not in valid_executions)
        failed = arms[failed_name]
        return PairValidity(ArmExecution.UNKNOWN.value, Correctness.UNKNOWN.value,
                            Comparability.UNKNOWN.value, PolicyEvidence.MISSING.value,
                            failed.get("first_failure_stage") or "ARM_TERMINAL_STATE",
                            failed.get("root_cause") or "UNRECOGNIZED_ARM_EXECUTION_STATE",
                            receipt_hashes, arm_states,
                            "An arm has an unrecognized execution state")
    if ArmExecution.UNKNOWN.value in executions:
        failed_name = next(name for name, status in execution_by_arm.items()
                           if status == ArmExecution.UNKNOWN.value)
        failed = arms[failed_name]
        return PairValidity(ArmExecution.UNKNOWN.value, Correctness.UNKNOWN.value,
                            Comparability.INCOMPLETE.value, PolicyEvidence.MISSING.value,
                            failed.get("first_failure_stage") or "ARM_TERMINAL_STATE",
                            failed.get("root_cause") or "STARTED_ATTEMPT_WITHOUT_TERMINAL_RECEIPT",
                            receipt_hashes, arm_states,
                            "At least one started arm has no terminal receipt; do not replay")
    if ArmExecution.TIMEOUT.value in executions:
        failed_name = next(name for name, status in execution_by_arm.items()
                           if status == ArmExecution.TIMEOUT.value)
        failed = arms[failed_name]
        return PairValidity(ArmExecution.TIMEOUT.value, Correctness.UNASSESSED.value,
                            Comparability.INCOMPLETE.value, PolicyEvidence.MISSING.value,
                            failed.get("first_failure_stage") or "ARM_DEADLINE",
                            failed.get("root_cause") or "ARM_HARD_TIMEOUT", receipt_hashes,
                            arm_states,
                            "A timeout is an execution failure, not candidate harm or a zero outcome")
    if ArmExecution.NOT_STARTED.value in executions:
        failed_name = next(name for name, status in execution_by_arm.items()
                           if status == ArmExecution.NOT_STARTED.value)
        failed = arms[failed_name]
        return PairValidity(ArmExecution.NOT_STARTED.value, Correctness.UNASSESSED.value,
                            Comparability.INCOMPLETE.value, PolicyEvidence.MISSING.value,
                            failed.get("first_failure_stage") or "ARM_NOT_STARTED",
                            failed.get("root_cause"), receipt_hashes,
                            arm_states,
                            "A planned arm has not started")
    if ArmExecution.FAILED.value in executions:
        failed_name = next(name for name, status in execution_by_arm.items()
                           if status == ArmExecution.FAILED.value)
        failed = arms[failed_name]
        return PairValidity(ArmExecution.FAILED.value, Correctness.UNASSESSED.value,
                            Comparability.INCOMPLETE.value, PolicyEvidence.MISSING.value,
                            failed.get("first_failure_stage") or "ARM_EXECUTION",
                            failed.get("root_cause") or "ARM_EXECUTION_FAILURE", receipt_hashes, arm_states,
                            "Execution failure is kept separate from scientific harm")

    receipt_hashes_valid = all(
        arms[name].get("receipt_verified") is True
        and isinstance(receipt_hashes[name], str)
        and len(receipt_hashes[name]) == 64
        and all(character in "0123456789abcdef" for character in receipt_hashes[name].lower())
        for name in ("S0", "S1")
    )
    if not receipt_hashes_valid:
        correctness_result = (
            Correctness.PASS.value if all(value == Correctness.PASS.value for value in correctness)
            else Correctness.UNKNOWN.value if Correctness.UNKNOWN.value in correctness
            else Correctness.UNASSESSED.value
        )
        return PairValidity(ArmExecution.COMPLETE.value, correctness_result,
                            Comparability.UNKNOWN.value, PolicyEvidence.MISSING.value,
                            "PAIR_RECEIPT_INTEGRITY", "VERIFIED_RECEIPT_DIGEST_MISSING",
                            receipt_hashes, arm_states,
                            "Eligibility requires both raw receipts verified by recover_attempt")

    if any(value in {Correctness.UNKNOWN.value, Correctness.UNASSESSED.value} for value in correctness):
        return PairValidity(ArmExecution.COMPLETE.value, Correctness.UNASSESSED.value,
                            Comparability.UNKNOWN.value, PolicyEvidence.MISSING.value,
                            "CORRECTNESS_EVIDENCE", "CORRECTNESS_NOT_ESTABLISHED", receipt_hashes,
                            arm_states,
                            "Both processes terminated but a correctness gate is missing")

    identity_fields = ("runtime_gpu_uuid", "inputs_sha256", "weights_sha256", "source_commit",
                       "config_digest", "context_digest")
    missing = [field for field in identity_fields if any(not arm.get(field) for arm in arms.values())]
    if missing:
        return PairValidity(ArmExecution.COMPLETE.value, Correctness.PASS.value,
                            Comparability.UNKNOWN.value, PolicyEvidence.MISSING.value,
                            "PAIR_IDENTITY", "REQUIRED_PAIR_IDENTITY_MISSING", receipt_hashes,
                            arm_states,
                            "Missing identity fields: " + ", ".join(missing))
    mismatched = [field for field in identity_fields if arms["S0"].get(field) != arms["S1"].get(field)]
    if mismatched:
        return PairValidity(ArmExecution.COMPLETE.value, Correctness.PASS.value,
                            Comparability.CONFOUNDED.value, PolicyEvidence.MISSING.value,
                            "PAIR_IDENTITY", "PAIRED_INPUT_SOURCE_OR_DEVICE_MISMATCH", receipt_hashes,
                            arm_states,
                            "Mismatched identity fields: " + ", ".join(mismatched))
    runtime_identities = {
        name: arms[name].get("runtime_identity") for name in ("S0", "S1")
    }
    if any(not isinstance(value, Mapping) for value in runtime_identities.values()):
        return PairValidity(ArmExecution.COMPLETE.value, Correctness.PASS.value,
                            Comparability.UNKNOWN.value, PolicyEvidence.MISSING.value,
                            "PAIR_RUNTIME_IDENTITY", "REQUIRED_RUNTIME_IDENTITY_MISSING",
                            receipt_hashes, arm_states,
                            "Both arm receipts must contain observed runtime identity")
    runtime_differences = cross_process_runtime_differences(
        dict(runtime_identities["S0"]), dict(runtime_identities["S1"]),
    )
    if runtime_differences:
        return PairValidity(ArmExecution.COMPLETE.value, Correctness.PASS.value,
                            Comparability.CONFOUNDED.value, PolicyEvidence.MISSING.value,
                            "PAIR_RUNTIME_IDENTITY", "CROSS_PROCESS_RUNTIME_IDENTITY_MISMATCH",
                            receipt_hashes, arm_states,
                            "Runtime differs between arms: " + ", ".join(sorted(runtime_differences)))
    if not isinstance(expected_gpu_uuid, str) or not expected_gpu_uuid.strip():
        return PairValidity(ArmExecution.COMPLETE.value, Correctness.PASS.value,
                            Comparability.UNKNOWN.value, PolicyEvidence.MISSING.value,
                            "PAIR_IDENTITY", "SELECTED_GPU_UUID_MISSING", receipt_hashes,
                            arm_states,
                            "A matched pair requires the physical GPU UUID frozen before either arm starts")
    if any(arms[name].get("runtime_gpu_uuid") != expected_gpu_uuid for name in ("S0", "S1")):
        return PairValidity(ArmExecution.COMPLETE.value, Correctness.PASS.value,
                            Comparability.CONFOUNDED.value, PolicyEvidence.MISSING.value,
                            "PAIR_IDENTITY", "RUNTIME_GPU_UUID_DIFFERS_FROM_SELECTED_DEVICE",
                            receipt_hashes, arm_states,
                            "Observed worker UUID does not match the UUID frozen for this pair")
    selected_gpu_differences = {
        name: differences
        for name in ("S0", "S1")
        if (differences := selected_gpu_identity_differences(
            dict(runtime_identities[name]), expected_gpu_uuid,
        ))
    }
    if selected_gpu_differences:
        return PairValidity(ArmExecution.COMPLETE.value, Correctness.PASS.value,
                            Comparability.CONFOUNDED.value, PolicyEvidence.MISSING.value,
                            "PAIR_RUNTIME_IDENTITY", "RUNTIME_IDENTITY_DIFFERS_FROM_SELECTED_DEVICE",
                            receipt_hashes, arm_states,
                            "Runtime GPU identity differs from the selected physical UUID")
    if exposure_match is None:
        return PairValidity(ArmExecution.COMPLETE.value, Correctness.PASS.value,
                            Comparability.UNKNOWN.value, PolicyEvidence.MISSING.value,
                            "GPU_EXPOSURE_TELEMETRY", "GPU_EXPOSURE_UNKNOWN", receipt_hashes,
                            arm_states,
                            "GPU co-tenancy/telemetry is not sufficient to establish a matched pair")
    if not exposure_match:
        return PairValidity(ArmExecution.COMPLETE.value, Correctness.PASS.value,
                            Comparability.CONFOUNDED.value, PolicyEvidence.MISSING.value,
                            "GPU_EXPOSURE_COMPARISON", "ARM_EXPOSURES_NOT_MATCHED", receipt_hashes,
                            arm_states,
                            "Raw outcomes are retained but the pair is not eligible for matched-effect inference")
    if mechanism_active is not True:
        reason = "NO_MECHANISTIC_EXPOSURE" if mechanism_active is False else "MECHANISM_EXPOSURE_UNASSESSED"
        return PairValidity(ArmExecution.COMPLETE.value, Correctness.PASS.value,
                            Comparability.MATCHED.value, PolicyEvidence.MISSING.value,
                            "S1_MECHANISM_ACTIVATION", reason, receipt_hashes, arm_states,
                            "A correct pair without verified channels-last execution is not evidence for the intervention")
    return PairValidity(ArmExecution.COMPLETE.value, Correctness.PASS.value,
                        Comparability.MATCHED.value, PolicyEvidence.ELIGIBLE.value,
                        None, None, receipt_hashes, arm_states,
                        "All execution, correctness, identity and exposure gates passed")
