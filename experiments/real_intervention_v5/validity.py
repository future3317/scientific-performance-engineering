"""Five independent validity dimensions for one V5 paired session."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping, Sequence


EXECUTION = {"NOT_STARTED", "STARTED", "COMPLETE", "ERROR", "UNKNOWN"}
SCIENTIFIC = {"PASS", "NUMERICAL_UNSAFE", "UNVERIFIED"}
PLATFORM = {"HEALTHY", "PLATFORM_FAILURE", "UNCERTAIN"}
EXPOSURE = {"MATCHED", "CONTENDED", "INCOMPLETE", "UNKNOWN"}
ANALYSIS = {"VALID_PAIRED", "NONCOMPARABLE", "EXCLUDED_PRESPECIFIED", "MISSING"}


def _execution(arm: Mapping[str, Any]) -> str:
    value = str(arm.get("execution_status", arm.get("arm_execution", "UNKNOWN"))).upper()
    return value if value in EXECUTION else "UNKNOWN"


def _scientific(arm: Mapping[str, Any]) -> str:
    if arm.get("outputs_match") is False:
        return "NUMERICAL_UNSAFE"
    value = str(arm.get("scientific_status", arm.get("scientific_correctness", "UNVERIFIED"))).upper()
    if value in SCIENTIFIC:
        return value
    if value in {"FAIL", "UNSAFE", "NUMERIC_MISMATCH"}:
        return "NUMERICAL_UNSAFE"
    if value in {"PASS", "VALID", "CORRECT"}:
        return "PASS"
    correctness = arm.get("correctness")
    if isinstance(correctness, Mapping):
        if correctness.get("outputs_match") is False:
            return "NUMERICAL_UNSAFE"
        if correctness.get("passed") is True:
            return "PASS"
        if correctness.get("passed") is False:
            return "NUMERICAL_UNSAFE"
    worker = arm.get("worker_result")
    if isinstance(worker, Mapping):
        nested = worker.get("scientific_status", worker.get("scientific_correctness"))
        if nested is not None:
            return _scientific({"scientific_status": nested})
        nested_correctness = worker.get("correctness")
        if isinstance(nested_correctness, Mapping):
            return _scientific({"correctness": nested_correctness})
    return "UNVERIFIED"


def _exposure_status(exposure: Mapping[str, Any]) -> str:
    explicit = str(exposure.get("status", exposure.get("exposure_status", ""))).upper()
    if explicit in EXPOSURE:
        return explicit
    if exposure.get("co_tenant_confirmed") is True or exposure.get("co_tenant_observed") is True:
        return "CONTENDED"
    if exposure.get("telemetry_complete") is False or exposure.get("telemetry_errors"):
        return "INCOMPLETE"
    if exposure.get("unattributed_activity") is True:
        return "UNKNOWN"
    if exposure.get("matched") is True:
        return "MATCHED"
    if exposure.get("matched") is False:
        return "CONTENDED"
    return "UNKNOWN"


def _identity_mismatch(arms: Sequence[Mapping[str, Any]], exposure: Mapping[str, Any]) -> bool:
    left, right = arms[0].get("arm_identity"), arms[-1].get("arm_identity")
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in ("session_id", "pair_id", "gpu_uuid", "family_id", "split", "context_digest"):
            if left.get(key) is not None and right.get(key) is not None and left[key] != right[key]:
                return True
    expected = exposure.get("expected_gpu_uuid")
    if expected is not None:
        for arm in arms:
            identity = arm.get("arm_identity")
            if isinstance(identity, Mapping) and identity.get("gpu_uuid") not in (None, expected):
                return True
    devices = exposure.get("gpu_uuids", exposure.get("devices"))
    if isinstance(devices, Mapping):
        left, right = devices.get("arm0", devices.get("S0")), devices.get("arm1", devices.get("S1"))
        if left is not None and right is not None and left != right:
            return True
    elif isinstance(devices, (list, tuple, set)) and len(set(devices)) > 1:
        return True
    return False


def _platform_status(arms: Sequence[Mapping[str, Any]], exposure: str) -> str:
    statuses = [str(arm.get("platform_status", "HEALTHY")).upper() for arm in arms]
    if "PLATFORM_FAILURE" in statuses or any(
        arm.get("error_code") for arm in arms
    ):
        return "PLATFORM_FAILURE"
    if exposure == "INCOMPLETE" or "UNCERTAIN" in statuses:
        return "UNCERTAIN"
    return "HEALTHY"


def classify_pair(arms: list[dict[str, Any]], exposure: dict[str, Any]) -> dict[str, Any]:
    """Derive orthogonal execution, scientific, platform, exposure and analysis states.

    `arms` must contain the ordered baseline then intervention records. Timing
    validity is not interpreted as scientific harm; numerical failure is kept
    in its own field and excluded from paired effect analysis.
    """
    if not arms:
        raise ValueError("session validity requires at least one ordered arm record")
    executions = [_execution(arm) for arm in arms]
    scientific_arms = [_scientific(arm) for arm in arms]
    exposure_status = _exposure_status(exposure)
    identity_mismatch = _identity_mismatch(arms, exposure)

    if all(value == "COMPLETE" for value in executions):
        execution_status = "COMPLETE"
    elif "ERROR" in executions:
        execution_status = "ERROR"
    elif "UNKNOWN" in executions or ("STARTED" in executions and "COMPLETE" in executions):
        execution_status = "UNKNOWN"
    elif "STARTED" in executions:
        execution_status = "STARTED"
    elif all(value == "NOT_STARTED" for value in executions):
        execution_status = "NOT_STARTED"
    else:
        execution_status = "UNKNOWN"

    if scientific_arms[0] != "PASS":
        scientific_status = "UNVERIFIED"
    elif any(value == "NUMERICAL_UNSAFE" for value in scientific_arms[1:]):
        scientific_status = "NUMERICAL_UNSAFE"
    elif all(value == "PASS" for value in scientific_arms):
        scientific_status = "PASS"
    else:
        scientific_status = "UNVERIFIED"

    platform_status = _platform_status(arms, exposure_status)
    if execution_status in {"ERROR", "UNKNOWN"} and any(
        arm.get("error_code") or arm.get("platform_status") == "PLATFORM_FAILURE" for arm in arms
    ):
        platform_status = "PLATFORM_FAILURE"

    if execution_status != "COMPLETE":
        analysis_status = "MISSING"
        first_gate = "EXECUTION"
    elif identity_mismatch:
        analysis_status = "NONCOMPARABLE"
        first_gate = "ARM_IDENTITY"
    elif scientific_status == "UNVERIFIED" and scientific_arms[0] != "PASS":
        analysis_status = "MISSING"
        first_gate = "BASELINE_CORRECTNESS"
    elif scientific_status == "NUMERICAL_UNSAFE":
        analysis_status = "EXCLUDED_PRESPECIFIED"
        first_gate = "SCIENTIFIC_CORRECTNESS"
    elif scientific_status != "PASS":
        analysis_status = "MISSING"
        first_gate = "SCIENTIFIC_CORRECTNESS"
    elif platform_status == "PLATFORM_FAILURE":
        analysis_status = "MISSING"
        first_gate = "PLATFORM"
    elif exposure_status == "CONTENDED":
        analysis_status = "EXCLUDED_PRESPECIFIED"
        first_gate = "GPU_EXPOSURE"
    elif exposure_status in {"INCOMPLETE", "UNKNOWN"}:
        analysis_status = "NONCOMPARABLE" if exposure_status == "UNKNOWN" else "MISSING"
        first_gate = "GPU_EXPOSURE"
    elif platform_status == "UNCERTAIN":
        analysis_status = "MISSING"
        first_gate = "PLATFORM"
    else:
        analysis_status = "VALID_PAIRED"
        first_gate = None

    raw = json.dumps({"arms": arms, "exposure": exposure}, sort_keys=True,
                     separators=(",", ":"), default=str).encode("utf-8")
    return {
        "execution_status": execution_status,
        "scientific_status": scientific_status,
        "platform_status": platform_status,
        "exposure_status": exposure_status,
        "analysis_status": analysis_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error_code": next((arm.get("error_code") for arm in arms if arm.get("error_code")), None),
        "first_failing_gate": first_gate,
        "arm_identity": [arm.get("arm_identity") for arm in arms],
        "raw_digest": hashlib.sha256(raw).hexdigest(),
    }
