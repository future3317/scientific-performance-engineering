"""Calibration bundle status classification."""

from __future__ import annotations

from typing import Any

from .state import derive_cell_state
from benchmark.harness.fingerprint import fingerprints_compatible
from benchmark.provenance import digest_mapping, file_digest


def calibration_envelope(
    *, producer_revision: str, task_package_digest: str, population_manifest_digest: str,
    harness_digest_value: str, calibration_runner_digest: str, noise_digest: str,
    raw_result_digest: str, fingerprint: dict[str, Any], task_id: str,
    outer_trial_id: str, seed: int, measurement_class: str,
    calibration_protocol_digest: str | None = None,
) -> dict[str, Any]:
    envelope = {
        "schema_version": 1, "task_id": str(task_id), "outer_trial_id": str(outer_trial_id),
        "seed": int(seed), "measurement_class": str(measurement_class),
        "producer_revision": str(producer_revision), "task_package_digest": str(task_package_digest),
        "population_manifest_digest": str(population_manifest_digest), "harness_digest": str(harness_digest_value),
        "calibration_runner_digest": str(calibration_runner_digest), "noise_digest": str(noise_digest),
        "raw_result_digest": str(raw_result_digest), "fingerprint": dict(fingerprint),
    }
    if calibration_protocol_digest is not None:
        envelope["calibration_protocol_digest"] = str(calibration_protocol_digest)
    envelope["envelope_digest"] = digest_mapping(envelope)
    return envelope


def validate_calibration_envelope(payload: dict[str, Any], expected: dict[str, Any] | None = None) -> list[str]:
    required = {
        "schema_version", "task_id", "outer_trial_id", "seed", "measurement_class",
        "producer_revision", "task_package_digest", "population_manifest_digest",
        "harness_digest", "calibration_runner_digest", "noise_digest", "raw_result_digest",
        "fingerprint", "envelope_digest",
    }
    errors = [f"missing {key}" for key in sorted(required - set(payload))]
    if payload.get("schema_version") != 1:
        errors.append("schema_version mismatch")
    if payload.get("envelope_digest") != digest_mapping({key: value for key, value in payload.items() if key != "envelope_digest"}):
        errors.append("envelope_digest mismatch")
    for key, value in (expected or {}).items():
        if key == "fingerprint":
            actual = payload.get("fingerprint")
            if not isinstance(actual, dict) or not isinstance(value, dict):
                errors.append("fingerprint missing or invalid")
            else:
                compatible, reasons = fingerprints_compatible(actual, value)
                if not compatible:
                    errors.append("fingerprint mismatch: " + "; ".join(reasons))
        elif key not in payload:
            errors.append(f"missing {key}")
        elif payload.get(key) != value:
            errors.append(f"{key} mismatch")
    return errors


def classify_result(raw: dict[str, Any]) -> str:
    """Classify a persisted cell before resume reuse."""
    state = derive_cell_state(raw)
    if raw.get("failure_class") == "infrastructure" or str(raw.get("failure_stage", "")) in {"executor", "worker", "agent"}:
        return "rerun"
    if state == "resource_blocked":
        return "rerun"
    if state == "invalid":
        return "blocked_requires_revision"
    if state == "eligible" or (state == "ineligible" and "calibration_status" in raw):
        return "reusable"
    return "rerun"
