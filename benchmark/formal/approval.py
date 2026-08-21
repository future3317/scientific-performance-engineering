"""Single approval validator shared by readiness, formal entry, and claim gates."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def validate_calibration_approval(
    report: dict[str, Any] | None,
    calibration: dict[str, Any] | None,
    approval: dict[str, Any] | None,
    *,
    active_manifest: dict[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> list[str]:
    """Validate the one approval contract used by every formal gate."""
    errors: list[str] = []
    if not isinstance(approval, dict):
        return ["calibration approval is missing or invalid"]
    if approval.get("schema_version") != 1:
        errors.append("calibration approval schema_version must be 1")
    if approval.get("approved") is not True:
        errors.append("calibration approval is not approved")
    for key in ("approval_digest", "approver", "timestamp", "population_report_digest", "pilot_calibration_digest"):
        if not approval.get(key):
            errors.append(f"calibration approval missing {key}")
    if approval.get("approved") is True:
        for key in ("empirical_digest", "calibration_protocol_digest", "benchmark_revision"):
            if not approval.get(key):
                errors.append(f"calibration approval missing {key}")
    if approval.get("approved") is True:
        body = {key: value for key, value in approval.items() if key != "approval_digest"}
        if approval.get("approval_digest") != _digest(body):
            errors.append("calibration approval digest mismatch")
    if repo_root is not None:
        try:
            current_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            current_revision = None
        if not approval.get("benchmark_revision"):
            errors.append("calibration approval missing benchmark_revision")
        elif current_revision and approval.get("benchmark_revision") != current_revision:
            errors.append("calibration approval benchmark_revision mismatch")
    if not isinstance(report, dict) or not isinstance(calibration, dict):
        errors.append("population report or pilot calibration is missing")
        return errors
    if approval.get("population_report_digest") != _digest(report):
        errors.append("calibration approval population digest mismatch")
    if approval.get("pilot_calibration_digest") != calibration.get("artifact_digest"):
        errors.append("calibration approval pilot digest mismatch")
    expected_empirical = report.get("empirical_calibration", {}).get("empirical_digest")
    if approval.get("empirical_digest") != expected_empirical:
        errors.append("calibration approval empirical digest mismatch")
    if approval.get("empirical_digest") != calibration.get("empirical_digest"):
        errors.append("calibration approval pilot empirical digest mismatch")
    active_ids = set(str(item) for item in (active_manifest or {}).get("task_ids", report.get("active_task_ids", [])))
    calibrated = {str(item.get("task_id")) for item in calibration.get("tasks", []) if isinstance(item, dict)}
    approved_tasks = set(str(item) for item in approval.get("approved_task_ids", []))
    if active_ids and calibrated != active_ids:
        errors.append("pilot calibration coverage does not match active population")
    if active_ids and approved_tasks != active_ids:
        errors.append("approval approved_task_ids do not match active population")
    if any(item.get("status") != "eligible" or item.get("eligibility") is not True for item in calibration.get("tasks", []) if isinstance(item, dict)):
        errors.append("all active tasks must have eligible empirical calibration")
    if report.get("retired_for_formal") and any(str(item.get("task_id")) in active_ids for item in report.get("retired_for_formal", [])):
        errors.append("retired task remains active")
    if repo_root is not None:
        root = Path(repo_root)
        bindings = {
            "benchmark_revision": root,
            "claims_digest": root / "CLAIMS.yaml",
            "statistical_protocol_digest": root / "references" / "STATISTICAL_PROTOCOL.md",
            "calibration_protocol_digest": root / "benchmark" / "calibration" / "calibration_protocol.json",
        }
        for field, path in bindings.items():
            if field in approval and field.endswith("_digest"):
                expected = _file_digest(path) if path.is_file() else None
                if expected is None or approval.get(field) != expected:
                    errors.append(f"calibration approval {field} mismatch")
    return errors


__all__ = ["validate_calibration_approval"]
