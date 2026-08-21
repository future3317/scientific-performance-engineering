"""Canonical calibration approval construction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.calibration.report import rebuild_calibration_views
from benchmark.provenance import benchmark_revision, file_digest, json_digest


def validate_calibration_approval(
    report: dict[str, Any] | None,
    calibration: dict[str, Any] | None,
    approval: dict[str, Any] | None,
    *,
    active_manifest: dict[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> list[str]:
    """Validate the approval contract used by formal entry and claim gates."""
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
        body = {key: value for key, value in approval.items() if key != "approval_digest"}
        if approval.get("approval_digest") != json_digest(body):
            errors.append("calibration approval digest mismatch")
    if repo_root is not None:
        current_revision = benchmark_revision(repo_root)
        if approval.get("benchmark_revision") != current_revision:
            errors.append("calibration approval benchmark_revision mismatch")
    if not isinstance(report, dict) or not isinstance(calibration, dict):
        errors.append("population report or pilot calibration is missing")
        return errors
    if approval.get("population_report_digest") != json_digest(report):
        errors.append("calibration approval population digest mismatch")
    if approval.get("pilot_calibration_digest") != calibration.get("artifact_digest"):
        errors.append("calibration approval pilot digest mismatch")
    expected_empirical = report.get("empirical_calibration", {}).get("empirical_digest")
    if approval.get("empirical_digest") != expected_empirical or approval.get("empirical_digest") != calibration.get("empirical_digest"):
        errors.append("calibration approval empirical digest mismatch")
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
        for field, path in {
            "claims_digest": root / "CLAIMS.yaml",
            "statistical_protocol_digest": root / "references" / "STATISTICAL_PROTOCOL.md",
            "calibration_protocol_digest": root / "benchmark" / "calibration" / "calibration_protocol.json",
        }.items():
            expected = file_digest(path) if path.is_file() else None
            if expected is None or approval.get(field) != expected:
                errors.append(f"calibration approval {field} mismatch")
    return errors


def issue_calibration_approval(
    *, report_path: Path, pilot_path: Path, empirical_path: Path,
    out_path: Path, repo_root: Path, approver: str, timestamp: str | None,
) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    empirical = json.loads(empirical_path.read_text(encoding="utf-8"))
    canonical_report, canonical_pilot, rebuild_errors = rebuild_calibration_views(
        tasks_root=repo_root / "benchmark" / "tasks",
        empirical_path=empirical_path,
        manifest_path=repo_root / "benchmark" / "pilot_population.json",
    )
    if rebuild_errors:
        raise ValueError("canonical calibration projections are not ready: " + "; ".join(rebuild_errors))
    if json_digest(report) != json_digest(canonical_report) or json_digest(pilot) != json_digest(canonical_pilot):
        raise ValueError("derived calibration projections do not match canonical rebuild")
    report = canonical_report
    pilot = canonical_pilot
    active_ids = [str(item) for item in report.get("active_task_ids", [])]
    if report.get("empirical_calibration", {}).get("calibration_gate") != "ready_for_review":
        raise ValueError("population report is not ready_for_review")
    if pilot.get("calibration_gate") != "ready_for_review":
        raise ValueError("pilot calibration is not ready_for_review")
    if set(str(item.get("task_id")) for item in pilot.get("tasks", [])) != set(active_ids):
        raise ValueError("pilot calibration does not cover the active population")
    if any(item.get("status") != "eligible" or item.get("eligibility") is not True for item in pilot.get("tasks", [])):
        raise ValueError("every active task must be eligible before approval")
    empirical_digest = json_digest(empirical)
    if report.get("empirical_calibration", {}).get("empirical_digest") != empirical_digest:
        raise ValueError("empirical digest does not match population report")
    if pilot.get("empirical_digest") != empirical_digest:
        raise ValueError("empirical digest does not match pilot calibration")
    protocol_path = repo_root / "benchmark" / "calibration" / "calibration_protocol.json"
    revision = benchmark_revision(repo_root)
    body: dict[str, object] = {
        "schema_version": 1, "approved": True, "status": "approved",
        "population_report_digest": json_digest(report),
        "pilot_calibration_digest": pilot.get("artifact_digest"),
        "empirical_digest": empirical_digest,
        "calibration_protocol_digest": file_digest(protocol_path),
        "benchmark_revision": revision,
        "claims_digest": file_digest(repo_root / "CLAIMS.yaml"),
        "statistical_protocol_digest": file_digest(repo_root / "references" / "STATISTICAL_PROTOCOL.md"),
        "approved_task_ids": active_ids, "blocked_task_ids": [], "replacement_task_ids": [],
        "approver": approver,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "validator_revision": "strict-formal-v2",
        "review_policy": "active-30-complete-current-revision",
    }
    candidate = dict(body)
    candidate["approval_digest"] = json_digest(body)
    errors = validate_calibration_approval(report, pilot, candidate, repo_root=repo_root)
    if errors:
        raise ValueError("approval validation failed: " + "; ".join(errors))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return candidate
