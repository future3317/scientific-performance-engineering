#!/usr/bin/env python3
"""Issue calibration approval only for a complete current-revision bundle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmark.formal import attest
from benchmark.formal.approval import _digest, validate_calibration_approval
from benchmark.taskgen.validate_population import _json_digest, rebuild_calibration_views


def issue(
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
    if _json_digest(report) != _json_digest(canonical_report) or _json_digest(pilot) != _json_digest(canonical_pilot):
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
    empirical_digest = _json_digest(empirical)
    if report.get("empirical_calibration", {}).get("empirical_digest") != empirical_digest:
        raise ValueError("empirical digest does not match population report")
    if pilot.get("empirical_digest") != empirical_digest:
        raise ValueError("empirical digest does not match pilot calibration")
    protocol_path = repo_root / "benchmark" / "calibration" / "calibration_protocol.json"
    revision = attest.benchmark_revision(repo_root)
    body: dict[str, object] = {
        "schema_version": 1,
        "approved": True,
        "status": "approved",
        "population_report_digest": _json_digest(report),
        "pilot_calibration_digest": pilot.get("artifact_digest"),
        "empirical_digest": empirical_digest,
        "calibration_protocol_digest": attest.file_digest(protocol_path),
        "benchmark_revision": revision,
        "claims_digest": attest.file_digest(repo_root / "CLAIMS.yaml"),
        "statistical_protocol_digest": attest.file_digest(repo_root / "references" / "STATISTICAL_PROTOCOL.md"),
        "approved_task_ids": active_ids,
        "blocked_task_ids": [],
        "replacement_task_ids": [],
        "approver": approver,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "validator_revision": "strict-formal-v2",
        "review_policy": "active-30-complete-current-revision",
    }
    candidate = dict(body)
    candidate["approval_digest"] = _digest(body)
    errors = validate_calibration_approval(report, pilot, candidate, repo_root=repo_root)
    if errors:
        raise ValueError("approval validation failed: " + "; ".join(errors))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--empirical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--approver", required=True)
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()
    issue(
        report_path=args.report, pilot_path=args.pilot, empirical_path=args.empirical,
        out_path=args.out, repo_root=args.repo_root.resolve(), approver=args.approver,
        timestamp=args.timestamp,
    )
    print(json.dumps({"approved": True, "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
