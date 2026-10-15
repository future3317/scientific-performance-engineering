"""Read historical formal receipts without inferring stages from zero scores."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TaskAttemptReceipt:
    task_id: str
    arm: str
    outer: str
    family: str | None
    stage: str
    failure_code: str | None
    execution_valid: bool
    verifier_invoked: bool
    correctness_pass: bool | None
    measurement_reached: bool
    verified_positive: bool | None
    input_tokens: int | None
    output_tokens: int | None
    tool_calls: int | None
    wall_seconds: float | None
    tool_trace_count: int | None
    benchmark_revision: str | None
    controller_revision: str | None
    source: str
    source_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_attempt(trial: Mapping[str, Any], result: Mapping[str, Any] | None,
                  receipt: Mapping[str, Any], *, source: str, source_digest: str) -> TaskAttemptReceipt:
    valid = trial.get("execution_validity") == "valid"
    host = receipt.get("host_controller") or {}
    usage = receipt.get("usage") or trial.get("agent_usage") or {}
    experiment = trial.get("experiment") or {}
    measured = bool(result and result.get("measurement"))
    correctness = result.get("correctness_pass") if result is not None else None
    positive = (result.get("verified_speedup") or {}).get("verified") if result else None
    if not valid:
        stage, reason = "nonlaunched", trial.get("execution_validity")
    elif result is None:
        stage, reason = "premature", host.get("outcome") or "missing_terminal_reason"
    elif (result.get("anticheat") or {}).get("hard_fail") is True:
        stage, reason = "correctness_failure", "anticheat_hard_fail"
    elif correctness is False:
        errors = " ".join(str(x) for x in result.get("errors", []))
        stage = "correctness_failure"
        reason = "api_contract_violation" if "API" in errors or "S2" in errors else "unclassified_correctness"
    elif measured:
        stage, reason = ("verified_positive", None) if positive is True else ("measurement_only", "speedup_unverified")
    else:
        stage, reason = "verifier_unresolved", "measurement_not_recorded"
    trace = host.get("tool_trace")
    return TaskAttemptReceipt(
        task_id=str(trial["task_id"]), arm=str(trial["condition"]),
        outer=str(trial["outer_trial_id"]), family=trial.get("family"),
        stage=stage, failure_code=reason, execution_valid=valid,
        verifier_invoked=result is not None, correctness_pass=correctness,
        measurement_reached=measured, verified_positive=positive,
        input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
        tool_calls=usage.get("tool_calls"), wall_seconds=usage.get("wall_time_s"),
        tool_trace_count=len(trace) if isinstance(trace, list) else None,
        benchmark_revision=experiment.get("benchmark_revision"),
        controller_revision=(experiment.get("agent_config") or {}).get("agent_code_commit"),
        source=source, source_digest=source_digest,
    )


def audit_root(root: Path) -> tuple[list[TaskAttemptReceipt], dict[str, Any]]:
    campaign_path = root / "campaign.json"
    campaign = json.loads(campaign_path.read_text())
    rows = []
    identities = set()
    digests = {}
    for path in sorted(root.glob("trials/*/*/trial.json")):
        trial = json.loads(path.read_text())
        identity = (trial["task_id"], trial["condition"], trial["outer_trial_id"])
        if identity in identities:
            raise ValueError(f"duplicate scheduled cell: {identity}")
        identities.add(identity)
        values = {}
        for name in ("trial.json", "result.json", "executor_receipt.json"):
            file = path.parent / name
            if file.exists():
                raw = file.read_bytes()
                digests[str(file.relative_to(root))] = hashlib.sha256(raw).hexdigest()
                values[name] = json.loads(raw)
        receipt = values.get("executor_receipt.json") or trial.get("executor_receipt") or {}
        rows.append(parse_attempt(trial, values.get("result.json"), receipt,
                                 source=str(path.relative_to(root)),
                                 source_digest=digests[str(path.relative_to(root))]))
    counts = Counter(row.stage for row in rows)
    summary = {
        "source_root": str(root), "source_campaign_sha256": hashlib.sha256(campaign_path.read_bytes()).hexdigest(),
        "expected_cells": campaign["schedule_size"], "observed_cells": len(rows),
        "execution_valid": sum(row.execution_valid for row in rows),
        "verifier_records": sum(row.verifier_invoked for row in rows),
        "stages": dict(counts), "failures": dict(Counter(row.failure_code for row in rows if row.failure_code)),
        "source_digests": digests, "provider_calls": 0,
        "interpretation": "Read-only retrospective diagnosis; no new efficacy evidence.",
    }
    if len(rows) != campaign["schedule_size"]:
        raise ValueError("schema_discrepancy: scheduled cells do not close")
    return rows, summary


def verify_frozen_counts(summary: Mapping[str, Any], frozen: Mapping[str, Any]) -> None:
    audit = frozen["verifier_and_failure_audit"]
    expected = {
        "observed_cells": frozen["matrix"]["expected_cells"],
        "execution_valid": audit["execution_valid_cells"],
        "verifier_records": audit["explicit_verifier_result_records"],
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise ValueError(f"schema_discrepancy: {key}: {summary[key]} != {value}")
    for stage, expected_count in {
        "nonlaunched": frozen["matrix"]["resource_blocked_cells"],
        "premature": audit["agent_outcome_failures_without_verifier_record"],
        "correctness_failure": audit["correctness_failures_among_verifier_records"],
        "measurement_only": audit["measurement_records"],
    }.items():
        if summary["stages"].get(stage, 0) != expected_count:
            raise ValueError(f"schema_discrepancy: {stage}")
