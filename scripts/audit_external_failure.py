"""Audit prior public-skill and hosted campaign artifacts without rerunning them."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


ROOT_CAUSES = (
    "protocol_failure",
    "intervention_failure",
    "context_schema_failure",
    "evidence_sparsity",
    "executor_failure",
    "treatment_no_effect",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _nested_count(report: Mapping[str, Any], names: tuple[str, ...]) -> int | None:
    direct = _first(report, *names)
    if direct is not None and not isinstance(direct, (dict, list)):
        return _number(direct)
    summary = report.get("policy_summary")
    if isinstance(summary, Mapping):
        for value in summary.values():
            if isinstance(value, Mapping):
                found = _first(value, *names)
                if found is not None and not isinstance(found, (dict, list)):
                    return _number(found)
    return None


def _contexts(rows: list[Any]) -> set[tuple[tuple[str, str], ...]]:
    combinations: set[tuple[tuple[str, str], ...]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        context = row.get("context") or row.get("public_context")
        if not isinstance(context, Mapping):
            continue
        combinations.add(tuple(sorted((str(k), str(v)) for k, v in context.items())))
    return combinations


def _predicate_rejections(report: Mapping[str, Any]) -> dict[str, int]:
    names = {
        "no_positive_support": 0,
        "contains_counterexample": 0,
        "insufficient_evidence": 0,
        "no_test_context_match": 0,
    }
    candidates = report.get("candidate_predicates", [])
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            status = str(candidate.get("status", "")).lower().replace("-", "_").replace(" ", "_")
            for name in names:
                if name in status:
                    names[name] += 1
    explicit = report.get("candidate_predicate_rejections")
    if isinstance(explicit, Mapping):
        for name in names:
            if name in explicit:
                names[name] = _number(explicit[name])
    return names


def _benchmark_report(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    if not isinstance(report, Mapping):
        raise ValueError(f"benchmark report must be an object: {path}")
    rows = report.get("rows", [])
    rows = rows if isinstance(rows, list) else []
    train_instances = _first(report, "train_evidence", "train_instances") or []
    train_evidence = len(train_instances) if isinstance(train_instances, list) else _number(train_instances)
    reason = _first(report, "full_acre_reason", "acre_zero_reason")
    if reason is None:
        policy_summary = report.get("policy_summary")
        acre = policy_summary.get("policies", {}).get("acre", {}) if isinstance(policy_summary, Mapping) else {}
        reason = acre.get("reason") if isinstance(acre, Mapping) else None
    if reason is None:
        reason = "not_persisted_in_artifact"

    values = {
        "train_evidence": train_evidence,
        "positive_anchors": _nested_count(report, ("positive_anchors", "anchors")),
        "counterexamples": _nested_count(report, ("counterexamples", "ces")),
        "unresolved": _nested_count(report, ("unresolved", "unresolved_count")),
        "repairs": _nested_count(report, ("repairs",)),
        "regressions": _nested_count(report, ("regressions",)),
        "both_pass": _nested_count(report, ("both_pass",)),
        "both_fail": _nested_count(report, ("both_fail",)),
    }
    return {
        **values,
        "unique_context_feature_combinations": len(_contexts(rows)),
        "candidate_predicates": len(report.get("candidate_predicates", []))
        if isinstance(report.get("candidate_predicates", []), list)
        else None,
        "candidate_predicate_rejections": _predicate_rejections(report),
        "full_acre_coverage": _first(report, "full_acre_coverage")
        or (report.get("policy_summary", {}).get("macro", {}).get("acre", {}).get("coverage")
            if isinstance(report.get("policy_summary"), Mapping)
            else None),
        "full_acre_zero_reason": reason,
        "source": str(path),
        "revision": report.get("revision"),
        "provider_calls": report.get("provider_calls"),
    }


def audit_skillgen_suite(path: Path) -> dict[str, Any]:
    """Summarize each benchmark report found below a prior suite artifact."""
    reports: dict[str, Any] = {}
    for report_path in sorted(path.glob("*/report.json")):
        benchmark = report_path.parent.name
        reports[benchmark] = _benchmark_report(report_path)
    return {
        "source": str(path),
        "benchmarks": reports,
        "benchmark_count": len(reports),
        "missing_benchmarks": sorted({"scienceworld", "mind2web", "toolbench", "mcp_bench"} - set(reports)),
        "provider_calls": 0,
        "root_cause_labels": list(ROOT_CAUSES),
    }


def _quantile_nearest_rank(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _hosted_rows(path: Path) -> list[Mapping[str, Any]]:
    if path.is_file():
        value = _read_json(path)
        return value if isinstance(value, list) else list(value.get("rows", [])) if isinstance(value, Mapping) else []
    rows: list[Mapping[str, Any]] = []
    for item in sorted(path.rglob("*.json")):
        value = _read_json(item)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, Mapping))
        elif isinstance(value, Mapping) and isinstance(value.get("rows"), list):
            rows.extend(row for row in value["rows"] if isinstance(row, Mapping))
    return rows


def audit_hosted_campaign(path: Path) -> dict[str, Any]:
    """Reconstruct hosted funnel counts from raw rows, retaining unknown statuses."""
    rows = _hosted_rows(path)
    by_condition: dict[str, Counter[str]] = defaultdict(Counter)
    tokens: list[int] = []
    turns: list[int] = []
    known = {
        "scheduled", "launched", "execution_valid", "reached_verifier", "contract_fail",
        "anticheat_fail", "turn_limit", "input_token_limit", "controlled_termination",
        "measurement_valid", "positive_effect",
    }
    for row in rows:
        condition = str(row.get("condition", "unknown"))
        status = str(row.get("status", row.get("termination_reason", "unknown"))).lower()
        status = status.replace("-", "_").replace(" ", "_")
        if status not in known:
            status = "unknown_status"
        by_condition[condition][status] += 1
        for target, output in (("input_tokens", tokens), ("turns", turns)):
            value = row.get(target)
            if isinstance(value, (int, float)) and math.isfinite(value):
                output.append(int(value))

    total = len(rows)
    all_counts = Counter()
    for counter in by_condition.values():
        all_counts.update(counter)
    for counter in by_condition.values():
        counter["scheduled"] = sum(counter.values())
    return {
        "source": str(path),
        "scheduled": total,
        "by_condition": {condition: dict(sorted(counter.items())) for condition, counter in sorted(by_condition.items())},
        "funnel": dict(sorted(all_counts.items())),
        "input_tokens": {
            "median": _quantile_nearest_rank(tokens, 0.5),
            "p90": _quantile_nearest_rank(tokens, 0.9),
            "n": len(tokens),
        },
        "turns": {
            "median": _quantile_nearest_rank(turns, 0.5),
            "p90": _quantile_nearest_rank(turns, 0.9),
            "n": len(turns),
        },
        "invariants": {"status_count_equals_scheduled": sum(all_counts.values()) == total},
        "root_cause_labels": list(ROOT_CAUSES),
    }


def write_failure_audit(out_dir: Path, suite_paths: Mapping[str, Path], hosted_path: Path | None) -> None:
    payload: dict[str, Any] = {"skillgen": {}, "hosted": None, "provider_calls": 0}
    for name, path in suite_paths.items():
        payload["skillgen"][name] = audit_skillgen_suite(path)
    if hosted_path is not None and hosted_path.exists():
        payload["hosted"] = audit_hosted_campaign(hosted_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "external_failure_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# External Failure Audit", "", "This is a read-only audit of prior artifacts; no provider calls were made.", ""]
    for name, report in payload["skillgen"].items():
        lines.append(f"## {name}")
        lines.append(f"- benchmark reports: {report['benchmark_count']}")
        lines.append(f"- missing benchmarks: {', '.join(report['missing_benchmarks']) or 'none'}")
    if payload["hosted"] is not None:
        lines.extend(["", "## Hosted campaign", f"- scheduled rows: {payload['hosted']['scheduled']}"])
    (out_dir / "external_failure_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
