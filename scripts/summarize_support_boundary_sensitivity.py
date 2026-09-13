"""Create a compact, machine-readable summary from a sensitivity artifact."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


METRICS = (
    "coverage",
    "unsupported_deployed",
    "unsupported_all",
    "support_precision",
    "support_recall",
    "support_decision_regret_mean",
    "anchor_count",
    "counterexample_count",
    "unresolved_count",
    "predicate_depth",
    "predicate_literals",
    "predicate_dl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["configuration_id"]), []).append(row)
    summary: dict[str, Any] = {}
    for configuration_id, members in sorted(grouped.items()):
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            values = _numeric([member.get(metric) for member in members])
            metrics[metric] = {
                "n": len(values),
                "mean": sum(values) / len(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        summary[configuration_id] = {
            "row_count": len(members),
            "families": sorted({str(member["family"]) for member in members}),
            "seeds": sorted({int(member["seed"]) for member in members}),
            "metrics": metrics,
        }
    return summary


def _grid_accounting(report: dict[str, Any]) -> dict[str, Any]:
    expected = [
        {"configuration_id": "default", "factor": "default", "parameter": "registered", "value": 0.95},
        *[
            {"configuration_id": f"confidence_{value:.2f}", "factor": "confidence", "parameter": "confidence_level", "value": value}
            for value in report["fixed_sensitivity_grid"]["confidence_level"]
        ],
        *[
            {"configuration_id": f"grammar_depth_{int(value):+d}", "factor": "grammar", "parameter": "depth_offset", "value": value}
            for value in report["fixed_sensitivity_grid"]["grammar_depth_offset"]
        ],
        *[
            {"configuration_id": f"grammar_literals_{int(value):+d}", "factor": "grammar", "parameter": "literal_offset", "value": value}
            for value in report["fixed_sensitivity_grid"]["grammar_literal_offset"]
        ],
        *[
            {"configuration_id": f"dose_{int(round(value * 100))}", "factor": "counterexample_dose", "parameter": "dose", "value": value}
            for value in report["doses"]
        ],
    ]
    executed = {item["configuration_id"] for item in report["configuration_registry"]}
    not_executed = [
        {
            **item,
            "status": "not_applicable",
            "reason": "grammar bound rejects this offset; no independent implementation exists at the requested +1 setting",
        }
        for item in expected
        if item["configuration_id"] not in executed
    ]
    return {
        "expected_configurations": expected,
        "executed_configuration_ids": sorted(executed),
        "not_executed": not_executed,
    }


def build_summary(artifact: Path) -> dict[str, Any]:
    report = json.loads(artifact.read_text(encoding="utf-8"))
    status_counts = Counter(str(row["status"]) for row in report["metric_rows"])
    return {
        "schema_version": 1,
        "analysis": report["analysis"],
        "source_artifact": str(artifact),
        "source_artifact_sha256": _sha256(artifact),
        "code_commit": report["code_commit"],
        "registered_input_artifact": report["registered_input_artifact"],
        "registered_input_artifact_sha256": report["registered_input_artifact_sha256"],
        "fixed_sensitivity_grid": report["fixed_sensitivity_grid"],
        "availability": report["availability"],
        "configuration_registry": report["configuration_registry"],
        "grid_accounting": _grid_accounting(report),
        "metric_summary": _aggregate(report["metric_rows"]),
        "dose_summary": _aggregate(report["dose_rows"]),
        "status_counts": dict(sorted(status_counts.items())),
        "stability": report["stability"],
        "hidden_truth_semantics": report["hidden_truth_semantics"],
        "underidentified_operational_semantics": report["underidentified_operational_semantics"],
        "claim_boundary": report["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.artifact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
