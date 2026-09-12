"""Derive family-level counterexample-dose summaries from a frozen artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping


DOSES = ("0.0", "0.25", "0.5", "1.0")


def _row_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    total = sum(int(row[name]) for name in ("tp", "fp", "fn", "tn"))
    deployed = int(row["tp"]) + int(row["fp"])
    return {
        "subset_seed": row["subset_seed"],
        "counterexample_count": row["counterexample_count"],
        "coverage": deployed / total if total else 0.0,
        "unsupported_risk_given_deployed": int(row["fp"]) / deployed if deployed else None,
        "unsafe_rate_all_contexts": row["unsafe_generalization_rate"],
        "description_length": row["description_length"],
        "deployed": deployed,
        "unsupported": int(row["fp"]),
        "total": total,
    }


def reanalyze_counterexample_dose(source: Mapping[str, Any]) -> dict[str, Any]:
    families: list[dict[str, Any]] = []
    for family_record in source["family_rows"]:
        dose_summaries: dict[str, Any] = {}
        for dose in DOSES:
            rows = [_row_summary(row) for row in family_record["counterexample_dose"][dose]["rows"]]
            risks = [row["unsupported_risk_given_deployed"] for row in rows if row["unsupported_risk_given_deployed"] is not None]
            lengths = [row["description_length"] for row in rows if row["description_length"] is not None]
            dose_summaries[dose] = {
                "subset_seeds": len(rows),
                "mean_coverage": mean(row["coverage"] for row in rows) if rows else None,
                "mean_unsupported_risk_given_deployed": mean(risks) if risks else None,
                "median_unsupported_risk_given_deployed": median(risks) if risks else None,
                "mean_unsafe_rate_all_contexts": mean(row["unsafe_rate_all_contexts"] for row in rows) if rows else None,
                "mean_description_length": mean(lengths) if lengths else None,
                "rows": rows,
            }
        risk_25 = dose_summaries["0.25"]["mean_unsupported_risk_given_deployed"]
        risk_100 = dose_summaries["1.0"]["mean_unsupported_risk_given_deployed"]
        families.append({
            "family": family_record["family"],
            "risk_25_minus_risk_100": risk_25 - risk_100 if risk_25 is not None and risk_100 is not None else None,
            "risk_non_increasing_25_to_100": risk_100 <= risk_25 if risk_25 is not None and risk_100 is not None else None,
            "doses": dose_summaries,
        })

    family_cluster_summary: dict[str, Any] = {}
    for dose in DOSES[1:]:
        values = [
            family["doses"][dose]["mean_unsupported_risk_given_deployed"]
            for family in families
            if family["doses"][dose]["mean_unsupported_risk_given_deployed"] is not None
        ]
        family_cluster_summary[dose] = {
            "family_count": len(values),
            "mean": mean(values) if values else None,
            "median": median(values) if values else None,
        }
    family_cluster_unsafe_rate_all_contexts: dict[str, Any] = {}
    for dose in DOSES[1:]:
        values = [
            family["doses"][dose]["mean_unsafe_rate_all_contexts"]
            for family in families
            if family["doses"][dose]["mean_unsafe_rate_all_contexts"] is not None
        ]
        family_cluster_unsafe_rate_all_contexts[dose] = {
            "family_count": len(values),
            "mean": mean(values) if values else None,
            "median": median(values) if values else None,
        }
    deltas = [family["risk_25_minus_risk_100"] for family in families if family["risk_25_minus_risk_100"] is not None]
    return {
        "schema_version": 1,
        "analysis": "counterexample_dose_family_reanalysis",
        "provider_calls": 0,
        "source_seed": source["seed"],
        "family_count": len(families),
        "subset_seeds": source["counterexample_dose"]["subset_seeds"],
        "risk_definition": "unsupported / deployed within each family and subset seed; zero-dose all-abstention has undefined conditional risk",
        "family_cluster_summary": family_cluster_summary,
        "family_cluster_unsafe_rate_all_contexts": family_cluster_unsafe_rate_all_contexts,
        "risk_25_minus_risk_100": {
            "family_count": len(deltas),
            "mean": mean(deltas) if deltas else None,
            "median": median(deltas) if deltas else None,
            "non_increasing_family_count": sum(bool(family["risk_non_increasing_25_to_100"]) for family in families),
        },
        "families": families,
        "interpretation": "provider-free mechanism reanalysis; family is the uncertainty unit; no hosted efficacy claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    report = reanalyze_counterexample_dose(source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
