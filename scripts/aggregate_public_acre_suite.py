#!/usr/bin/env python3
"""Aggregate public fixed-skill reports by benchmark (macro, not pooled)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = (
    "coverage",
    "unsupported_deployed_rate",
    "unsupported_all_rate",
    "repair_recall",
    "support_decision_regret",
    "net_score_delta",
)


def aggregate(root: Path) -> dict[str, Any]:
    reports = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*/report.json"))}
    policies = sorted({name for report in reports.values() for name in report["policy_summary"]["policies"]})
    macro: dict[str, dict[str, float | None]] = {}
    for policy in policies:
        macro[policy] = {}
        for metric in METRICS:
            values = [float(report["policy_summary"]["policies"][policy][metric]) for report in reports.values() if report["policy_summary"]["policies"][policy].get(metric) is not None]
            macro[policy][metric] = mean(values) if values else None
    return {
        "schema_version": 1,
        "suite_revision": json.loads((root / "manifest.json").read_text(encoding="utf-8"))["revision"],
        "benchmarks": sorted(reports),
        "benchmark_count": len(reports),
        "macro_unit": "benchmark x model cell",
        "macro": macro,
        "per_benchmark": {name: report["policy_summary"]["policies"] for name, report in reports.items()},
        "provider_calls": sum(int(report["provider_calls"]) for report in reports.values()),
        "test_instances": sum(len(report["test_instances"]) for report in reports.values()),
        "claim_boundary": "public fixed-intervention audit; not prospective SkillGen generation efficacy",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.root)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
