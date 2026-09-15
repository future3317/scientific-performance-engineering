#!/usr/bin/env python3
"""Run the calibration-frozen provider-free policy frontier."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.boundary.policy_eval import clustered_frontier
from benchmark.families import PILOT_FAMILIES
from scripts.run_rscb_static_frontier import _rows_for_family, run_frozen_frontier


METRICS = ("coverage", "unsupported_all", "unsupported_deployed", "recall", "incremental_policy_utility")


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap(rows: Sequence[Mapping[str, Any]], reps: int, seed: int) -> dict[str, Any]:
    clusters: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[(row.get("family"), row.get("seed"))].append(row)
    cluster_rows = list(clusters.values())
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for _ in range(reps):
        sampled = [row for cluster in (rng.choice(cluster_rows) for _ in cluster_rows) for row in cluster]
        for metric in METRICS:
            values = [float(row[metric]) for row in sampled if isinstance(row.get(metric), (int, float))]
            if values:
                draws[metric].append(mean(values))
    return {
        "replicates": reps,
        "cluster_unit": "family_seed",
        "interval": {metric: [_quantile(values, 0.025), _quantile(values, 0.975)] for metric, values in draws.items()},
    }


def _permutation_summary(
    *,
    families: Sequence[str],
    seeds: Sequence[int],
    surface_count: int,
    reps: int,
    seed: int = 0,
) -> dict[str, Any]:
    observed: list[int] = []
    null_counts: list[int] = []
    rng = random.Random(seed)
    for family in families:
        for split_seed in seeds:
            split = _rows_for_family(family, surface_count, split_seed)
            labels = [
                bool(record["tau"] > 0.0 and record["scientific_gate"])
                for record in split["score_records"].values()
            ]
            observed.append(sum(labels))
            for _ in range(reps):
                shuffled = list(labels)
                rng.shuffle(shuffled)
                null_counts.append(sum(shuffled))
    return {
        "target": "counterexample_assignment_only",
        "observed_supported_counts": observed,
        "null_supported_count_interval": [_quantile(null_counts, 0.025), _quantile(null_counts, 0.975)],
        "replicates": reps,
    }


def _aurc(rows: Sequence[Mapping[str, Any]]) -> float | None:
    points = sorted(
        (
            float(row["coverage"]),
            float(row["unsupported_deployed"]),
        )
        for row in rows
        if isinstance(row.get("coverage"), (int, float))
        and isinstance(row.get("unsupported_deployed"), (int, float))
    )
    if len(points) < 2:
        return 0.0 if points else None
    return sum((right[0] - left[0]) * (left[1] + right[1]) / 2.0 for left, right in zip(points, points[1:]))


def run_controlled_p0(
    *,
    surface_count: int = 100,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    families: Sequence[str] = PILOT_FAMILIES,
    bootstrap_reps: int = 10000,
    permutation_reps: int = 100,
) -> dict[str, Any]:
    reports = [
        run_frozen_frontier(surface_count=surface_count, seed=int(seed), families=families)
        for seed in seeds
    ]
    rows = [row for report in reports for row in report["rows"]]
    summaries: list[dict[str, Any]] = []
    for policy in sorted({str(row["policy"]) for row in rows}):
        policy_rows = [row for row in rows if row["policy"] == policy]
        summary = {"policy": policy, **clustered_frontier(policy_rows, cluster_keys=("family", "seed"))}
        summary["bootstrap"] = _bootstrap(policy_rows, bootstrap_reps, seed=0)
        summaries.append(summary)

    frontier_rows = [row for row in rows if str(row["policy"]).startswith("rscb_frontier_")]
    acre_rows = [row for row in rows if row["policy"] == "acre_full"]
    matched_rows = [row for row in rows if row["policy"] == "rscb_matched"]
    matched_pairs = []
    for acre in acre_rows:
        candidates = [
            row for row in matched_rows
            if row.get("family") == acre.get("family") and row.get("seed") == acre.get("seed")
        ]
        if candidates:
            matched = candidates[0]
            matched_pairs.append({
                "family": acre.get("family"),
                "seed": acre.get("seed"),
                "acre_coverage": acre.get("coverage"),
                "acre_risk": acre.get("unsupported_deployed"),
                "rscb_coverage": matched.get("coverage"),
                "rscb_risk": matched.get("unsupported_deployed"),
            })

    return {
        "schema_version": 1,
        "analysis": "controlled_calibration_frozen_policy_frontier",
        "provider_calls": 0,
        "surface_count": int(surface_count),
        "seeds": [int(seed) for seed in seeds],
        "families": list(families),
        "rows": rows,
        "summaries": summaries,
        "risk_coverage_frontier": frontier_rows,
        "risk_at_acre_matched_coverage": matched_pairs,
        "aurc_by_family_seed": {
            f"{family}|{seed}": _aurc([
                row for row in frontier_rows if row.get("family") == family and row.get("seed") == seed
            ])
            for family in families for seed in seeds
        },
        "permutation": _permutation_summary(
            families=families,
            seeds=seeds,
            surface_count=surface_count,
            reps=permutation_reps,
        ),
        "statistics": {
            "bootstrap_replicates": int(bootstrap_reps),
            "permutation_replicates": int(permutation_reps),
            "cluster_unit": "family_seed",
            "permutation_target": "counterexample_assignment_only",
        },
        "claim_boundary": "provider-free calibration-frozen support-policy comparison; no hosted efficacy claim",
    }


def write_controlled_p0(out_root: Path, report: Mapping[str, Any]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "controlled_p0.json").write_text(
        json.dumps(dict(report), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (out_root / "controlled_p0_manifest.json").write_text(
        json.dumps({
            "analysis": report["analysis"],
            "provider_calls": report["provider_calls"],
            "surface_count": report["surface_count"],
            "seeds": report["seeds"],
            "families": report["families"],
            "statistics": report["statistics"],
            "claim_boundary": report["claim_boundary"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-count", type=int, default=100)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/external_real/controlled_p0")
    args = parser.parse_args()
    report = run_controlled_p0(surface_count=args.surface_count)
    write_controlled_p0(args.out_root, report)
    print(json.dumps({"status": "complete", "provider_calls": 0, "out_root": str(args.out_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
