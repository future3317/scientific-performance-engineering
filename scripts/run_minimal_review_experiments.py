#!/usr/bin/env python3
"""Produce the minimal provider-free evidence package for reviewer follow-up.

The applicability and relation reports are reaggregations of persisted raw
evidence.  Only lifecycle replicas execute the existing provider-free episode
harness; no provider, hosted, formal, or sealed path is reachable here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping

from benchmark.harness.evolution import run_episode
from benchmark.interaction.factorial_bench import relation_recovery_metrics
from scripts.run_pilot_surface_experiments import run_budget_aware_ablation, run_budget_aware_reserve_stress


EPISODES = (
    ("poison", "benchmark/tasks/EVOL-EPISODE-POISON-10/episodes/poison_episode.yaml"),
    ("compiler_drift", "benchmark/tasks/EVOL-COMPILER-DRIFT-20/episodes/compiler_drift_episode.yaml"),
    ("equivariant_specialization", "benchmark/tasks/EVOL-EQUIVARIANT-SPECIALIZE-30/episodes/equivariant_specialization_episode.yaml"),
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _risk_row(metrics: Mapping[str, Any]) -> dict[str, Any]:
    total = int(metrics.get("tp", 0)) + int(metrics.get("fp", 0)) + int(metrics.get("fn", 0)) + int(metrics.get("tn", 0))
    covered = int(metrics.get("tp", 0)) + int(metrics.get("fp", 0))
    return {
        "coverage": covered / total if total else None,
        "unsafe_risk": int(metrics.get("fp", 0)) / covered if covered else None,
        "unsafe_generalization_rate": metrics.get("unsafe_generalization_rate"),
        "abstention_rate": metrics.get("abstention_rate"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "total": total,
    }


def build_risk_coverage_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Reaggregate held-out confusion counts without adding a threshold."""
    family_results = raw.get("family_results")
    if not isinstance(family_results, list) or not family_results:
        raise ValueError("applicability raw evidence lacks family_results")
    policies = [str(item) for item in raw.get("policies", [])]
    prefixes = [str(item) for item in raw.get("evidence_prefixes", [])]
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for policy in policies:
        summary[policy] = {}
        for prefix in prefixes:
            rows: list[dict[str, Any]] = []
            for family in family_results:
                metrics = family["prefixes"][prefix]["policies"][policy]
                rows.append(_risk_row(metrics))
            summary[policy][prefix] = {
                "family_cluster_mean": {
                    key: _mean([float(row[key]) for row in rows if row[key] is not None])
                    for key in ("coverage", "unsafe_risk", "unsafe_generalization_rate", "abstention_rate", "recall", "f1")
                },
                "family_count": len(rows),
                "source_metric": "persisted held-out confusion counts",
                "new_threshold": False,
            }
    return {
        "schema_version": 1,
        "analysis": "descriptive_risk_coverage_reaggregation",
        "provider_calls": 0,
        "coverage_definition": "(tp+fp)/(tp+fp+fn+tn)",
        "unsafe_risk_definition": "fp/(tp+fp) among covered predictions",
        "summary": summary,
    }


def aggregate_relation_budget_curve(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate persisted per-block relation estimates; never rerun factorials."""
    blocks = [int(item) for item in raw.get("block_schedule", [])]
    surfaces = raw.get("surface_results")
    if not blocks or not isinstance(surfaces, list) or not surfaces:
        raise ValueError("relation raw evidence lacks block_schedule or surface_results")
    curves: list[dict[str, Any]] = []
    for block in blocks:
        details: list[dict[str, object]] = []
        for surface in surfaces:
            estimate = next((item for item in surface.get("estimates", []) if int(item["blocks"]) == block), None)
            if estimate is None:
                raise ValueError(f"missing persisted estimate at block {block}")
            details.append({
                "hidden_relation": surface["hidden_relation"],
                "predicted_relation": estimate["decision"],
            })
        metrics = relation_recovery_metrics(details)
        curves.append({"blocks": block, **metrics})
    return {
        "schema_version": 1,
        "analysis": "persisted_relation_block_reaggregation",
        "provider_calls": 0,
        "block_schedule": blocks,
        "surface_count": len(surfaces),
        "curves": curves,
        "factorial_rerun": False,
    }


def _episode_metric(result: Mapping[str, Any], condition: str, name: str) -> float | None:
    value = result[condition]["metrics"].get(name)
    if isinstance(value, Mapping):
        value = value.get("total")
    return float(value) if isinstance(value, (int, float)) else None


def run_lifecycle_replicates(repo_root: Path, output_root: Path, replicas_per_episode: int = 20) -> dict[str, Any]:
    if replicas_per_episode < 1:
        raise ValueError("replicas_per_episode must be positive")
    rows: list[dict[str, Any]] = []
    raw_root = output_root / "lifecycle_raw"
    for episode_index, (episode_name, relative_path) in enumerate(EPISODES):
        episode_path = repo_root / relative_path
        for replicate in range(replicas_per_episode):
            seed = 20260910 + episode_index * 1000 + replicate
            replicate_id = f"{episode_name}-rep{replicate:02d}"
            row: dict[str, Any] = {"episode": episode_name, "replicate": replicate, "replicate_id": replicate_id, "seed": seed}
            for condition in ("C", "D"):
                out_dir = raw_root / episode_name / replicate_id / condition
                try:
                    result = run_episode(
                        episode_path,
                        condition,
                        out_dir,
                        core_repo=repo_root,
                        context_mode="reset",
                        seed=seed,
                    )
                    metrics = result["metrics"]
                    row[condition] = {
                        "status": "complete",
                        "negative_transfer_rate": metrics.get("negative_transfer_rate"),
                        "transfer_gain": metrics.get("transfer_gain"),
                        "poisoning_survival_rate": metrics.get("poisoning_survival_rate"),
                        "rule_precision": metrics.get("rule_precision"),
                        "drift_recovery_latency": metrics.get("drift_recovery_latency"),
                        "evolution_regret": metrics.get("evolution_regret", {}).get("total"),
                    }
                except Exception as exc:
                    row[condition] = {"status": "failed", "error": repr(exc)}
            rows.append(row)
    if any(row.get(condition, {}).get("status") != "complete" for row in rows for condition in ("C", "D")):
        raise RuntimeError("one or more lifecycle replicas failed; raw records were retained")

    aggregates: dict[str, Any] = {}
    metric_names = ("negative_transfer_rate", "transfer_gain", "poisoning_survival_rate", "rule_precision", "drift_recovery_latency", "evolution_regret")
    for episode_name, _ in EPISODES:
        aggregates[episode_name] = {}
        episode_rows = [row for row in rows if row["episode"] == episode_name]
        for condition in ("C", "D"):
            aggregates[episode_name][condition] = {
                metric: {
                    "mean": _mean([float(row[condition][metric]) for row in episode_rows if row[condition].get(metric) is not None]),
                    "n": sum(row[condition].get(metric) is not None for row in episode_rows),
                }
                for metric in metric_names
            }
    return {
        "schema_version": 1,
        "analysis": "seed_disjoint_provider_free_lifecycle_replication",
        "provider_calls": 0,
        "replicas_per_episode": replicas_per_episode,
        "seed_policy": "20260910 + episode_index*1000 + replicate",
        "episodes": aggregates,
        "rows": rows,
    }


def run_resource_quality_ablation(repo_root: Path, seed: int = 7) -> dict[str, Any]:
    compact = run_budget_aware_ablation(root=repo_root, seed=seed)
    stress = run_budget_aware_reserve_stress(seed=seed, schedule_count=256)
    acquisition = compact["acquisition"]
    legacy = acquisition["legacy_acre"]
    agreement: dict[str, bool] = {}
    for name, row in acquisition.items():
        if not isinstance(row, Mapping) or name == "provider_calls":
            continue
        agreement[name] = (
            row.get("target_reached") == legacy.get("target_reached")
            and row.get("final_error") == legacy.get("final_error")
        )
    return {
        "schema_version": 1,
        "analysis": "provider_free_resource_quality_ablation",
        "provider_calls": 0,
        "decision_agreement_with_legacy": agreement,
        "quality_fields": "target_reached and final_error are the registered fixture quality fields; unsafe rate, recall, and regret are unavailable in this fixture",
        "compact_ablation": compact,
        "reserve_stress": stress,
    }


def write_lifecycle_csv(report: Mapping[str, Any], path: Path) -> None:
    rows = report["rows"]
    fields = ["episode", "replicate", "replicate_id", "seed"]
    for condition in ("C", "D"):
        fields.extend(f"{condition}_{name}" for name in ("status", "negative_transfer_rate", "transfer_gain", "poisoning_survival_rate", "rule_precision", "drift_recovery_latency", "evolution_regret"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in ("episode", "replicate", "replicate_id", "seed")}
            for condition in ("C", "D"):
                flat.update({f"{condition}_{key}": value for key, value in row[condition].items()})
            writer.writerow(flat)


def run_all(repo_root: Path, output_root: Path, method_source: Path, relation_source: Path, replicas_per_episode: int) -> dict[str, Any]:
    method_raw = _read_json(method_source)
    relation_raw = _read_json(relation_source)
    lifecycle = run_lifecycle_replicates(repo_root, output_root, replicas_per_episode)
    report = {
        "schema_version": 1,
        "analysis": "minimal_reviewer_experiments",
        "provider_calls": 0,
        "benchmark_revision": "runtime-at-command",
        "sources": {
            "method": {"path": str(method_source), "sha256": _sha256(method_source)},
            "relation": {"path": str(relation_source), "sha256": _sha256(relation_source)},
        },
        "risk_coverage": build_risk_coverage_summary(method_raw),
        "relation_budget": aggregate_relation_budget_curve(relation_raw),
        "resource_quality": run_resource_quality_ablation(repo_root),
        "lifecycle_replication": lifecycle,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "minimal_review_experiments.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    write_lifecycle_csv(lifecycle, output_root / "lifecycle_replication.csv")
    return report


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-source", type=Path, required=True)
    parser.add_argument("--relation-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicas-per-episode", type=int, default=20)
    args = parser.parse_args()
    report = run_all(repo_root, args.out, args.method_source, args.relation_source, args.replicas_per_episode)
    print(json.dumps({"status": "complete", "provider_calls": report["provider_calls"], "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
