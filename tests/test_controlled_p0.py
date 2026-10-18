from __future__ import annotations

from scripts.run_controlled_p0 import run_controlled_p0


def test_controlled_p0_records_frozen_seeds_and_provider_free_statistics() -> None:
    report = run_controlled_p0(
        surface_count=12,
        seeds=(0,),
        families=("autograd",),
        bootstrap_reps=20,
        permutation_reps=5,
    )

    assert report["provider_calls"] == 0
    assert report["seeds"] == [0]
    assert report["families"] == ["autograd"]
    assert report["statistics"]["bootstrap_replicates"] == 20
    assert report["statistics"]["permutation_replicates"] == 5
    assert report["statistics"]["permutation_target"] == "counterexample_assignment_only"
    assert report["summaries"]


def test_controlled_p0_emits_policy_metrics_needed_for_the_funnel() -> None:
    report = run_controlled_p0(
        surface_count=12,
        seeds=(0,),
        families=("autograd",),
        bootstrap_reps=10,
        permutation_reps=3,
    )

    acre = next(row for row in report["summaries"] if row["policy"] == "acre_full")
    assert "coverage" in acre["cluster_means"]
    assert "unsupported_deployed" in acre["cluster_means"]
    assert "recall" in acre["cluster_means"]
    assert "incremental_policy_utility" in acre["cluster_means"]
    assert acre["bootstrap"]["replicates"] == 10
