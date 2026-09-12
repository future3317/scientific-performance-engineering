from __future__ import annotations

from scripts.run_rscb_static_frontier import run_frozen_frontier


def test_frontier_emits_shared_decisions_and_freezes_threshold_before_evaluation() -> None:
    report = run_frozen_frontier(surface_count=12, seed=7, families=("compile",))
    assert report["provider_calls"] == 0
    assert report["parameter_freeze"]["evaluation_used_for_threshold_selection"] is False
    assert report["rows"]
    for row in report["rows"]:
        assert row["n"] > 0
        assert all(decision["action"] in {"reuse", "abstain"} for decision in row["decisions"])
        assert all("reason" in decision for decision in row["decisions"])
    policies = {row["policy"] for row in report["rows"]}
    assert "rscb_default" in policies
    assert "rscb_matched" in policies
    assert "static_test_gate" in policies


def test_frontier_reports_family_seed_cluster_units() -> None:
    report = run_frozen_frontier(surface_count=12, seed=7, families=("compile", "episode"))
    summary = next(row for row in report["summaries"] if row["policy"] == "rscb_default")
    assert summary["cluster_keys"] == ["family", "seed"]
    assert summary["n_clusters"] == 2

