from __future__ import annotations

from scripts.run_context_indexed_frontier import run_context_indexed_frontier


def test_frontier_uses_one_evaluation_set_for_all_provider_free_policies() -> None:
    report = run_context_indexed_frontier(
        surface_count=24,
        generator_seeds=(0,),
        families=("compile",),
        doses=(1.0,),
        permutation_reps=3,
        bootstrap_reps=20,
    )
    assert report["provider_calls"] == 0
    rows = [row for row in report["rows"] if row["family"] == "compile"]
    assert {row["policy"] for row in rows} >= {
        "global_reuse",
        "positive_only",
        "nearest_experience",
        "matched_positive",
        "cart_ltt",
        "catboost_ltt",
        "acre_full",
    }
    assert len({tuple(row["evaluation_ids"]) for row in rows}) == 1


def test_permutation_null_is_separate_from_registered_rows() -> None:
    report = run_context_indexed_frontier(
        surface_count=24,
        generator_seeds=(0,),
        families=("compile",),
        doses=(1.0,),
        permutation_reps=3,
        bootstrap_reps=20,
    )
    assert len(report["permutation_null"]) == 3
    assert all("support_precision" in row for row in report["permutation_null"])
    assert report["registered_input_artifact"] != report["permutation_null"]


def test_frontier_records_cluster_bootstrap_and_denominators() -> None:
    report = run_context_indexed_frontier(
        surface_count=24,
        generator_seeds=(0, 1),
        families=("compile",),
        doses=(1.0,),
        permutation_reps=1,
        bootstrap_reps=20,
    )
    row = next(item for item in report["rows"] if item["policy"] == "acre_full")
    assert row["unsupported_all"] is not None
    assert "unsupported_deployed" in row
    assert report["cluster_bootstrap"]["acre_full"]["n_clusters"] == 2
