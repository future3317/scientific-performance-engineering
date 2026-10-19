from pathlib import Path

from experiments.real_intervention_v4.contracts import (
    build_frozen_manifest,
    family_split,
    registered_family_ids,
    split_family_counts,
)
from experiments.real_intervention_v4.run_all import _gate


def _config():
    return {
        "study_id": "test-v4",
        "intervention_id": "CHANNELS_LAST_EAGER_LAYOUT_V1",
        "seed": 20261019,
        "batches": [1, 8],
        "horizons": [1],
        "sessions_per_context": 2,
        "shape": [3, 224, 224],
        "split": {
            "minimum_total_families_for_confirmatory": 23,
            "minimum_development_families": 8,
            "minimum_calibration_families": 10,
            "minimum_final_families": 5,
            "underpowered_development_family_cap": 8,
            "underpowered_calibration_family_cap": 10,
        },
    }


def _universe(n=19):
    return [
        {"candidate_id": f"model-{i:02d}", "family_id": f"family-{i:02d}",
         "source": "test", "model": f"model-{i:02d}", "availability": "available"}
        for i in range(n)
    ]


def _qualifications(universe, eligible_count):
    rows = []
    for index, candidate in enumerate(universe):
        eligible = index < eligible_count
        for batch in (1, 8):
            rows.append({
                "candidate_id": candidate["candidate_id"],
                "family_id": candidate["family_id"],
                "batch": batch,
                "eligibility": (
                    "ELIGIBLE_CHANNELS_LAST_CORRECTNESS_ONLY" if eligible
                    else "NOT_CHANNELS_LAST_APPLICABLE"
                ),
                "model_summary": {
                    "weights_sha256": f"weights-{index}",
                    "inputs_sha256": f"inputs-{index}-{batch}",
                    "reference_sha256": f"reference-{index}-{batch}",
                    "reference_path": f"reference-{index}-{batch}.pt",
                },
            })
    return rows


def _manifest(universe, eligible_count, assignments):
    return build_frozen_manifest(
        config=_config(), source_commit="source", config_path=Path("PRIMARY_LAYOUT_STUDY.json"),
        selected_gpu_uuid="GPU-test", runtime_identity={}, universe=universe,
        qualifications=_qualifications(universe, eligible_count),
        frozen_family_split=assignments,
    )


def test_underpowered_split_preserves_rule_and_final_family_minima_in_denominator():
    universe = _universe()
    family_ids = registered_family_ids(universe)
    assignments = family_split(
        family_ids, seed=20261019, split_config=_config()["split"],
    )

    assert split_family_counts(assignments) == {
        "development": 8, "calibration": 10, "final": 1,
    }
    manifest = _manifest(universe, eligible_count=7, assignments=assignments)
    assert set(manifest["family_split"]) == set(family_ids)
    assert len(manifest["registered_families"]) == 19
    assert len(manifest["eligible_families"]) == 7
    assert len(manifest["unqualified_families"]) == 12
    assert manifest["split_family_counts"] == {
        "development": 8, "calibration": 10, "final": 1,
    }
    assert manifest["confirmatory_split_eligible"] is False


def test_qualification_changes_never_reassign_registered_family_splits():
    universe = _universe()
    assignments = family_split(
        registered_family_ids(universe), seed=20261019,
        split_config=_config()["split"],
    )

    first = _manifest(universe, eligible_count=7, assignments=assignments)
    second = _manifest(universe, eligible_count=13, assignments=assignments)

    assert second["family_split"] == first["family_split"] == assignments
    assert second["eligible_family_counts_by_split"] != first["eligible_family_counts_by_split"]


def test_confirmatory_split_requires_at_least_twenty_three_registered_families():
    config = _config()
    assignments = family_split(
        [f"family-{i:02d}" for i in range(23)], seed=config["seed"],
        split_config=config["split"],
    )

    assert split_family_counts(assignments) == {
        "development": 8, "calibration": 10, "final": 5,
    }


def test_final_gate_counts_registered_and_eligible_families_separately():
    split_counts = {"development": 8, "calibration": 10, "final": 1}
    manifest = {
        "registered_families": [f"family-{i}" for i in range(19)],
        "eligible_families": [f"family-{i}" for i in range(19)],
        "split_family_counts": split_counts,
        "eligible_family_counts_by_split": split_counts,
    }
    rows = [
        {"family_id": f"family-{i}", "split": split,
         "classification": {"policy_evidence": "ELIGIBLE"}}
        for split, n in (("development", 8), ("calibration", 10))
        for i in range(n)
    ]
    result = _gate(
        manifest, rows,
        {"passes": True},
        {"split": _config()["split"],
         "acre": {"minimum_development_families": 8,
                  "minimum_calibration_deploy_families": 10,
                  "minimum_calibration_support_families": 5,
                  "minimum_calibration_non_support_families": 5}},
        dev_cal_terminal=True, policy_freeze={"status": "FROZEN"},
    )

    assert result["final_opened"] is False
    assert result["checks"]["registered_family_roster_minimum"] is False
    assert result["checks"]["final_split_minimum"] is False
