from __future__ import annotations

import copy
import math

from experiments.real_intervention_v4.policy_eval import (
    _fit_supervised_stump,
    _score,
    freeze_policy_set,
)


def _pair(family: str, horizon: int, benefit: float | None, *,
          split: str = "development", status: str = "valid",
          correctness: str = "PASS"):
    return {
        "pair_id": f"{family}-{horizon}",
        "family_id": family,
        "split": split,
        "effect_status": status,
        "log_benefit": benefit,
        "cold_seconds_saved_per_pair": 0.25 if benefit is not None else None,
        "context": {
            "horizon": horizon,
            "batch": None,
            "conv2d_module_count": None,
            "input_channels": None,
            "actual_input_shapes": {"input": [1, 3, 224, 224]},
            "qualification": {
                "eligibility": "ELIGIBLE_CHANNELS_LAST_CORRECTNESS_ONLY",
                "conv2d_count": 1,
            },
        },
        "classification": {"scientific_correctness": correctness},
        "S0": {
            "arm_execution": "COMPLETE",
            "scientific_correctness": "PASS",
        },
        "S1": {"scientific_correctness": correctness},
    }


def test_supervised_cv_holds_out_whole_families_and_never_selects_on_held_out_rows():
    rows = [
        _pair("family-a", 1, 1.0),
        _pair("family-a", 1, 0.8),
        _pair("family-b", 2, -1.0),
        _pair("family-c", 3, 0.5),
    ]
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["family_id"] == "family-c":
            row["context"]["horizon"] = 10_000
            row["log_benefit"] = -100.0

    original = _fit_supervised_stump(rows)
    altered = _fit_supervised_stump(changed)
    original_fold = next(
        fold for fold in original["cross_validation"]["folds"]
        if fold["held_out_family"] == "family-c"
    )
    altered_fold = next(
        fold for fold in altered["cross_validation"]["folds"]
        if fold["held_out_family"] == "family-c"
    )

    assert original_fold["training_family_ids"] == ["family-a", "family-b"]
    assert altered_fold["training_family_ids"] == ["family-a", "family-b"]
    assert (original_fold["selected_feature"], original_fold["selected_threshold"],
            original_fold["selected_direction"]) == (
                altered_fold["selected_feature"], altered_fold["selected_threshold"],
                altered_fold["selected_direction"])


def test_freeze_uses_development_only_for_fitting_and_comparator_selection():
    development = [
        _pair("family-a", 1, 0.4),
        _pair("family-b", 2, -0.2),
        _pair("family-c", 3, 0.1),
    ]
    calibration = [_pair("family-cal", 500, 50.0, split="calibration")]
    final = [_pair("family-final", 50_000, 500.0, split="final")]

    first = freeze_policy_set(
        development, calibration_pairs=calibration, acre_status="ACRE_NOT_FITTED",
    )
    second = freeze_policy_set(
        [*development, *final],
        calibration_pairs=[_pair("family-cal", -500, -50.0, split="calibration")],
        acre_status="ACRE_NOT_FITTED",
    )

    assert first["policies"] == second["policies"]
    assert first["strongest_simple_baseline"] == second["strongest_simple_baseline"]
    assert first["strongest_simple_selection_split"] == "development"
    assert first["strongest_simple_selection_method"] == "leave_one_independent_family_out"
    assert first["calibration_outcomes_read"] is False
    assert first["final_outcomes_read"] is False


def test_invalid_development_families_stay_in_comparator_denominator():
    development = [
        _pair("family-a", 1, 1.0),
        _pair("family-b", 2, 1.0),
        _pair("family-c", 3, 1.0),
        _pair("family-risk", 4, None, status="unsafe", correctness="FAIL"),
        _pair("family-risk", 5, None, status="unknown", correctness="UNKNOWN"),
        _pair("family-baseline-invalid", 6, None, status="baseline_invalid"),
    ]
    development[-1]["S0"]["scientific_correctness"] = "FAIL"

    # Filtering to valid rows alone makes ALWAYS look like a safe, positive comparator.
    valid_only = [row for row in development if row["effect_status"] == "valid"]
    valid_only_freeze = freeze_policy_set(
        valid_only, calibration_pairs=[], acre_status="ACRE_NOT_FITTED",
    )
    assert valid_only_freeze["strongest_simple_baseline"] == "ALWAYS"

    full_freeze = freeze_policy_set(
        development, calibration_pairs=[], acre_status="ACRE_NOT_FITTED",
    )
    cv = full_freeze["strongest_simple_cross_validation"]
    assert cv["fold_count"] == 5
    assert full_freeze["strongest_simple_baseline"] is None
    risk_fold = next(fold for fold in cv["folds"]
                     if fold["held_out_family"] == "family-risk")
    assert risk_fold["scores"]["ALWAYS"]["known_unsafe_deployments"] == 1
    assert risk_fold["scores"]["ALWAYS"]["missing_deployed_outcomes"] == 1


def test_missing_outcomes_keep_complete_case_metrics_and_explicit_bounds():
    complete = _pair("family-complete", 1, 0.2)
    missing = _pair("family-missing", 2, None, status="unknown")
    unsafe = _pair(
        "family-unsafe", 3, None, status="unsafe", correctness="FAIL",
    )

    score = _score(
        "ALWAYS", [complete, missing, unsafe],
        models={}, acre_status="ACRE_NOT_FITTED",
    )
    complete_utility = math.tanh(0.2 / math.log(2.0))

    assert score["known_valid_policy_families"] == 1
    assert score["family_equal_mean_net_bounded_utility_complete_case"] == complete_utility
    assert score["missing_deployed_outcomes"] == 1
    assert score["known_unsafe_deployments"] == 1
    assert score["incomplete_families"] == ["family-missing", "family-unsafe"]
    bounds = score["unresolved_benefit_bounds"]
    assert bounds["interval"] == [
        (complete_utility - 2.0) / 3.0,
        (complete_utility + 2.0) / 3.0,
    ]
    assert bounds["predeclared_context_bounds"]["missing_deployed_outcome"] == [-1.0, 1.0]
    assert bounds["predeclared_context_bounds"]["unsafe_deployment"] == [-1.0, 1.0]
    assert bounds["unsafe_deployments_remain_a_separate_hard_gate"] is True


def test_unknown_policy_decision_is_not_relabelled_as_missing_or_unsafe():
    pair = _pair("family-unknown", 1, 0.4)
    score = _score(
        "SIMPLE_SUPERVISED", [pair],
        models={"SIMPLE_SUPERVISED": {"status": "NOT_FITTED"}},
        acre_status="ACRE_NOT_FITTED",
    )

    assert score["unknown_policy_decisions"] == 1
    assert score["missing_deployed_outcomes"] == 0
    assert score["known_unsafe_deployments"] == 0
    assert score["unresolved_benefit_bounds"]["interval"] == [-1.0, 1.0]
