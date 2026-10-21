import pytest

from experiments.real_intervention_v5.analysis import (
    evaluate_policies,
    freeze_policies,
    summarize,
)


def _pair(family, effect, *, split="development", **values):
    return {
        "pair_id": values.pop("pair_id", f"{family}-{split}-{values.get('session', 0)}"),
        "family_id": family,
        "split": split,
        "batch": 1,
        "horizon": 16,
        "conv_count": 10,
        "execution_status": "COMPLETE",
        "scientific_status": "PASS",
        "platform_status": "HEALTHY",
        "exposure_status": "MATCHED",
        "analysis_status": "VALID_PAIRED",
        "log_benefit": effect,
        **values,
    }


def test_family_effects_equal_weight_contexts_not_repeated_sessions():
    pairs = [
        _pair("a", 0.0, horizon=1),
        _pair("a", 2.0, horizon=16, session=0),
        _pair("a", 2.0, horizon=16, session=1),
        _pair("a", 2.0, horizon=16, session=2),
        _pair("b", 1.0, horizon=1),
    ]

    result = summarize(pairs)["overall"]

    assert result["family_mean_log_benefit"] == {"a": 1.0, "b": 1.0}
    assert result["family_equal_mean_log_benefit"] == 1.0


def test_summaries_separate_contrasts_and_stratify_split_horizon_batch():
    pairs = [
        _pair("a", 1.0, split="development", contrast="COMPILE_COLD", horizon=16, batch=1),
        _pair("a", -1.0, split="development", contrast="PERSISTENT_R10", horizon=128, batch=8),
    ]

    result = summarize(pairs)

    assert result["overall"]["status"] == "NOT_POOLED_DISTINCT_CONTRASTS"
    assert set(result["by_contrast"]) == {"COMPILE_COLD", "PERSISTENT_R10"}
    cold = result["by_contrast"]["COMPILE_COLD"]
    assert "development|H=16|B=1" in cold["by_split_horizon_batch"]


def test_nested_features_and_raw_conv_threshold_prediction():
    dev = [
        _pair(f"positive-{index}", 1.0, horizon=16, features={
            "parameters": 1000, "conv2d": 25, "layers": 5, "batch": 1,
        })
        for index in range(2)
    ] + [
        _pair(f"negative-{index}", -1.0, horizon=1, features={
            "parameters": 500, "conv2d": 2, "layers": 2, "batch": 1,
        })
        for index in range(2)
    ]
    final_spec = {
        "pair_id": "final-1", "split": "final", "horizon": 16,
        "features": {"parameters": 1000, "conv2d": 25, "batch": 1},
    }

    frozen = freeze_policies(dev, [], [final_spec])

    assert frozen["final_predictions"][0]["predictions"]["STRUCTURE_CONV_COUNT"] is True
    assert frozen["final_predictions"][0]["feature_identity"]["parameter_count"] > 0
    assert frozen["final_outcomes_accessed"] is False


def test_final_policy_evaluation_requires_frozen_pair_and_features():
    dev = [
        _pair(f"positive-{index}", 1.0, horizon=16)
        for index in range(2)
    ] + [
        _pair(f"negative-{index}", -1.0, horizon=1)
        for index in range(2)
    ]
    spec = {"pair_id": "final-1", "split": "final", "horizon": 16,
            "batch": 1, "conv_count": 5}
    frozen = freeze_policies(dev, [], [spec])
    final = _pair("final-family", -0.5, split="final", pair_id="final-1",
                  horizon=16, batch=1, conv_count=5, s0_s=2.0, s1_s=3.0)

    rows = evaluate_policies([final], frozen)
    always = next(row for row in rows if row["policy"] == "ALWAYS")
    assert always["net_seconds_saved"] == -1.0
    assert always["negative_deployed_pairs"] == 1
    assert always["strongest_simple_policy_on_development"] == frozen[
        "strongest_simple_policy_on_development"
    ]

    with pytest.raises(ValueError, match="feature identity changed"):
        evaluate_policies([dict(final, horizon=128)], frozen)
    with pytest.raises(ValueError, match="pair_ids do not match"):
        evaluate_policies([dict(final, pair_id="not-frozen")], frozen)


def test_low_heterogeneity_leaves_predictive_policies_not_fitted():
    frozen = freeze_policies([_pair("positive", 1.0), _pair("negative", -1.0)], [], [])

    assert frozen["policies"]["NEVER"]["status"] == "FITTED"
    assert frozen["policies"]["ALWAYS"]["status"] == "FITTED"
    assert frozen["policies"]["HORIZON_THRESHOLD"]["status"] == "NOT_FITTED"
    assert frozen["policies"]["SIMPLE_SUPERVISED_RIDGE"]["status"] == "NOT_FITTED"


def test_empty_policy_evaluation_is_not_estimable():
    frozen = freeze_policies([], [], [])

    rows = evaluate_policies([], frozen)

    assert rows
    by_name = {row["policy"]: row for row in rows}
    assert by_name["NEVER"]["status"] == "NOT_ESTIMABLE_EMPTY"
    assert by_name["ALWAYS"]["status"] == "NOT_ESTIMABLE_EMPTY"
    for row in rows:
        if row["status"] in {"NOT_FITTED", "ORIGINAL_ACRE_NOT_FITTED"}:
            continue
        assert row["status"] == "NOT_ESTIMABLE_EMPTY"
        assert row["family_equal_policy_utility"] is None
        assert row["missingness_bounds"] == {
            "lower": "NOT_ESTIMABLE",
            "upper": "NOT_ESTIMABLE",
        }
