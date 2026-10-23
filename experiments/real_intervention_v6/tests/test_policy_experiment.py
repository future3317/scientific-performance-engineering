import pytest

from experiments.real_intervention_v6.policy_experiment import freeze_policies, summarize


def _rows(families, split, effect_fn):
    return [
        {
            "family_id": family,
            "context_id": f"{family}-h{horizon}",
            "split": split,
            "horizon": horizon,
            "reuse_count": reuse,
            "features": {
                "parameters": 1000 + horizon,
                "conv_fraction": 0.25 if horizon < 1024 else 0.75,
            },
            "log_benefit": effect_fn(index, horizon),
            "s0_s": 2.0,
            "s1_s": 1.0,
        }
        for index, family in enumerate(families)
        for horizon, reuse in ((128, 1), (4096, 10))
    ]


def test_policy_freeze_uses_disjoint_splits_and_final_features_only():
    dev = _rows([f"d{i}" for i in range(6)], "development",
                lambda index, horizon: 0.2 if horizon == 4096 else -0.1)
    cal = _rows([f"c{i}" for i in range(4)], "calibration",
                lambda index, horizon: 0.1 if horizon == 4096 else -0.05)
    final = [
        {"family_id": "f0", "context_id": "f0-h128", "horizon": 128, "reuse_count": 1,
         "features": {"parameters": 1128, "conv_fraction": 0.25}},
        {"family_id": "f0", "context_id": "f0-h4096", "horizon": 4096, "reuse_count": 10,
         "features": {"parameters": 5096, "conv_fraction": 0.75}},
        {"family_id": "f0", "context_id": "f0-h8192", "horizon": 8192, "reuse_count": 10,
         "features": {"parameters": 9192, "conv_fraction": 0.75}},
    ]
    frozen = freeze_policies(dev, cal, final)

    assert set(frozen["policies"]) == {"NEVER", "ALWAYS", "H_ONLY", "SIMPLE", "ACRE_SYS_RESEARCH"}
    assert "ORACLE" not in frozen["policies"]
    assert len(frozen["final_predictions"]) == 3
    assert frozen["final_outcomes_accessed"] is False
    assert frozen["original_acre_status"] == "NOT_CERTIFIED"
    research_model = frozen["policies"]["ACRE_SYS_RESEARCH"]["model"]
    assert research_model["training_feature_min"]
    assert research_model["training_feature_max"]
    assert frozen["final_predictions"][-1]["predictions"]["ACRE_SYS_RESEARCH"] is False
    assert frozen["final_predictions"][-1]["research_context_status"] == "OOD_NEVER"
    assert all(set(item) == {
        "family_id", "context_id", "routing_overhead_ns", "research_context_status", "predictions"
    }
               for item in frozen["final_predictions"])

    with pytest.raises(ValueError, match="pre-execution only"):
        freeze_policies(dev, cal, [{**final[0], "log_benefit": 0.4}])
    with pytest.raises(ValueError, match="must be disjoint"):
        freeze_policies(dev, cal, [{**final[0], "family_id": "d0"}])


def test_summary_keeps_missing_outcomes_and_reports_family_paired_policy_ci():
    dev = _rows([f"d{i}" for i in range(6)], "development",
                lambda index, horizon: 0.2 if horizon == 4096 else -0.1)
    cal = _rows([f"c{i}" for i in range(4)], "calibration",
                lambda index, horizon: 0.1 if horizon == 4096 else -0.05)
    final_contexts = [
        {"family_id": f"f{i}", "context_id": f"f{i}-h{h}", "horizon": h,
         "reuse_count": 1 if h == 128 else 10,
         "features": {"parameters": 1000 + h, "conv_fraction": 0.25 if h == 128 else 0.75}}
        for i in range(4) for h in (128, 4096)
    ]
    frozen = freeze_policies(dev, cal, final_contexts)
    outcomes = [
        {**context, "log_benefit": (0.3 if context["horizon"] == 4096 else -0.2),
         "s0_s": 2.0, "s1_s": 1.0}
        for context in final_contexts
    ]
    outcomes[-1]["log_benefit"] = None
    outcomes[-1]["s1_s"] = None

    result = summarize(outcomes, frozen)
    assert "ORACLE" in result["policies"]
    assert result["overall"]["missing_rows"] == 1
    assert result["overall"]["planned_families"] == 4
    assert result["overall"]["raw_s0_seconds"] == 16.0
    assert result["overall"]["missingness_bounds"] == {"lower": "-unbounded", "upper": "+unbounded"}
    assert result["policies"]["ALWAYS"]["missing_rows"] == 1
    assert result["research_minus_best_simple"]["raw_seconds_net_benefit_difference"]["paired_family_contexts"] <= 8


def test_frozen_policy_seconds_subtract_measured_routing_overhead_and_pair_contexts():
    dev = _rows([f"d{i}" for i in range(6)], "development",
                lambda index, horizon: 0.2 if horizon == 4096 else -0.1)
    cal = _rows([f"c{i}" for i in range(4)], "calibration",
                lambda index, horizon: 0.1 if horizon == 4096 else -0.05)
    contexts = [
        {"family_id": f"f{i}", "context_id": f"f{i}-h{h}", "horizon": h,
         "reuse_count": 1 if h == 128 else 10,
         "features": {"parameters": 1000 + h, "conv_fraction": 0.25 if h == 128 else 0.75}}
        for i in range(4) for h in (128, 4096)
    ]
    frozen = freeze_policies(dev, cal, contexts)
    outcomes = [
        {**context, "log_benefit": 0.01, "s0_s": 10.0, "s1_s": 9.0}
        for context in contexts
    ]
    overhead = {"H_ONLY": 0.001, "SIMPLE": 0.002, "ACRE_SYS_RESEARCH": 0.01}

    result = summarize(outcomes, frozen, routing_overhead_s=overhead)
    assert result["policies"]["ALWAYS"]["seconds"]["net_benefit_s"] == pytest.approx(8.0)
    for name, cost in overhead.items():
        seconds = result["policies"][name]["seconds"]
        assert seconds["status"] == "COMPLETE"
        assert seconds["net_benefit_s"] == pytest.approx(
            seconds["gross_savings_s"] - seconds["routing_overhead_s"]
        )
        assert seconds["routing_overhead_s"] == pytest.approx(cost * len(contexts))
    difference = result["research_minus_best_simple"]["raw_seconds_net_benefit_difference"]
    assert difference["paired_family_contexts"] == len(contexts)
    assert len(difference["family_bootstrap_ci95"]) == 2
    log_difference = result["research_minus_best_simple"]["log_benefit_difference"]
    assert log_difference["paired_family_contexts"] == len(contexts)
    assert len(log_difference["family_bootstrap_ci95"]) == 2


def test_router_seconds_are_not_estimable_without_measured_overhead():
    dev = _rows([f"d{i}" for i in range(6)], "development",
                lambda index, horizon: 0.2 if horizon == 4096 else -0.1)
    cal = _rows([f"c{i}" for i in range(4)], "calibration",
                lambda index, horizon: 0.1 if horizon == 4096 else -0.05)
    contexts = [
        {"family_id": f"f{i}", "context_id": f"f{i}-h{h}", "horizon": h,
         "reuse_count": 1 if h == 128 else 10,
         "features": {"parameters": 1000 + h, "conv_fraction": 0.25 if h == 128 else 0.75}}
        for i in range(2) for h in (128, 4096)
    ]
    frozen = freeze_policies(dev, cal, contexts)
    outcomes = [{**row, "log_benefit": 0.1, "s0_s": 10.0, "s1_s": 9.0} for row in contexts]
    result = summarize(outcomes, frozen)
    assert result["policies"]["ACRE_SYS_RESEARCH"]["seconds"]["net_benefit_s"] is None
    assert result["policies"]["ACRE_SYS_RESEARCH"]["seconds"]["status"] != "COMPLETE"


def test_summary_weights_families_equally_and_bootstrap_is_deterministic():
    rows = [
        {"family_id": "a", "context_id": "a1", "log_benefit": 1.0},
        {"family_id": "a", "context_id": "a1", "log_benefit": 3.0},
        {"family_id": "a", "context_id": "a2", "log_benefit": 5.0},
        {"family_id": "b", "context_id": "b1", "log_benefit": -1.0},
        {"family_id": "c", "context_id": "c1", "log_benefit": None},
    ]
    first = summarize(rows)["overall"]
    second = summarize(rows)["overall"]

    assert first["family_equal_mean_log_benefit"] == 1.25
    assert first["family_mean_log_benefit"] == {"a": 3.5, "b": -1.0}
    assert first["planned_families"] == 3
    assert first["missing_families"] == 1
    assert first["estimate_status"] == "INCOMPLETE_OBSERVED_ONLY"
    assert first["family_bootstrap_ci95"] == second["family_bootstrap_ci95"]
