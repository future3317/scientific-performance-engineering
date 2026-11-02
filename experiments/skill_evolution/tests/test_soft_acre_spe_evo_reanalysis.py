from experiments.skill_evolution.soft_acre_spe_evo_reanalysis import _policy_metrics


def test_fixed_policy_predictions_remain_observation_specific_when_contexts_collide():
    rows = [
        {"id": "first", "context": {"scale": "low"}},
        {"id": "second", "context": {"scale": "low"}},
    ]
    scores = {
        "first": {"tau": 0.2, "scientific_gate": True},
        "second": {"tau": -0.2, "scientific_gate": True},
    }

    result = _policy_metrics(
        None, rows, scores, reuse_by_id={"first": True, "second": False}
    )

    assert result["deployed"] == 1
    assert result["utility_sum"] == 0.2
    assert result["harmful_deployed"] == 0
