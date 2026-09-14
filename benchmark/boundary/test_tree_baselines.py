from __future__ import annotations

import pytest

from benchmark.boundary.policy_eval import Decision
from benchmark.boundary.tree_baselines import fit_cart_policy, fit_catboost_ltt_policy
from benchmark.boundary.typed_features import TypedFeatureSchema


def _rows() -> tuple[list[dict], list[dict], TypedFeatureSchema]:
    contexts = [
        {"workload": {"logical_steps": 64, "dynamic_shape_rate": 0.0, "graph_size": 64}},
        {"workload": {"logical_steps": 128, "dynamic_shape_rate": 0.0, "graph_size": 128}},
        {"workload": {"logical_steps": 192, "dynamic_shape_rate": 0.6, "graph_size": 256}},
        {"workload": {"logical_steps": 192, "dynamic_shape_rate": 0.6, "graph_size": 320}},
    ]
    schema = TypedFeatureSchema.from_calibration("compile", contexts)
    synthesis = [
        {"context": contexts[0], "label": True},
        {"context": contexts[1], "label": True},
        {"context": contexts[2], "label": False},
        {"context": contexts[3], "label": False},
    ]
    calibration = [
        {"context": contexts[0], "unsupported": False},
        {"context": contexts[2], "unsupported": True},
    ]
    return synthesis, calibration, schema


@pytest.mark.parametrize("factory", [fit_cart_policy, fit_catboost_ltt_policy])
def test_tree_policy_returns_shared_decision_contract(factory):
    synthesis, calibration, schema = _rows()
    policy = factory(synthesis, calibration, schema, seed=7)
    decision = policy.decide({"workload": {"n": 1, "difficulty_tier": "easy"}})
    assert isinstance(decision, Decision)
    assert decision.action in {"reuse", "abstain"}
    assert policy.to_dict()["fit_source"] == "synthesis_then_calibration"


def test_tree_policy_is_deterministic_and_does_not_accept_evaluation_rows():
    synthesis, calibration, schema = _rows()
    first = fit_cart_policy(synthesis, calibration, schema, max_depth=2, seed=11)
    second = fit_cart_policy(synthesis, calibration, schema, max_depth=2, seed=11)
    contexts = [
        {"workload": {"logical_steps": 64, "dynamic_shape_rate": 0.0, "graph_size": 64}},
        {"workload": {"logical_steps": 192, "dynamic_shape_rate": 0.6, "graph_size": 320}},
    ]
    assert [first.decide(c) for c in contexts] == [second.decide(c) for c in contexts]
    with pytest.raises(TypeError):
        fit_cart_policy(synthesis, calibration, schema, evaluation_rows=[])


def test_catboost_policy_has_explicit_dependency_failure_message(monkeypatch):
    synthesis, calibration, schema = _rows()
    monkeypatch.setattr("benchmark.boundary.tree_baselines.CatBoostClassifier", None)
    with pytest.raises(RuntimeError, match="CatBoost"):
        fit_catboost_ltt_policy(synthesis, calibration, schema, seed=0)
