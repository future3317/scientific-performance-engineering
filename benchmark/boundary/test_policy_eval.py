from __future__ import annotations

from benchmark.boundary.policy_eval import Decision, clustered_frontier, evaluate_policy


def test_evaluator_keeps_all_and_deployed_denominators_separate() -> None:
    evaluation = [
        {"id": "a", "family": "f1", "seed": 1, "context": {"x": 0}},
        {"id": "b", "family": "f1", "seed": 1, "context": {"x": 1}},
        {"id": "c", "family": "f2", "seed": 2, "context": {"x": 2}},
    ]
    scores = {
        "a": {"tau": 0.4, "scientific_gate": True},
        "b": {"tau": -0.1, "scientific_gate": True},
        "c": {"tau": -0.8, "scientific_gate": True},
    }

    report = evaluate_policy(
        lambda context: Decision("reuse" if context["x"] != 1 else "abstain", 1.0, 0.0, "test"),
        calibration=[],
        evaluation=evaluation,
        score_records=scores,
        delta=0.0,
    )

    assert report["n"] == 3
    assert report["deployed"] == 2
    assert report["coverage"] == 2 / 3
    assert report["unsupported"] == 1
    assert report["unsupported_all"] == 1 / 3
    assert report["unsupported_deployed"] == 1 / 2
    assert report["unsupported_deployed_defined"] is True


def test_zero_deployment_has_undefined_conditional_risk() -> None:
    evaluation = [{"id": "a", "family": "f", "seed": 1, "context": {}}]
    scores = {"a": {"tau": 0.5, "scientific_gate": True}}
    report = evaluate_policy(
        lambda context: Decision("abstain", 0.0, 0.0, "no_support"),
        calibration=[],
        evaluation=evaluation,
        score_records=scores,
        delta=0.0,
    )
    assert report["deployed"] == 0
    assert report["coverage"] == 0.0
    assert report["unsupported_deployed"] is None
    assert report["unsupported_deployed_defined"] is False


def test_support_decision_regret_respects_scientific_gate_and_margin() -> None:
    evaluation = [
        {"id": "supported", "family": "f", "seed": 1, "context": {"x": 1}},
        {"id": "gate_fail", "family": "f", "seed": 1, "context": {"x": 2}},
        {"id": "below_margin", "family": "f", "seed": 1, "context": {"x": 3}},
    ]
    scores = {
        "supported": {"tau": 0.4, "scientific_gate": True},
        "gate_fail": {"tau": 1.0, "scientific_gate": False},
        "below_margin": {"tau": 0.05, "scientific_gate": True},
    }
    report = evaluate_policy(
        lambda context: Decision("abstain", 0.0, 0.0, "conservative"),
        calibration=[],
        evaluation=evaluation,
        score_records=scores,
        delta=0.1,
    )
    assert report["support_decision_regret"] == 0.4
    assert report["support_decision_regret_mean"] == 0.4 / 3
    assert report["oracle_supported"] == 1


def test_policy_receives_public_context_only() -> None:
    seen: list[dict] = []

    def policy(context: dict) -> Decision:
        seen.append(context)
        assert "tau" not in context
        assert "scientific_gate" not in context
        assert "oracle_label" not in context
        return Decision("abstain", 0.0, 0.0, "isolated")

    evaluation = [{
        "id": "a",
        "family": "f",
        "seed": 3,
        "context": {"workload": {"x": 1}},
        "tau": 99.0,
        "scientific_gate": False,
        "oracle_label": True,
    }]
    report = evaluate_policy(
        policy,
        calibration=[{"context": {"workload": {"x": 0}}, "hidden": "calibration-only"}],
        evaluation=evaluation,
        score_records={"a": {"tau": 0.2, "scientific_gate": True}},
        delta=0.0,
    )
    assert report["n"] == 1
    assert seen == [{"workload": {"x": 1}}]


def test_clustered_frontier_uses_family_seed_as_units() -> None:
    rows = [
        {"family": "f1", "seed": 0, "coverage": 0.5, "unsupported_all": 0.0},
        {"family": "f1", "seed": 0, "coverage": 1.0, "unsupported_all": 1.0},
        {"family": "f2", "seed": 0, "coverage": 0.0, "unsupported_all": 0.0},
    ]
    report = clustered_frontier(rows, cluster_keys=("family", "seed"))
    assert report["n_rows"] == 3
    assert report["n_clusters"] == 2
    assert report["cluster_means"]["coverage"] == (0.75 + 0.0) / 2
