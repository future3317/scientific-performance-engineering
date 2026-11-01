import pytest

from experiments.scientific_skill_evolution import (
    DecisionModel,
    ExperimentOption,
    plan_discriminative_experiment,
)


def _model():
    return DecisionModel(
        priors={"real_speedup": 0.5, "cache_artifact": 0.5},
        losses={
            "apply_skill": {"real_speedup": 0.0, "cache_artifact": 1.0},
            "abstain": {"real_speedup": 0.4, "cache_artifact": 0.4},
        },
    )


def test_selects_cheap_test_that_changes_evolution_decision():
    plan = plan_discriminative_experiment(_model(), (
        ExperimentOption("cold_vs_steady", 2.0, {
            "real_speedup": {"speedup": 0.9, "no_speedup": 0.1},
            "cache_artifact": {"speedup": 0.1, "no_speedup": 0.9},
        }),
        ExperimentOption("prompt_rewrite", 1.0, {
            "real_speedup": {"same": 1.0},
            "cache_artifact": {"same": 1.0},
        }),
    ), budget=2.0)

    assert plan.status == "SELECTED"
    assert plan.experiment_id == "cold_vs_steady"
    assert plan.current_action == "abstain"
    assert plan.action_by_outcome == {"no_speedup": "abstain", "speedup": "apply_skill"}
    assert plan.expected_value > 0


def test_abstains_when_no_safe_or_feasible_discriminating_experiment():
    option = ExperimentOption("unsafe_probe", 0.1, {
        "real_speedup": {"yes": 1.0}, "cache_artifact": {"no": 1.0},
    }, safe=False)
    plan = plan_discriminative_experiment(_model(), (option,), budget=1.0)

    assert (plan.status, plan.reason, plan.experiment_id) == (
        "ABSTAIN", "NO_SAFE_DISCRIMINATING_TEST", None,
    )


def test_abstains_when_test_cannot_change_best_decision():
    model = DecisionModel(
        priors={"h1": 0.5, "h2": 0.5},
        losses={"always_keep": {"h1": 0.0, "h2": 0.0}, "abstain": {"h1": 1.0, "h2": 1.0}},
    )
    option = ExperimentOption("separating_but_irrelevant", 1.0, {
        "h1": {"a": 1.0}, "h2": {"b": 1.0},
    })
    plan = plan_discriminative_experiment(model, (option,), budget=1.0)

    assert (plan.status, plan.reason, plan.expected_value) == (
        "ABSTAIN", "NO_DECISION_VALUE", 0.0,
    )


def test_rejects_incomplete_or_invalid_probability_models():
    with pytest.raises(ValueError, match="sum to one"):
        DecisionModel(priors={"h": 0.9}, losses={"a": {"h": 0.0}})
    with pytest.raises(ValueError, match="every hypothesis"):
        plan_discriminative_experiment(_model(), (
            ExperimentOption("partial", 1.0, {"real_speedup": {"yes": 1.0}}),
        ), budget=1.0)
