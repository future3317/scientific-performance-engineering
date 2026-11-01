"""Decision-focused value-of-information baseline for choosing an experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class DecisionModel:
    """Prior mechanism probabilities and losses for available evolution actions."""

    priors: Mapping[str, float]
    losses: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        hypotheses = set(self.priors)
        if not hypotheses or any(not name for name in hypotheses):
            raise ValueError("decision model requires named mechanism hypotheses")
        if any(not math.isfinite(float(p)) or float(p) < 0 for p in self.priors.values()):
            raise ValueError("hypothesis priors must be finite and nonnegative")
        if not math.isclose(sum(map(float, self.priors.values())), 1.0, abs_tol=1e-9):
            raise ValueError("hypothesis priors must sum to one")
        if not self.losses:
            raise ValueError("decision model requires at least one evolution action")
        for action, by_hypothesis in self.losses.items():
            if not action or set(by_hypothesis) != hypotheses:
                raise ValueError("each action must define a loss for every hypothesis")
            if any(not math.isfinite(float(loss)) or float(loss) < 0 for loss in by_hypothesis.values()):
                raise ValueError("decision losses must be finite and nonnegative")


@dataclass(frozen=True)
class ExperimentOption:
    """A safe experiment's predicted observation likelihood under each hypothesis."""

    experiment_id: str
    cost: float
    likelihoods: Mapping[str, Mapping[str, float]]
    safe: bool = True
    feasible: bool = True

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must be nonempty")
        if not math.isfinite(float(self.cost)) or self.cost <= 0:
            raise ValueError("experiment cost must be finite and positive")
        if not self.likelihoods:
            raise ValueError("experiment requires predicted observation likelihoods")
        for hypothesis, outcomes in self.likelihoods.items():
            if not hypothesis or not outcomes or any(not outcome for outcome in outcomes):
                raise ValueError("experiment predictions require named hypotheses and outcomes")
            if any(not math.isfinite(float(p)) or float(p) < 0 for p in outcomes.values()):
                raise ValueError("observation likelihoods must be finite and nonnegative")
            if not math.isclose(sum(map(float, outcomes.values())), 1.0, abs_tol=1e-9):
                raise ValueError("observation likelihoods must sum to one per hypothesis")


@dataclass(frozen=True)
class ExperimentPlan:
    status: str
    reason: str
    experiment_id: str | None
    current_action: str
    expected_value: float
    value_per_cost: float
    action_by_outcome: Mapping[str, str]


def condition_on_observation(
    model: DecisionModel,
    experiment: ExperimentOption,
    observation: str,
) -> DecisionModel:
    """Update mechanism beliefs from one released observation."""
    hypotheses = set(model.priors)
    if set(experiment.likelihoods) != hypotheses:
        raise ValueError("experiment must predict observations under every hypothesis")
    if not any(observation in outcomes for outcomes in experiment.likelihoods.values()):
        raise ValueError("observation is outside the experiment's prediction space")
    unnormalized = {
        hypothesis: float(model.priors[hypothesis])
        * float(experiment.likelihoods[hypothesis].get(observation, 0.0))
        for hypothesis in hypotheses
    }
    evidence_probability = sum(unnormalized.values())
    if evidence_probability == 0:
        raise ValueError("observation has zero probability under every hypothesis")
    posterior = {hypothesis: probability / evidence_probability
                 for hypothesis, probability in unnormalized.items()}
    return DecisionModel(posterior, model.losses)


def plan_discriminative_experiment(
    model: DecisionModel,
    experiments: tuple[ExperimentOption, ...],
    *,
    budget: float,
) -> ExperimentPlan:
    """Choose the feasible experiment with greatest expected decision value per cost.

    Value is the reduction in Bayes decision loss after observing an experiment,
    not generic information gain. The function only plans; it never runs a test,
    changes a Skill, or certifies a mechanism.
    """
    if not math.isfinite(float(budget)) or budget < 0:
        raise ValueError("budget must be finite and nonnegative")
    ids = [option.experiment_id for option in experiments]
    if len(ids) != len(set(ids)):
        raise ValueError("experiment IDs must be unique")
    hypotheses = set(model.priors)
    prior = {name: float(value) for name, value in model.priors.items()}
    current_action, current_loss = _best_action(model.losses, prior)

    eligible = []
    for option in experiments:
        if set(option.likelihoods) != hypotheses:
            raise ValueError("every experiment must predict observations under every hypothesis")
        if option.safe and option.feasible and option.cost <= budget:
            eligible.append(option)
    if not eligible:
        reason = "NO_SAFE_DISCRIMINATING_TEST" if not any(
            option.safe and option.feasible for option in experiments
        ) else "NO_TEST_WITHIN_BUDGET"
        return ExperimentPlan("ABSTAIN", reason, None, current_action, 0.0, 0.0, {})

    scored: list[tuple[float, float, str, ExperimentOption, dict[str, str]]] = []
    for option in eligible:
        outcome_probabilities: dict[str, float] = {}
        joint: dict[str, dict[str, float]] = {}
        for hypothesis, probability in prior.items():
            for outcome, likelihood in option.likelihoods[hypothesis].items():
                joint.setdefault(outcome, {})[hypothesis] = probability * float(likelihood)
                outcome_probabilities[outcome] = outcome_probabilities.get(outcome, 0.0) + probability * float(likelihood)

        expected_post_loss = 0.0
        actions: dict[str, str] = {}
        for outcome in sorted(outcome_probabilities):
            probability = outcome_probabilities[outcome]
            if probability == 0:
                continue
            posterior = {hypothesis: joint[outcome].get(hypothesis, 0.0) / probability
                         for hypothesis in hypotheses}
            action, loss = _best_action(model.losses, posterior)
            actions[outcome] = action
            expected_post_loss += probability * loss

        value = max(0.0, current_loss - expected_post_loss)
        scored.append((value / float(option.cost), value, option.experiment_id, option, actions))

    value_per_cost, value, _, selected, actions = max(scored, key=lambda row: (row[0], row[1], row[2]))
    if value <= 1e-12:
        return ExperimentPlan("ABSTAIN", "NO_DECISION_VALUE", None, current_action, value, value_per_cost, {})
    return ExperimentPlan("SELECTED", "POSITIVE_EXPECTED_DECISION_VALUE", selected.experiment_id,
                          current_action, value, value_per_cost, actions)


def _best_action(losses: Mapping[str, Mapping[str, float]], probabilities: Mapping[str, float]) -> tuple[str, float]:
    risks = {
        action: sum(float(probabilities[hypothesis]) * float(loss) for hypothesis, loss in by_hypothesis.items())
        for action, by_hypothesis in losses.items()
    }
    action = min(risks, key=lambda name: (risks[name], name))
    return action, risks[action]
