from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping, Sequence

from core.acre.cegis import BoundaryObservation, SynthesisResult, synthesize_boundary
from core.predicates import match_predicate
from core.sequential_stats import paired_repetition_interval

from .models import FALLBACK_TO_BASE_AGENT, PairedOutcome, PolicyDecision, SupportEvidence


def evidence_budget_grid(available_extra: int | None) -> tuple[int, ...]:
    """Return the preregistered absolute budget or its percentage fallback."""
    if available_extra is None or available_extra >= 64:
        return (0, 8, 16, 32, 64)
    if available_extra < 0:
        raise ValueError("available_extra must be non-negative")
    return tuple(ceil(available_extra * fraction) for fraction in (0.0, 0.10, 0.25, 0.50, 1.0))


def count_evidence_sources(evidence: Sequence[SupportEvidence]) -> dict[str, int]:
    counts = {"upstream_verification": 0, "acre_additional_query": 0}
    for item in evidence:
        counts[item.evidence_source] += 1
    return counts


def support_evidence_from_outcome(
    outcome: PairedOutcome,
    evidence_source: str,
) -> SupportEvidence:
    if not outcome.execution_valid:
        raise ValueError("invalid paired outcomes cannot become support evidence")
    return SupportEvidence(
        task_id=outcome.task_id,
        repeat_seed=outcome.repeat_seed,
        baseline_score=outcome.baseline_score,
        skill_score=outcome.skill_score,
        baseline_success=outcome.baseline_success,
        skill_success=outcome.skill_success,
        effect=outcome.skill_score - outcome.baseline_score,
        evidence_source=evidence_source,  # type: ignore[arg-type]
        provenance={"task_id": outcome.task_id, "repeat_seed": outcome.repeat_seed},
    )


@dataclass(frozen=True)
class SupportDecision:
    deploy: bool
    reason: str


@dataclass(frozen=True)
class SupportModel:
    synthesis: SynthesisResult
    training_task_ids: tuple[str, ...]
    feature_fields: tuple[str, ...]

    def decide(self, context: Mapping[str, object]) -> SupportDecision:
        if self.synthesis.status != "identified" or not self.synthesis.predicate:
            return SupportDecision(False, FALLBACK_TO_BASE_AGENT)
        if match_predicate(self.synthesis.predicate, context):
            return SupportDecision(True, "identified_support")
        return SupportDecision(False, FALLBACK_TO_BASE_AGENT)

    def to_dict(self) -> dict[str, object]:
        return {
            "synthesis": self.synthesis.to_dict(),
            "training_task_ids": list(self.training_task_ids),
            "feature_fields": list(self.feature_fields),
        }


def decide_policies(
    support_model: SupportModel,
    test_tasks: Sequence[Any],
    *,
    policy_name: str,
    decision_timestamp: str,
) -> list[PolicyDecision]:
    decisions: list[PolicyDecision] = []
    for task in test_tasks:
        support = support_model.decide(task.public_context)
        decisions.append(
            PolicyDecision.create(
                task_id=task.task_id,
                policy_name=policy_name,
                deploy=support.deploy,
                reason=support.reason if support.deploy else FALLBACK_TO_BASE_AGENT,
                decision_timestamp=decision_timestamp,
            )
        )
    return decisions

def _grammar(contexts: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], tuple[str, ...]]:
    fields = sorted({str(key) for context in contexts for key in context})
    features = []
    for field in fields:
        values = [context.get(field) for context in contexts]
        kind = "numeric" if values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values) else "categorical"
        features.append({"path": field, "type": kind})
    return {
        "schema_version": 1,
        "features": features,
        "max_depth": 3,
        "max_literals": 4,
    }, tuple(fields)


def fit_support_model(
    train_outcomes: Sequence[Mapping[str, object]],
    public_contexts: Mapping[str, Mapping[str, object]],
    *,
    delta: float = 0.05,
) -> SupportModel:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in train_outcomes:
        task_id = str(row.get("task_id", ""))
        context = public_contexts.get(task_id)
        if not task_id or context is None:
            raise ValueError("train outcome lacks a matching public context")
        if not bool(row.get("execution_valid", False)):
            continue
        grouped.setdefault(task_id, []).append(row)
    observations: list[BoundaryObservation] = []
    contexts: list[Mapping[str, object]] = []
    for task_id in sorted(grouped):
        rows = grouped[task_id]
        effects = [float(row["effect"]) for row in rows]
        lower, upper = paired_repetition_interval(effects, delta)
        context = dict(public_contexts[task_id])
        contexts.append(context)
        observations.append(
            BoundaryObservation(
                observation_id=task_id,
                context=context,
                effect=sum(effects) / len(effects),
                gate_passed=all(bool(row.get("scientific_gate", False)) for row in rows),
                effect_lower=lower,
                effect_upper=upper,
            )
        )
    if not observations:
        empty = synthesize_boundary([], {"schema_version": 1, "features": [{"path": "missing", "type": "categorical"}]})
        return SupportModel(empty, (), ())
    grammar, fields = _grammar(contexts)
    result = synthesize_boundary(observations, grammar, decision_contexts=contexts, delta=delta)
    return SupportModel(result, tuple(sorted(grouped)), fields)
