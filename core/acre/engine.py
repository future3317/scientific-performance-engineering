"""Single public orchestration façade for ACRE core semantics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from core.governance import EvolutionDecision
from core.models import EvidenceEvent, RelationSpec, RelationState, RuleSpec, RuleState, TaskContext

from .router import ConservativeCausalRouter, RoutingDecision
from .controller import AcreController


class AcreEngine:
    def __init__(
        self,
        *,
        rule_specs: Sequence[RuleSpec] = (),
        rule_states: Mapping[str, RuleState] | None = None,
        relation_specs: Sequence[RelationSpec] = (),
        relation_states: Mapping[str, RelationState] | None = None,
        query_proposer: Callable[[TaskContext | Mapping[str, Any], Sequence[EvidenceEvent]], Any] | None = None,
    ) -> None:
        self.rule_specs = tuple(rule_specs)
        self.rule_states = dict(rule_states or {})
        self.relation_specs = tuple(relation_specs)
        self.relation_states = dict(relation_states or {})
        self._query_proposer = query_proposer
        self._controller = AcreController()

    def observe(self, event: EvidenceEvent | Mapping[str, Any]) -> EvidenceEvent:
        return self._controller.observe(event)

    def propose_experiment(self, context: TaskContext | Mapping[str, Any]) -> Any:
        if self._query_proposer is None:
            return None
        return self._query_proposer(context, self._controller.events)

    def evolve(self, subject_id: str) -> EvolutionDecision:
        if subject_id in self.rule_states:
            subject_type, state = "rule", self.rule_states[subject_id]
        elif subject_id in self.relation_states:
            subject_type, state = "relation", self.relation_states[subject_id]
        else:
            return EvolutionDecision("rule", subject_id, "NO_OP", "rejected", "none", "unknown subject")
        assessment = self._controller.assess()
        evidence_ids = tuple(event.event_id for event in self._controller.events)
        if assessment.specialization_event_ids:
            return EvolutionDecision(subject_type, subject_id, "SPECIALIZE", "review_required", "human-review", "adversarial evidence requires specialization or quarantine review", evidence_ids)
        if state.status == "retired":
            return EvolutionDecision(subject_type, subject_id, "RETIRE", "review_required", "human-review", "subject is retired", evidence_ids)
        if state.drift_state != "stable":
            return EvolutionDecision(subject_type, subject_id, "REVALIDATE", "review_required", "human-review", "subject drift requires revalidation", evidence_ids)
        return EvolutionDecision(subject_type, subject_id, "NO_OP", "approved", "bounded-auto", "subject state is stable", evidence_ids)

    def route(self, context: TaskContext | Mapping[str, Any], token_budget: int | None = None) -> RoutingDecision:
        budget = token_budget if token_budget is not None else (context.token_budget if isinstance(context, TaskContext) else 4096)
        return ConservativeCausalRouter(token_budget=budget).route(
            self.rule_specs, self.rule_states, self.relation_specs, self.relation_states, context
        )
