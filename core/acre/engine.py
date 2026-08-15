"""Single public orchestration façade for ACRE core semantics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from core.governance import EvolutionDecision
from core.models import EvidenceEvent, RelationSpec, RelationState, RuleSpec, RuleState, TaskContext

from .router import ConservativeCausalRouter, RoutingDecision


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
        self._events: list[EvidenceEvent] = []

    def observe(self, event: EvidenceEvent | Mapping[str, Any]) -> EvidenceEvent:
        canonical = event if isinstance(event, EvidenceEvent) else EvidenceEvent.from_dict(dict(event))
        self._events.append(canonical)
        return canonical

    def propose_query(self, context: TaskContext | Mapping[str, Any]) -> Any:
        if self._query_proposer is None:
            return None
        return self._query_proposer(context, tuple(self._events))

    def update_rule(self, rule_id: str) -> EvolutionDecision:
        state = self.rule_states.get(rule_id)
        if state is None:
            return EvolutionDecision("rule", rule_id, "revalidate", "rejected", "none", "unknown rule")
        status = "approved" if state.status != "retired" and state.drift_state == "stable" else "review_required"
        mode = "bounded-auto" if status == "approved" else "human-review"
        return EvolutionDecision("rule", rule_id, "revalidate", status, mode, "rule state evaluated")

    def update_relation(self, relation_id: str) -> EvolutionDecision:
        state = self.relation_states.get(relation_id)
        if state is None:
            return EvolutionDecision("relation", relation_id, "revalidate", "rejected", "none", "unknown relation")
        status = "approved" if state.status != "retired" and state.drift_state == "stable" else "review_required"
        mode = "bounded-auto" if status == "approved" else "human-review"
        return EvolutionDecision("relation", relation_id, "revalidate", status, mode, "relation state evaluated")

    def route(self, context: TaskContext | Mapping[str, Any], token_budget: int | None = None) -> RoutingDecision:
        budget = token_budget if token_budget is not None else (context.token_budget if isinstance(context, TaskContext) else 4096)
        return ConservativeCausalRouter(token_budget=budget).route(
            self.rule_specs, self.rule_states, self.relation_specs, self.relation_states, context
        )
