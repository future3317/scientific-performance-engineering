"""Matched condition adapters for the formal driver and evolution harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from benchmark.harness.experience_retrieval import RawExperienceRetriever
from core.acre.engine import AcreEngine
from core.models import RuleSpec, RuleState, TaskContext


class FormalConditionAdapter:
    """Expose C raw retrieval and D governed routing through one interface."""

    def __init__(self, condition: str, store: str | Path, *, token_budget: int = 4096) -> None:
        self.condition = condition.upper()
        self.store = Path(store)
        self.token_budget = token_budget
        if self.condition not in {"C", "C_STRESS", "D"}:
            raise ValueError("formal condition adapter supports C, C_STRESS, or D")

    def _governed_engine(self) -> AcreEngine:
        specs: list[RuleSpec] = []
        states: dict[str, RuleState] = {}
        for path in sorted((self.store / "rules").glob("*.json")):
            try:
                card = json.loads(path.read_text(encoding="utf-8"))
                spec = RuleSpec.from_dict(card)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            specs.append(spec)
            states[spec.rule_id] = RuleState(
                rule_id=spec.rule_id, version=spec.version, status="canonical",
                effect={"utility": 0.2}, confidence_sequence={"lcb": 0.2}, retrieval_utility=0.2,
            )
        return AcreEngine(rule_specs=specs, rule_states=states)

    def propose_interventions(self, context: TaskContext | Mapping[str, Any]) -> list[str]:
        query = str(context.workload.get("mechanism", "")) if isinstance(context, TaskContext) else str(context.get("mechanism", ""))
        if self.condition in {"C", "C_STRESS"}:
            return RawExperienceRetriever(self.store, token_budget=self.token_budget).propose_interventions(query=query)
        engine = self._governed_engine()
        routed = engine.route(context, token_budget=self.token_budget)
        specs = {spec.rule_id: spec for spec in engine.rule_specs}
        return [str(specs[item].intervention.get("action", item)) for item in routed.selected_rule_ids if item in specs]
