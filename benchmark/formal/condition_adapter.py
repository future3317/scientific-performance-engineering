"""Matched condition adapters for the formal driver and evolution harness."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Mapping

from benchmark.harness.experience_retrieval import RawExperienceRetriever
from core.acre.engine import AcreEngine
from core.models import TaskContext


def _canonical_public_context(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: dict(value[key]) if isinstance(value.get(key), dict) else value[key]
        for key in ("domain", "workload", "hardware", "software", "evidence")
        if key in value
    }
    workload = result.setdefault("workload", {})
    nested = workload.pop("family_parameters", None)
    if isinstance(nested, dict):
        result["workload"] = {**nested, **workload}
    return result


class FormalConditionAdapter:
    """Expose C raw retrieval and D governed routing through one interface."""

    def __init__(self, condition: str, store: str | Path, *, token_budget: int = 4096) -> None:
        self.condition = condition.upper()
        self.store = Path(store)
        self.token_budget = token_budget
        if self.condition not in {"C", "C_STRESS", "D"}:
            raise ValueError("formal condition adapter supports C, C_STRESS, or D")

    def _governed_engine(self) -> AcreEngine:
        return AcreEngine.from_store(self.store)

    @staticmethod
    def _task_context(context: TaskContext | Mapping[str, Any]) -> TaskContext:
        if isinstance(context, TaskContext):
            return context
        public = _canonical_public_context(context)
        workload = public.get("workload") if isinstance(public.get("workload"), dict) else dict(public)
        return TaskContext(
            domain=str(public.get("domain", "scientific-performance")),
            workload=workload,
            hardware=dict(public.get("hardware", {})),
            software=dict(public.get("software", {})),
            evidence=dict(public.get("evidence", {})),
            token_budget=int(context.get("token_budget", 4096)),
        )

    def retrieved_context(self, context: TaskContext | Mapping[str, Any]) -> dict[str, Any]:
        typed_context = self._task_context(context)
        if self.condition in {"C", "C_STRESS"}:
            query = json.dumps(typed_context.workload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            experiences = RawExperienceRetriever(self.store, token_budget=self.token_budget).retrieve(
                query=query
            )
            actions = RawExperienceRetriever(self.store, token_budget=self.token_budget).propose_interventions(
                query=query
            )
            return {
                "schema_version": 1,
                "condition": self.condition,
                "context": typed_context.to_dict(),
                "retrieved_experiences": experiences,
                "proposed_interventions": actions,
            }
        engine = self._governed_engine()
        routed = engine.route(typed_context, token_budget=self.token_budget)
        specs = {spec.rule_id: spec for spec in engine.rule_specs}
        actions = [str(specs[item].intervention.get("action", item)) for item in routed.selected_rule_ids if item in specs]
        return {
            "schema_version": 1,
            "condition": self.condition,
            "context": typed_context.to_dict(),
            "selected_rule_ids": list(routed.selected_rule_ids),
            "selected_relation_ids": [],
            "proposed_interventions": actions,
        }

    def propose_interventions(self, context: TaskContext | Mapping[str, Any]) -> list[str]:
        return list(self.retrieved_context(context).get("proposed_interventions", []))
