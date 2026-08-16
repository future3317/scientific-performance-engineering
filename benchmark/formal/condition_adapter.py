"""Matched condition adapters for the formal driver and evolution harness."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Mapping

from benchmark.harness.experience_retrieval import RawExperienceRetriever
from benchmark.formal.schedule import PendingCandidateScheduler
from core.acre.engine import AcreEngine
from core.models import TaskContext
from core.cost import PromptCostModel


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
            query = typed_context.to_dict()
            retriever = RawExperienceRetriever(self.store, token_budget=self.token_budget)
            experiences = retriever.retrieve(query=query)
            actions = retriever.propose_from_records(experiences)
            rendered = {"experiences": experiences, "proposed_interventions": actions}
            return {
                "schema_version": 1,
                "condition": self.condition,
                "context": typed_context.to_dict(),
                "retrieved_experiences": experiences,
                "proposed_interventions": actions,
                "token_cost": PromptCostModel().cost(rendered),
            }
        engine = self._governed_engine()
        routed = engine.route(typed_context, token_budget=self.token_budget)
        specs = {spec.rule_id: spec for spec in engine.rule_specs}
        actions = [str(specs[item].intervention.get("action", item)) for item in routed.selected_rule_ids if item in specs]
        rule_views = []
        for item in routed.selected_rule_ids:
            spec = specs.get(item)
            state = engine.rule_states.get(item)
            if spec is None or state is None:
                continue
            rule_views.append({
                "rule_id": spec.rule_id,
                "version": spec.version,
                "action": dict(spec.intervention),
                "expected_mechanism": spec.expected_mechanism,
                "applicability": dict(spec.applicability),
                "scientific_invariants": list(spec.scientific_invariants),
                "abstain_conditions": dict(spec.abstain_conditions),
                "effect_lcb": float(state.confidence_sequence.get("utility_effect_lcb", state.retrieval_utility)),
                "provenance": dict(spec.provenance_policy),
            })
        cost_model = PromptCostModel()
        while rule_views and cost_model.cost({"rule_views": rule_views}) > self.token_budget:
            rule_views.pop()
        selected_set = set(routed.selected_rule_ids)
        relation_ids = [
            spec.relation_id for spec in engine.relation_specs
            if set(spec.endpoints.values()).issubset(selected_set)
            and engine.relation_states.get(spec.relation_id) is not None
            and engine.relation_states[spec.relation_id].status == "canonical"
        ]
        pending: list[dict[str, Any]] = []
        candidate_dir = self.store / "evolution" / "candidates"
        if candidate_dir.is_dir():
            scheduler = PendingCandidateScheduler()
            for path in sorted(candidate_dir.glob("*.json")):
                try:
                    candidate = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(candidate, dict) or candidate.get("status") != "collecting_evidence":
                    continue
                family_id = candidate.get("family_id")
                if isinstance(family_id, str):
                    pending.extend(scheduler.for_candidate(candidate, family_id))
        return {
            "schema_version": 1,
            "condition": self.condition,
            "context": typed_context.to_dict(),
            "selected_rule_ids": list(routed.selected_rule_ids),
            "selected_relation_ids": relation_ids,
            "proposed_interventions": actions,
            "rule_views": rule_views,
            "routing": {"optimizer_mode": routed.optimizer_mode, "objective": routed.objective, "blockers": list(routed.blockers)},
            "pending_replay_contexts": pending,
            "token_cost": cost_model.cost({"rule_views": rule_views, "relations": relation_ids, "pending": pending}),
        }

    def propose_interventions(self, context: TaskContext | Mapping[str, Any]) -> list[str]:
        return list(self.retrieved_context(context).get("proposed_interventions", []))
