"""Matched condition adapters for the formal driver and evolution harness."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Mapping

from benchmark.harness.experience_retrieval import RawExperienceRetriever
from benchmark.formal.schedule import PendingCandidateScheduler
from core.acre.engine import AcreEngine
from core.models import TaskContext
from core.cost import BudgetedContextRenderer
from core.public_context import build_public_context

class FormalConditionAdapter:
    """Expose C raw retrieval and D governed routing through one interface."""

    def __init__(self, condition: str, store: str | Path, *, token_budget: int = 4096, family_id: str | None = None) -> None:
        self.condition = condition.upper()
        self.store = Path(store)
        self.token_budget = token_budget
        self.family_id = family_id
        if self.condition not in {"B", "C", "C_STRESS", "D"}:
            raise ValueError("formal condition adapter supports B, C, C_STRESS, or D")

    def _governed_engine(self) -> AcreEngine:
        return AcreEngine.from_store(self.store)

    @staticmethod
    def _task_context(context: TaskContext | Mapping[str, Any]) -> TaskContext:
        if isinstance(context, TaskContext):
            return context
        public = build_public_context(context)
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
            experiences = retriever.retrieve(query=query, family_id=self.family_id)
            renderer = BudgetedContextRenderer(self.token_budget)
            # Actions are derived only after the renderer has selected the
            # retained records; a dropped experience cannot influence routing.
            while True:
                actions = retriever.propose_from_records(experiences)
                try:
                    rendered = renderer.render(
                        {
                            "schema_version": 1,
                            "condition": self.condition,
                            "context": typed_context.to_dict(),
                            "retrieved_experiences": experiences,
                            "proposed_interventions": actions,
                        },
                        entries_key="retrieved_experiences",
                    )
                    break
                except ValueError:
                    if not experiences:
                        raise
                    experiences = experiences[:-1]
            retained = rendered.get("retrieved_experiences", [])
            rendered["proposed_interventions"] = retriever.propose_from_records(retained)
            return rendered
        # B and D share the same router semantics.  B simply loads the
        # frozen snapshot and never runs maintenance; D may update it.
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
        renderer = BudgetedContextRenderer(self.token_budget)
        routing = {
            "optimizer_mode": routed.optimizer_mode,
            "objective": routed.objective,
            "blockers": list(routed.blockers),
            "required_experiments": list(routed.required_experiments),
        }
        rendered = renderer.render({
            "schema_version": 1,
            "condition": self.condition,
            "context": typed_context.to_dict(),
            "rule_views": rule_views,
            "relations": relation_ids,
            "pending_replay_contexts": pending,
            "proposed_interventions": actions,
            "routing": routing,
        }, entries_keys=("rule_views", "pending_replay_contexts"))
        retained_rule_views = rendered.get("rule_views", [])
        retained_rule_ids = [str(item["rule_id"]) for item in retained_rule_views if isinstance(item, dict) and item.get("rule_id")]
        actions = [str(item.get("action", {}).get("action", item["rule_id"])) for item in retained_rule_views if isinstance(item, dict) and isinstance(item.get("action"), dict)]
        retained_set = set(retained_rule_ids)
        retained_relations = [
            relation_id for relation_id in relation_ids
            if relation_id in {spec.relation_id for spec in engine.relation_specs
                               if set(spec.endpoints.values()).issubset(retained_set)}
        ]
        rendered = renderer.render({
            "schema_version": 1,
            "condition": self.condition,
            "context": typed_context.to_dict(),
            "rule_views": retained_rule_views,
            "relations": retained_relations,
            "pending_replay_contexts": rendered.get("pending_replay_contexts", []),
            "selected_rule_ids": retained_rule_ids,
            "selected_relation_ids": retained_relations,
            "proposed_interventions": actions,
            "routing": routing,
        }, entries_keys=("rule_views", "pending_replay_contexts"))
        return {
            **rendered,
        }

    def propose_interventions(self, context: TaskContext | Mapping[str, Any]) -> list[str]:
        return list(self.retrieved_context(context).get("proposed_interventions", []))
