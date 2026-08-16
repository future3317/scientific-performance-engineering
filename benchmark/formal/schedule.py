"""Deterministic condition/context/task schedule construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import random

from benchmark.harness import miniyaml, split
from core.sequential_stats import minimum_all_successes


class FamilyReplayExecutor:
    """Harness executor for representative paired replay contexts.

    It evaluates the declared FamilyEnvironment action and returns repeated
    bounded measurements; no hidden applicability label is consulted.
    """

    def __init__(self, family_id: str, action_id: str, *, repetitions: int = 512, transformation_state: Any = None) -> None:
        if repetitions < 2:
            raise ValueError("replay requires at least two paired repetitions")
        self.family_id = str(family_id)
        self.action_id = str(action_id)
        self.repetitions = int(repetitions)
        self.transformation_state = transformation_state

    def execute(self, context: Mapping[str, Any], *, arm: str = "on") -> Mapping[str, Any]:
        from benchmark.families.environment import FamilyEnvironment
        env = FamilyEnvironment(self.family_id)
        context = context.get("context", context) if isinstance(context.get("context", context), Mapping) else context
        deployed = [self.action_id] if arm == "on" else []
        outcome = env.evaluate(context, deployed, self.transformation_state)
        return {
            "utility": float(outcome.utility),
            "scientific_ok": all(outcome.scientific_gates.values()),
            "measurements": [float(outcome.utility)] * self.repetitions,
            "oracle_bundle": list(outcome.oracle_bundle),
        }


class PromotionReplayScheduler:
    """Select representative contexts for promotion replay only."""

    def __init__(self, *, p_min: float = 0.8, delta: float = 0.05) -> None:
        self.p_min = float(p_min)
        self.delta = float(delta)
        self.minimum_groups = minimum_all_successes(self.p_min, self.delta) if self.p_min > 0.0 else 1

    def pending_contexts(
        self,
        family_id: str,
        *,
        seen_group_ids: set[str] | None = None,
        seed: int = 0,
    ) -> list[dict[str, Any]]:
        """Return preregistered representative/query contexts not yet replayed."""
        from benchmark.families import family_views

        seen = {str(value) for value in (seen_group_ids or set())}
        # The task stream is only the proposal trigger.  Independent replay
        # groups come from preregistered family contexts and therefore remain
        # available even when a family has fewer public anchor tasks.
        pools = family_views(family_id, count=max(3 * self.minimum_groups, 24), seed=seed)
        contexts: list[dict[str, Any]] = []
        for instance in pools["representative_pool"]:
            group_id = f"family-{instance.instance_id}"
            if group_id in seen:
                continue
            contexts.append({
                "context_id": instance.instance_id,
                "independence_group": group_id,
                "query_type": "representative",
                "context": {"workload": dict(instance.parameters)},
                "experiment_cost": 1.0,
            })
            if len(contexts) >= self.minimum_groups:
                break
        return contexts

    def execute(
        self,
        subject_id: str,
        family_id: str,
        *,
        executor: Any,
        record_case: Any,
        update_certificate: Any,
        seen_group_ids: set[str] | None = None,
        seed: int = 0,
    ) -> Any:
        """Execute representative replay groups through the Core paired API."""
        from core.acre.experiments import ExperimentPlan, execute_paired_plan

        contexts = self.pending_contexts(family_id, seen_group_ids=seen_group_ids, seed=seed)
        plan = ExperimentPlan(subject_id=str(subject_id), contexts=tuple(contexts), max_groups=self.minimum_groups)
        return execute_paired_plan(
            plan,
            executor,
            record_case=record_case,
            update_certificate=update_certificate,
        )


class PendingCandidateScheduler(PromotionReplayScheduler):
    """Harness-owned view of collecting candidates awaiting more evidence."""

    def for_candidate(
        self,
        candidate: Mapping[str, Any],
        family_id: str,
        *,
        seen_group_ids: set[str] | None = None,
        seed: int = 0,
    ) -> list[dict[str, Any]]:
        pending = candidate.get("replay_schedule", {}).get("pending_contexts") if isinstance(candidate.get("replay_schedule"), Mapping) else None
        if isinstance(pending, list):
            return [dict(item) for item in pending if isinstance(item, Mapping)]
        return self.pending_contexts(family_id, seen_group_ids=seen_group_ids, seed=seed)


class SynthesisAcquisitionScheduler:
    """Plan active/boundary contexts used to shrink the CEGIS version space."""

    def pending_contexts(self, family_id: str, *, seen_context_ids: set[str] | None = None, seed: int = 0) -> list[dict[str, Any]]:
        from benchmark.families import family_views
        seen = {str(item) for item in (seen_context_ids or set())}
        pools = family_views(family_id, count=24, seed=seed)
        return [
            {
                "context_id": item.instance_id,
                "query_type": "active_query",
                "context": {"workload": dict(item.parameters)},
                "experiment_cost": 1.0,
            }
            for item in pools["active_query_pool"]
            if item.instance_id not in seen
        ]


class ValidationScheduler:
    """Plan disjoint replication/transfer/boundary/adversarial validation."""

    def contexts(self, family_id: str, *, seed: int = 0) -> dict[str, list[dict[str, Any]]]:
        from benchmark.families import family_views
        pools = family_views(family_id, count=24, seed=seed)
        return {
            "replication": [{"context_id": item.instance_id, "context": {"workload": dict(item.parameters)}} for item in pools["representative_pool"]],
            "transfer": [{"context_id": item.instance_id, "context": {"workload": dict(item.parameters)}} for item in pools["active_query_pool"]],
            "boundary": [{"context_id": item.instance_id, "context": {"workload": dict(item.parameters)}} for item in pools["sealed_boundary_pool"]],
            "adversarial": [],
        }


class RelationExperimentScheduler:
    """Harness-owned scheduler for relation evidence.

    A relation proposal is never sent through node utility replay.  It first
    receives a complete factorial design; the resulting contrast certificate
    is the only input that can later reach relation governance.
    """

    def schedule(self, candidate: Mapping[str, Any], family_id: str, *, seed: int = 0) -> dict[str, Any]:
        from benchmark.families import family_views
        pools = family_views(family_id, count=24, seed=seed)
        contexts = [item for item in (*pools["representative_pool"], *pools["active_query_pool"])]
        return {
            "evidence_type": "factorial_contrast",
            "relation_id": str(candidate.get("relation_id") or candidate.get("id") or ""),
            "family_id": family_id,
            "arms": ["00", "10", "01", "11"],
            "pending_contexts": [
                {"context_id": item.instance_id, "context": {"workload": dict(item.parameters)}, "independence_group": f"relation-{item.instance_id}"}
                for item in contexts
            ],
            "status": "scheduled",
        }

    def execute(
        self,
        candidate: Mapping[str, Any],
        family_id: str,
        *,
        block_executor: Any,
        maintainer: Any,
        seed: int = 0,
        delta: float = 0.05,
        practical_margin: float = 0.05,
    ) -> dict[str, Any]:
        """Materialize scheduled contexts and run factorial inference in Core.

        ``block_executor`` is the only harness callback: it executes one
        context and returns canonical ``FactorialBlock`` values.  Scheduling
        and semantic inference are otherwise not represented as metadata.
        """
        plan = self.schedule(candidate, family_id, seed=seed)
        context_blocks: dict[str, list[Any]] = {}
        for item in plan["pending_contexts"]:
            context_id = str(item["context_id"])
            blocks = block_executor(dict(item["context"]), context_id=context_id)
            if not isinstance(blocks, (list, tuple)) or not blocks:
                raise ValueError(f"relation context produced no factorial blocks: {context_id}")
            context_blocks[context_id] = list(blocks)
        identification = maintainer.execute_relation_experiment(
            context_blocks, delta=delta, practical_margin=practical_margin,
        )
        plan["status"] = "executed"
        plan["identification"] = {
            "decision": identification.decision,
            "context_decisions": dict(identification.context_decisions),
            "applicability_predicate": identification.applicability_predicate,
            "confidence": dict(identification.confidence),
        }
        endpoint_versions = candidate.get("endpoint_versions")
        if isinstance(endpoint_versions, Mapping) and endpoint_versions:
            certificates = maintainer.relation_certificates(
                context_blocks,
                endpoint_versions={str(key): int(value) for key, value in endpoint_versions.items()},
                delta=delta,
                practical_margin=practical_margin,
            )
            plan["relation_evidence_certificates"] = {
                key: value.to_dict() for key, value in certificates.items()
            }
        return plan


def task_order(split_path: str | Path) -> list[tuple[int, str]]:
    manifest = split.load_split_manifest(split_path)
    ordered: list[tuple[int, str]] = []
    for phase in sorted(manifest["phases"], key=lambda item: int(item["index"])):
        ordered.extend((int(phase["index"]), str(task_id)) for task_id in phase.get("tasks", []))
    if len({task_id for _, task_id in ordered}) != len(ordered):
        raise ValueError("formal task schedule cannot contain duplicate task IDs")
    return ordered


def build_schedule(
    split_path: str | Path,
    *,
    conditions: tuple[str, ...] = ("A", "B", "C", "D"),
    context_modes: tuple[str, ...] = ("reset",),
    outer_trials: int = 3,
) -> list[dict[str, Any]]:
    if outer_trials < 1:
        raise ValueError("outer_trials must be positive")
    if not conditions:
        raise ValueError("at least one condition is required")
    if any(condition not in {"A", "B", "C", "C_STRESS", "D"} for condition in conditions):
        raise ValueError("unknown formal condition")
    if any(mode not in {"reset", "carry"} for mode in context_modes):
        raise ValueError("unknown context mode")
    tasks = task_order(split_path)
    schedule: list[dict[str, Any]] = []
    index = 0
    for outer_trial in range(outer_trials):
        # Rotate the condition stream between outer trials.  Task order within
        # a condition remains fixed, while thermal/cache and queue effects are
        # not deterministically assigned to one treatment.
        # This is preregistered blocked randomization with a rotation and a
        # seeded shuffle; it is intentionally not called a Latin square.
        rotation = outer_trial % len(conditions)
        rotated = conditions[rotation:] + conditions[:rotation]
        rng = random.Random(outer_trial)
        order = list(rotated)
        rng.shuffle(order)
        ordered_conditions = tuple(order)
        for condition in ordered_conditions:
            for context_mode in context_modes:
                stream_id = f"trial-{outer_trial:03d}-{condition}-{context_mode}"
                for phase, task_id in tasks:
                    schedule.append(
                        {
                            "schedule_index": index,
                            "stream_id": stream_id,
                            "outer_trial_id": f"outer-{outer_trial:03d}",
                            "outer_trial_index": outer_trial,
                            "condition": condition,
                            "condition_block_order": list(ordered_conditions),
                            "randomization_seed": outer_trial,
                            "context_mode": context_mode,
                            "phase": phase,
                            "task_id": task_id,
                        }
                    )
                    index += 1
    return schedule
