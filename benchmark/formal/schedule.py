"""Deterministic condition/context/task schedule construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from benchmark.harness import miniyaml, split
from core.sequential_stats import minimum_all_successes


class PromotionReplayScheduler:
    """Select public family contexts until the promotion gate is reachable."""

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
        pools = family_views(family_id, count=max(3 * self.minimum_groups, 24), seed=seed)
        contexts: list[dict[str, Any]] = []
        for instance in [*pools["representative_pool"], *pools["active_query_pool"]]:
            group_id = f"family-{instance.instance_id}"
            if group_id in seen:
                continue
            contexts.append({
                "context_id": instance.instance_id,
                "independence_group": group_id,
                "context": {"workload": dict(instance.parameters)},
            })
            if len(contexts) >= self.minimum_groups:
                break
        return contexts


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
        for condition in conditions:
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
                            "context_mode": context_mode,
                            "phase": phase,
                            "task_id": task_id,
                        }
                    )
                    index += 1
    return schedule
