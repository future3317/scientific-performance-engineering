"""Deterministic condition/context/task schedule construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.harness import miniyaml, split


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
