from __future__ import annotations

from typing import Mapping


PUBLIC_FIELDS = (
    "task_family",
    "goal_operation",
    "object_category",
    "material_category",
    "target_state_change",
    "requires_measurement",
    "requires_container",
    "variation_bucket",
)

FORBIDDEN_FIELDS = frozenset(
    {
        "gold_trajectory",
        "reference_trajectory",
        "reference_actions",
        "final_score",
        "baseline_score",
        "skill_score",
        "oracle_support",
        "judge_result",
    }
)


def _all_keys(value: Mapping[str, object]) -> set[str]:
    keys = set(value)
    for nested in value.values():
        if isinstance(nested, Mapping):
            keys.update(_all_keys(nested))
    return keys


def extract_public_context(task: Mapping[str, object]) -> dict[str, object]:
    forbidden = sorted(_all_keys(task) & FORBIDDEN_FIELDS)
    if forbidden:
        raise ValueError(f"forbidden ScienceWorld task fields: {', '.join(forbidden)}")
    return {field: task[field] for field in PUBLIC_FIELDS if field in task}
