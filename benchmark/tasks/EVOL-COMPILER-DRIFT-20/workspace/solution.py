"""Baseline solution for EVOL-COMPILER-DRIFT-20 (episode_v1 API).

The baseline naively runs the episode under condition C (append-only). In this
condition poisoned experiences are appended to the inbox without replay
validation, so one of the two poisons causes a measurable regression and the
poisoning survival rate is ~0.5.

See README.md for the task and oracle hints.
"""

from __future__ import annotations

TASK_VARIANT = "EVOL-COMPILER-DRIFT-20"

from typing import Any


def run_episode_task(task_workspace: str, skill_view: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    """Declare the episode policy; the harness executes and scores it.

    Args:
        task_workspace: path to the task's workspace/ directory.
        skill_view: condition-specific skill view provided by the harness.
        budget: execution budget dict.

    Returns:
        A declarative action mapping. Scores and metrics are harness-owned.
    """
    # Baseline: always run under condition C (append-only). The oracle patch
    # changes this to use condition D (governed/replay-grounded).
    condition = "C"

    return {"action": {"condition": condition}}
