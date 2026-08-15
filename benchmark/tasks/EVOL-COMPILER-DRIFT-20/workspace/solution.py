"""Baseline solution for EVOL-COMPILER-DRIFT-20 (episode_v1 API).

The baseline naively runs the episode under condition C (append-only). In this
condition poisoned experiences are appended to the inbox without replay
validation, so one of the two poisons causes a measurable regression and the
poisoning survival rate is ~0.5.

See README.md for the task and oracle hints.
"""

from __future__ import annotations

TASK_VARIANT = "EVOL-COMPILER-DRIFT-20"

import tempfile
from pathlib import Path
from typing import Any


def run_episode_task(task_workspace: str, skill_view: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    """Run the prototype episode and return a score for the verifier.

    Args:
        task_workspace: path to the task's workspace/ directory.
        skill_view: condition-specific skill view provided by the harness.
        budget: execution budget dict.

    Returns:
        dict with at least ``episode_score`` (float, 0..1) and
        ``episode_metrics``.
    """
    # Baseline: always run under condition C (append-only). The oracle patch
    # changes this to use condition D (governed/replay-grounded).
    condition = "C"

    # Import the harness episode runner. The repo root is on sys.path when the
    # harness evaluates this module, so the top-level benchmark package is
    # importable.
    from benchmark.harness import evolution

    episode_yaml = Path(task_workspace).parents[0] / "episodes" / "compiler_drift_episode.yaml"
    out_dir = Path(tempfile.mkdtemp(prefix="spe_evo_episode_"))
    result = evolution.run_episode(str(episode_yaml), condition, out_dir)
    metrics = result.get("metrics", {})
    survival = metrics.get("poisoning_survival_rate")
    score = float(survival) if survival is not None else 0.0
    return {
        "episode_score": score,
        "episode_metrics": metrics,
        "condition_used": condition,
        "out_dir": str(out_dir),
    }
