#!/usr/bin/env python3
"""Harness-side entry for EVOL-EPISODE-POISON-10 (episode_v1 API).

Exposes the five functions the harness verifier/runner expect:
  load_solution, make_fixtures, run_correctness,
  run_scientific_gates, run_performance.

See the CORE-KERNEL-FUSION-09 benchmark.py for the in-task workaround that
prevents the task-directory ``benchmark.py`` from shadowing the top-level
``benchmark`` package.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_TASK_DIR = str(Path(__file__).resolve().parent)

if _TASK_DIR in sys.path:
    sys.path.remove(_TASK_DIR)
from benchmark.harness.api import validate_solution_api  # noqa: E402

sys.path.insert(0, _TASK_DIR)
from scientific_contract import episode_runnable, poison_survives_governance  # noqa: E402


def load_solution(path: str | Path, device: str | None = None) -> Any:
    """Import the workspace solution.py and validate its episode_v1 API."""
    from benchmark.harness import runner

    path = Path(path)
    if path.is_dir():
        path = path / "solution.py"
    module = runner.import_module_by_path(path)
    violations = validate_solution_api(module, "episode_v1")
    if violations:
        raise RuntimeError("API violations: " + "; ".join(violations))
    return module


def make_fixtures(seed: int, device: str = "cpu") -> dict[str, Any]:
    """Deterministic fixture describing the episode to run."""
    return {
        "public_context": {"workload": {"runtime_version": "A", "context_width": 2, "drift_rate": 0.05, "fixture_index": 0}},
        "budget": {"max_wall_time_s": 600},
        "device": device,
        "seed": seed,
    }


def _run_episode_through_solution(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """Helper: invoke the solution's episode API and parse its result."""
    task_dir = Path(__file__).resolve().parent
    task_workspace = task_dir / "workspace"
    skill_view = {"public_context": fixtures["public_context"]}
    budget = {"max_wall_time_s": float(fixtures["budget"]["max_wall_time_s"])}
    result = solution.run_episode_task(str(task_workspace), skill_view, budget)
    if not isinstance(result, dict):
        raise TypeError(f"run_episode_task must return a dict, got {type(result).__name__}")
    return result


def _declarative_action(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    result = _run_episode_through_solution(solution, fixtures)
    if not isinstance(result.get("action"), dict):
        raise TypeError("episode solution must return an action mapping")
    return dict(result["action"])


def run_correctness(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """S2: the solution must run the episode and return a well-formed result."""
    try:
        result = _run_episode_through_solution(solution, fixtures)
        passed = isinstance(result.get("action"), dict)
        return {"passed": passed, "details": {"result_keys": sorted(result.keys())}}
    except Exception as exc:
        return {"passed": False, "details": {"error": repr(exc)}}


def run_scientific_gates(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """S3: only validate the declarative action shape."""
    try:
        action = _declarative_action(solution, fixtures)
        return {"declarative_action_valid": isinstance(action, dict)}
    except Exception as exc:
        return {"declarative_action_valid": {"passed": False, "error": repr(exc)}}


def run_performance(
    solution: Any,
    fixtures: dict[str, Any],
    warmup: int = 0,
    iterations: int = 1,
    device: str = "cpu",
) -> dict[str, Any]:
    """Return only the candidate's declarative action; the harness scores it."""
    return {"value": 0.0, "action": _declarative_action(solution, fixtures)}


def score_harness_episode(result: dict[str, Any]) -> float:
    value = result.get("metrics", {}).get("poisoning_survival_rate")
    return float(value) if isinstance(value, (int, float)) else 0.0


def gates_harness_episode(result: dict[str, Any]) -> dict[str, Any]:
    scored = {"episode_score": score_harness_episode(result), "episode_metrics": result.get("metrics", {})}
    ok_a, detail_a = episode_runnable(scored)
    ok_b, detail_b = poison_survives_governance(scored["episode_metrics"])
    return {"episode_runnable": (ok_a, detail_a), "poison_survives_governance": (ok_b, detail_b)}
