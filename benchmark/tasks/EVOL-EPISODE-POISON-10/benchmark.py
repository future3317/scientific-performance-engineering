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

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_TASK_DIR = str(Path(__file__).resolve().parent)

if _TASK_DIR in sys.path:
    sys.path.remove(_TASK_DIR)
from benchmark.harness.api import validate_solution_api  # noqa: E402
from benchmark.harness import evolution  # noqa: E402

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
    task_dir = Path(__file__).resolve().parent
    episode_yaml = task_dir / "episodes" / "poison_episode.yaml"
    return {
        "episode_yaml": str(episode_yaml),
        "condition": "C",
        "budget": {"max_wall_time_s": 120},
        "device": device,
        "seed": seed,
    }


def _run_episode_through_solution(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """Helper: invoke the solution's episode API and parse its result."""
    task_dir = Path(__file__).resolve().parent
    task_workspace = task_dir / "workspace"
    skill_view = {"condition": fixtures["condition"], "episode_yaml": fixtures["episode_yaml"]}
    budget = fixtures["budget"]
    budget = {**budget, "seed": int(fixtures.get("seed", 0))}
    result = solution.run_episode_task(str(task_workspace), skill_view, budget)
    if not isinstance(result, dict):
        raise TypeError(f"run_episode_task must return a dict, got {type(result).__name__}")
    return result


def run_correctness(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """S2: the solution must run the episode and return a well-formed result."""
    try:
        result = _run_episode_through_solution(solution, fixtures)
        passed = isinstance(result.get("episode_score"), (int, float))
        return {"passed": passed, "details": {"result_keys": sorted(result.keys())}}
    except Exception as exc:
        return {"passed": False, "details": {"error": repr(exc)}}


def run_scientific_gates(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """S3: episode-specific scientific gates."""
    result = _run_episode_through_solution(solution, fixtures)
    metrics = result.get("episode_metrics", {})

    gates = {}
    passed, details = episode_runnable(result)
    gates["episode_runnable"] = (passed, details)
    passed, details = poison_survives_governance(metrics)
    gates["poison_survives_governance"] = (passed, details)
    return gates


def run_performance(
    solution: Any,
    fixtures: dict[str, Any],
    warmup: int = 0,
    iterations: int = 1,
    device: str = "cpu",
) -> dict[str, Any]:
    """S5: score the solution's episode handling.

    The metric is the episode's poisoning survival rate (higher is better).
    Baseline (condition C) scores ~0.5; oracle (condition D) scores 1.0.
    """
    start = time.perf_counter()
    result = _run_episode_through_solution(solution, fixtures)
    wall_s = time.perf_counter() - start
    score = float(result.get("episode_score", 0.0))
    ok_a, detail_a = episode_runnable(result)
    ok_b, detail_b = poison_survives_governance(result.get("episode_metrics", {}))
    gates = {"episode_runnable": (ok_a, detail_a), "poison_survives_governance": (ok_b, detail_b)}
    return {
        "value": score,
        "work_units": {"episode_runs": 1},
        "output_checksums": {"result": json.dumps(result, sort_keys=True, default=str)},
        "timing": {"wall_time_s": wall_s},
        "episode_result": result,
        "episode_gates": {name: bool(value[0] if isinstance(value, (tuple, list)) else value) for name, value in gates.items()},
        "episode_gate_details": {name: (value[1] if isinstance(value, (tuple, list)) and len(value) > 1 else {}) for name, value in gates.items()},
    }
