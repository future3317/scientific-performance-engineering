#!/usr/bin/env python3
"""Public smoke test for EVOL-COMPILER-DRIFT-20.

Agent-visible sanity check: import the workspace solution, run the episode task,
and assert the result shape. Contains no hidden-verifier logic.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]
    workspace = Path(__file__).resolve().parents[1] / "workspace"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(workspace))
    try:
        import solution

        task_workspace = str(workspace)
        skill_view = {"condition": "C"}
        budget = {"max_wall_time_s": 60}
        result = solution.run_episode_task(task_workspace, skill_view, budget)
    finally:
        if str(workspace) in sys.path:
            sys.path.remove(str(workspace))
        if str(repo_root) in sys.path:
            sys.path.remove(str(repo_root))

    assert isinstance(result, dict), "run_episode_task must return a dict"
    assert "episode_score" in result, "result must contain 'episode_score'"
    score = result["episode_score"]
    assert isinstance(score, (int, float)), "episode_score must be numeric"
    assert 0.0 <= score <= 1.0, f"episode_score {score} out of [0,1]"
    assert "episode_metrics" in result, "result must contain 'episode_metrics'"

    print(f"smoke_test: OK (episode_score={score:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
