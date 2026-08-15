#!/usr/bin/env python3
"""Reset mode does not carry ordinary task context across tasks."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.harness.context import TaskContext
from benchmark.harness import conditions
from scripts.render_skill_view import render_skill_view


def main() -> None:
    reset = TaskContext("reset")
    reset.begin_task("TASK-A")
    reset.record("trajectory", [1, 2, 3])
    state = reset.begin_task("TASK-B")
    assert state == {"task_id": "TASK-B"}

    carry = TaskContext("carry")
    carry.begin_task("TASK-A")
    carry.record("trajectory", [1, 2, 3])
    assert carry.begin_task("TASK-B")["trajectory"] == [1, 2, 3]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "skill"
        root.mkdir()
        (root / "SKILL.md").write_text("# skill\n", encoding="utf-8")
        bundle = Path(tmp) / "bundle"
        render_skill_view(root, bundle)
        manifest = conditions.materialize_condition("A", None, Path(tmp) / "a", context_mode="reset")
        assert manifest["context_mode"] == "reset"

    print("test_context_mode: OK")


if __name__ == "__main__":
    main()
