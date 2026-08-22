#!/usr/bin/env python3
"""Regression tests for module loading and task-package import isolation."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.harness import runner


def test_task_module_imports_are_isolated() -> None:
    root = Path("benchmark/tasks")
    before = list(sys.path)
    runner.import_module_by_path(root / "EVOL-COMPILER-DRIFT-20" / "benchmark.py")
    runner.import_module_by_path(root / "EVOL-EPISODE-POISON-10" / "benchmark.py")
    module = runner.import_module_by_path(root / "SCIML-GRAPH-REBUILD-08R2" / "benchmark.py")
    assert module.__name__.startswith("spe_evo_benchmark_")
    assert "SCIML-GRAPH-REBUILD-08R2" in module.__doc__
    assert sys.path == before


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task_dir = Path(tmp)
        (task_dir / "helper.py").write_text("VALUE = 7\n", encoding="utf-8")
        (task_dir / "benchmark.py").write_text(
            "from benchmark.harness import runner\n"
            "from helper import VALUE\n"
            "LOADED = (runner.__name__, VALUE)\n",
            encoding="utf-8",
        )
        before = list(sys.path)
        module = runner.import_module_by_path(task_dir / "benchmark.py")
        assert module.LOADED == ("benchmark.harness.runner", 7)
        assert sys.path == before

    print("test_runner: OK")


if __name__ == "__main__":
    main()
