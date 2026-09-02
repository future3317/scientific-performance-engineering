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
    module = runner.import_module_by_path(root / "SCIML-GRAPH-REBUILD-08R2" / "benchmark.py")
    assert module.__name__.startswith("spe_evo_benchmark_")
    assert "SCIML-GRAPH-REBUILD-08R2" in module.__doc__
    assert sys.path == before


def test_thread_topology_configuration_is_idempotent() -> None:
    result = runner.run_python_subprocess(
        snippet=(
            "from benchmark.harness.runner import configure_thread_topology; "
            "topology = {"
            "'omp_num_threads': 8, 'mkl_num_threads': 8, "
            "'openblas_num_threads': 8, 'numexpr_num_threads': 8, "
            "'torch_num_threads': 8, 'torch_num_interop_threads': 1, "
            "'compiler_threads': 2}; "
            "configure_thread_topology(topology); "
            "configure_thread_topology(topology); "
            "print('ok')"
        ),
        timeout=10.0,
    )

    assert result["exit_code"] == 0, result["stderr"]
    assert result["stdout"].strip() == "ok"


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
