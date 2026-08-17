import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest


def _load_compile_benchmark(task_id: str):
    task_dir = Path(__file__).parents[1] / "tasks" / task_id
    spec = importlib.util.spec_from_file_location(
        f"compile_calibration_{task_id}", task_dir / "benchmark.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, task_dir


def test_compile_solution_loads_get_fresh_inductor_and_triton_caches():
    module, task_dir = _load_compile_benchmark("CORE-COMPILE-TINY-12")
    solution_path = task_dir / "workspace" / "solution.py"

    module.load_solution(str(solution_path))
    first = (
        os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        os.environ.get("TRITON_CACHE_DIR"),
    )
    import torch._inductor.config as inductor_config

    assert inductor_config.compile_threads == 2
    module.load_solution(str(solution_path))
    second = (
        os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        os.environ.get("TRITON_CACHE_DIR"),
    )

    assert first[0] and first[1]
    assert second[0] and second[1]
    assert first != second


@pytest.mark.parametrize(
    "task_id",
    [
        "CORE-COMPILE-RECOMPILE-04",
        "CORE-COMPILE-DYNAMIC-11",
        "CORE-COMPILE-TINY-12",
    ],
)
def test_reference_patch_reproduces_executable_oracle(task_id):
    task_dir = Path(__file__).parents[1] / "tasks" / task_id
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        shutil.copy2(task_dir / "workspace" / "solution.py", temp_path / "solution.py")
        subprocess.run(
            ["git", "apply", str(task_dir / "oracle" / "reference_patch.diff")],
            cwd=temp_path,
            check=True,
        )
        patched = (temp_path / "solution.py").read_text().splitlines()
    oracle = (task_dir / "oracle" / "solution_oracle.py").read_text().splitlines()
    assert patched == oracle


@pytest.mark.skipif(
    sys.platform == "win32" and shutil.which("cl") is None,
    reason="TorchInductor CPU compilation requires the MSVC compiler on Windows",
)
def test_dynamic_compile_activation_is_harness_owned_and_contrastive():
    module, task_dir = _load_compile_benchmark("CORE-COMPILE-DYNAMIC-11")
    fixtures = module.make_fixtures(7)
    candidate = module.load_solution(str(task_dir / "oracle" / "solution_oracle.py"))
    baseline = module.load_solution(str(task_dir / "workspace" / "solution.py"))

    evidence = module.run_activation_evidence(candidate, baseline, fixtures)

    assert set(evidence) == {"candidate_metrics", "baseline_metrics"}
    assert evidence["candidate_metrics"]["compile_cache"] != evidence["baseline_metrics"]["compile_cache"]
    assert evidence["candidate_metrics"]["dynamic_guard_stable"] is True
    assert evidence["baseline_metrics"]["dynamic_guard_stable"] is False
