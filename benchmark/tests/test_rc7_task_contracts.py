from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def _load_benchmark(task_dir: Path):
    name = f"benchmark_{task_dir.name.lower().replace('-', '_')}"
    sys.path.insert(0, str(task_dir))
    try:
        spec = importlib.util.spec_from_file_location(name, task_dir / "benchmark.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(task_dir))


def test_blocked_empirical_record_cannot_project_to_eligible() -> None:
    from benchmark.calibration.report import build_pilot_calibration

    task_id = "CORE-SCALAR-SYNC-LOW-CADENCE-21"
    report = {
        "population_id": "test",
        "empirical_calibration": {
            "status": "observed",
            "calibration_gate": "blocked",
            "records_by_task": {
                task_id: {
                    "task_id": task_id,
                    "revision": "test-revision",
                    "calibration_status": "blocked",
                    "outer_trials": [],
                }
            },
            "missing_task_ids": [],
        },
        "empirical_rejection_flags": {},
        "semantic_gate_failures": [],
    }

    pilot = build_pilot_calibration(report, ROOT / "benchmark" / "tasks")
    projected = next(item for item in pilot["tasks"] if item["task_id"] == task_id)
    assert projected["status"] == "blocked"
    assert projected["eligibility"] is False


def test_counterexample_noise_floor_does_not_apply_positive_effect_cap() -> None:
    from benchmark.harness.verifier import noise_floor_blocks_calibration

    assert noise_floor_blocks_calibration({"kind": "positive", "oracle": {"expected_speedup_range": [0.9, 1.1]}}, 30.0)
    assert not noise_floor_blocks_calibration({"kind": "counterexample", "oracle": {"expected_speedup_range": [0.9, 1.1]}}, 30.0)


def test_crystal_oracle_uses_fp64_score_path(tmp_path: Path) -> None:
    import torch

    from benchmark.calibration.campaign import _copy_oracle

    task_dir = ROOT / "benchmark" / "tasks" / "SCIML-CRYSTAL-DIFFUSION-07R2"
    solution_dir = tmp_path / "crystal"
    _copy_oracle(task_dir, solution_dir)
    benchmark = _load_benchmark(task_dir)
    solution = benchmark.load_solution(solution_dir / "solution.py", device="cpu")
    sampler = solution.build_sampler(benchmark.make_fixtures(0, device="cpu"))
    assert next(sampler["score"].parameters()).dtype == torch.float64


@pytest.mark.parametrize("task_id", [
    "CORE-MEM-RETAINED-GRAPH-13R3",
    "CORE-REPEATED-BACKBONE-02R2",
])
def test_positive_tasks_expose_contrastive_activation_evidence(task_id: str) -> None:
    benchmark = _load_benchmark(ROOT / "benchmark" / "tasks" / task_id)
    assert callable(getattr(benchmark, "run_activation_evidence", None))


@pytest.mark.parametrize(
    "task_id",
    [
        "CORE-MEM-RETAINED-GRAPH-13R3",
        "CORE-AUTOGRAD-BATCHED-VJP-15R2",
        "CORE-CHECKPOINT-AMPLE-MEM-14R2",
        "CORE-AUTOGRAD-VJP-SMALL-25R2",
        "SCIML-CRYSTAL-DIFFUSION-07R2",
        "SCIML-CRYSTAL-STATIC-SAMPLING-29R2",
    ],
)
def test_rc7_oracle_materialization_matches_task_contract(task_id: str, tmp_path: Path) -> None:
    from benchmark.calibration.campaign import _copy_oracle

    task_dir = ROOT / "benchmark" / "tasks" / task_id
    solution_dir = tmp_path / task_id
    _copy_oracle(task_dir, solution_dir)
    benchmark = _load_benchmark(task_dir)
    solution = benchmark.load_solution(solution_dir / "solution.py", device="cpu")
    fixtures = benchmark.make_fixtures(0, device="cpu")
    result = benchmark.run_correctness(solution, fixtures)
    assert result["passed"], result


def test_equivariance_tempting_cache_patch_is_executable_and_rejected(tmp_path: Path) -> None:
    task_dir = ROOT / "benchmark" / "tasks" / "SCIML-EQUIV-RECOMPUTE-06R2"
    solution_dir = tmp_path / "solution"
    shutil.copytree(task_dir / "workspace", solution_dir)
    patch = task_dir / "oracle" / "tempting_patch.diff"
    # Apply the package patch to an isolated task-shaped copy.
    isolated_task = tmp_path / "task"
    shutil.copytree(task_dir, isolated_task)
    shutil.copytree(solution_dir, isolated_task / "workspace", dirs_exist_ok=True)
    applied = subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch)],
        cwd=isolated_task,
        text=True,
        capture_output=True,
        check=False,
    )
    assert applied.returncode == 0, applied.stderr or applied.stdout
    benchmark = _load_benchmark(task_dir)
    solution = benchmark.load_solution(isolated_task / "workspace" / "solution.py", device="cpu")
    fixtures = benchmark.make_fixtures(0, device="cpu")
    correctness = benchmark.run_correctness(solution, fixtures)
    gates = benchmark.run_scientific_gates(solution, fixtures)
    assert not correctness["passed"] or not all(
        bool(value[0] if isinstance(value, tuple) else value)
        for value in gates.values()
    ), (correctness, gates)
