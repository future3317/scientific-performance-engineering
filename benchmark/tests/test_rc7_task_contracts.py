from __future__ import annotations

import importlib.util
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


@pytest.mark.parametrize(
    "task_id",
    [
        "CORE-MEM-RETAINED-GRAPH-13R2",
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
