from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[2]


NEGATIVE_ANCHORS = (
    "CORE-AUTOGRAD-VJP-SMALL-25R2",
    "CORE-CHECKPOINT-AMPLE-MEM-14R2",
    "CORE-REPEATED-BACKBONE-LOW-REUSE-22",
    "CORE-SCALAR-SYNC-LOW-CADENCE-21",
    "SCIML-CRYSTAL-HIGH-GUIDANCE-28",
    "SCIML-GRAPH-REBUILD-08R2",
    "SCIML-EQUIV-RECOMPUTE-06R2",
)


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
    "CORE-CHECKPOINT-HIGH-PRESSURE-24R2",
    "CORE-REPEATED-BACKBONE-02R2",
])
def test_positive_tasks_expose_contrastive_activation_evidence(task_id: str) -> None:
    benchmark = _load_benchmark(ROOT / "benchmark" / "tasks" / task_id)
    assert callable(getattr(benchmark, "run_activation_evidence", None))


@pytest.mark.parametrize(
    ("task_id", "measured_iterations", "noise_floor_percent"),
    [
        ("CORE-CHECKPOINT-HIGH-PRESSURE-24R2", 240, 3.0),
        ("CORE-REPEATED-BACKBONE-02R2", 800, 2.0),
    ],
)
def test_positive_task_measurement_contract_preserves_gate_and_horizon(
    task_id: str, measured_iterations: int, noise_floor_percent: float
) -> None:
    from benchmark.harness.miniyaml import load

    spec = load(str(ROOT / "benchmark" / "tasks" / task_id / "task.yaml"))
    measurement = spec["measurement"]
    assert measurement["repetitions"] == 5
    assert measurement["min_improvement_percent"] == 5.0
    assert measurement["noise_floor_percent"] == noise_floor_percent
    assert measurement["measured_iterations"] == measured_iterations


@pytest.mark.parametrize(
    "task_id",
    [
        "CORE-CHECKPOINT-HIGH-PRESSURE-24R2",
        "CORE-REPEATED-BACKBONE-02R2",
    ],
)
def test_positive_task_activation_evidence_is_contrastive(task_id: str) -> None:
    task_dir = ROOT / "benchmark" / "tasks" / task_id
    benchmark = _load_benchmark(task_dir)
    baseline = benchmark.load_solution(task_dir / "workspace" / "solution.py")
    oracle = benchmark.load_solution(task_dir / "oracle" / "solution_oracle.py")
    evidence = benchmark.run_activation_evidence(oracle, baseline, benchmark.make_fixtures(0))

    assert set(evidence) == {"candidate_metrics", "baseline_metrics"}
    if task_id == "CORE-CHECKPOINT-HIGH-PRESSURE-24R2":
        assert evidence["candidate_metrics"]["checkpoint_recompute_calls"] > 0
        assert evidence["baseline_metrics"]["checkpoint_recompute_calls"] == 0
    else:
        assert evidence["candidate_metrics"]["backbone_calls"] == 1
        assert evidence["baseline_metrics"]["backbone_calls"] == 4
        assert evidence["candidate_metrics"]["fixed_projection_calls"] == 1
        assert evidence["baseline_metrics"]["fixed_projection_calls"] == 4


@pytest.mark.parametrize(
    "task_id",
    [
        "CORE-MEM-RETAINED-GRAPH-13R3",
        "CORE-AUTOGRAD-BATCHED-VJP-15R2",
        "CORE-CHECKPOINT-AMPLE-MEM-14R2",
        "CORE-AUTOGRAD-VJP-SMALL-25R2",
        "SCIML-CRYSTAL-DIFFUSION-07R2",
        "SCIML-CRYSTAL-STATIC-SAMPLING-29R2",
        "CORE-H2D-PIPELINE-INTERIOR-32",
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


@pytest.mark.parametrize("task_id", NEGATIVE_ANCHORS)
def test_negative_anchor_tempting_intervention_applies_cleanly(tmp_path: Path, task_id: str) -> None:
    task_dir = ROOT / "benchmark" / "tasks" / task_id
    patch = task_dir / "oracle" / "tempting_patch.diff"
    assert patch.is_file(), f"{task_id} must ship an executable tempting intervention"

    isolated_task = tmp_path / task_id
    shutil.copytree(task_dir, isolated_task)
    original = isolated_task / "workspace" / "solution.py"
    before = original.read_text(encoding="utf-8")
    patch_text = patch.read_text(encoding="utf-8")
    patch_root = isolated_task if "a/workspace/" in patch_text else isolated_task / "workspace"
    applied = subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch)],
        cwd=patch_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert applied.returncode == 0, applied.stderr or applied.stdout
    assert original.read_text(encoding="utf-8") != before


def test_scalar_sync_abstention_keeps_baseline_and_oracle_identical() -> None:
    task_dir = ROOT / "benchmark" / "tasks" / "CORE-SCALAR-SYNC-LOW-CADENCE-21"
    baseline = (task_dir / "workspace" / "solution.py").read_text(encoding="utf-8")
    oracle = (task_dir / "oracle" / "solution_oracle.py").read_text(encoding="utf-8")
    assert baseline == oracle
    assert "no source mutation" in (task_dir / "oracle" / "reference_patch.diff").read_text(encoding="utf-8")


def test_h2d_interior_anchor_matches_declared_generator_point() -> None:
    from benchmark.harness.miniyaml import load

    task_dir = ROOT / "benchmark" / "tasks" / "CORE-H2D-PIPELINE-INTERIOR-32"
    spec = load(str(task_dir / "task.yaml"))
    assert spec["family_id"] == "h2d_pipeline"
    assert spec["mechanism"] == "h2d_blocking"
    assert spec["kind"] == "positive"
    assert spec["family_parameters"] == {
        "batch_size": 112,
        "worker_count": 4,
        "prefetch_factor": 4,
        "pin_memory": True,
    }
    assert spec["lineage"]["mutation_template_id"] == "MT-H2D-INTERIOR-32-V1"
    assert spec["oracle_fix_pattern_id"] == "FIX-H2D-INTERIOR-32-ASYNC-TRANSFER"


def test_ample_checkpoint_abstention_keeps_checkpointing_disabled() -> None:
    task_dir = ROOT / "benchmark" / "tasks" / "CORE-CHECKPOINT-AMPLE-MEM-14R2"
    benchmark = _load_benchmark(task_dir)
    baseline = benchmark.load_solution(task_dir / "workspace" / "solution.py")
    oracle = benchmark.load_solution(task_dir / "oracle" / "solution_oracle.py")
    fixtures = benchmark.make_fixtures(0)

    def checkpoint_calls(solution):
        checkpoint_fn = getattr(solution, "checkpoint", None)
        if checkpoint_fn is None:
            return 0
        calls = {"count": 0}

        def counted_checkpoint(*args, **kwargs):
            calls["count"] += 1
            return checkpoint_fn(*args, **kwargs)

        solution.checkpoint = counted_checkpoint
        try:
            model = solution.build_model(fixtures)
            optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["optimizer_config"]["lr"])
            solution.train_step(model, (fixtures["inputs"][:16], fixtures["targets"][:16]), optimizer)
        finally:
            solution.checkpoint = checkpoint_fn
        return calls["count"]

    assert checkpoint_calls(baseline) == 0
    assert checkpoint_calls(oracle) == 0
