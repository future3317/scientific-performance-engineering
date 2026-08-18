from __future__ import annotations

from pathlib import Path

from types import SimpleNamespace

from core.acre.engine import AcreEngine
from core.acre.experiments import ExperimentPlan
from core.acre.maintainer import AcreMaintainer, MaintenanceInput
from core.acre.factorial import FactorialBlock
from core.acre.router import RequiredExperiment
from benchmark.formal.schedule import FamilyReplayExecutor, RelationExperimentScheduler
from benchmark.harness import runner


def test_core_paired_plan_executes_family_replay_and_records_measurements() -> None:
    maintainer = AcreMaintainer(AcreEngine())
    executor = FamilyReplayExecutor("compile", "reuse_compile_cache", repetitions=8)
    recorded: list[dict] = []

    result = maintainer.execute_node_experiment(
        ExperimentPlan(
            subject_id="RULE-CLOSURE",
            contexts=({
                "context_id": "compile-closure",
                "independence_group": "closure-group",
                "context": {"workload": {"logical_steps": 256, "graph_size": 128, "dynamic_shape_rate": 0.2}},
            },),
            max_groups=1,
        ),
        executor,
        record_case=recorded.append,
        update_certificate=lambda cases: {"status": "collecting", "n": len(cases)},
    )

    assert result.groups_executed == 1
    assert result.stop_reason == "plan_exhausted"
    assert recorded[0]["paired_replay"] is True
    assert len(recorded[0]["intervention_measurements"]) == 8
    assert len(recorded[0]["baseline_measurements"]) == 8


def test_maintainer_serializes_evidence_assessment() -> None:
    transition = AcreMaintainer(AcreEngine()).run(MaintenanceInput())
    assert transition.assessment["representative_count"] == 0
    assert transition.assessment["adversarial_count"] == 0


def test_relation_scheduler_executes_factorial_blocks_through_core() -> None:
    maintainer = AcreMaintainer(AcreEngine())
    scheduler = RelationExperimentScheduler()

    def blocks(_context: dict, *, context_id: str) -> list[FactorialBlock]:
        return [
            FactorialBlock(
                f"{context_id}-{index}",
                {"00": 0.10, "10": 0.25, "01": 0.25, "11": 0.80},
            )
            for index in range(8)
        ]

    result = scheduler.execute(
        {"relation_id": "REL-CLOSURE"},
        "compile",
        block_executor=blocks,
        maintainer=maintainer,
    )

    assert result["status"] == "executed"
    assert result["identification"]["context_decisions"]


def test_required_experiment_is_typed_and_serializable() -> None:
    experiment = RequiredExperiment(
        "three-way:R1:R2:R3",
        "three_way_factorial",
        ("R1", "R2", "R3"),
        ("000", "001", "010", "011", "100", "101", "110", "111"),
        "higher-order certificate is required",
    )
    payload = experiment.to_dict()
    assert payload["experiment_type"] == "three_way_factorial"
    assert payload["required_arms"][-1] == "111"


def test_h2d_fixture_reuse_keeps_identity_and_arm_isolation(tmp_path) -> None:
    def make_fixtures(seed: int, device: str = "cpu") -> dict:
        return {"mutable": [seed], "device": device}

    def clone_fixtures(fixtures: dict) -> dict:
        return {"mutable": list(fixtures["mutable"]), "device": fixtures["device"]}

    def load_solution(path: str, device: str = "cpu") -> str:
        return "candidate" if "candidate" in path else "baseline"

    def run_performance(
        solution: str,
        fixtures: dict,
        warmup: int,
        iterations: int,
        device: str = "cpu",
    ) -> dict:
        fixtures["mutable"][0] += 1
        return {
            "value": 1.0 if solution == "baseline" else 0.5,
            "work_units": {"steps": iterations},
            "output_checksums": {},
        }

    module = SimpleNamespace(
        make_fixtures=make_fixtures,
        clone_fixtures=clone_fixtures,
        load_solution=load_solution,
        run_performance=run_performance,
    )
    baseline = tmp_path / "baseline.py"
    candidate = tmp_path / "candidate.py"
    record = runner.run_paired_measurement(
        module,
        baseline,
        candidate,
        {"warmup_iterations": 1, "measured_iterations": 2, "repetitions": 2},
        reuse_fixture_per_repetition=True,
    )

    assert record["fixture_hashes"]["baseline:0"] == record["fixture_hashes"]["candidate:0"]
    assert record["fixture_hashes"]["baseline:1"] == record["fixture_hashes"]["candidate:1"]
    assert record["fixture_build_time_s"] >= 0.0
    assert record["fixture_hash_time_s"] >= 0.0


def test_s5_exception_is_protocol_invalid(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace
    from benchmark.harness import verifier

    task_dir = Path(__file__).resolve().parents[1] / "tasks" / "CORE-COMPILE-TINY-12"
    solution_dir = tmp_path / "solution"
    solution_dir.mkdir()
    (solution_dir / "solution.py").write_text("x = 1\n", encoding="utf-8")
    fake_module = SimpleNamespace(make_fixtures=lambda **kwargs: {}, load_solution=lambda **kwargs: {}, run_scientific_gates=lambda **kwargs: {})
    monkeypatch.setattr(verifier.runner, "select_device", lambda requires_cuda: ("cpu", True))
    monkeypatch.setattr(verifier.runner, "import_module_by_path", lambda path: fake_module)
    monkeypatch.setattr(verifier, "_fresh_input_correctness", lambda *args, **kwargs: {"passed": True, "per_input": [], "output_checksums": {}})
    monkeypatch.setattr(verifier.runner, "call_benchmark_fn", lambda *args, **kwargs: {"gate": {"passed": True, "details": {}}})
    def raise_s5(*args, **kwargs):
        raise RuntimeError("synthetic S5 infrastructure failure")
    monkeypatch.setattr(verifier.runner, "run_paired_measurement", raise_s5)

    result = verifier.verify_task(task_dir, solution_dir, out_path=tmp_path / "result.json")

    assert result.get("protocol_failure") is True, result.get("errors")
    assert result["validity"] == "invalid"
    assert result["verdict"] == "error"
