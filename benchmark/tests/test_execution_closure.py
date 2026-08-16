from __future__ import annotations

from core.acre.engine import AcreEngine
from core.acre.experiments import ExperimentPlan
from core.acre.maintainer import AcreMaintainer
from core.acre.factorial import FactorialBlock
from core.acre.router import RequiredExperiment
from benchmark.formal.schedule import FamilyReplayExecutor, RelationExperimentScheduler


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
