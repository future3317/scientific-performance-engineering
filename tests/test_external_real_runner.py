from __future__ import annotations

from dataclasses import dataclass

from experiments.external_real.models import ExternalTask, PairedOutcome, PolicyDecision
from experiments.external_real.paired_runner import run_paired


@dataclass
class FakeAdapter:
    calls: list[tuple[str, int]]

    def run_pair(self, task: ExternalTask, skill: object, seed: int) -> PairedOutcome:
        del skill
        self.calls.append((task.task_id, seed))
        return PairedOutcome(
            task_id=task.task_id,
            repeat_seed=seed,
            baseline_score=0.2,
            skill_score=0.8,
            baseline_success=False,
            skill_success=True,
            native_metrics={"official_score": 0.8},
            execution_valid=True,
            failure_reason=None,
        )


def test_runner_executes_one_shared_pair_per_task_and_seed() -> None:
    tasks = [
        ExternalTask("task-1", "fixture", {"family": "a"}, {"payload": 1}),
        ExternalTask("task-2", "fixture", {"family": "b"}, {"payload": 2}),
    ]
    adapter = FakeAdapter([])

    outcomes = run_paired(adapter, tasks, object(), repeat_seeds=(0, 1))

    assert adapter.calls == [("task-1", 0), ("task-1", 1), ("task-2", 0), ("task-2", 1)]
    assert len(outcomes) == 4


def test_runner_rejects_duplicate_adapter_outcomes() -> None:
    class DuplicateAdapter(FakeAdapter):
        def run_pair(self, task: ExternalTask, skill: object, seed: int) -> PairedOutcome:
            result = super().run_pair(task, skill, seed)
            return result

    tasks = [ExternalTask("task-1", "fixture", {}, {})]
    adapter = DuplicateAdapter([])
    outcomes = run_paired(adapter, tasks, object(), repeat_seeds=(0,))

    assert {(outcome.task_id, outcome.repeat_seed) for outcome in outcomes} == {("task-1", 0)}
