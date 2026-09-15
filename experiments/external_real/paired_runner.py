from __future__ import annotations

from typing import Protocol, Sequence

from .models import ExternalTask, PairedOutcome


class NativeAdapter(Protocol):
    def run_pair(self, task: ExternalTask, skill: object, seed: int) -> PairedOutcome:
        ...


def run_paired(
    adapter: NativeAdapter,
    tasks: Sequence[ExternalTask],
    skill: object,
    repeat_seeds: Sequence[int],
) -> list[PairedOutcome]:
    outcomes: list[PairedOutcome] = []
    seen: set[tuple[str, int]] = set()
    for task in tasks:
        for seed in repeat_seeds:
            outcome = adapter.run_pair(task, skill, int(seed))
            key = (outcome.task_id, outcome.repeat_seed)
            if key in seen:
                raise ValueError(f"duplicate paired outcome: {key}")
            if outcome.task_id != task.task_id or outcome.repeat_seed != int(seed):
                raise ValueError(f"adapter returned mismatched pair for {task.task_id}/{seed}")
            seen.add(key)
            outcomes.append(outcome)
    return outcomes
