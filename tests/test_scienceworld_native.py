from __future__ import annotations

from dataclasses import dataclass
import sys
from types import SimpleNamespace

import pytest

from experiments.external_real.adapters.scienceworld_native import ScienceWorldAdapter
from experiments.external_real.models import ExternalTask
from experiments.external_real.run_scienceworld import ScienceWorldProviderBackend


@dataclass
class FakeScienceWorldBackend:
    calls: list[tuple[str, bool, int]]

    def execute(self, native_payload: dict[str, object], skill: object | None, seed: int) -> dict[str, object]:
        self.calls.append((str(native_payload["task_id"]), skill is not None, seed))
        return {
            "official_score": 0.8 if skill is not None else 0.4,
            "success": skill is not None,
            "execution_valid": True,
            "native_metrics": {"task_score": 0.8 if skill is not None else 0.4},
        }


def _adapter(backend: FakeScienceWorldBackend) -> ScienceWorldAdapter:
    return ScienceWorldAdapter(
        {
            "test": [
                {
                    "task_id": "sw-test-1",
                    "task_family": "navigation",
                    "goal_operation": "move",
                    "object_category": "plant",
                    "material_category": "organic",
                    "target_state_change": "located",
                    "requires_measurement": False,
                    "requires_container": False,
                    "variation_bucket": "v1",
                    "task_text": "Move the plant.",
                }
            ]
        },
        backend=backend,
        benchmark_revision="scienceworld-1.2.3-5d569f9",
        evaluator_revision="official-task-score",
    )


def test_public_context_contains_only_pre_execution_typed_fields() -> None:
    adapter = _adapter(FakeScienceWorldBackend([]))
    task = adapter.load_tasks("test")[0]

    assert task.public_context["task_family"] == "navigation"
    assert "task_text" not in task.public_context
    assert "gold_trajectory" not in task.public_context


def test_native_pair_uses_same_task_seed_and_official_scores() -> None:
    backend = FakeScienceWorldBackend([])
    adapter = _adapter(backend)
    task = adapter.load_tasks("test")[0]

    outcome = adapter.run_pair(task, object(), 7)

    assert backend.calls == [("sw-test-1", False, 7), ("sw-test-1", True, 7)]
    assert outcome.baseline_score == 0.4
    assert outcome.skill_score == 0.8
    assert outcome.baseline_success is False
    assert outcome.skill_success is True
    assert outcome.execution_valid is True


def test_forbidden_scoring_oracle_fields_are_rejected_from_public_task() -> None:
    backend = FakeScienceWorldBackend([])
    adapter = ScienceWorldAdapter(
        {"test": [{"task_id": "bad", "task_text": "x", "gold_trajectory": ["look"]}]},
        backend=backend,
        benchmark_revision="scienceworld-1.2.3-5d569f9",
        evaluator_revision="official-task-score",
    )

    with pytest.raises(ValueError, match="gold_trajectory"):
        adapter.load_tasks("test")


@pytest.mark.parametrize(
    ("completed", "score", "expected_success"),
    [
        (True, 1.0, False),
        (True, 100.0, True),
        (False, 100.0, False),
    ],
)
def test_provider_backend_uses_scienceworld_full_score_for_success(
    tmp_path, monkeypatch, completed: bool, score: float, expected_success: bool
) -> None:
    class FakeScienceWorldEnv:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def load(self, *_args, **_kwargs) -> None:
            pass

        def reset(self):
            return "initial observation", {}

        def get_possible_actions(self):
            return ["do the task"]

        def step(self, _action: str):
            return "final observation", 0.0, completed, {"score": score}

        def close(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "scienceworld", SimpleNamespace(ScienceWorldEnv=FakeScienceWorldEnv))
    monkeypatch.setattr(
        "benchmark.formal.host_agent._api_call_with_retry",
        lambda *_args, **_kwargs: {
            "choices": [{"message": {"content": '{"actions": ["do the task"]}'}}]
        },
    )

    backend = ScienceWorldProviderBackend(scienceworld_root=tmp_path, model_id="fake-model")
    result = backend.execute(
        {"task_name": "fake-task", "variation_idx": 0, "task_description": "Do the task."},
        None,
        seed=0,
    )

    assert result["official_score"] == score
    assert result["success"] is expected_success
    assert backend.provider_calls == 1
