from __future__ import annotations

import math
from typing import Any, Mapping, Protocol, Sequence

from ..context.scienceworld import PUBLIC_FIELDS, extract_public_context
from ..models import ExternalTask, PairedOutcome


class ScienceWorldBackend(Protocol):
    def execute(self, native_payload: dict[str, object], skill: object | None, seed: int) -> Mapping[str, object]:
        ...


class ScienceWorldAdapter:
    def __init__(
        self,
        task_manifest: Mapping[str, Sequence[Mapping[str, object]]],
        *,
        backend: ScienceWorldBackend,
        benchmark_revision: str,
        evaluator_revision: str,
    ) -> None:
        self._task_manifest = task_manifest
        self._backend = backend
        self._benchmark_revision = benchmark_revision
        self._evaluator_revision = evaluator_revision

    def protocol_manifest(self) -> dict[str, object]:
        return {
            "benchmark": "ScienceWorld",
            "benchmark_revision": self._benchmark_revision,
            "evaluator_revision": self._evaluator_revision,
            "public_context_fields": list(PUBLIC_FIELDS),
            "execution": "official simulator and official task score",
            "claim_boundary": "benchmark-native prospective fixed-skill evaluation",
        }

    def load_tasks(self, split: str) -> list[ExternalTask]:
        if split not in self._task_manifest:
            raise ValueError(f"missing ScienceWorld split: {split}")
        tasks: list[ExternalTask] = []
        seen: set[str] = set()
        for raw in self._task_manifest[split]:
            if not isinstance(raw, Mapping):
                raise ValueError("ScienceWorld task metadata must be an object")
            task_id = str(raw.get("task_id", ""))
            if not task_id:
                raise ValueError("ScienceWorld task lacks task_id")
            if task_id in seen:
                raise ValueError(f"duplicate ScienceWorld task_id: {task_id}")
            seen.add(task_id)
            public_context = extract_public_context(raw)
            tasks.append(
                ExternalTask(
                    task_id=task_id,
                    benchmark="scienceworld",
                    public_context=public_context,
                    native_payload=dict(raw),
                )
            )
        return tasks

    def public_context(self, task: ExternalTask) -> Mapping[str, object]:
        return dict(task.public_context)

    @staticmethod
    def native_score(raw_result: Mapping[str, object]) -> dict[str, object]:
        if "official_score" not in raw_result or "success" not in raw_result:
            raise ValueError("ScienceWorld backend must return official_score and success")
        score = float(raw_result["official_score"])
        if not math.isfinite(score):
            raise ValueError("ScienceWorld official score must be finite")
        execution_valid = bool(raw_result.get("execution_valid", False))
        native_metrics = raw_result.get("native_metrics", {})
        if not isinstance(native_metrics, Mapping):
            raise ValueError("ScienceWorld native_metrics must be an object")
        return {
            "official_score": score,
            "success": bool(raw_result["success"]),
            "execution_valid": execution_valid,
            "native_metrics": dict(native_metrics),
            "failure_reason": raw_result.get("failure_reason"),
        }

    def run_pair(self, task: ExternalTask, skill: object, seed: int) -> PairedOutcome:
        baseline_raw = self.native_score(self._backend.execute(task.native_payload, None, seed))
        skill_raw = self.native_score(self._backend.execute(task.native_payload, skill, seed))
        valid = bool(baseline_raw["execution_valid"]) and bool(skill_raw["execution_valid"])
        failures = [
            str(raw["failure_reason"])
            for raw in (baseline_raw, skill_raw)
            if raw.get("failure_reason")
        ]
        return PairedOutcome(
            task_id=task.task_id,
            repeat_seed=seed,
            baseline_score=float(baseline_raw["official_score"]),
            skill_score=float(skill_raw["official_score"]),
            baseline_success=bool(baseline_raw["success"]),
            skill_success=bool(skill_raw["success"]),
            native_metrics={"baseline": baseline_raw["native_metrics"], "skill": skill_raw["native_metrics"]},
            execution_valid=valid,
            failure_reason=";".join(failures) if failures else None,
        )
