"""Small, shared contracts for pre-outcome policy decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


OUTCOME_FIELDS = frozenset(
    {
        "score",
        "judge_score",
        "official_score",
        "baseline_score",
        "skill_score",
        "trajectory",
        "verifier_result",
        "hidden_score",
        "gold_actions",
        "future_outcome",
    }
)


@dataclass(frozen=True)
class DecisionRecord:
    run_id: str
    condition: str
    task_id: str
    task_index: int
    stage: str
    policy: str
    decision_id: str
    reuse: bool
    score: float | None
    uncertainty: float | None
    reason: str
    certificate_id: str | None
    skill_version: str | None
    public_context_digest: str
    available_evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DecisionRecord":
        required = {
            "run_id", "condition", "task_id", "task_index", "stage", "policy",
            "decision_id", "reuse", "reason", "public_context_digest",
            "available_evidence_ids",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"decision record missing fields: {missing}")
        validate_no_outcome_fields(value)
        return cls(
            run_id=str(value["run_id"]),
            condition=str(value["condition"]),
            task_id=str(value["task_id"]),
            task_index=int(value["task_index"]),
            stage=str(value["stage"]),
            policy=str(value["policy"]),
            decision_id=str(value["decision_id"]),
            reuse=bool(value["reuse"]),
            score=None if value.get("score") is None else float(value["score"]),
            uncertainty=None if value.get("uncertainty") is None else float(value["uncertainty"]),
            reason=str(value["reason"]),
            certificate_id=None if value.get("certificate_id") is None else str(value["certificate_id"]),
            skill_version=None if value.get("skill_version") is None else str(value["skill_version"]),
            public_context_digest=str(value["public_context_digest"]),
            available_evidence_ids=tuple(str(item) for item in value["available_evidence_ids"]),
        )


def validate_no_outcome_fields(record: Mapping[str, Any]) -> None:
    leaked = sorted(set(record) & OUTCOME_FIELDS - {"score"})
    if leaked:
        raise ValueError(f"decision record contains outcome fields: {leaked}")


def validate_public_context(context: Mapping[str, Any], scoring_only_fields: set[str] | frozenset[str]) -> None:
    leaked = sorted(set(context) & set(scoring_only_fields))
    if leaked:
        raise ValueError(f"public context contains scoring-only fields: {leaked}")


def validate_task_split(task_split: Mapping[str, list[str] | tuple[str, ...]]) -> None:
    names = ("synthesis", "calibration", "evaluation")
    seen: dict[str, str] = {}
    for name in names:
        if name not in task_split:
            raise ValueError(f"task split missing {name}")
        for task_id in task_split[name]:
            task_id = str(task_id)
            if task_id in seen:
                raise ValueError(f"task split overlap: {task_id} in {seen[task_id]} and {name}")
            seen[task_id] = name

