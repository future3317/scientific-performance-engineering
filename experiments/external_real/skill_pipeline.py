from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any, Callable, Mapping, Sequence

from .models import PairedOutcome, WITHHOLD_SKILL


def _canonical(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


@dataclass(frozen=True)
class FrozenSkill:
    benchmark: str
    generation_seed: int
    body: str
    scripts: tuple[str, ...]
    resources: tuple[str, ...]
    semantic_abstract: str
    skill_hash: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationReport:
    skill_hash: str
    outcome: str
    pair_count: int
    execution_valid_count: int
    skill_success_count: int
    failure_reasons: tuple[str, ...]

    @property
    def admitted(self) -> bool:
        return self.outcome == "ADMITTED"


def freeze_skill(
    body: str,
    scripts: Sequence[str],
    resources: Sequence[str],
    semantic_abstract: str,
    benchmark: str,
    generation_seed: int,
) -> FrozenSkill:
    payload = {
        "benchmark": benchmark,
        "body": body,
        "generation_seed": int(generation_seed),
        "resources": list(resources),
        "scripts": list(scripts),
        "semantic_abstract": semantic_abstract,
    }
    return FrozenSkill(
        benchmark=benchmark,
        generation_seed=int(generation_seed),
        body=body,
        scripts=tuple(scripts),
        resources=tuple(resources),
        semantic_abstract=semantic_abstract,
        skill_hash=hashlib.sha256(_canonical(payload)).hexdigest(),
    )


def verify_skill(
    adapter: Any,
    skill: FrozenSkill,
    verification_tasks: Sequence[Any],
    *,
    repeat_seed: int = 0,
) -> VerificationReport:
    """Admit only a skill that passes the native verification gate."""
    outcomes: list[PairedOutcome] = [
        adapter.run_pair(task, skill, repeat_seed) for task in verification_tasks
    ]
    valid = [outcome for outcome in outcomes if outcome.execution_valid]
    successful = [outcome for outcome in valid if outcome.skill_success]
    reasons = tuple(sorted({
        str(outcome.failure_reason)
        for outcome in outcomes
        if outcome.failure_reason
    }))
    admitted = bool(outcomes) and len(valid) == len(outcomes) and len(successful) == len(outcomes)
    return VerificationReport(
        skill_hash=skill.skill_hash,
        outcome="ADMITTED" if admitted else WITHHOLD_SKILL,
        pair_count=len(outcomes),
        execution_valid_count=len(valid),
        skill_success_count=len(successful),
        failure_reasons=reasons,
    )
