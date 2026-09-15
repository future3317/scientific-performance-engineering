from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


@dataclass(frozen=True)
class ExternalTask:
    task_id: str
    benchmark: str
    public_context: dict[str, object]
    native_payload: dict[str, object]


@dataclass(frozen=True)
class PairedOutcome:
    task_id: str
    repeat_seed: int
    baseline_score: float
    skill_score: float
    baseline_success: bool
    skill_success: bool
    native_metrics: dict[str, object]
    execution_valid: bool
    failure_reason: str | None


@dataclass(frozen=True)
class PolicyDecision:
    task_id: str
    policy_name: str
    deploy: bool
    reason: str
    decision_hash: str
    decision_timestamp: str

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        policy_name: str,
        deploy: bool,
        reason: str,
        decision_timestamp: str,
    ) -> "PolicyDecision":
        payload = {
            "task_id": task_id,
            "policy_name": policy_name,
            "deploy": bool(deploy),
            "reason": reason,
            "decision_timestamp": decision_timestamp,
        }
        decision_hash = hashlib.sha256(_canonical(payload)).hexdigest()
        return cls(decision_hash=decision_hash, **payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def content_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("decision_hash")
        return hashlib.sha256(_canonical(payload)).hexdigest()
