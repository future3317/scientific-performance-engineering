from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .models import PolicyDecision


def _parse_timestamp(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def assert_decision_before_scoring(decision_timestamp: str, scoring_join_timestamp: str) -> None:
    if _parse_timestamp(decision_timestamp) >= _parse_timestamp(scoring_join_timestamp):
        raise ValueError("policy decision must be before scoring join")


def freeze_decisions(path: Path, decisions: Sequence[PolicyDecision]) -> str:
    ordered = sorted(decisions, key=lambda decision: (decision.task_id, decision.policy_name))
    keys = [(decision.task_id, decision.policy_name) for decision in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate task/policy decision")
    for decision in ordered:
        if decision.content_hash() != decision.decision_hash:
            raise ValueError(f"decision hash mismatch for {decision.task_id}/{decision.policy_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for decision in ordered:
            handle.write(json.dumps(decision.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()
