"""Auditable, monotonic decisions for one evolution episode."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


_TRANSITIONS = {
    "candidate": {"evaluated"},
    "evaluated": {"promoted", "rejected"},
    "promoted": {"revalidating", "retired"},
    "revalidating": {"promoted", "retired"},
    "rejected": set(),
    "retired": set(),
}


@dataclass
class Decision:
    rule_id: str
    version: int
    replay_digest: str
    status: str
    utility: float | None = None


class EvolutionDecisionLedger:
    def __init__(self) -> None:
        self._decisions: dict[tuple[str, int, str], Decision] = {}

    def record(self, rule_id: str, version: int, replay_digest: str, status: str, utility: float | None = None) -> Decision:
        key = (str(rule_id), int(version), str(replay_digest))
        if key not in self._decisions:
            if status != "candidate":
                raise ValueError("a new ledger key must start at candidate")
            self._decisions[key] = Decision(*key, status, utility)
            return self._decisions[key]
        current = self._decisions[key]
        if status != current.status and status not in _TRANSITIONS[current.status]:
            raise ValueError(f"invalid ledger transition {current.status} -> {status}")
        current.status = status
        if utility is not None:
            current.utility = float(utility)
        return current

    def decisions(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self._decisions.values()]

    def has(self, rule_id: str, version: int) -> bool:
        return any(item.rule_id == str(rule_id) and item.version == int(version) for item in self._decisions.values())

    def count(self, *statuses: str) -> int:
        return sum(item.status in statuses for item in self._decisions.values())

    def precision(self) -> float | None:
        admitted = self.count("evaluated", "promoted", "revalidating", "retired", "rejected")
        survived = self.count("promoted", "revalidating")
        return survived / admitted if admitted else None
