"""Auditable, monotonic decisions for one evolution episode."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import json
from pathlib import Path
import hashlib


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
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._decisions: dict[tuple[str, int, str], Decision] = {}
        self._sequence = 0
        self._prev_digest = ""
        if self.path and self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and (value.get("subject_id") or value.get("rule_id")):
                    subject = value.get("subject_id", value.get("rule_id"))
                    item = Decision(str(subject), int(value["version"]), str(value["replay_digest"]), str(value["status"]), value.get("utility"))
                    self._decisions[(item.rule_id, item.version, item.replay_digest)] = item
                    self._sequence = max(self._sequence, int(value.get("sequence", 0)))
                    self._prev_digest = str(value.get("record_digest", self._prev_digest))

    def record(self, rule_id: str, version: int, replay_digest: str, status: str, utility: float | None = None) -> Decision:
        key = (str(rule_id), int(version), str(replay_digest))
        if key not in self._decisions:
            if status != "candidate":
                raise ValueError("a new ledger key must start at candidate")
            self._decisions[key] = Decision(*key, status, utility)
            self._persist(self._decisions[key])
            return self._decisions[key]
        current = self._decisions[key]
        if status != current.status and status not in _TRANSITIONS[current.status]:
            raise ValueError(f"invalid ledger transition {current.status} -> {status}")
        current.status = status
        if utility is not None:
            current.utility = float(utility)
        self._persist(current)
        return current

    def _persist(self, decision: Decision) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence += 1
        record = {**asdict(decision), "subject_id": decision.rule_id, "sequence": self._sequence, "prev_digest": self._prev_digest, "policy_version": "evolution-ledger-v1"}
        record["record_digest"] = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        self._prev_digest = record["record_digest"]
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    def decisions(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self._decisions.values()]

    def has(self, rule_id: str, version: int) -> bool:
        return any(item.rule_id == str(rule_id) and item.version == int(version) for item in self._decisions.values())

    def has_replay(self, rule_id: str, version: int, replay_digest: str) -> bool:
        """Return whether this exact rule revision/replay has already advanced."""
        key = (str(rule_id), int(version), str(replay_digest))
        return key in self._decisions

    def count(self, *statuses: str) -> int:
        return sum(item.status in statuses for item in self._decisions.values())

    def precision(self) -> float | None:
        admitted = self.count("evaluated", "promoted", "revalidating", "retired", "rejected")
        survived = self.count("promoted", "revalidating")
        return survived / admitted if admitted else None


class CandidateEvidenceLedger:
    """Append-only evidence index for a candidate revision.

    Candidate cards are mutable workflow projections; this ledger is the
    durable record that prevents a later task from replacing earlier paired
    evidence.  The identity includes the replay context digest, so repeated
    submissions of the same context are idempotent while new tasks append.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @staticmethod
    def context_digest(case: dict[str, Any]) -> str:
        body = {key: case.get(key) for key in ("same_fixture_id", "context", "source_id", "independence_group")}
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()

    def append(self, subject_id: str, version: int, case: dict[str, Any], *, action_digest: str = "") -> dict[str, Any] | None:
        digest = self.context_digest(case)
        case_for_hash = {key: value for key, value in case.items() if key != "case_path"}
        case_bytes = json.dumps(case_for_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        record = {
            "subject_id": str(subject_id), "version": int(version),
            "replay_context_digest": digest, "case_id": case.get("case_id"),
            "case_sha256": hashlib.sha256(case_bytes).hexdigest(),
            "case_path": str(case.get("case_path", "")),
            "action_digest": str(action_digest),
        }
        existing: set[tuple[str, int, str]] = set()
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    existing.add((str(value.get("subject_id")), int(value.get("version", 0)), str(value.get("replay_context_digest"))))
        key = (record["subject_id"], record["version"], digest)
        if key in existing:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        return record

    def members(self, subject_id: str, version: int, *, action_digest: str | None = None) -> list[dict[str, Any]]:
        """Return immutable evidence memberships for a candidate revision."""
        if not self.path.is_file():
            return []
        values: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and str(value.get("subject_id")) == str(subject_id) and int(value.get("version", 0)) == int(version) and (action_digest is None or str(value.get("action_digest", "")) == str(action_digest)):
                values.append(value)
        return values
