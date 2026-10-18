"""Decision-before-outcome ledger for paired external treatment runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import DecisionRecord, validate_no_outcome_fields


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


class DecisionLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._rows: list[dict[str, Any]] = []
        self._decisions: dict[str, DecisionRecord] = {}
        self._outcomes: set[str] = set()
        self._prev_digest = ""
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                body = {key: item for key, item in value.items() if key != "record_digest"}
                expected = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
                if value.get("record_digest") != expected or value.get("prev_digest", "") != self._prev_digest:
                    raise ValueError("public decision ledger digest chain is invalid")
                self._rows.append(value)
                self._prev_digest = str(value["record_digest"])
                if value.get("event") == "decision":
                    decision = DecisionRecord.from_mapping(value["decision"])
                    self._decisions[decision.decision_id] = decision
                elif value.get("event") == "outcome":
                    self._outcomes.add(str(value["decision_id"]))

    def append(self, decision: DecisionRecord) -> None:
        if decision.decision_id in self._decisions:
            if self._decisions[decision.decision_id] != decision:
                raise ValueError(f"decision is immutable: {decision.decision_id}")
            return
        validate_no_outcome_fields(decision.to_dict())
        self._append({"event": "decision", "decision": decision.to_dict()})
        self._decisions[decision.decision_id] = decision

    def append_outcome(self, decision_id: str, outcome: dict[str, Any]) -> None:
        decision_id = str(decision_id)
        if decision_id not in self._decisions:
            raise ValueError("outcome requires an existing decision")
        if decision_id in self._outcomes:
            raise ValueError(f"outcome already exists: {decision_id}")
        if int(outcome.get("task_index", self._decisions[decision_id].task_index)) > self._decisions[decision_id].task_index:
            raise ValueError("future outcome cannot be appended to a decision")
        if "future_outcome" in outcome:
            raise ValueError("future outcome cannot be appended")
        self._append({"event": "outcome", "decision_id": decision_id, "outcome": dict(outcome)})
        self._outcomes.add(decision_id)

    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def _append(self, body: dict[str, Any]) -> None:
        record = {**body, "sequence": len(self._rows) + 1, "prev_digest": self._prev_digest}
        record["record_digest"] = hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        self._rows.append(record)
        self._prev_digest = record["record_digest"]

