"""Raw-experience retrieval for the C control condition.

This module deliberately returns source records only; it never constructs a
RuleSpec or invokes replay/governance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def retrieve_raw_experiences(store: str | Path, query: str = "", token_budget: int = 4096) -> list[dict[str, Any]]:
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    records: list[dict[str, Any]] = []
    used = 0
    for path in sorted((Path(store) / "experience" / "inbox").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        text = json.dumps(record, sort_keys=True, ensure_ascii=False)
        if query and query.lower() not in text.lower():
            continue
        cost = max(1, (len(text) + 3) // 4)
        if used + cost > token_budget:
            break
        records.append(record)
        used += cost
    return records


class RawExperienceRetriever:
    """Matched-budget C adapter; returns actions, never typed rule objects."""

    def __init__(self, store: str | Path, *, token_budget: int = 4096) -> None:
        self.store = Path(store)
        self.token_budget = token_budget

    def retrieve(self, query: str = "") -> list[dict[str, Any]]:
        return retrieve_raw_experiences(self.store, query=query, token_budget=self.token_budget)

    def propose_interventions(self, query: str = "") -> list[str]:
        actions: list[str] = []
        for record in self.retrieve(query):
            if record.get("record_type") != "causal_evidence":
                lesson = record.get("lesson")
                proposed = lesson.get("proposed_interventions", []) if isinstance(lesson, dict) else []
                if isinstance(proposed, list):
                    actions.extend(str(item) for item in proposed if item)
                continue
            intervention = record.get("assignment", {}).get("interventions", {})
            if isinstance(intervention, dict):
                actions.extend(str(action) for action, value in intervention.items() if value == 1)
        return list(dict.fromkeys(actions))
