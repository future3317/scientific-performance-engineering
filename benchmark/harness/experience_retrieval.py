"""Raw-experience retrieval for the C control condition.

This module deliberately returns source records only; it never constructs a
RuleSpec or invokes replay/governance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value} if prefix else {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten(item, path))
    return result


def _context_distance(record: Mapping[str, Any], query: Mapping[str, Any]) -> float:
    stored = record.get("public_context") if isinstance(record.get("public_context"), Mapping) else record.get("context")
    if not isinstance(stored, Mapping):
        return float("inf")
    left, right = _flatten(stored), _flatten(query)
    shared = set(left) | set(right)
    if not shared:
        return float("inf")
    distance = 0.0
    for key in shared:
        if key not in left or key not in right:
            distance += 1.0
        elif isinstance(left[key], (int, float)) and isinstance(right[key], (int, float)) and not isinstance(left[key], bool) and not isinstance(right[key], bool):
            distance += abs(float(left[key]) - float(right[key])) / max(1.0, abs(float(left[key])), abs(float(right[key])))
        else:
            distance += 0.0 if left[key] == right[key] else 1.0
    return distance / len(shared)


def retrieve_raw_experiences(store: str | Path, query: str | Mapping[str, Any] = "", token_budget: int = 4096) -> list[dict[str, Any]]:
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    records: list[tuple[float, str, dict[str, Any], int]] = []
    for path in sorted((Path(store) / "experience" / "inbox").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        text = json.dumps(record, sort_keys=True, ensure_ascii=False)
        distance = _context_distance(record, query) if isinstance(query, Mapping) else 0.0
        if isinstance(query, str) and query and query.lower() not in text.lower():
            continue
        cost = max(1, (len(text) + 3) // 4)
        records.append((distance, str(path), record, cost))
    records.sort(key=lambda item: (item[0], item[1]))
    selected: list[dict[str, Any]] = []
    used = 0
    for _, _, record, cost in records:
        if used + cost > token_budget:
            continue
        selected.append(record)
        used += cost
    return selected


class RawExperienceRetriever:
    """Matched-budget C adapter; returns actions, never typed rule objects."""

    def __init__(self, store: str | Path, *, token_budget: int = 4096) -> None:
        self.store = Path(store)
        self.token_budget = token_budget

    def retrieve(self, query: str | Mapping[str, Any] = "") -> list[dict[str, Any]]:
        return retrieve_raw_experiences(self.store, query=query, token_budget=self.token_budget)

    def propose_interventions(self, query: str | Mapping[str, Any] = "") -> list[str]:
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
