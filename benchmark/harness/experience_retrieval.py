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
