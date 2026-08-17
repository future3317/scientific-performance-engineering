"""One prompt-cost model for raw records and governed RuleViews."""

from __future__ import annotations

import json
import hashlib
from typing import Any


class PromptCostModel:
    def __init__(self, *, bytes_per_token: float = 4.0) -> None:
        if bytes_per_token <= 0:
            raise ValueError("bytes_per_token must be positive")
        self.bytes_per_token = float(bytes_per_token)

    def cost(self, value: Any) -> int:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return max(1, int((len(payload) + self.bytes_per_token - 1) // self.bytes_per_token))


class BudgetedContextRenderer:
    """Render the complete worker payload, then trim whole entries once."""

    VERSION = "budgeted-context-renderer-2"

    def __init__(self, budget: int, *, cost_model: PromptCostModel | None = None) -> None:
        if budget < 1:
            raise ValueError("context budget must be positive")
        self.budget = int(budget)
        self.cost_model = cost_model or PromptCostModel()

    def render(
        self,
        payload: dict[str, Any],
        *,
        entries_key: str = "rule_views",
        entries_keys: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Return the exact worker-visible payload within ``self.budget``.

        Dropped entries are represented only by compact audit metadata.  Their
        payload is never copied into the worker-visible result, and every list
        named in ``entries_keys`` participates in the same final cap.
        """
        value = dict(payload)
        keys = tuple(entries_keys or (entries_key,))
        entries = {
            key: list(value.get(key, [])) if isinstance(value.get(key), list) else []
            for key in keys
        }
        dropped: list[tuple[str, Any]] = []
        for key, items in entries.items():
            value[key] = items
        while self.cost_model.cost(value) > self.budget:
            key = next((name for name in keys if entries[name]), None)
            if key is None:
                break
            item = entries[key].pop()
            dropped.append((key, item))
            value[key] = entries[key]
        dropped_ids: list[str] = []
        dropped_digests: list[str] = []
        for key, item in dropped:
            if isinstance(item, dict):
                identifier = item.get("id") or item.get("rule_id") or item.get("context_id") or item.get("candidate_id")
                if identifier is not None:
                    dropped_ids.append(f"{key}:{identifier}")
            dropped_digests.append(hashlib.sha256(
                json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
            ).hexdigest())
        value["renderer"] = {
            "version": self.VERSION,
            "budget": self.budget,
            "dropped_count": len(dropped),
            "dropped_ids": dropped_ids,
            "dropped_digests": dropped_digests,
        }
        value["token_cost"] = self.cost_model.cost(value)
        if value["token_cost"] > self.budget:
            raise ValueError("rendered context exceeds token budget after entry trimming")
        return value


__all__ = ["PromptCostModel", "BudgetedContextRenderer"]
