"""One prompt-cost model for raw records and governed RuleViews."""

from __future__ import annotations

import json
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

    VERSION = "budgeted-context-renderer-1"

    def __init__(self, budget: int, *, cost_model: PromptCostModel | None = None) -> None:
        if budget < 1:
            raise ValueError("context budget must be positive")
        self.budget = int(budget)
        self.cost_model = cost_model or PromptCostModel()

    def render(self, payload: dict[str, Any], *, entries_key: str = "rule_views") -> dict[str, Any]:
        value = dict(payload)
        entries = list(value.get(entries_key, [])) if isinstance(value.get(entries_key), list) else []
        dropped: list[Any] = []
        while self.cost_model.cost(value) > self.budget and entries:
            dropped.insert(0, entries.pop())
            value[entries_key] = entries
        value["renderer"] = {"version": self.VERSION, "budget": self.budget, "dropped_entries": dropped}
        value["token_cost"] = self.cost_model.cost(value)
        if value["token_cost"] > self.budget:
            raise ValueError("rendered context exceeds token budget after entry trimming")
        return value


__all__ = ["PromptCostModel", "BudgetedContextRenderer"]
