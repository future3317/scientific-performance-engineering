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


__all__ = ["PromptCostModel"]
