"""BoundaryBench evaluation only; synthesis remains owned by core.acre."""

from __future__ import annotations

from typing import Any

from core.predicates import match_predicate


def sealed_errors(predicate: dict[str, Any] | None, sealed_pool: list[Any]) -> int:
    if predicate is None:
        return len(sealed_pool)
    return sum(int(bool(match_predicate(predicate, item.context)) != item.expected_applicable) for item in sealed_pool)
