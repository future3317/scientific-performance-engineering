"""Small typed predicate DSL for rule applicability."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _lookup(context: Mapping[str, Any], path: str) -> Any:
    if "." not in path and isinstance(context.get(path), Mapping) and path in context[path]:
        # Preserve the legacy ``TaskContext(workload={"workload": ...})``
        # spelling while the canonical root remains the full TaskContext.
        return context[path][path]
    value: Any = context
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def match_predicate(predicate: Any, context: Mapping[str, Any]) -> bool:
    """Evaluate all/any/not, comparisons, version, and exact predicates."""
    if predicate is None:
        return True
    if isinstance(predicate, str):
        return bool(_lookup(context, predicate))
    if isinstance(predicate, list):
        return all(match_predicate(item, context) for item in predicate)
    if not isinstance(predicate, Mapping):
        return False
    if "all" in predicate and not all(match_predicate(item, context) for item in predicate["all"]):
        return False
    if "any" in predicate and not any(match_predicate(item, context) for item in predicate["any"]):
        return False
    if "not" in predicate and match_predicate(predicate["not"], context):
        return False
    for path, condition in predicate.get("equals", {}).items():
        if _lookup(context, path) != condition:
            return False
    for path, condition in predicate.get("compare", {}).items():
        actual = _lookup(context, path)
        if actual is None or not isinstance(condition, Mapping):
            return False
        for operator, expected in condition.items():
            try:
                if operator == "lt" and not actual < expected: return False
                if operator == "lte" and not actual <= expected: return False
                if operator == "gt" and not actual > expected: return False
                if operator == "gte" and not actual >= expected: return False
                if operator == "in" and actual not in expected: return False
            except TypeError:
                return False
    for path, expected in predicate.get("version", {}).items():
        if _lookup(context, path) != expected:
            return False
    # A bare mapping is an exact field/path predicate.
    reserved = {"all", "any", "not", "equals", "compare", "version"}
    for path, expected in predicate.items():
        if path not in reserved and _lookup(context, path) != expected:
            return False
    return True
