"""Construction of the common public context visible to every condition."""

from __future__ import annotations

from typing import Any, Mapping


def build_public_context(
    value: Mapping[str, Any] | None = None,
    *,
    workload: Mapping[str, Any] | None = None,
    hardware: Mapping[str, Any] | None = None,
    software: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    """Build one public context without manufacturing hidden/default fields."""
    source = dict(value or {})
    if workload is not None:
        source["workload"] = dict(workload)
    elif "workload" not in source:
        family_parameters = source.get("family_parameters")
        source["workload"] = dict(family_parameters) if isinstance(family_parameters, Mapping) else dict(value or {})
    elif isinstance(source.get("workload"), Mapping) and isinstance(source["workload"].get("family_parameters"), Mapping):
        source["workload"] = dict(source["workload"]["family_parameters"])
    source.pop("family_parameters", None)
    for key, replacement in (("hardware", hardware), ("software", software), ("evidence", evidence)):
        if replacement is not None:
            source[key] = dict(replacement)
    if domain is not None:
        source["domain"] = domain
    result = {key: dict(source[key]) if isinstance(source.get(key), Mapping) else source[key] for key in ("domain", "workload", "hardware", "software", "evidence") if key in source}
    return result


__all__ = ["build_public_context"]
