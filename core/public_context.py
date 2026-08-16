"""Construction of the common public context visible to every condition."""

from __future__ import annotations

from typing import Any, Mapping


def build_public_context(
    workload: Mapping[str, Any],
    *,
    hardware: Mapping[str, Any] | None = None,
    software: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    domain: str = "scientific-performance",
) -> dict[str, Any]:
    return {
        "domain": domain,
        "workload": dict(workload),
        "hardware": dict(hardware or {}),
        "software": dict(software or {}),
        "evidence": dict(evidence or {}),
    }


__all__ = ["build_public_context"]
