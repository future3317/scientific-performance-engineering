"""Dependency-aware reuse of existing certificate clauses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


_CLAUSE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "A": ("context_fields", "model_id", "data_id", "objective_id"),
    "E": ("model_id", "data_id", "objective_id", "hardware", "software"),
    "S": ("scientific_policy", "objective_id"),
    "R": ("relation_partner_set", "model_id", "data_id", "objective_id", "hardware", "software"),
    "L": ("context_fields", "hardware", "software"),
}


@dataclass(frozen=True)
class CertificateDependencyResult:
    reusable: bool
    reused_clauses: tuple[str, ...]
    invalidated_clauses: tuple[str, ...]
    receipt: Mapping[str, Any]


def compare_certificate_dependencies(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    evidence_id: str,
) -> CertificateDependencyResult:
    """Compare only dependencies that can affect each certificate clause."""
    if not evidence_id:
        raise ValueError("evidence_id must be non-empty")
    changed = tuple(sorted({
        key for key in set(previous) | set(current)
        if previous.get(key) != current.get(key)
    }))
    invalidated = tuple(sorted(
        clause for clause, keys in _CLAUSE_DEPENDENCIES.items()
        if any(key in changed for key in keys)
    ))
    reused = tuple(clause for clause in ("A", "E", "S", "R", "L") if clause not in invalidated)
    receipt = {
        "type": "certificate_dependency_comparison",
        "source_evidence_id": evidence_id,
        "changed_dependencies": list(changed),
        "reused_clauses": list(reused),
        "invalidated_clauses": list(invalidated),
        "append_only": True,
    }
    return CertificateDependencyResult(not invalidated, reused, invalidated, receipt)


__all__ = ["CertificateDependencyResult", "compare_certificate_dependencies"]
