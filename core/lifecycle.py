"""State reducer for governed rule and relation lifecycle decisions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .governance import EvolutionDecision
from .models import RelationState, RuleState


def apply_lifecycle_decision(
    decision: EvolutionDecision,
    state: RuleState | RelationState,
    journal: Any | None = None,
) -> RuleState | RelationState:
    """Apply a decision to canonical in-memory state."""
    state_id = state.rule_id if isinstance(state, RuleState) else state.relation_id
    if decision.subject_id != state_id:
        raise ValueError("lifecycle decision subject does not match state")
    changes: dict[str, Any] = {}
    if decision.operation == "RETIRE":
        changes.update(status="retired", drift_state="stale")
    elif decision.operation in {"SPECIALIZE", "SPLIT", "QUARANTINE"}:
        changes.update(status="candidate", drift_state="revalidating")
    elif decision.operation == "REVALIDATE":
        changes["drift_state"] = "revalidating"
    updated = replace(state, **changes) if changes else state
    if journal is not None and changes:
        journal.append("update_state", decision.subject_id, version=int(updated.version), digest=decision.operation.lower())
    return updated


__all__ = ["apply_lifecycle_decision"]
