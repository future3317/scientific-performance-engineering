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
    state_store: Any | None = None,
) -> RuleState | RelationState:
    """Apply a decision to canonical in-memory state."""
    state_id = state.rule_id if isinstance(state, RuleState) else state.relation_id
    if decision.subject_id != state_id:
        raise ValueError("lifecycle decision subject does not match state")
    changes: dict[str, Any] = {}
    if decision.operation == "RETIRE":
        changes.update(status="retired", drift_state="stale")
    elif decision.operation in {"SPECIALIZE", "SPLIT"}:
        # The canonical parent revision is immutable.  A child candidate is
        # created by governance; the parent only becomes ineligible through
        # its drift state.
        changes.update(status=state.status, drift_state="stale")
    elif decision.operation == "QUARANTINE":
        changes.update(status=state.status, drift_state="stale")
    elif decision.operation == "REVALIDATE":
        changes["drift_state"] = "revalidating"
    updated = replace(state, **changes) if changes else state
    if changes and state_store is not None:
        state_store.apply_transition(state, updated, decision=decision, journal=journal)
        if decision.operation == "RETIRE":
            import json
            registry_path = state_store.root / "registry" / ("rules.json" if isinstance(state, RuleState) else "relations.json")
            if registry_path.is_file():
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
                key = "rules" if isinstance(state, RuleState) else "relations"
                id_key = "rule_id" if isinstance(state, RuleState) else "relation_id"
                registry[key] = [entry for entry in registry.get(key, []) if entry.get(id_key) != state_id]
                registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                if journal is not None:
                    import hashlib
                    journal.append(
                        "activate_registry",
                        state_id,
                        version=int(updated.version),
                        artifact_path=str(registry_path.relative_to(state_store.root)).replace("\\", "/"),
                        digest=hashlib.sha256(registry_path.read_bytes()).hexdigest(),
                        old_digest="",
                        operation_detail="retire_revision",
                    )
    elif journal is not None and changes:
        journal.append("update_state", decision.subject_id, version=int(updated.version), digest=decision.operation.lower(), operation_detail=decision.operation)
    return updated


__all__ = ["apply_lifecycle_decision"]
