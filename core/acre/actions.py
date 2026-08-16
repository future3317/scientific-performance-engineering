"""Semantic action projection shared by formal and evolution paths."""

from __future__ import annotations

from typing import Any, Mapping

from core.models import ActionSpec
import hashlib
import json


class RealizationValidator:
    """Validate that a proposal names one registered semantic action.

    Source patches are execution details; they are not accepted as a causal
    action identity until the harness maps them to a declared FamilySpec
    action.  This keeps patch variants from sharing an evidence ledger entry
    merely because they target the same family.
    """

    @staticmethod
    def validate(family_id: str | None, action: ActionSpec) -> ActionSpec:
        if action.action_id.startswith("patch-") and family_id:
            raise ValueError("source patch has no proven registered ActionSpec")
        if family_id:
            try:
                from benchmark.families import FAMILY_SPECS, resolve_family_id
                family = FAMILY_SPECS[resolve_family_id(family_id)]
            except (ImportError, KeyError, ValueError) as exc:
                raise ValueError(f"unknown action family: {family_id}") from exc
            if action.action_id not in family.action_specs:
                raise ValueError(f"action {action.action_id} is not registered for family {family_id}")
        return action


def action_from_proposal(family_id: str | None, proposal: Mapping[str, Any]) -> ActionSpec:
    explicit = proposal.get("action_spec")
    if not isinstance(explicit, Mapping):
        explicit = proposal.get("intervention") if isinstance(proposal.get("intervention"), Mapping) else {}
    if explicit.get("action"):
        action_id = str(explicit["action"])
        parameters = dict(explicit.get("parameters") or {})
    elif isinstance(proposal.get("intervention"), Mapping):
        # A source patch is not a semantic action label.  Keep its identity
        # separate until the harness can prove a unique registered action;
        # never collapse all patches in one family into one ledger key.
        patch = dict(proposal["intervention"])
        action_id = "patch-" + hashlib.sha256(json.dumps(patch, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
        parameters = patch
    else:
        # A family default is a workload-level prior, not observed activation
        # evidence.  Treating it as the classified intervention would let a
        # raw patch enter the causal ledger without an S4 action certificate.
        raise ValueError("proposal must provide an explicit action_spec or source intervention")
    action = ActionSpec(action_id=action_id, family=str(family_id or "runtime"), parameters=parameters)
    return RealizationValidator.validate(family_id, action)


__all__ = ["action_from_proposal", "RealizationValidator"]
