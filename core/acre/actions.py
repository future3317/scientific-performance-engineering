"""Semantic action projection shared by formal and evolution paths."""

from __future__ import annotations

from typing import Any, Mapping

from core.models import ActionSpec
import hashlib
import json


_FAMILY_ACTIONS = {
    "compile": "reuse_compile_cache",
    "graph_cache": "reuse_graph_cache",
    "h2d_pipeline": "pin_memory_pipeline",
    "checkpoint": "checkpoint_recompute",
    "scalar_sync": "aggregate_scalars",
}


def action_from_proposal(family_id: str | None, proposal: Mapping[str, Any]) -> ActionSpec:
    explicit = proposal.get("action_spec")
    if not isinstance(explicit, Mapping):
        explicit = proposal.get("intervention") if isinstance(proposal.get("intervention"), Mapping) else {}
    if not explicit.get("action") and not family_id and isinstance(proposal.get("intervention"), Mapping):
        patch = dict(proposal["intervention"])
        action_id = "patch-" + hashlib.sha256(json.dumps(patch, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
        parameters = patch
    else:
        action_id = str(explicit.get("action") or _FAMILY_ACTIONS.get(str(family_id or ""), "measure"))
        parameters = dict(explicit.get("parameters") or {})
    return ActionSpec(action_id=action_id, family=str(family_id or "runtime"), parameters=parameters)


__all__ = ["action_from_proposal"]
