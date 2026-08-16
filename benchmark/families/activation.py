"""Harness-owned S4 activation classification for registered actions."""

from __future__ import annotations

from typing import Any, Mapping


_REQUIRED_METRICS = {
    "compile_cache_guard_hit": "compile_cache_hit",
    "compile_dynamic_guard_stability": "dynamic_guard_stable",
    "kernel_fusion_operator_trace": "fused_operator_count",
    "graph_cache_hit_without_rebuild": "cache_hit_without_rebuild",
    "graph_cache_rebuild_trace": "graph_rebuild_count",
    "neighbor_graph_reuse_trace": "neighbor_graph_reused",
    "batched_force_vjp_trace": "batched_vjp_calls",
    "h2d_pin_nonblocking_trace": "pin_nonblocking_overlap",
    "h2d_prefetch_fanout_trace": "prefetch_workers",
    "checkpoint_recompute_trace": "checkpoint_recompute_calls",
    "retained_graph_trace": "retained_graph_edges",
    "scalar_aggregation_sync_count": "aggregated_scalar_syncs",
    "scalar_deferred_sync_count": "deferred_scalar_syncs",
    "backbone_reuse_trace": "backbone_reused",
    "batched_vjp_trace": "batched_vjp_calls",
    "equivariant_path_trace": "equivariant_path_calls",
    "crystal_sampling_trace": "crystal_sampler_calls",
    "graph_rebuild_trace": "graph_rebuild_count",
    "episode_transition_trace": "transition_applied",
}


def classify_activation(family_id: str, action_specs: Mapping[str, Mapping[str, Any]], metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return an S4 certificate only when exactly one action is observed."""
    matches: list[str] = []
    for action_id, spec in action_specs.items():
        validator = str(spec.get("activation_validator", ""))
        metric = _REQUIRED_METRICS.get(validator)
        if metric and bool(metrics.get(metric, False)):
            matches.append(str(action_id))
    if len(matches) != 1:
        return {"status": "rejected", "matched_actions": matches, "reason": "activation must match exactly one registered action"}
    action_id = matches[0]
    validator = str(action_specs[action_id].get("activation_validator", ""))
    return {
        "status": "passed",
        "matched_actions": matches,
        "action_id": action_id,
        "validator": validator,
        "metrics": dict(metrics),
    }


__all__ = ["classify_activation"]
