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


def _contrastive_observed(validator: str, candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    """Require an action-specific change relative to the control trace."""
    metric = _REQUIRED_METRICS.get(validator)
    if not metric:
        return False
    c = candidate.get(metric)
    b = baseline.get(metric)
    if isinstance(c, bool) or isinstance(b, bool):
        return bool(c) and not bool(b)
    if not isinstance(c, (int, float)) or not isinstance(b, (int, float)):
        return False
    if "sync_count" in validator or "deferred_sync" in validator:
        return float(c) < float(b)
    if "rebuild" in validator or "operator" in validator or "worker" in validator or "prefetch" in validator:
        return float(c) > float(b)
    return float(c) != float(b)


def classify_activation(
    family_id: str,
    action_specs: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an S4 certificate only when one action changes the control trace."""
    if baseline_metrics is None:
        return {"status": "rejected", "matched_actions": [], "reason": "contrastive baseline trace is required"}
    matches: list[str] = []
    for action_id, spec in action_specs.items():
        validator = str(spec.get("activation_validator", ""))
        metric = _REQUIRED_METRICS.get(validator)
        if metric and _contrastive_observed(validator, metrics, baseline_metrics):
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
        "baseline_metrics": dict(baseline_metrics),
    }


__all__ = ["classify_activation"]
