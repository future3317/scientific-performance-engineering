"""Report-only bounds for resource-conditioned population estimates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def compute_resource_sensitivity(
    *,
    registered_weights: Mapping[str, float],
    observed_delta: float,
    utility_bounds: tuple[float, float] = (-1.0, 1.0),
) -> dict[str, Any]:
    """Return the worst-case full-population interval for excluded task weight.

    ``registered_weights`` is the preregistered weight map for tasks excluded
    from the resource-feasible estimate.  The observed value is already the
    resource-conditioned contribution; this function never changes eligibility.
    """
    if not isinstance(registered_weights, Mapping) or not registered_weights:
        raise ValueError("registered_weights must contain excluded tasks")
    lower, upper = (float(value) for value in utility_bounds)
    observed = float(observed_delta)
    if not all(math.isfinite(value) for value in (lower, upper, observed)) or lower > upper:
        raise ValueError("utility bounds and observed delta must be finite and ordered")
    weights = {str(task_id): float(weight) for task_id, weight in registered_weights.items()}
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("registered task weights must be finite and non-negative")
    missing_weight = sum(weights.values())
    if missing_weight > 1.0 + 1e-12:
        raise ValueError("excluded task weights cannot exceed one")
    contrast_bounds = (lower - upper, upper - lower)
    return {
        "registered_weighting": "equal_task_then_equal_lineage",
        "excluded_task_weights": dict(sorted(weights.items())),
        "missing_weight": missing_weight,
        "observed_resource_conditioned_contribution": observed,
        "utility_bounds": [lower, upper],
        "contrast_bounds": list(contrast_bounds),
        "full_population_worst_case": [
            observed + missing_weight * contrast_bounds[0],
            observed + missing_weight * contrast_bounds[1],
        ],
    }
