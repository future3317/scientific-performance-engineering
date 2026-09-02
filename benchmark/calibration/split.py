"""Condition-blind, seed-disjoint calibration population construction."""

from __future__ import annotations

from typing import Any, Sequence

from benchmark.families import family_instances


def build_calibration_split(
    families: Sequence[str],
    *,
    calibration_seed: int,
    final_seed: int,
    count: int = 8,
) -> dict[str, Any]:
    """Build IDs and namespaces without exposing family outcome labels."""
    if calibration_seed == final_seed:
        raise ValueError("calibration and final seeds must be disjoint")
    if count < 1:
        raise ValueError("calibration count must be positive")
    calibration_ids = [
        item.instance_id
        for family in families
        for item in family_instances(str(family), count=count, seed=int(calibration_seed))
    ]
    final_ids = [
        item.instance_id
        for family in families
        for item in family_instances(str(family), count=count, seed=int(final_seed))
    ]
    calibration_evidence = [f"calibration/{item}" for item in calibration_ids]
    final_evidence = [f"confirmatory/{item}" for item in final_ids]
    promotion_ids = [f"promotion/{item}" for item in final_ids]
    return {
        "schema_version": 1,
        "calibration_seed": int(calibration_seed),
        "final_seed": int(final_seed),
        "families": [str(item) for item in families],
        "calibration_instance_ids": sorted(calibration_ids),
        "final_instance_ids": sorted(final_ids),
        "calibration_evidence_ids": sorted(calibration_evidence),
        "final_evidence_ids": sorted(final_evidence),
        "promotion_instance_ids": sorted(promotion_ids),
        "seed_disjoint": int(calibration_seed) != int(final_seed) and not (set(calibration_ids) & set(final_ids)),
        "promotion_disjoint": not (set(calibration_ids) & set(promotion_ids)),
        "evidence_disjoint": not (set(calibration_evidence) & set(final_evidence)),
        "labels_exposed_to_policy": False,
        "confirmatory_estimand": "excluded",
    }


__all__ = ["build_calibration_split"]
