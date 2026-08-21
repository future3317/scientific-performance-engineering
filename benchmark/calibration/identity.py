"""Calibration identity and JSON digest primitives."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def canonical_cell_identity(
    *, task_id: str, outer_trial_id: str, seed: int,
    measurement_family: str, task_package_digest: str,
    population_manifest_digest: str,
) -> dict[str, Any]:
    """Describe one cell at its raw-result and envelope identity layers."""
    family = str(measurement_family)
    if family not in {"atomic_performance", "evolution"}:
        raise ValueError(f"unsupported calibration measurement family: {family}")
    return {
        "task_id": str(task_id),
        "outer_trial_id": str(outer_trial_id),
        "seed": int(seed),
        "measurement_family": family,
        "raw_measurement_class": "episode_bounded_score" if family == "evolution" else family,
        "envelope_measurement_class": family,
        "task_package_digest": str(task_package_digest),
        "population_manifest_digest": str(population_manifest_digest),
    }

