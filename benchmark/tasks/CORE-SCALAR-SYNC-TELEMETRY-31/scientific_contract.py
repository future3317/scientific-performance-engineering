"""Scientific contract for the telemetry-sync positive anchor."""

from __future__ import annotations

REQUIRED_GATES = (
    "final_model_equivalence",
    "loss_trajectory_equivalence",
    "telemetry_vector_equivalence",
    "cadence_preserved",
    "work_units_preserved",
)
