"""Registered measurement-noise replay for provider-free support evidence."""

from __future__ import annotations

import random
from typing import Any, Mapping

from core.acre.cegis import BoundaryObservation
from core.sequential_stats import paired_repetition_interval


def repeated_observation(
    *,
    observation_id: str,
    context: Mapping[str, Any],
    effect: float,
    gate_passed: bool,
    seed: int,
    repetitions: int,
    noise_scale: float,
    delta: float,
) -> BoundaryObservation:
    if int(repetitions) < 2 or float(noise_scale) <= 0.0:
        raise ValueError("non-degenerate replay requires repetitions >= 2 and positive noise_scale")
    rng = random.Random(f"{observation_id}|{int(seed)}")
    values = [max(-1.0, min(1.0, float(effect) + rng.uniform(-float(noise_scale), float(noise_scale)))) for _ in range(int(repetitions))]
    lower, upper = paired_repetition_interval(values, float(delta))
    return BoundaryObservation(
        str(observation_id),
        dict(context),
        sum(values) / len(values),
        bool(gate_passed),
        lower,
        upper,
    )


def classify_observation(observation: BoundaryObservation, practical_margin: float) -> str:
    if not observation.gate_passed or observation.effect_upper <= float(practical_margin):
        return "counterexample"
    if observation.effect_lower > float(practical_margin):
        return "positive_anchor"
    return "unresolved"

