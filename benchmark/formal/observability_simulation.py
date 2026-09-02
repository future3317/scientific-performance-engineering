"""Provider-free simulation of treatment-coupled outcome observability."""

from __future__ import annotations

import math
import random
from statistics import fmean
from typing import Sequence


def _validate_outcomes(values: Sequence[float], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in values):
        raise ValueError(f"{name} must contain finite values in [0, 1]")


def partial_identification_bounds(
    d_outcomes: Sequence[float],
    b_outcomes: Sequence[float],
    observed_d: Sequence[bool],
) -> dict[str, float | int | None]:
    """Return the true effect and Manski-style bounds for censored D outcomes."""
    _validate_outcomes(d_outcomes, "d_outcomes")
    _validate_outcomes(b_outcomes, "b_outcomes")
    if len(d_outcomes) != len(b_outcomes) or len(d_outcomes) != len(observed_d):
        raise ValueError("outcomes and observation mask must have equal length")
    if any(not isinstance(flag, bool) for flag in observed_d):
        raise ValueError("observed_d must contain booleans")

    population_size = len(d_outcomes)
    observed_sum = sum(float(value) for value, observed in zip(d_outcomes, observed_d) if observed)
    observed_count = sum(observed_d)
    b_mean = fmean(float(value) for value in b_outcomes)
    true_effect = fmean(float(value) for value in d_outcomes) - b_mean
    lower = observed_sum / population_size - b_mean
    upper = (observed_sum + population_size - observed_count) / population_size - b_mean
    complete_case_effect = observed_sum / observed_count - b_mean if observed_count else None
    return {
        "population_size": population_size,
        "observed_count": observed_count,
        "censoring_rate": 1.0 - observed_count / population_size,
        "true_effect": true_effect,
        "complete_case_effect": complete_case_effect,
        "complete_case_bias": complete_case_effect - true_effect if complete_case_effect is not None else None,
        "lower": lower,
        "upper": upper,
        "interval_width": upper - lower,
        "covered": lower - 1e-12 <= true_effect <= upper + 1e-12,
    }


def _fixed_population(population_size: int, seed: int) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    d_outcomes = [0.2 + 0.6 * rng.random() for _ in range(population_size)]
    b_outcomes = [0.1 + 0.6 * rng.random() for _ in range(population_size)]
    return d_outcomes, b_outcomes


def _observation_mask(
    d_outcomes: Sequence[float],
    observed_count: int,
    *,
    mechanism: str,
    seed: int,
) -> list[bool]:
    rng = random.Random(seed)
    indices = list(range(len(d_outcomes)))
    if mechanism == "independent":
        selected = set(rng.sample(indices, observed_count))
    elif mechanism == "treatment_coupled":
        ranked = sorted(indices, key=lambda index: (d_outcomes[index], rng.random()))
        selected = set(ranked[:observed_count])
    else:
        raise ValueError(f"unknown observation mechanism: {mechanism}")
    return [index in selected for index in indices]


def run_observability_simulation(
    *,
    censoring_rates: Sequence[float] = (0.0, 0.10, 0.20, 0.35, 0.50),
    population_size: int = 200,
    repetitions: int = 200,
    seed: int = 7,
) -> dict[str, object]:
    """Run the fixed finite-population observability grid without provider calls."""
    rates = [float(rate) for rate in censoring_rates]
    if not rates or any(not 0.0 <= rate <= 1.0 for rate in rates):
        raise ValueError("censoring_rates must be non-empty and lie in [0, 1]")
    if population_size < 1 or repetitions < 1:
        raise ValueError("population_size and repetitions must be positive")

    d_outcomes, b_outcomes = _fixed_population(population_size, seed)
    true_effect = fmean(d_outcomes) - fmean(b_outcomes)
    rows: list[dict[str, object]] = []
    for rate_index, rate in enumerate(rates):
        observed_count = population_size - round(population_size * rate)
        for mechanism_index, mechanism in enumerate(("independent", "treatment_coupled")):
            replicates = []
            for repetition in range(repetitions):
                mask = _observation_mask(
                    d_outcomes,
                    observed_count,
                    mechanism=mechanism,
                    seed=seed * 100000 + rate_index * 1000 + mechanism_index * 100 + repetition,
                )
                replicates.append(partial_identification_bounds(d_outcomes, b_outcomes, mask))
            complete_case_biases = [
                float(item["complete_case_bias"])
                for item in replicates
                if item["complete_case_bias"] is not None
            ]
            rows.append({
                "mechanism": mechanism,
                "censoring_rate": rate,
                "observed_count": observed_count,
                "repetitions": repetitions,
                "true_effect": true_effect,
                "mean_complete_case_effect": fmean(
                    float(item["complete_case_effect"])
                    for item in replicates
                    if item["complete_case_effect"] is not None
                ) if complete_case_biases else None,
                "mean_complete_case_bias": fmean(complete_case_biases) if complete_case_biases else None,
                "mean_interval_width": fmean(float(item["interval_width"]) for item in replicates),
                "coverage": sum(bool(item["covered"]) for item in replicates) / repetitions,
            })
    return {
        "schema_version": 1,
        "provider_calls": 0,
        "seed": seed,
        "population_size": population_size,
        "repetitions": repetitions,
        "censoring_rates": rates,
        "mechanisms": ["independent", "treatment_coupled"],
        "true_effect": true_effect,
        "results": rows,
    }
