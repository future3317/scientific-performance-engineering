"""Frozen RSCB-LCB probabilistic reuse gate for provider-free evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .policy_eval import Decision
from .typed_features import TypedFeatureSchema


@dataclass(frozen=True)
class RSCBConfig:
    k: float
    theta: float
    max_iterations: int
    tolerance: float
    threshold_grid: tuple[float, ...]

    @classmethod
    def default(cls) -> "RSCBConfig":
        return cls(
            k=1.64485362695147,
            theta=0.90,
            max_iterations=100,
            tolerance=1e-9,
            threshold_grid=(0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95),
        )


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True)
class RSCBModel:
    schema: TypedFeatureSchema
    config: RSCBConfig
    coefficients: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    iterations: int

    @classmethod
    def fit(
        cls,
        calibration_rows: Sequence[Mapping[str, Any]],
        schema: TypedFeatureSchema,
        config: RSCBConfig,
    ) -> "RSCBModel":
        if not calibration_rows:
            raise ValueError("RSCB calibration requires at least one row")
        x = np.asarray([
            [1.0, *schema.transform(row["context"])] for row in calibration_rows
        ], dtype=float)
        y = np.asarray([1.0 if bool(row["label"]) else 0.0 for row in calibration_rows], dtype=float)
        dimension = x.shape[1]
        prior_precision = np.ones(dimension, dtype=float)
        prior_precision[0] = 0.01  # N(0, 100) intercept prior
        beta = np.zeros(dimension, dtype=float)
        iterations = 0
        for iterations in range(1, config.max_iterations + 1):
            logits = x @ beta
            probabilities = np.asarray([_sigmoid(float(logit)) for logit in logits])
            weights = np.maximum(probabilities * (1.0 - probabilities), 1e-12)
            hessian = (x.T * weights) @ x + np.diag(prior_precision)
            gradient = x.T @ (y - probabilities) - prior_precision * beta
            step = np.linalg.solve(hessian, gradient)
            beta_next = beta + step
            if float(np.max(np.abs(step))) <= config.tolerance:
                beta = beta_next
                break
            beta = beta_next
        logits = x @ beta
        probabilities = np.asarray([_sigmoid(float(logit)) for logit in logits])
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-12)
        observed_hessian = (x.T * weights) @ x + np.diag(prior_precision)
        covariance = np.linalg.inv(observed_hessian)
        return cls(
            schema=schema,
            config=config,
            coefficients=tuple(float(value) for value in beta),
            covariance=tuple(tuple(float(value) for value in row) for row in covariance),
            iterations=iterations,
        )

    def _phi(self, context: Mapping[str, Any]) -> np.ndarray:
        return np.asarray([1.0, *self.schema.transform(context)], dtype=float)

    def posterior(self, context: Mapping[str, Any]) -> tuple[float, float]:
        phi = self._phi(context)
        beta = np.asarray(self.coefficients, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        probability = _sigmoid(float(phi @ beta))
        variance = max(0.0, float(phi @ covariance @ phi))
        uncertainty = probability * (1.0 - probability) * math.sqrt(variance)
        return probability, uncertainty

    def lcb(self, context: Mapping[str, Any], k: float | None = None) -> float:
        probability, uncertainty = self.posterior(context)
        penalty = self.config.k if k is None else float(k)
        return max(0.0, min(1.0, probability - penalty * uncertainty))

    def decide(self, context: Mapping[str, Any]) -> Decision:
        probability, uncertainty = self.posterior(context)
        lower_bound = max(0.0, min(1.0, probability - self.config.k * uncertainty))
        admitted = lower_bound >= self.config.theta
        return Decision(
            "reuse" if admitted else "abstain",
            lower_bound,
            uncertainty,
            "lcb_pass" if admitted else "lcb_below_threshold",
        )

    def with_threshold(self, theta: float) -> "RSCBModel":
        return replace(self, config=replace(self.config, theta=float(theta)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema.to_dict(),
            "config": {
                "k": self.config.k,
                "theta": self.config.theta,
                "max_iterations": self.config.max_iterations,
                "tolerance": self.config.tolerance,
                "threshold_grid": list(self.config.threshold_grid),
            },
            "coefficients": list(self.coefficients),
            "covariance": [list(row) for row in self.covariance],
            "iterations": self.iterations,
        }


def matched_threshold(
    model: RSCBModel,
    calibration_contexts: Sequence[Mapping[str, Any]],
    target_coverage: float,
    thresholds: Sequence[float] | None = None,
) -> float:
    """Select theta by calibration coverage only; ties prefer higher theta."""

    if not 0.0 <= float(target_coverage) <= 1.0:
        raise ValueError("target coverage must lie in [0, 1]")
    grid = tuple(model.config.threshold_grid if thresholds is None else thresholds)
    if not grid:
        raise ValueError("matched threshold grid must not be empty")
    lcb_values = [model.lcb(context) for context in calibration_contexts]
    candidates = []
    for theta in grid:
        coverage = sum(value >= float(theta) for value in lcb_values) / len(lcb_values) if lcb_values else 0.0
        candidates.append((abs(coverage - float(target_coverage)), -float(theta), float(theta)))
    return min(candidates)[2]


__all__ = ["RSCBConfig", "RSCBModel", "matched_threshold"]
