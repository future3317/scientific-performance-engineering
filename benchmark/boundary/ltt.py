"""Calibration-only selective-risk logistic reuse policy.

The policy is deliberately small: it fits the existing public typed-feature
logistic scorer on synthesis rows, then selects the lowest score threshold
whose calibration unsupported/deployed risk has a one-sided exact upper bound
below the registered target.  Evaluation rows are never consumed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from scipy.special import betaincinv

from .policy_eval import Decision
from .rscb import RSCBConfig, RSCBModel
from .typed_features import TypedFeatureSchema


@dataclass(frozen=True)
class LTTConfig:
    threshold_grid: tuple[float, ...]
    target_risk: float
    confidence: float

    @classmethod
    def default(cls) -> "LTTConfig":
        return cls(
            threshold_grid=(0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95),
            target_risk=0.05,
            confidence=0.95,
        )

    def __post_init__(self) -> None:
        if not self.threshold_grid:
            raise ValueError("LTT threshold grid must not be empty")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.threshold_grid):
            raise ValueError("LTT thresholds must lie in [0, 1]")
        if not 0.0 <= float(self.target_risk) <= 1.0:
            raise ValueError("LTT target risk must lie in [0, 1]")
        if not 0.0 < float(self.confidence) < 1.0:
            raise ValueError("LTT confidence must lie in (0, 1)")


def _upper_binomial_bound(errors: int, total: int, confidence: float) -> float:
    if total <= 0:
        return 1.0
    if errors <= 0:
        return 1.0 - (1.0 - confidence) ** (1.0 / total)
    if errors >= total:
        return 1.0
    return float(betaincinv(errors + 1, total - errors, confidence))


@dataclass(frozen=True)
class LTTLogisticPolicy:
    schema: TypedFeatureSchema
    config: LTTConfig
    coefficients: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    threshold: float | None
    calibration_risk_upper: float | None
    calibration_deployed: int
    calibration_unsupported: int
    fit_source: str = "synthesis_then_calibration"

    @classmethod
    def fit(
        cls,
        synthesis_rows: Sequence[Mapping[str, Any]],
        calibration_rows: Sequence[Mapping[str, Any]],
        schema: TypedFeatureSchema,
        config: LTTConfig,
    ) -> "LTTLogisticPolicy":
        if not synthesis_rows:
            raise ValueError("LTT synthesis requires at least one row")
        scorer_rows = [
            {"context": row["context"], "label": bool(row["label"])}
            for row in synthesis_rows
        ]
        scorer = RSCBModel.fit(
            scorer_rows,
            schema,
            RSCBConfig(
                k=0.0,
                theta=0.0,
                max_iterations=100,
                tolerance=1e-9,
                threshold_grid=config.threshold_grid,
            ),
        )
        scored_calibration = [
            (scorer.posterior(row["context"])[0], bool(row["unsupported"]))
            for row in calibration_rows
        ]
        chosen: tuple[float, float, int, int] | None = None
        thresholds = sorted(set(float(value) for value in config.threshold_grid))
        simultaneous_confidence = 1.0 - (1.0 - config.confidence) / len(thresholds)
        for threshold in thresholds:
            selected = [item for item in scored_calibration if item[0] >= threshold]
            if not selected:
                continue
            errors = sum(int(unsupported) for _, unsupported in selected)
            upper = _upper_binomial_bound(errors, len(selected), simultaneous_confidence)
            if upper <= config.target_risk:
                chosen = (threshold, upper, len(selected), errors)
                break
        return cls(
            schema=schema,
            config=config,
            coefficients=scorer.coefficients,
            covariance=scorer.covariance,
            threshold=chosen[0] if chosen else None,
            calibration_risk_upper=chosen[1] if chosen else None,
            calibration_deployed=chosen[2] if chosen else 0,
            calibration_unsupported=chosen[3] if chosen else 0,
        )

    def _scorer(self) -> RSCBModel:
        return RSCBModel(
            schema=self.schema,
            config=RSCBConfig(
                k=0.0,
                theta=0.0,
                max_iterations=100,
                tolerance=1e-9,
                threshold_grid=self.config.threshold_grid,
            ),
            coefficients=self.coefficients,
            covariance=self.covariance,
            iterations=0,
        )

    def decide(self, context: Mapping[str, Any]) -> Decision:
        if self.threshold is None:
            return Decision("abstain", 0.0, 0.0, "risk_not_certified")
        probability, _ = self._scorer().posterior(context)
        admitted = probability >= self.threshold
        return Decision(
            "reuse" if admitted else "abstain",
            probability,
            0.0,
            "risk_calibrated_pass" if admitted else "risk_calibrated_below_threshold",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": "ltt_logistic",
            "fit_source": self.fit_source,
            "threshold_error_control": "Bonferroni across the fixed threshold grid; binomial bound assumes independent calibration observations",
            "config": {
                "threshold_grid": list(self.config.threshold_grid),
                "target_risk": self.config.target_risk,
                "confidence": self.config.confidence,
            },
            "schema": self.schema.to_dict(),
            "coefficients": list(self.coefficients),
            "covariance": [list(row) for row in self.covariance],
            "threshold": self.threshold,
            "calibration_risk_upper": self.calibration_risk_upper,
            "calibration_deployed": self.calibration_deployed,
            "calibration_unsupported": self.calibration_unsupported,
        }


__all__ = ["LTTConfig", "LTTLogisticPolicy"]
