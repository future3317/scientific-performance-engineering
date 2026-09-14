"""Tree-based selective-risk baselines for the shared boundary evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.tree import DecisionTreeClassifier

try:  # Optional at import time so provider-free tests can report availability.
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - exercised through the explicit error path
    CatBoostClassifier = None  # type: ignore[assignment,misc]

from .ltt import LTTConfig, _upper_binomial_bound
from .policy_eval import Decision
from .typed_features import TypedFeatureSchema


def _matrix(rows: Sequence[Mapping[str, Any]], schema: TypedFeatureSchema) -> np.ndarray:
    if not rows:
        raise ValueError("tree policy fitting requires at least one synthesis row")
    return np.asarray([schema.transform(row["context"]) for row in rows], dtype=float)


def _labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([1 if bool(row["label"]) else 0 for row in rows], dtype=int)


def _positive_probability(model: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = [int(value) for value in getattr(model, "classes_", (0, 1))]
    if 1 not in classes:
        return np.zeros(len(features), dtype=float)
    return probabilities[:, classes.index(1)]


def _select_threshold(
    model: Any,
    calibration_rows: Sequence[Mapping[str, Any]],
    schema: TypedFeatureSchema,
    config: LTTConfig,
) -> tuple[float | None, float | None, int, int]:
    if not calibration_rows:
        return None, None, 0, 0
    scores = _positive_probability(
        model,
        np.asarray([schema.transform(row["context"]) for row in calibration_rows], dtype=float),
    )
    chosen: tuple[float, float, int, int] | None = None
    for threshold in sorted(float(value) for value in config.threshold_grid):
        selected = scores >= threshold
        deployed = int(np.sum(selected))
        errors = int(sum(bool(row["unsupported"]) for row, keep in zip(calibration_rows, selected) if keep))
        upper = _upper_binomial_bound(errors, deployed, config.confidence) if deployed else 0.0
        if upper <= config.target_risk:
            chosen = (threshold, upper, deployed, errors)
            break
    return chosen if chosen is not None else (None, None, 0, 0)


@dataclass(frozen=True)
class TreeSelectivePolicy:
    model: Any
    schema: TypedFeatureSchema
    config: LTTConfig
    threshold: float | None
    calibration_risk_upper: float | None
    calibration_deployed: int
    calibration_unsupported: int
    policy_name: str
    model_config: Mapping[str, Any]
    fit_source: str = "synthesis_then_calibration"

    def decide(self, context: Mapping[str, Any]) -> Decision:
        if self.threshold is None:
            return Decision("abstain", 0.0, 0.0, "risk_not_certified")
        features = np.asarray([self.schema.transform(context)], dtype=float)
        probability = float(_positive_probability(self.model, features)[0])
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
            "policy": self.policy_name,
            "fit_source": self.fit_source,
            "config": {
                "threshold_grid": list(self.config.threshold_grid),
                "target_risk": self.config.target_risk,
                "confidence": self.config.confidence,
            },
            "model_config": dict(self.model_config),
            "schema": self.schema.to_dict(),
            "threshold": self.threshold,
            "calibration_risk_upper": self.calibration_risk_upper,
            "calibration_deployed": self.calibration_deployed,
            "calibration_unsupported": self.calibration_unsupported,
        }


def _fit_policy(
    model: Any,
    *,
    policy_name: str,
    model_config: Mapping[str, Any],
    synthesis_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    schema: TypedFeatureSchema,
    config: LTTConfig | None,
) -> TreeSelectivePolicy:
    if config is None:
        config = LTTConfig.default()
    model.fit(_matrix(synthesis_rows, schema), _labels(synthesis_rows))
    threshold, upper, deployed, unsupported = _select_threshold(model, calibration_rows, schema, config)
    return TreeSelectivePolicy(
        model=model,
        schema=schema,
        config=config,
        threshold=threshold,
        calibration_risk_upper=upper,
        calibration_deployed=deployed,
        calibration_unsupported=unsupported,
        policy_name=policy_name,
        model_config=model_config,
    )


def fit_cart_policy(
    synthesis_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    schema: TypedFeatureSchema,
    *,
    max_depth: int = 5,
    seed: int = 0,
    config: LTTConfig | None = None,
) -> TreeSelectivePolicy:
    model = DecisionTreeClassifier(max_depth=int(max_depth), random_state=int(seed))
    return _fit_policy(
        model,
        policy_name="cart_ltt",
        model_config={"max_depth": int(max_depth), "seed": int(seed)},
        synthesis_rows=synthesis_rows,
        calibration_rows=calibration_rows,
        schema=schema,
        config=config,
    )


def fit_catboost_ltt_policy(
    synthesis_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    schema: TypedFeatureSchema,
    *,
    depth: int = 5,
    trees: int = 300,
    learning_rate: float = 0.05,
    seed: int = 0,
    config: LTTConfig | None = None,
) -> TreeSelectivePolicy:
    if CatBoostClassifier is None:
        raise RuntimeError("CatBoost is required for the CatBoost-LTT baseline")
    model = CatBoostClassifier(
        depth=int(depth),
        iterations=int(trees),
        learning_rate=float(learning_rate),
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )
    return _fit_policy(
        model,
        policy_name="catboost_ltt",
        model_config={
            "depth": int(depth),
            "trees": int(trees),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
        },
        synthesis_rows=synthesis_rows,
        calibration_rows=calibration_rows,
        schema=schema,
        config=config,
    )


__all__ = ["TreeSelectivePolicy", "fit_cart_policy", "fit_catboost_ltt_policy"]
