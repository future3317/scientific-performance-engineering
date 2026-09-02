"""BoundaryBench evaluation only; synthesis remains owned by core.acre."""

from __future__ import annotations

from typing import Any, Mapping

from core.predicates import match_predicate


def sealed_errors(predicate: dict[str, Any] | None, sealed_pool: list[Any]) -> int:
    if predicate is None:
        return len(sealed_pool)
    return sum(int(bool(match_predicate(predicate, item.context)) != item.expected_applicable) for item in sealed_pool)


def evaluate_applicability(
    predictions: Mapping[str, bool], truth: Mapping[str, bool]
) -> dict[str, float | int | None]:
    """Evaluate predictions only after a policy has finished selecting evidence.

    ``truth`` is deliberately kept in this offline evaluator.  Policy code
    receives no reference to it, which keeps held-out labels out of synthesis
    and routing decisions.
    """
    if set(predictions) != set(truth):
        raise ValueError("predictions and truth must cover the same observations")
    tp = sum(bool(predictions[key]) and bool(truth[key]) for key in truth)
    fp = sum(bool(predictions[key]) and not bool(truth[key]) for key in truth)
    fn = sum(not bool(predictions[key]) and bool(truth[key]) for key in truth)
    tn = sum(not bool(predictions[key]) and not bool(truth[key]) for key in truth)
    predicted_positive = tp + fp
    actual_positive = tp + fn
    precision = tp / predicted_positive if predicted_positive else None
    recall = tp / actual_positive if actual_positive else None
    f1 = (2.0 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    total = len(truth)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "unsafe_generalization_rate": fp / total if total else 0.0,
        "abstention_rate": (fn + tn) / total if total else 0.0,
    }
