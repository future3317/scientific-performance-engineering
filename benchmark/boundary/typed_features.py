"""Calibration-frozen public feature encoding for boundary policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


_FORBIDDEN_CALIBRATION_KEYS = {
    "expected_applicable",
    "scientific_truth",
    "scientific_gate",
    "tau",
    "oracle_label",
    "counterexample",
    "split",
    "outcome",
    "outcome_vector",
    "seed",
    "context_id",
}


def _get_path(context: Mapping[str, Any], path: str) -> tuple[Any, bool]:
    current: Any = context
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None, False
        current = current[part]
    return current, True


def _canonical_family(family: str) -> str:
    return {
        "graph_cache_geometry_motion": "graph_cache",
        "compile_horizon": "compile",
    }.get(family, family)


@dataclass(frozen=True)
class TypedFeatureSchema:
    """A deterministic, calibration-only schema for public contexts."""

    family: str
    paths: tuple[str, ...]
    features: tuple[dict[str, Any], ...]
    output_width: int
    fit_source: str = "calibration"

    @classmethod
    def from_calibration(
        cls,
        family: str,
        calibration_contexts: Sequence[Mapping[str, Any]],
    ) -> "TypedFeatureSchema":
        canonical = _canonical_family(family)
        from benchmark.families import family_predicate_grammar

        try:
            grammar = family_predicate_grammar(canonical)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"unknown family: {family}") from exc
        if not grammar or not grammar.get("features"):
            raise ValueError(f"family {family!r} has no public predicate features")
        for context in calibration_contexts:
            if not isinstance(context, Mapping):
                raise ValueError("calibration records must be a public context mapping")
            if _FORBIDDEN_CALIBRATION_KEYS.intersection(context):
                raise ValueError("calibration context must contain public context only")
            if "context" in context:
                raise ValueError("calibration context must contain public context only")

        raw_features = sorted(
            (dict(feature) for feature in grammar["features"]),
            key=lambda feature: str(feature["path"]),
        )
        fitted: list[dict[str, Any]] = []
        width = 0
        for raw in raw_features:
            path = str(raw["path"])
            kind = str(raw["type"]).lower()
            values = [_get_path(context, path)[0] for context in calibration_contexts if _get_path(context, path)[1]]
            if kind == "numeric":
                numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))]
                if not numeric:
                    center, scale = 0.0, 1.0
                else:
                    center = sum(numeric) / len(numeric)
                    variance = sum((value - center) ** 2 for value in numeric) / len(numeric)
                    scale = math.sqrt(variance) or 1.0
                fitted.append({"path": path, "type": kind, "mean": center, "std": scale})
                width += 2  # standardized value plus missing indicator
            elif kind in {"categorical", "category", "string"}:
                vocabulary = sorted({str(value) for value in values if value is not None})
                vocabulary.append("<UNK>")
                fitted.append({"path": path, "type": "categorical", "vocabulary": vocabulary})
                width += len(vocabulary) + 1  # one-hot plus missing indicator
            elif kind in {"boolean", "bool"}:
                fitted.append({"path": path, "type": "boolean"})
                width += 2
            else:
                raise ValueError(f"unsupported public feature type: {kind!r}")
        return cls(
            family=canonical,
            paths=tuple(item["path"] for item in fitted),
            features=tuple(fitted),
            output_width=width,
        )

    def transform(self, context: Mapping[str, Any]) -> tuple[float, ...]:
        """Encode only registered public paths; extra fields are ignored."""

        values: list[float] = []
        for feature in self.features:
            raw, present = _get_path(context, feature["path"])
            kind = feature["type"]
            if kind == "numeric":
                valid = present and isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(float(raw))
                values.append((float(raw) - feature["mean"]) / feature["std"] if valid else 0.0)
                values.append(0.0 if valid else 1.0)
            elif kind == "categorical":
                category = str(raw) if present and raw is not None else "<UNK>"
                vocabulary = feature["vocabulary"]
                category = category if category in vocabulary else "<UNK>"
                values.extend(1.0 if item == category else 0.0 for item in vocabulary)
                values.append(0.0 if present and raw is not None else 1.0)
            elif kind == "boolean":
                valid = present and isinstance(raw, bool)
                values.append(1.0 if valid and raw else 0.0)
                values.append(0.0 if valid else 1.0)
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "family": self.family,
            "fit_source": self.fit_source,
            "paths": list(self.paths),
            "features": [dict(feature) for feature in self.features],
            "output_width": self.output_width,
        }


__all__ = ["TypedFeatureSchema"]
