"""Finite predicate grammar and auditable AST complexity metrics."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any, Mapping


SYNTHESIZER_VERSION = "acre-predicate-synth-v2"


def _lookup(context: Mapping[str, Any], path: str) -> Any:
    value: Any = context
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def predicate_complexity(predicate: Mapping[str, Any]) -> dict[str, int]:
    """Return deterministic AST depth, literal count, and description length."""
    if not isinstance(predicate, Mapping) or not predicate:
        raise ValueError("predicate must be a non-empty mapping")
    if "not" in predicate:
        child = predicate_complexity(predicate["not"])
        depth, literals = child["depth"] + 1, child["literals"]
    elif "all" in predicate or "any" in predicate:
        key = "all" if "all" in predicate else "any"
        children = predicate[key]
        if not isinstance(children, list) or not children:
            raise ValueError(f"{key} must contain at least one predicate")
        metrics = [predicate_complexity(child) for child in children]
        depth = 1 + max(item["depth"] for item in metrics)
        literals = sum(item["literals"] for item in metrics)
    else:
        reserved = {"equals", "compare", "version"}
        literal_groups = [predicate[key] for key in reserved if key in predicate]
        if not literal_groups and any(not isinstance(key, str) for key in predicate):
            raise ValueError("predicate paths must be strings")
        depth, literals = 1, sum(len(group) if isinstance(group, Mapping) else 1 for group in literal_groups)
        if literals == 0:
            literals = len(predicate)
    return {"depth": depth, "literals": literals, "description_length": len(_key(predicate))}


@dataclass(frozen=True)
class PredicateGrammar:
    schema_version: int
    features: tuple[dict[str, str], ...]
    max_depth: int = 3
    max_literals: int = 4

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PredicateGrammar":
        if value.get("schema_version") != 1:
            raise ValueError("predicate grammar schema_version must be 1")
        raw_features = value.get("features")
        if not isinstance(raw_features, list) or not raw_features:
            raise ValueError("predicate grammar requires features")
        features: list[dict[str, str]] = []
        seen: set[str] = set()
        for feature in raw_features:
            if not isinstance(feature, dict) or feature.get("type") not in {"numeric", "categorical", "version"}:
                raise ValueError("feature type must be numeric, categorical, or version")
            path = feature.get("path")
            if not isinstance(path, str) or not path or path in seen:
                raise ValueError("feature paths must be non-empty and unique")
            seen.add(path)
            features.append({"path": path, "type": str(feature["type"])})
        max_depth = int(value.get("max_depth", 3))
        max_literals = int(value.get("max_literals", 4))
        if not 1 <= max_depth <= 3 or not 1 <= max_literals <= 4:
            raise ValueError("predicate grammar bounds must be max_depth<=3 and max_literals<=4")
        return cls(1, tuple(features), max_depth=max_depth, max_literals=max_literals)

    def _within_bounds(self, predicate: dict[str, Any]) -> bool:
        complexity = predicate_complexity(predicate)
        return complexity["depth"] <= self.max_depth and complexity["literals"] <= self.max_literals

    def _atoms(self, contexts: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        atoms: list[dict[str, Any]] = []
        for feature in self.features:
            path, kind = feature["path"], feature["type"]
            values = [_lookup(context, path) for context in contexts]
            values = [value for value in values if value is not None]
            if kind == "numeric":
                numeric = sorted({float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)})
                for threshold in sorted({(left + right) / 2.0 for left, right in zip(numeric, numeric[1:])}):
                    atoms.extend(({"compare": {path: {"lte": threshold}}}, {"compare": {path: {"gte": threshold}}}))
            elif kind == "version":
                atoms.extend({"version": {path: value}} for value in sorted(set(values), key=_key))
            else:
                atoms.extend({"equals": {path: value}} for value in sorted(set(values), key=_key))
        return sorted(atoms, key=_key)

    def candidates(self, contexts: list[Mapping[str, Any]], parent_predicate: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        atoms = self._atoms(contexts)
        fragments = atoms + [{"not": atom} for atom in atoms]
        candidates: list[dict[str, Any]] = []
        if parent_predicate is not None and self._within_bounds(parent_predicate):
            candidates.extend((parent_predicate, {"not": parent_predicate}))
        for width in range(1, min(self.max_literals, len(fragments)) + 1):
            for selected in itertools.combinations(fragments, width):
                if parent_predicate is not None:
                    candidates.append({"all": [parent_predicate, *selected]})
                    candidates.append({"any": [parent_predicate, *selected]})
                else:
                    candidates.append(selected[0] if width == 1 else {"all": list(selected)})
                    if width > 1:
                        candidates.append({"any": list(selected)})
        unique = {_key(candidate): candidate for candidate in candidates if self._within_bounds(candidate)}
        return [unique[key] for key in sorted(unique)]
