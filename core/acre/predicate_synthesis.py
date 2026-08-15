"""Finite, deterministic predicate language used by Statistical CEGIS."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any, Mapping

from core.predicates import match_predicate

SYNTHESIZER_VERSION = "acre-predicate-synth-v1"


def _lookup(context: Mapping[str, Any], path: str) -> Any:
    value: Any = context
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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

    def _atoms(self, contexts: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        atoms: list[dict[str, Any]] = []
        for feature in self.features:
            path, kind = feature["path"], feature["type"]
            values = [_lookup(context, path) for context in contexts]
            values = [value for value in values if value is not None]
            if kind == "numeric":
                numeric = sorted({float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)})
                thresholds = sorted({(left + right) / 2.0 for left, right in zip(numeric, numeric[1:])})
                for threshold in thresholds:
                    atoms.append({"compare": {path: {"lte": threshold}}})
                    atoms.append({"compare": {path: {"gte": threshold}}})
            elif kind == "version":
                # Version values are treated as ordered strings only at the
                # grammar boundary; the core predicate evaluator keeps exact
                # version equality as its stable compatibility semantics.
                for value in sorted(set(values), key=_key):
                    atoms.append({"version": {path: value}})
            else:
                for value in sorted(set(values), key=_key):
                    atoms.append({"equals": {path: value}})
        return sorted(atoms, key=_key)

    def candidates(
        self,
        contexts: list[Mapping[str, Any]],
        parent_predicate: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Enumerate canonical conjunctions using only observed adjacent midpoints."""
        atoms = self._atoms(contexts)
        if not atoms and parent_predicate is not None:
            return [parent_predicate]
        candidates: list[dict[str, Any]] = []
        for atom in atoms:
            candidates.extend((atom, {"not": atom}))
        for width in range(1, min(self.max_literals, len(atoms)) + 1):
            for selected in itertools.combinations(atoms, width):
                clauses: list[dict[str, Any]] = []
                if parent_predicate is not None:
                    clauses.append(parent_predicate)
                clauses.extend(selected)
                if len(clauses) == 1:
                    candidates.append(clauses[0])
                else:
                    candidates.append({"all": clauses})
                    if parent_predicate is None and width > 1:
                        candidates.append({"any": list(selected)})
        unique = {_key(candidate): candidate for candidate in candidates}
        return [unique[key] for key in sorted(unique)]


@dataclass(frozen=True)
class SynthesisResult:
    status: str
    predicate: dict[str, Any] | None
    certified_counterexamples: tuple[str, ...]
    positive_anchors: tuple[str, ...]
    synthesizer_version: str = SYNTHESIZER_VERSION
    provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "predicate": self.predicate,
            "certified_counterexamples": list(self.certified_counterexamples),
            "positive_anchors": list(self.positive_anchors),
            "synthesizer_version": self.synthesizer_version,
            "provenance": self.provenance,
        }
