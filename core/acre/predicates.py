"""Finite predicate grammar and auditable AST complexity metrics."""

from __future__ import annotations

import itertools
import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

from core.predicates import match_predicate


SYNTHESIZER_VERSION = "acre-predicate-synth-v3"


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
class PredicateSearchResult:
    candidates: tuple[dict[str, Any], ...]
    search_exhausted: bool
    candidates_examined: int
    elapsed_seconds: float
    vocabulary_digest: str


@dataclass(frozen=True)
class PredicateGrammar:
    schema_version: int
    features: tuple[dict[str, str], ...]
    max_depth: int = 3
    max_literals: int = 4
    threshold_universe: dict[str, tuple[float, ...]] | None = None
    allow_observed_thresholds: bool = True

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
        raw_thresholds = value.get("threshold_universe", {})
        if raw_thresholds is not None and not isinstance(raw_thresholds, dict):
            raise ValueError("threshold_universe must be a mapping")
        threshold_universe = {
            str(path): tuple(sorted({float(item) for item in values}))
            for path, values in (raw_thresholds or {}).items()
            if isinstance(values, list) and values
        }
        formal = bool(value.get("formal", False) or value.get("mode") == "formal")
        return cls(1, tuple(features), max_depth=max_depth, max_literals=max_literals,
                   threshold_universe=threshold_universe, allow_observed_thresholds=not formal)

    def _within_bounds(self, predicate: dict[str, Any]) -> bool:
        complexity = predicate_complexity(predicate)
        return complexity["depth"] <= self.max_depth and complexity["literals"] <= self.max_literals

    def _atoms(self, contexts: list[Mapping[str, Any]], *, deadline: float | None = None) -> list[dict[str, Any]]:
        atoms: list[dict[str, Any]] = []
        for feature in self.features:
            if deadline is not None and time.monotonic() >= deadline:
                break
            path, kind = feature["path"], feature["type"]
            values = [_lookup(context, path) for context in contexts]
            values = [value for value in values if value is not None]
            if kind == "numeric":
                configured = (self.threshold_universe or {}).get(path)
                numeric = sorted({float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)})
                if configured is None and not self.allow_observed_thresholds:
                    raise ValueError(f"formal predicate grammar requires frozen thresholds for {path}")
                thresholds = configured if configured else sorted({(left + right) / 2.0 for left, right in zip(numeric, numeric[1:])})
                for threshold in thresholds:
                    if deadline is not None and time.monotonic() >= deadline:
                        return sorted(atoms, key=_key)
                    atoms.extend(({"compare": {path: {"lte": threshold}}}, {"compare": {path: {"gte": threshold}}}))
            elif kind == "version":
                atoms.extend({"version": {path: value}} for value in sorted(set(values), key=_key))
            else:
                atoms.extend({"equals": {path: value}} for value in sorted(set(values), key=_key))
        return sorted(atoms, key=_key)

    def search_candidates(
        self,
        contexts: list[Mapping[str, Any]],
        parent_predicate: dict[str, Any] | None = None,
        *,
        max_candidates: int | None = None,
        max_wall_time: float | None = 30.0,
        comparison_contexts: list[Mapping[str, Any]] | None = None,
        preserve_distinct: bool = False,
    ) -> PredicateSearchResult:
        if max_candidates is not None and max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if max_wall_time is not None and max_wall_time <= 0:
            raise ValueError("max_wall_time must be positive")
        started = time.monotonic()

        def expired() -> bool:
            return max_wall_time is not None and time.monotonic() - started >= max_wall_time

        deadline = started + max_wall_time if max_wall_time is not None else None
        atoms = self._atoms(contexts, deadline=deadline)
        if expired():
            return PredicateSearchResult((), True, 0, time.monotonic() - started, "")
        fragments = atoms + [{"not": atom} for atom in atoms]
        if max_candidates is None:
            # Bound default work by the grammar's complete combination count;
            # explicit caller caps remain unchanged.
            max_candidates = sum(
                math.comb(len(fragments), width)
                for width in range(1, min(self.max_literals, len(fragments)) + 1)
            )
        signature_contexts = contexts + list(comparison_contexts or [])
        context_masks = []
        for fragment in fragments:
            if expired():
                return PredicateSearchResult((), True, 0, time.monotonic() - started, "")
            context_masks.append(sum(1 << index for index, context in enumerate(signature_contexts) if match_predicate(fragment, context)))
        full_mask = (1 << len(signature_contexts)) - 1
        fragment_keys = [_key(fragment) for fragment in fragments]
        fragment_complexities = []
        for fragment in fragments:
            if expired():
                return PredicateSearchResult((), True, 0, time.monotonic() - started, "")
            fragment_complexities.append(predicate_complexity(fragment))
        representatives: dict[Any, tuple[tuple[int, int, int, str], dict[str, Any]]] = {}
        vocabulary_hasher = hashlib.sha256()

        def add(candidate: dict[str, Any], signature: int, candidate_key: str, depth: int, literals: int) -> None:
            if depth > self.max_depth or literals > self.max_literals:
                return
            vocabulary_hasher.update(candidate_key.encode("utf-8") + b"\n")
            priority = (len(candidate_key), depth, literals, candidate_key)
            representative_key: Any = candidate_key if preserve_distinct else signature
            current = representatives.get(representative_key)
            if current is None or priority < current[0]:
                representatives[representative_key] = (priority, candidate)

        if parent_predicate is not None:
            parent_key = _key(parent_predicate)
            parent_complexity = predicate_complexity(parent_predicate)
            parent_mask = sum(
                1 << index for index, context in enumerate(signature_contexts)
                if match_predicate(parent_predicate, context)
            )
            # A specialization may preserve the parent or narrow it; it must
            # never introduce a complement/OR branch that broadens coverage.
            add(parent_predicate, parent_mask, parent_key, parent_complexity["depth"], parent_complexity["literals"])
        else:
            parent_key = None
            parent_complexity = None
            parent_mask = None
        examined = 0
        search_exhausted = expired()
        for width in range(1, min(self.max_literals, len(fragments)) + 1):
            if search_exhausted:
                break
            for selected_indices in itertools.combinations(range(len(fragments)), width):
                if expired() or (max_candidates is not None and examined >= max_candidates):
                    search_exhausted = True
                    break
                examined += 1
                selected = tuple(fragments[index] for index in selected_indices)
                selected_keys = tuple(fragment_keys[index] for index in selected_indices)
                all_mask = full_mask
                any_mask = 0
                selected_depth = 1
                for index in selected_indices:
                    all_mask &= context_masks[index]
                    any_mask |= context_masks[index]
                    selected_depth = max(selected_depth, fragment_complexities[index]["depth"])
                if parent_predicate is not None:
                    all_key = '{"all":[' + parent_key + ',' + ','.join(selected_keys) + ']}'
                    all_depth = 1 + max(parent_complexity["depth"], selected_depth)
                    add({"all": [parent_predicate, *selected]}, parent_mask & all_mask, all_key, all_depth, parent_complexity["literals"] + width)
                    if width > 1:
                        # A disjunction is safe only inside the parent
                        # predicate: parent AND (a OR b) is still a true
                        # specialization, whereas (parent OR a) is not.
                        any_key = '{"any":[' + ','.join(selected_keys) + ']}'
                        nested_all_key = '{"all":[' + parent_key + ',' + any_key + ']}'
                        nested_all_depth = 1 + max(parent_complexity["depth"], 1 + selected_depth)
                        add({"all": [parent_predicate, {"any": list(selected)}]}, parent_mask & any_mask, nested_all_key, nested_all_depth, parent_complexity["literals"] + width)
                        one_key = '{"all":[' + parent_key + ',' + selected_keys[0] + ']}'
                        one_depth = 2 + max(parent_complexity["depth"], fragment_complexities[selected_indices[0]]["depth"])
                        add({"any": [{"all": [parent_predicate, selected[0]]}]}, parent_mask & context_masks[selected_indices[0]], '{"any":[' + one_key + ']}', one_depth, parent_complexity["literals"] + 1)
                else:
                    if width == 1:
                        add(selected[0], all_mask, selected_keys[0], fragment_complexities[selected_indices[0]]["depth"], 1)
                    else:
                        add({"all": list(selected)}, all_mask, '{"all":[' + ','.join(selected_keys) + ']}', 1 + selected_depth, width)
                    if width > 1:
                        add({"any": list(selected)}, any_mask, '{"any":[' + ','.join(selected_keys) + ']}', 1 + selected_depth, width)
        search_exhausted = search_exhausted or expired()
        return PredicateSearchResult(
            candidates=tuple(item[1] for item in sorted(representatives.values(), key=lambda item: item[0][3])),
            search_exhausted=search_exhausted,
            candidates_examined=examined,
            elapsed_seconds=time.monotonic() - started,
            vocabulary_digest=vocabulary_hasher.hexdigest(),
        )

    def candidates(
        self,
        contexts: list[Mapping[str, Any]],
        parent_predicate: dict[str, Any] | None = None,
        *,
        max_candidates: int | None = None,
        max_wall_time: float | None = None,
    ) -> list[dict[str, Any]]:
        """Enumerate exactly for existing inspection callers, or honor explicit caps.

        Production synthesis uses ``search_candidates`` with mandatory limits.
        This original exact-enumeration API never returns a truncated set.
        """
        result = self.search_candidates(
            contexts,
            parent_predicate,
            max_candidates=max_candidates,
            max_wall_time=max_wall_time,
        )
        if result.search_exhausted:
            raise RuntimeError("predicate search exhausted its configured budget")
        return list(result.candidates)
