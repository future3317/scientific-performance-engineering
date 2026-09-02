"""Provider-free applicability policies over public boundary evidence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Mapping

from core.acre.cegis import BoundaryObservation, StatisticalCEGIS
from core.acre.predicates import PredicateGrammar, predicate_complexity
from core.predicates import match_predicate


def _workload(observation: BoundaryObservation) -> Mapping[str, Any]:
    context = observation.context
    workload = context.get("workload") if isinstance(context, Mapping) else None
    return workload if isinstance(workload, Mapping) else context


def _context_key(observation: BoundaryObservation) -> str:
    return json.dumps(dict(observation.context), sort_keys=True, separators=(",", ":"), default=str)


def _distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    keys = set(left) | set(right)
    distance = 0.0
    for key in keys:
        if key not in left or key not in right:
            distance += 1.0
            continue
        a, b = left[key], right[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
            distance += abs(float(a) - float(b)) / max(1.0, abs(float(a)), abs(float(b)))
        else:
            distance += float(a != b)
    return distance


def _grammar_for(family: str) -> PredicateGrammar:
    from benchmark.families import family_predicate_grammar

    grammar = family_predicate_grammar(family)
    if not grammar:
        raise ValueError(f"family {family!r} has no predicate grammar")
    return PredicateGrammar.from_dict(grammar)


def _shortest_positive_rule(visible: Sequence[BoundaryObservation], family: str) -> dict[str, Any] | None:
    positives = [item for item in visible if item.positive_anchor()]
    if not positives:
        return None
    grammar = _grammar_for(family)
    candidates = grammar.candidates([item.context for item in positives])
    consistent = [
        candidate
        for candidate in candidates
        if all(match_predicate(candidate, item.context) for item in positives)
    ]
    return min(
        consistent,
        key=lambda candidate: (
            predicate_complexity(candidate)["description_length"],
            predicate_complexity(candidate)["depth"],
            predicate_complexity(candidate)["literals"],
            json.dumps(candidate, sort_keys=True, separators=(",", ":")),
        ),
    ) if consistent else None


def predict_applicability(
    policy: str,
    visible: Sequence[BoundaryObservation],
    heldout: Sequence[BoundaryObservation],
    *,
    family: str | None = None,
) -> dict[str, bool]:
    """Return held-out predictions using only visible observations.

    The function intentionally accepts ``BoundaryObservation`` rather than a
    ``BoundaryCase`` so hidden expected labels are not part of the policy API.
    """
    if policy == "always_apply":
        return {item.observation_id: True for item in heldout}

    if policy == "observed_positive_only":
        positive_keys = {_context_key(item) for item in visible if item.positive_anchor()}
        return {_item.observation_id: _context_key(_item) in positive_keys for _item in heldout}

    if policy == "nearest_experience":
        if not visible:
            return {item.observation_id: False for item in heldout}
        return {
            item.observation_id: min(
                visible,
                key=lambda candidate: (_distance(_workload(candidate), _workload(item)), candidate.observation_id),
            ).positive_anchor()
            for item in heldout
        }

    if policy == "untyped_positive_summary":
        positive_tiers = {
            _workload(item).get("difficulty_tier")
            for item in visible
            if item.positive_anchor() and "difficulty_tier" in _workload(item)
        }
        return {
            item.observation_id: _workload(item).get("difficulty_tier") in positive_tiers
            for item in heldout
        }

    if policy == "shortest_without_counterexamples":
        if family is None:
            raise ValueError("family is required for shortest_without_counterexamples")
        predicate = _shortest_positive_rule(visible, family)
        return {
            item.observation_id: bool(predicate and match_predicate(predicate, item.context))
            for item in heldout
        }

    if policy == "acre_full":
        if family is None:
            raise ValueError("family is required for acre_full")
        positives = [item for item in visible if item.positive_anchor()]
        counterexamples = [item for item in visible if item.certified_counterexample()]
        if not positives or not counterexamples:
            return {item.observation_id: False for item in heldout}
        synthesis = StatisticalCEGIS(_grammar_for(family)).synthesize(
            positive=positives,
            counterexamples=counterexamples,
            parent_predicate=None,
        )
        return {
            item.observation_id: bool(synthesis.predicate and match_predicate(synthesis.predicate, item.context))
            for item in heldout
        }

    raise ValueError(f"unknown applicability policy: {policy}")
