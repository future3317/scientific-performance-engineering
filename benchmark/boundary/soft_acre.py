"""Research-only cost-sensitive support selection for categorical contexts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import sqrt
from statistics import stdev
from typing import Any, Mapping, Sequence

from .policy_eval import Decision


@dataclass(frozen=True)
class SoftAcrePolicy:
    route: str
    literals: tuple[tuple[str, Any], ...]
    risk_weight: float
    complexity_penalty: float
    calibration_utility: float
    calibration_families: int
    switching_standard_error: float = 0.0

    def decide(self, context: Mapping[str, Any]) -> Decision:
        if self.route == "abstain" or context.get("scientific_gate") is not True:
            return Decision("abstain", 0.0, 0.0, "soft_acre_abstain")
        matched = self.route == "global" or all(
            context.get(field) == value for field, value in self.literals
        )
        return Decision(
            "reuse" if matched else "abstain",
            1.0 if matched else 0.0,
            0.0,
            "soft_acre_global" if self.route == "global" else (
                "soft_acre_predicate" if matched else "soft_acre_outside_support"
            ),
        )


@dataclass(frozen=True)
class CandidateSelection:
    name: str
    calibration_utility: float
    baseline_name: str
    switching_standard_error: float


def select_soft_acre_candidate(
    calibration: Sequence[Mapping[str, Any]],
    score_records: Mapping[str, Mapping[str, Any]],
    candidate_masks: Mapping[str, Mapping[str, bool]],
    *,
    risk_weight: float = 1.0,
    complexity_costs: Mapping[str, float] | None = None,
    switching_margin_se: float = 1.0,
) -> CandidateSelection:
    """Select a candidate only when its paired calibration gain clears an SE margin.

    Candidate masks must be produced without calibration outcomes. The calibration
    cohort chooses among them; final outcomes remain outside this function.
    """
    if not calibration:
        raise ValueError("calibration must contain at least one row")
    if risk_weight < 0 or switching_margin_se < 0:
        raise ValueError("risk_weight and switching_margin_se must be non-negative")
    if not {"global", "abstain"}.issubset(candidate_masks):
        raise ValueError("candidate_masks must include global and abstain")

    ids = {str(row["id"]) for row in calibration}
    if len(ids) != len(calibration):
        raise ValueError("calibration IDs must be unique")
    if any(set(mask) != ids for mask in candidate_masks.values()):
        raise ValueError("each candidate mask must cover exactly the calibration IDs")
    costs = complexity_costs or {}
    if any(cost < 0 for cost in costs.values()):
        raise ValueError("complexity costs must be non-negative")

    families = sorted({str(row["family"]) for row in calibration})
    row_utility: dict[str, float] = {}
    for row in calibration:
        identifier = str(row["id"])
        score = score_records.get(identifier)
        if score is None or "tau" not in score or "scientific_gate" not in score:
            raise ValueError(f"missing tau/scientific_gate for {identifier!r}")
        tau = float(score["tau"])
        row_utility[identifier] = (
            tau - risk_weight * max(-tau, 0.0)
            if bool(score["scientific_gate"]) else 0.0
        )

    def utility(name: str) -> float:
        by_family: dict[str, list[float]] = {family: [] for family in families}
        for row in calibration:
            identifier = str(row["id"])
            reward = row_utility[identifier] if candidate_masks[name][identifier] else 0.0
            by_family[str(row["family"])].append(reward)
        mean = sum(sum(values) / len(values) for values in by_family.values()) / len(families)
        return mean - float(costs.get(name, 0.0))

    scores = {name: utility(name) for name in candidate_masks}
    baseline = "global" if scores["global"] > 0.0 else "abstain"
    alternatives = [name for name in candidate_masks if name not in {"global", "abstain"}]
    if not alternatives:
        return CandidateSelection(baseline, scores[baseline], baseline, 0.0)

    best = max(alternatives, key=lambda name: (scores[name], name))
    family_differences: dict[str, list[float]] = {family: [] for family in families}
    row_differences: list[float] = []
    for row in calibration:
        identifier = str(row["id"])
        delta = (
            row_utility[identifier] *
            (int(candidate_masks[best][identifier]) - int(candidate_masks[baseline][identifier]))
        )
        family_differences[str(row["family"])].append(delta)
        row_differences.append(delta)

    independent_differences = (
        [sum(values) / len(values) for values in family_differences.values()]
        if len(families) > 1 else row_differences
    )
    standard_error = (
        stdev(independent_differences) / sqrt(len(independent_differences))
        if len(independent_differences) > 1 else 0.0
    )
    if scores[best] - scores[baseline] > switching_margin_se * standard_error:
        return CandidateSelection(best, scores[best], baseline, standard_error)
    return CandidateSelection(baseline, scores[baseline], baseline, standard_error)


def fit_soft_acre(
    calibration: Sequence[Mapping[str, Any]],
    score_records: Mapping[str, Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    risk_weight: float = 1.0,
    complexity_penalty: float = 0.01,
    max_literals: int = 1,
    switching_margin_se: float = 1.0,
) -> SoftAcrePolicy:
    """Choose global, conjunctive selective, or abstaining reuse on calibration.

    Outcomes are family-balanced. A failed scientific gate always contributes
    zero utility and blocks deployment. Negative transfer receives an
    additional risk penalty; predicate literals incur a fixed complexity cost.
    The result is an experimental routing policy, not an ACRE certificate.
    """
    if not calibration:
        raise ValueError("calibration must contain at least one row")
    if risk_weight < 0 or complexity_penalty < 0:
        raise ValueError("risk_weight and complexity_penalty must be non-negative")
    if max_literals < 1 or max_literals > len(feature_names):
        raise ValueError("max_literals must lie between one and the feature count")

    rows: list[tuple[str, str, Mapping[str, Any], float, bool]] = []
    for row in calibration:
        observation_id = str(row["id"])
        family = str(row.get("family") or "")
        context = row.get("context")
        if not family or not isinstance(context, Mapping):
            raise ValueError("each row requires a family and mapping context")
        score = score_records.get(observation_id)
        if score is None or "tau" not in score or "scientific_gate" not in score:
            raise ValueError(f"missing tau/scientific_gate for {observation_id!r}")
        rows.append((observation_id, family, context, float(score["tau"]), bool(score["scientific_gate"])))

    candidates: list[tuple[str, tuple[tuple[str, Any], ...]]] = [("abstain", ()), ("global", ())]
    levels_by_field = {
        field: sorted(
            {row[2][field] for row in rows if field in row[2] and isinstance(row[2][field], (str, int, bool))},
            key=lambda value: (type(value).__name__, str(value)),
        )
        for field in feature_names
    }
    for literal_count in range(1, max_literals + 1):
        for fields in combinations(feature_names, literal_count):
            levels = [levels_by_field[field] for field in fields]
            candidates.extend(("selective", tuple(zip(fields, values))) for values in product(*levels))

    family_count = len({row[1] for row in rows})
    if family_count == 0:
        raise ValueError("calibration must contain at least one independent family")

    named_candidates: dict[str, tuple[str, tuple[tuple[str, Any], ...]]] = {
        "abstain": ("abstain", ()),
        "global": ("global", ()),
    }
    masks: dict[str, dict[str, bool]] = {
        "abstain": {identifier: False for identifier, *_ in rows},
        "global": {identifier: True for identifier, *_ in rows},
    }
    costs = {"abstain": 0.0, "global": 0.0}
    for index, (candidate_route, literals) in enumerate(candidates[2:]):
        name = f"predicate_{index}"
        named_candidates[name] = (candidate_route, literals)
        masks[name] = {
            identifier: all(context.get(field) == value for field, value in literals)
            for identifier, _, context, _, _ in rows
        }
        costs[name] = complexity_penalty * len(literals)

    selection = select_soft_acre_candidate(
        calibration, score_records, masks, risk_weight=risk_weight,
        complexity_costs=costs, switching_margin_se=switching_margin_se,
    )
    route, literals = named_candidates[selection.name]
    return SoftAcrePolicy(
        route, literals, risk_weight, complexity_penalty,
        selection.calibration_utility, family_count, selection.switching_standard_error,
    )
