"""Research-only cost-sensitive support selection for categorical contexts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
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


def fit_soft_acre(
    calibration: Sequence[Mapping[str, Any]],
    score_records: Mapping[str, Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    risk_weight: float = 1.0,
    complexity_penalty: float = 0.01,
    max_literals: int = 1,
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

    candidates: list[tuple[str, tuple[tuple[str, Any], ...]]] = [
        ("abstain", ()),
        ("global", ()),
    ]
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
            candidates.extend(
                ("selective", tuple(zip(fields, values)))
                for values in product(*levels)
            )

    family_count = len({row[1] for row in rows})
    if family_count == 0:
        raise ValueError("calibration must contain at least one independent family")

    def score_candidate(route: str, literals: tuple[tuple[str, Any], ...]) -> float:
        family_utility: dict[str, list[float]] = {}
        for _, family, context, tau, gate in rows:
            selected = route == "global" or (
                route == "selective" and all(context.get(field) == value for field, value in literals)
            )
            utility = 0.0
            if selected and gate:
                utility = tau - risk_weight * max(-tau, 0.0)
            family_utility.setdefault(family, []).append(utility)
        mean = sum(sum(values) / len(values) for values in family_utility.values()) / family_count
        return mean - complexity_penalty * len(literals)

    ranked = []
    for route, literals in candidates:
        utility = score_candidate(route, literals)
        tie_preference = 2 if route == "abstain" and utility <= 0 else int(route == "global")
        ranked.append((utility, -len(literals), tie_preference, route, literals))
    utility, _, _, route, literals = max(ranked, key=lambda item: (item[0], item[1], item[2], repr(item[4])))
    return SoftAcrePolicy(route, literals, risk_weight, complexity_penalty, utility, family_count)
