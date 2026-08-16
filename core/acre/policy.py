"""Canonical confidence-gated relation decision policy."""

from __future__ import annotations

from typing import Mapping


class RelationDecisionPolicy:
    """Map independently estimated contrast confidence sets to one relation."""

    def __init__(self, practical_margin: float = 0.05, equivalence_margin: float | None = None) -> None:
        if not 0.0 <= practical_margin <= 1.0:
            raise ValueError("practical_margin must be in [0, 1]")
        self.practical_margin = practical_margin
        self.equivalence_margin = practical_margin if equivalence_margin is None else equivalence_margin

    def decide(
        self,
        intervals: Mapping[str, tuple[float, float]],
        scientific: Mapping[str, bool],
        *,
        kind_hint: str | None = None,
    ) -> str:
        required = {"gamma"}
        if kind_hint == "prerequisite":
            required |= {"delta_a_given_b0", "delta_a_given_b1", "delta_b_given_a0", "delta_b_given_a1"}
        elif kind_hint == "redundancy":
            required |= {"delta_a_given_b0", "delta_b_given_a0", "redundancy"}
        elif kind_hint is None:
            required |= {"delta_a_given_b0", "delta_a_given_b1", "delta_b_given_a0", "delta_b_given_a1", "redundancy"}
        if not required.issubset(intervals):
            return "unresolved"
        margin = self.practical_margin
        gamma_lcb, gamma_ucb = intervals["gamma"]

        def positive(name: str) -> bool:
            return intervals[name][0] > margin

        def null(name: str) -> bool:
            lower, upper = intervals[name]
            return lower >= -margin and upper <= margin

        if scientific.get("11") is False and all(scientific.get(arm, False) for arm in ("00", "10", "01")):
            return "semantic_conflict"
        if kind_hint == "redundancy":
            if positive("delta_a_given_b0") and positive("delta_b_given_a0"):
                lower, upper = intervals["redundancy"]
                if lower >= -self.equivalence_margin and upper <= self.equivalence_margin:
                    return "confirmed_redundancy"
            return "unresolved"
        if gamma_lcb > margin:
            if {"delta_b_given_a1", "delta_b_given_a0"}.issubset(intervals) and positive("delta_b_given_a1") and null("delta_b_given_a0"):
                return "prerequisite_a_to_b"
            if {"delta_a_given_b1", "delta_a_given_b0"}.issubset(intervals) and positive("delta_a_given_b1") and null("delta_a_given_b0"):
                return "prerequisite_b_to_a"
            return "confirmed_synergy"
        if gamma_ucb < -margin:
            return "confirmed_antagonism"
        if gamma_lcb >= -margin and gamma_ucb <= margin:
            return "confirmed_independence"
        return "unresolved"
