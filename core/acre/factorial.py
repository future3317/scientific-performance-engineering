"""Small, bounded 2x2 interaction estimator for ACRE experiments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Mapping


_ARMS = ("00", "10", "01", "11")
CANONICAL_RELATIONS = (
    "unresolved", "confirmed_synergy", "confirmed_antagonism", "confirmed_independence",
    "prerequisite_a_to_b", "prerequisite_b_to_a", "semantic_conflict",
)


@dataclass(frozen=True)
class FactorialBlock:
    """One complete randomized block with utility values in ``[-1, 1]``."""

    block_id: str
    outcomes: Mapping[str, float]
    scientific_gates: Mapping[str, bool] = field(default_factory=lambda: {arm: True for arm in _ARMS})

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("block_id must be non-empty")
        if set(self.outcomes) != set(_ARMS):
            raise ValueError("a factorial block must be complete: 00, 10, 01, 11")
        for value in self.outcomes.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("factorial outcomes must be finite numbers")
            if not -1.0 <= float(value) <= 1.0:
                raise ValueError("factorial outcomes must be bounded in [-1, 1]")
        if set(self.scientific_gates) != set(_ARMS) or any(not isinstance(value, bool) for value in self.scientific_gates.values()):
            raise ValueError("scientific_gates must provide boolean values for all factorial arms")

    @property
    def normalized_interaction(self) -> float:
        values = self.outcomes
        return (values["11"] - values["10"] - values["01"] + values["00"]) / 4.0


@dataclass(frozen=True)
class FactorialEstimate:
    gamma: float
    gamma_lcb: float
    gamma_ucb: float
    delta_a_given_b0: float
    delta_a_given_b1: float
    delta_b_given_a0: float
    delta_b_given_a1: float
    decision: str
    blocks: int
    scientific_00: bool
    scientific_10: bool
    scientific_01: bool
    scientific_11: bool


@dataclass(frozen=True)
class CoverageResult:
    true_interaction: float
    coverage: float
    covered: int
    repetitions: int


def _bounded_radius(n: int, delta: float) -> float:
    """Hoeffding radius for a predeclared, fixed block set."""
    return math.sqrt(2.0 * math.log(2.0 / delta) / n)


def _kl_radius(samples: list[float], delta: float) -> float:
    """KL confidence radius for a bounded mean mapped from [-1,1] to [0,1]."""
    n = len(samples)
    q = (sum(samples) / n + 1.0) / 2.0
    threshold = math.log(2.0 / delta) / n
    eps = 1e-12
    def kl(p: float) -> float:
        p = min(1.0 - eps, max(eps, p)); qq = min(1.0 - eps, max(eps, q))
        return qq * math.log(qq / p) + (1.0 - qq) * math.log((1.0 - qq) / (1.0 - p))
    lo, hi = eps, q
    for _ in range(48):
        mid = (lo + hi) / 2.0
        if kl(mid) > threshold: lo = mid
        else: hi = mid
    lower = hi
    lo, hi = q, 1.0 - eps
    for _ in range(48):
        mid = (lo + hi) / 2.0
        if kl(mid) > threshold: hi = mid
        else: lo = mid
    upper = lo
    return max(q - lower, upper - q)


class FactorialEngine:
    def __init__(self, *, delta: float = 0.05, practical_margin: float = 0.05, look_count: int = 1) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if practical_margin < 0.0 or practical_margin > 1.0:
            raise ValueError("practical_margin must be in [0, 1]")
        self.delta = delta
        self.practical_margin = practical_margin
        if look_count < 1:
            raise ValueError("look_count must be positive")
        self.look_count = look_count
        self._blocks: list[FactorialBlock] = []
        self._block_ids: set[str] = set()

    def add_block(self, block: FactorialBlock) -> None:
        if block.block_id in self._block_ids:
            raise ValueError(f"duplicate factorial block: {block.block_id}")
        self._blocks.append(block)
        self._block_ids.add(block.block_id)

    def estimate(self) -> FactorialEstimate:
        if not self._blocks:
            raise ValueError("at least one factorial block is required")
        n = len(self._blocks)
        values = {arm: [float(block.outcomes[arm]) for block in self._blocks] for arm in _ARMS}
        means = {arm: sum(samples) / n for arm, samples in values.items()}
        interaction_samples = [block.normalized_interaction for block in self._blocks]
        gamma = sum(interaction_samples) / n
        # Sequential callers predeclare the number of looks.  Spending
        # delta/look_count at every look makes adaptive confirmation valid.
        look_delta = self.delta / self.look_count
        variance = sum((value - gamma) ** 2 for value in interaction_samples) / max(1, n - 1)
        log_term = math.log(3.0 / look_delta)
        empirical_radius = math.sqrt(2.0 * variance * log_term / n) + 3.0 * log_term / n
        radius = min(1.0, _bounded_radius(n, look_delta), empirical_radius)
        gamma_lcb, gamma_ucb = max(-1.0, gamma - radius), min(1.0, gamma + radius)
        delta_a_b0 = means["10"] - means["00"]
        delta_a_b1 = means["11"] - means["01"]
        delta_b_a0 = means["01"] - means["00"]
        delta_b_a1 = means["11"] - means["10"]
        margin = self.practical_margin
        scientific = {arm: all(block.scientific_gates[arm] for block in self._blocks) for arm in _ARMS}
        # Use the same bounded radius for the conditional effects.  A
        # prerequisite is directional only when the present-partner effect is
        # confidently above the margin and the absent-partner effect is
        # confidently inside the practical-null interval.
        def confidently_positive(value: float) -> bool:
            return value - radius > margin

        def confidently_null(value: float) -> bool:
            return -margin < value - radius and value + radius < margin

        # Relation decisions are confidence-gated.  Prerequisite direction is
        # identified by a positive conditional effect only after the other
        # intervention is present, while its absent-partner effect stays in
        # the practical null region.
        if scientific["11"] is False and scientific["10"] and scientific["01"] and scientific["00"]:
            decision = "semantic_conflict"
        elif gamma_lcb > margin:
            if confidently_positive(delta_b_a1) and confidently_null(delta_b_a0):
                decision = "prerequisite_a_to_b"
            elif confidently_positive(delta_a_b1) and confidently_null(delta_a_b0):
                decision = "prerequisite_b_to_a"
            else:
                decision = "confirmed_synergy"
        elif gamma_ucb < -margin:
            decision = "confirmed_antagonism"
        elif gamma_lcb >= -margin and gamma_ucb <= margin:
            decision = "confirmed_independence"
        else:
            decision = "unresolved"
        return FactorialEstimate(
            gamma=gamma,
            gamma_lcb=gamma_lcb,
            gamma_ucb=gamma_ucb,
            delta_a_given_b0=delta_a_b0,
            delta_a_given_b1=delta_a_b1,
            delta_b_given_a0=delta_b_a0,
            delta_b_given_a1=delta_b_a1,
            decision=decision,
            blocks=n,
            scientific_00=scientific["00"],
            scientific_10=scientific["10"],
            scientific_01=scientific["01"],
            scientific_11=scientific["11"],
        )


def simulate_coverage(
    *, true_interaction: float, noise: float, blocks: int, repetitions: int, seed: int = 0, delta: float = 0.05
) -> CoverageResult:
    """Run a deterministic bounded-noise coverage check for the CI contract."""
    if blocks < 1 or repetitions < 1 or noise < 0.0:
        raise ValueError("blocks and repetitions must be positive and noise non-negative")
    if not -1.0 <= true_interaction <= 1.0:
        raise ValueError("true_interaction must be in [-1, 1]")
    rng = random.Random(seed)
    covered = 0
    # Keep the latent cells away from the bounds so the bounded perturbation
    # remains a valid utility observation.
    baseline, effect_a, effect_b = -0.2, 0.12, 0.10
    for _ in range(repetitions):
        engine = FactorialEngine(delta=delta)
        for block_index in range(blocks):
            latent = {
                "00": baseline,
                "10": baseline + effect_a,
                "01": baseline + effect_b,
                "11": baseline + effect_a + effect_b + 4.0 * true_interaction,
            }
            outcomes = {arm: value + rng.uniform(-noise, noise) for arm, value in latent.items()}
            engine.add_block(FactorialBlock(f"{block_index}", outcomes))
        estimate = engine.estimate()
        if estimate.gamma_lcb <= true_interaction <= estimate.gamma_ucb:
            covered += 1
    return CoverageResult(true_interaction, covered / repetitions, covered, repetitions)
