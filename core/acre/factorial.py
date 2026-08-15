"""Small, bounded 2x2 interaction estimator for ACRE experiments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Mapping


_ARMS = ("00", "10", "01", "11")
_THREE_WAY_ARMS = tuple(f"{a}{b}{c}" for a in (0, 1) for b in (0, 1) for c in (0, 1))
CANONICAL_RELATIONS = (
    "unresolved", "confirmed_synergy", "confirmed_antagonism", "confirmed_independence",
    "prerequisite_a_to_b", "prerequisite_b_to_a", "confirmed_redundancy", "context_dependent_relation", "semantic_conflict",
)


def canonical_relation_label(value: str) -> str:
    """Normalize estimator labels and typed relation-kind labels for reports."""
    aliases = {
        "synergy": "confirmed_synergy",
        "antagonism": "confirmed_antagonism",
        "independence": "confirmed_independence",
        "redundancy": "confirmed_redundancy",
        "context_dependent_interaction": "context_dependent_relation",
    }
    return aliases.get(value, value)


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
    utility_intervals: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    contrast_intervals: Mapping[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class CoverageResult:
    true_interaction: float
    coverage: float
    covered: int
    repetitions: int


@dataclass(frozen=True)
class HigherOrderEstimate:
    residual: float
    residual_lcb: float
    residual_ucb: float
    blocks: int
    status: str
    raw_residual: float = 0.0
    normalized_residual: float = 0.0


@dataclass(frozen=True)
class ThreeWayBlock:
    block_id: str
    outcomes: Mapping[str, float]

    def __post_init__(self) -> None:
        if set(self.outcomes) != set(_THREE_WAY_ARMS):
            raise ValueError("a three-way block must contain all 2^3 arms")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not -1.0 <= float(value) <= 1.0 for value in self.outcomes.values()):
            raise ValueError("three-way outcomes must be finite and bounded in [-1, 1]")


def estimate_higher_order(blocks: list[ThreeWayBlock], *, delta: float = 0.05, look_count: int = 1, practical_margin: float = 0.05) -> HigherOrderEstimate:
    if not blocks:
        raise ValueError("at least one three-way block is required")
    if not 0.0 < delta < 1.0 or look_count < 1:
        raise ValueError("invalid confidence configuration")
    residuals = []
    for block in blocks:
        u = block.outcomes
        residuals.append(u["111"] - u["110"] - u["101"] - u["011"] + u["100"] + u["010"] + u["001"] - u["000"])
    n = len(residuals)
    raw_residual = sum(residuals) / n
    # Inclusion-exclusion of eight utilities in [-1, 1] is bounded by 8.
    # Normalize before constructing a bounded confidence sequence.
    normalized_samples = [value / 8.0 for value in residuals]
    residual = sum(normalized_samples) / n
    look_delta = delta / look_count
    variance = sum((value - residual) ** 2 for value in normalized_samples) / max(1, n - 1)
    log_term = math.log(3.0 / look_delta)
    radius = min(_bounded_radius(n, look_delta), math.sqrt(2.0 * variance * log_term / n) + 3.0 * log_term / n)
    lcb, ucb = max(-1.0, residual - radius), min(1.0, residual + radius)
    if lcb > practical_margin or ucb < -practical_margin:
        status = "confirmed_nonzero"
    elif lcb >= -practical_margin and ucb <= practical_margin:
        status = "confirmed_negligible"
    else:
        status = "unresolved"
    return HigherOrderEstimate(residual, lcb, ucb, n, status, raw_residual=raw_residual, normalized_residual=residual)


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
        contrast_samples = {
            "gamma": interaction_samples,
            "delta_a_given_b0": [(block.outcomes["10"] - block.outcomes["00"]) / 2.0 for block in self._blocks],
            "delta_a_given_b1": [(block.outcomes["11"] - block.outcomes["01"]) / 2.0 for block in self._blocks],
            "delta_b_given_a0": [(block.outcomes["01"] - block.outcomes["00"]) / 2.0 for block in self._blocks],
            "delta_b_given_a1": [(block.outcomes["11"] - block.outcomes["10"]) / 2.0 for block in self._blocks],
            "redundancy": [(block.outcomes["11"] - max(block.outcomes["10"], block.outcomes["01"])) / 2.0 for block in self._blocks],
        }
        contrast_intervals: dict[str, tuple[float, float]] = {}
        contrast_scale = {"gamma": 1.0, "delta_a_given_b0": 2.0, "delta_a_given_b1": 2.0, "delta_b_given_a0": 2.0, "delta_b_given_a1": 2.0, "redundancy": 2.0}
        contrast_delta = self.delta / (self.look_count * len(contrast_samples))
        for name, samples in contrast_samples.items():
            mean = sum(samples) / n
            variance = sum((value - mean) ** 2 for value in samples) / max(1, n - 1)
            log_term = math.log(3.0 / contrast_delta)
            empirical_radius = math.sqrt(2.0 * variance * log_term / n) + 3.0 * log_term / n
            radius = min(1.0, _bounded_radius(n, contrast_delta), empirical_radius)
            scale = contrast_scale[name]
            contrast_intervals[name] = (scale * max(-1.0, mean - radius), scale * min(1.0, mean + radius))
        gamma_lcb, gamma_ucb = contrast_intervals["gamma"]
        # Arm intervals are retained for redundancy diagnostics.  They are
        # separate from the decision contrasts above.
        arm_delta = self.delta / (self.look_count * (len(contrast_samples) + len(_ARMS)))
        utility_intervals = {}
        for arm, samples in values.items():
            mean = means[arm]
            radius = min(1.0, _bounded_radius(n, arm_delta))
            utility_intervals[arm] = (max(-1.0, mean - radius), min(1.0, mean + radius))
        delta_a_b0 = means["10"] - means["00"]
        delta_a_b1 = means["11"] - means["01"]
        delta_b_a0 = means["01"] - means["00"]
        delta_b_a1 = means["11"] - means["10"]
        scientific = {arm: all(block.scientific_gates[arm] for block in self._blocks) for arm in _ARMS}
        estimate = FactorialEstimate(
            gamma=gamma,
            gamma_lcb=gamma_lcb,
            gamma_ucb=gamma_ucb,
            delta_a_given_b0=delta_a_b0,
            delta_a_given_b1=delta_a_b1,
            delta_b_given_a0=delta_b_a0,
            delta_b_given_a1=delta_b_a1,
            decision="unresolved",
            blocks=n,
            scientific_00=scientific["00"],
            scientific_10=scientific["10"],
            scientific_01=scientific["01"],
            scientific_11=scientific["11"],
            utility_intervals=utility_intervals,
            contrast_intervals=contrast_intervals,
        )
        # The semantic policy is shared with cross-context RelationIdentifier;
        # the estimator itself only creates arm and contrast confidence sets.
        from .policy import RelationDecisionPolicy

        return replace(estimate, decision=RelationDecisionPolicy(self.practical_margin).decide(contrast_intervals, scientific))


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
