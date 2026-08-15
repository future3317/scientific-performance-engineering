"""Small, bounded 2x2 interaction estimator for ACRE experiments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping


_ARMS = ("00", "10", "01", "11")


@dataclass(frozen=True)
class FactorialBlock:
    """One complete randomized block with utility values in ``[-1, 1]``."""

    block_id: str
    outcomes: Mapping[str, float]

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

    @property
    def normalized_interaction(self) -> float:
        values = self.outcomes
        return (values["11"] - values["10"] - values["01"] + values["00"]) / 4.0


@dataclass(frozen=True)
class FactorialEstimate:
    interaction: float
    ci_low: float
    ci_high: float
    classification: str
    main_effect_a: float
    main_effect_b: float
    joint_effect: float
    blocks: int


@dataclass(frozen=True)
class CoverageResult:
    true_interaction: float
    coverage: float
    covered: int
    repetitions: int


def _time_uniform_radius(n: int, delta: float) -> float:
    # Allocate delta over all future sample counts, so every inspected
    # endpoint remains covered without a fixed stopping horizon.
    delta_n = delta / (n * (n + 1))
    return math.sqrt(2.0 * math.log(2.0 / delta_n) / n)


class FactorialEngine:
    def __init__(self, *, delta: float = 0.05, practical_margin: float = 0.05) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if practical_margin < 0.0 or practical_margin > 1.0:
            raise ValueError("practical_margin must be in [0, 1]")
        self.delta = delta
        self.practical_margin = practical_margin
        self._blocks: list[FactorialBlock] = []

    def add_block(self, block: FactorialBlock) -> None:
        if any(existing.block_id == block.block_id for existing in self._blocks):
            raise ValueError(f"duplicate factorial block: {block.block_id}")
        self._blocks.append(block)

    def estimate(self) -> FactorialEstimate:
        if not self._blocks:
            raise ValueError("at least one factorial block is required")
        n = len(self._blocks)
        values = {arm: [float(block.outcomes[arm]) for block in self._blocks] for arm in _ARMS}
        means = {arm: sum(samples) / n for arm, samples in values.items()}
        interaction_samples = [block.normalized_interaction for block in self._blocks]
        interaction = sum(interaction_samples) / n
        radius = min(1.0, _time_uniform_radius(n, self.delta))
        ci_low, ci_high = max(-1.0, interaction - radius), min(1.0, interaction + radius)
        main_a = means["10"] - means["00"]
        main_b = means["01"] - means["00"]
        joint = means["11"] - means["00"]
        # A prerequisite has a beneficial joint effect while one arm alone
        # has no detectable effect.  Check this before the generic interaction
        # labels because prerequisites necessarily create positive contrast.
        prerequisite = joint > self.practical_margin and (main_a <= self.practical_margin or main_b <= self.practical_margin)
        if prerequisite:
            classification = "prerequisite"
        # The categorical estimate is a descriptive effect label; the
        # interval remains the separately reported evidence gate.  This lets
        # a complete block identify a large interaction while callers can
        # require the CI to clear a margin before promotion.
        elif interaction > self.practical_margin:
            classification = "synergy"
        elif interaction < -self.practical_margin:
            classification = "antagonism"
        else:
            classification = "independence"
        return FactorialEstimate(
            interaction=interaction,
            ci_low=ci_low,
            ci_high=ci_high,
            classification=classification,
            main_effect_a=main_a,
            main_effect_b=main_b,
            joint_effect=joint,
            blocks=n,
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
        if estimate.ci_low <= true_interaction <= estimate.ci_high:
            covered += 1
    return CoverageResult(true_interaction, covered / repetitions, covered, repetitions)
