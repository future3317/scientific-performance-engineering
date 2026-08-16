"""Small, bounded 2x2 interaction estimator for ACRE experiments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Mapping
from .budget import StatisticalBudget


_ARMS = ("00", "10", "01", "11")
_THREE_WAY_ARMS = tuple(f"{a}{b}{c}" for a in (0, 1) for b in (0, 1) for c in (0, 1))
CANONICAL_RELATIONS = (
    "unresolved", "confirmed_synergy", "confirmed_antagonism", "confirmed_independence",
    "prerequisite_a_to_b", "prerequisite_b_to_a", "confirmed_redundancy", "context_dependent_relation", "semantic_conflict",
)


@dataclass(frozen=True)
class RelationEvidenceCertificate:
    """Typed factorial evidence required before relation promotion."""

    contrast_cs: Mapping[str, Mapping[str, float]]
    alpha_budget: float
    look_schedule: tuple[int, ...]
    scientific_arm_gates: Mapping[str, bool]
    applicability_provenance: Mapping[str, object]
    endpoint_versions: Mapping[str, int]

    def __post_init__(self) -> None:
        if not 0.0 < float(self.alpha_budget) < 1.0 or not self.look_schedule:
            raise ValueError("relation evidence certificate needs alpha budget and look schedule")
        if set(self.scientific_arm_gates) != set(_ARMS) or not all(isinstance(value, bool) for value in self.scientific_arm_gates.values()):
            raise ValueError("relation certificate must gate all factorial arms")
        if not self.endpoint_versions or any(int(version) < 1 for version in self.endpoint_versions.values()):
            raise ValueError("relation certificate endpoint versions are required")

    def to_dict(self) -> dict[str, object]:
        return {
            "contrast_cs": dict(self.contrast_cs), "alpha_budget": self.alpha_budget,
            "look_schedule": list(self.look_schedule), "scientific_arm_gates": dict(self.scientific_arm_gates),
            "applicability_provenance": dict(self.applicability_provenance), "endpoint_versions": dict(self.endpoint_versions),
        }

    def validate_for(self, relation_spec: object, endpoint_states: Mapping[str, object]) -> None:
        """Validate that a certificate actually supports its typed relation."""
        endpoints = getattr(relation_spec, "endpoints", {})
        if not isinstance(endpoints, Mapping) or set(endpoints) != {"left", "right"}:
            raise ValueError("relation certificate endpoints are invalid")
        expected_versions = {}
        for rule_id, state in endpoint_states.items():
            version = getattr(state, "version", None)
            if version is None and isinstance(state, Mapping):
                version = state.get("version")
            if version is not None:
                expected_versions[str(rule_id)] = int(version)
        if any(self.endpoint_versions.get(rule_id) != expected_versions.get(rule_id) for rule_id in endpoints.values()):
            raise ValueError("relation certificate endpoint versions do not match current endpoints")
        if not self.applicability_provenance.get("source"):
            raise ValueError("relation certificate applicability provenance is required")
        kind = str(getattr(relation_spec, "kind", ""))
        if kind == "semantic_conflict":
            if not (
                self.scientific_arm_gates.get("00") is True
                and self.scientific_arm_gates.get("10") is True
                and self.scientific_arm_gates.get("01") is True
                and self.scientific_arm_gates.get("11") is False
            ):
                raise ValueError("semantic_conflict requires only the joint arm to fail")
        elif not all(self.scientific_arm_gates.values()):
            raise ValueError("relation certificate scientific arm gates must pass")
        required = {"gamma"}
        if kind == "prerequisite":
            required |= {"delta_a_given_b0", "delta_a_given_b1", "delta_b_given_a0", "delta_b_given_a1"}
        if kind == "redundancy":
            required.add("redundancy")
        missing = required - set(self.contrast_cs)
        if missing:
            raise ValueError("relation certificate missing contrasts: " + ", ".join(sorted(missing)))
        for name, interval in self.contrast_cs.items():
            if not isinstance(interval, Mapping) or not {"lcb", "ucb"}.issubset(interval):
                raise ValueError(f"relation certificate contrast {name} needs lcb and ucb")
            lcb, ucb = float(interval["lcb"]), float(interval["ucb"])
            if not math.isfinite(lcb) or not math.isfinite(ucb) or lcb > ucb or lcb < -1.0 or ucb > 1.0:
                raise ValueError(f"relation certificate contrast {name} interval is invalid")
        from .policy import RelationDecisionPolicy
        intervals = {
            name: (float(value["lcb"]), float(value["ucb"]))
            for name, value in self.contrast_cs.items()
        }
        policy = RelationDecisionPolicy(float(getattr(relation_spec, "practical_margin", 0.05)))
        kind = str(getattr(relation_spec, "kind", ""))
        if kind == "context_dependent_interaction":
            raise ValueError("context-dependent relations require cross-context evidence")
        decision = policy.decide(intervals, self.scientific_arm_gates, kind_hint=kind)
        expected = {
            "synergy": "confirmed_synergy",
            "antagonism": "confirmed_antagonism",
            "independence": "confirmed_independence",
            "redundancy": "confirmed_redundancy",
            "semantic_conflict": "semantic_conflict",
        }.get(kind)
        if kind == "prerequisite":
            orientation = str(getattr(relation_spec, "orientation", ""))
            expected = {
                "left_to_right": "prerequisite_a_to_b",
                "right_to_left": "prerequisite_b_to_a",
            }.get(orientation)
            if expected is None:
                raise ValueError("prerequisite relation certificate requires directed orientation")
        if expected is None or decision != expected:
            raise ValueError(f"relation certificate does not support declared relation kind: {kind} ({decision})")


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
    raw_contrasts: Mapping[str, float] = field(default_factory=dict)


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
class HigherOrderCertificate:
    """Typed certificate for a three-rule bundle frontier."""

    bundle_versions: Mapping[str, int]
    context_predicate: Mapping[str, object]
    regime_digest: str
    residual_lcb: float
    residual_ucb: float
    normalized_residual: float
    raw_residual: float
    status: str
    scientific_arm_gates: Mapping[str, bool] = field(default_factory=lambda: {arm: True for arm in _THREE_WAY_ARMS})
    estimator_version: str = "higher-order-cs-v1"

    def __post_init__(self) -> None:
        if len(self.bundle_versions) != 3 or any(int(version) < 1 for version in self.bundle_versions.values()):
            raise ValueError("higher-order certificates require exactly three versioned rules")
        if self.status not in {"pairwise_certified", "hyperedge_required", "unresolved"}:
            raise ValueError("invalid higher-order certificate status")
        if not -1.0 <= self.residual_lcb <= self.residual_ucb <= 1.0:
            raise ValueError("higher-order residual interval must be normalized and bounded")
        if set(self.scientific_arm_gates) != set(_THREE_WAY_ARMS) or any(not isinstance(value, bool) for value in self.scientific_arm_gates.values()):
            raise ValueError("higher-order certificates require all eight scientific arm gates")
        if not all(self.scientific_arm_gates.values()) and self.status == "pairwise_certified":
            raise ValueError("scientific failure cannot be pairwise_certified")

    def to_dict(self) -> dict[str, object]:
        return {
            "certificate_type": "higher_order",
            "bundle_ids": sorted(self.bundle_versions),
            "bundle_versions": dict(self.bundle_versions),
            "context_predicate": dict(self.context_predicate),
            "regime_digest": self.regime_digest,
            "residual_lcb": self.residual_lcb,
            "residual_ucb": self.residual_ucb,
            "normalized_residual": self.normalized_residual,
            "raw_residual": self.raw_residual,
            "status": self.status,
            "scientific_arm_gates": dict(self.scientific_arm_gates),
            "estimator_version": self.estimator_version,
        }


@dataclass(frozen=True)
class ThreeWayBlock:
    block_id: str
    outcomes: Mapping[str, float]
    scientific_gates: Mapping[str, bool] = field(default_factory=lambda: {arm: True for arm in _THREE_WAY_ARMS})

    def __post_init__(self) -> None:
        if set(self.outcomes) != set(_THREE_WAY_ARMS):
            raise ValueError("a three-way block must contain all 2^3 arms")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not -1.0 <= float(value) <= 1.0 for value in self.outcomes.values()):
            raise ValueError("three-way outcomes must be finite and bounded in [-1, 1]")
        if set(self.scientific_gates) != set(_THREE_WAY_ARMS) or any(not isinstance(value, bool) for value in self.scientific_gates.values()):
            raise ValueError("three-way scientific_gates must cover all arms")


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
    # Formal certificates use the bounded Hoeffding radius only.  The empirical
    # Bernstein estimate is intentionally not mixed into the coverage-critical
    # interval until a separate joint-coverage result is available.
    radius = _bounded_radius(n, look_delta)
    lcb, ucb = max(-1.0, residual - radius), min(1.0, residual + radius)
    all_scientific = all(all(block.scientific_gates.values()) for block in blocks)
    if not all_scientific:
        status = "unresolved"
    elif lcb > practical_margin or ucb < -practical_margin:
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
    def __init__(self, *, delta: float = 0.05, practical_margin: float = 0.05, look_count: int = 1, statistical_budget: StatisticalBudget | None = None) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if practical_margin < 0.0 or practical_margin > 1.0:
            raise ValueError("practical_margin must be in [0, 1]")
        self.delta = delta
        self.statistical_budget = statistical_budget or StatisticalBudget(delta_total=float(delta))
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
        for contrast_index, (name, samples) in enumerate(contrast_samples.items(), start=1):
            mean = sum(samples) / n
            contrast_delta = self.statistical_budget.group_delta(contrast_index) / self.look_count
            radius = _bounded_radius(n, contrast_delta)
            # Every decision contrast is normalized to [-1, 1].  In
            # particular, conditional effects and redundancy are divided by
            # two before their confidence sets are compared with the same
            # practical margin as gamma.  Raw effects remain available in
            # ``raw_contrasts`` for reporting.
            contrast_intervals[name] = (max(-1.0, mean - radius), min(1.0, mean + radius))
        gamma_lcb, gamma_ucb = contrast_intervals["gamma"]
        # Arm intervals are retained for redundancy diagnostics.  They are
        # separate from the decision contrasts above.
        arm_delta = self.statistical_budget.validation / (self.look_count * len(_ARMS))
        utility_intervals = {}
        for arm, samples in values.items():
            mean = means[arm]
            radius = min(1.0, _bounded_radius(n, arm_delta))
            utility_intervals[arm] = (max(-1.0, mean - radius), min(1.0, mean + radius))
        raw_contrasts = {
            "gamma": gamma * 4.0,
            "delta_a_given_b0": means["10"] - means["00"],
            "delta_a_given_b1": means["11"] - means["01"],
            "delta_b_given_a0": means["01"] - means["00"],
            "delta_b_given_a1": means["11"] - means["10"],
            "redundancy": means["11"] - max(means["10"], means["01"]),
        }
        delta_a_b0 = raw_contrasts["delta_a_given_b0"] / 2.0
        delta_a_b1 = raw_contrasts["delta_a_given_b1"] / 2.0
        delta_b_a0 = raw_contrasts["delta_b_given_a0"] / 2.0
        delta_b_a1 = raw_contrasts["delta_b_given_a1"] / 2.0
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
            raw_contrasts=raw_contrasts,
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
