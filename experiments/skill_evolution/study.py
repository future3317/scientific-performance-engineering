"""Synthetic qualification study; it does not invoke a model or provider."""

from __future__ import annotations

import math
import random
from statistics import mean, stdev
from typing import Any

from core.acre.resource_budget import PairReservation


PREDEFINED_SEEDS = (17, 41, 83)
REGIMES = ("broad_benefit", "selective_transfer", "null_effect")
CLUSTERS_PER_SEED = 80
TASKS_PER_CLUSTER = 4
ALPHA = 0.05
PAIR_RESERVATION_UNITS = 3.0
POWER_CLUSTER_COUNTS = (20, 40, 80, 160)
POWER_EFFECTS = (0.05, 0.10, 0.20)
POWER_DISCORDANCE_RATE = 0.25

HYPOTHESES = {
    "H1_BROAD": "Under broad positive transfer, global reuse improves paired task success over base.",
    "H2_SELECTIVE": "In selective transfer, a predeclared observable context gate reduces harm versus global reuse while retaining supported-context benefit.",
    "H3_NULL": "Under a null effect, the fixture does not systematically produce a positive paired effect; intervals are reported without a forced pass decision.",
}

PHASE_I_CONFIG = {
    "status": "NOT_RUN",
    "model": "deepseek-flash",
    "thinking": "enabled",
    "max_completion_tokens_per_call": 8192,
    "arms": ("legacy_stateless", "stateful"),
    "split": "train_only",
    "matching_keys": ("task_or_game_id", "environment_seed"),
    "matched_pairs": 6,
    "pairs_per_family": 2,
    "families": ("clean", "heat", "cool"),
    "max_steps": 60,
    "provider_call_cap": 720,
    "token_cap": 2_000_000,
    "retries": 0,
    "verifier": "same deterministic task verifier in both arms; executable readiness pending",
    "budget_status": "proposed_not_approved",
    "budget_note": "upper bounds do not authorize provider execution; pair-level reservation and provider resource approval are still required",
}


def required_fixture_task_execution_budget() -> float:
    return float(
        len(REGIMES) * len(PREDEFINED_SEEDS) * CLUSTERS_PER_SEED
        * TASKS_PER_CLUSTER * PAIR_RESERVATION_UNITS
    )


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def power_table() -> list[dict[str, float | int]]:
    """Normal-approximation planning values, not results from observations."""
    z_critical = 1.959963984540054
    rows = []
    for n_clusters in POWER_CLUSTER_COUNTS:
        for effect in POWER_EFFECTS:
            variance = POWER_DISCORDANCE_RATE - effect * effect
            se = math.sqrt(max(variance, 1e-12) / n_clusters)
            noncentrality = effect / se
            power = _normal_cdf(-z_critical - noncentrality) + 1.0 - _normal_cdf(z_critical - noncentrality)
            rows.append({
                "independent_clusters": n_clusters,
                "paired_effect_assumption": effect,
                "discordant_pair_rate_assumption": POWER_DISCORDANCE_RATE,
                "alpha_two_sided": ALPHA,
                "approximate_power": power,
            })
    return rows


def _regime_effect(regime: str, supported: bool, policy: str) -> float:
    if policy == "base":
        return 0.0
    if regime == "broad_benefit":
        return 0.14 if policy == "global_reuse" else 0.14
    if regime == "selective_transfer":
        if policy == "global_reuse":
            return 0.20 if supported else -0.18
        return 0.20 if supported else 0.0
    return 0.0


def _cluster_interval(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    estimate = mean(values) if values else 0.0
    se = stdev(values) / math.sqrt(n) if n > 1 else 0.0
    return {"estimate": estimate, "lower_95": estimate - 1.96 * se, "upper_95": estimate + 1.96 * se, "independent_clusters": n}


def _simulate_regime(seed: int, regime: str) -> dict[str, Any]:
    rng = random.Random(seed * 1009 + REGIMES.index(regime) * 7919)
    cluster_effects: dict[str, list[float]] = {"global_reuse": [], "context_gate": []}
    global_harm_clusters: list[float] = []
    gate_harm_clusters: list[float] = []
    stratum_effects = {
        policy: {stratum: [] for stratum in ("supported", "unsupported")}
        for policy in cluster_effects
    }
    for cluster in range(CLUSTERS_PER_SEED):
        cluster_shift = rng.uniform(-0.12, 0.12)
        outcomes: dict[str, list[float]] = {"base": [], "global_reuse": [], "context_gate": []}
        harms: dict[str, list[float]] = {"global_reuse": [], "context_gate": []}
        for task in range(TASKS_PER_CLUSTER):
            supported = task < TASKS_PER_CLUSTER // 2
            baseline_probability = min(0.9, max(0.1, 0.45 + cluster_shift + (0.04 if supported else -0.04)))
            shared_draw = rng.random()
            base_success = float(shared_draw < baseline_probability)
            outcomes["base"].append(base_success)
            for policy in ("global_reuse", "context_gate"):
                treatment_probability = min(0.98, max(0.02, baseline_probability + _regime_effect(regime, supported, policy)))
                treatment_success = float(shared_draw < treatment_probability)
                outcomes[policy].append(treatment_success)
                harms[policy].append(float(base_success > treatment_success))
        for policy in cluster_effects:
            cluster_effects[policy].append(mean(outcomes[policy]) - mean(outcomes["base"]))
            for stratum, indices in (
                ("supported", range(TASKS_PER_CLUSTER // 2)),
                ("unsupported", range(TASKS_PER_CLUSTER // 2, TASKS_PER_CLUSTER)),
            ):
                stratum_effects[policy][stratum].append(
                    mean(outcomes[policy][index] - outcomes["base"][index] for index in indices)
                )
        global_harm_clusters.append(mean(harms["global_reuse"]))
        gate_harm_clusters.append(mean(harms["context_gate"]))
    return {
        "regime": regime,
        "seed": seed,
        "statistical_unit": "independent task-family cluster; four matched task pairs per cluster",
        "clusters": CLUSTERS_PER_SEED,
        "paired_task_pairs": CLUSTERS_PER_SEED * TASKS_PER_CLUSTER,
        "contrasts": {
            policy: _cluster_interval(values)
            for policy, values in cluster_effects.items()
        },
        "stratum_contrasts": {
            policy: {stratum: _cluster_interval(values) for stratum, values in strata.items()}
            for policy, strata in stratum_effects.items()
        },
        "negative_transfer_rate": {
            "global_reuse": _cluster_interval(global_harm_clusters),
            "context_gate": _cluster_interval(gate_harm_clusters),
        },
        "synthetic_truth_used_for_generation_only": {
            "supported_contexts": "first two task slots in each cluster",
            "gate_policy": "predeclared context rule; this fixture is an oracle-mechanism ceiling, not learned applicability",
        },
    }


def run_fixture_study(
    *,
    task_execution_budget: float | None = None,
    evolution_search_budget: float = 0.0,
) -> dict[str, Any]:
    """Run fixed-seed stochastic potential-outcome fixtures without providers."""
    budget = required_fixture_task_execution_budget() if task_execution_budget is None else task_execution_budget
    pair_ids = tuple(
        (
            f"{regime}-seed-{seed}-cluster-{cluster:03d}-task-{task:02d}",
            PAIR_RESERVATION_UNITS,
        )
        for regime in REGIMES
        for seed in PREDEFINED_SEEDS
        for cluster in range(CLUSTERS_PER_SEED)
        for task in range(TASKS_PER_CLUSTER)
    )
    reservation = PairReservation.reserve_all_pairs(
        pair_ids,
        task_execution_budget=budget,
        evolution_search_budget=evolution_search_budget,
    )
    if not reservation.launch_allowed:
        return {
            "study_kind": "synthetic_stochastic_qualification_fixture",
            "synthetic_only": True,
            "provider_calls": 0,
            "launch_status": "NOT_LAUNCHED_INCOMPLETE_PAIR_RESERVATION",
            "reservation": reservation.as_dict(),
            "real_phase_i": {"status": "NOT_RUN", "reason": "fixture budget cannot authorize a real study"},
        }

    rows = [_simulate_regime(seed, regime) for regime in REGIMES for seed in PREDEFINED_SEEDS]
    for pair_id, _ in pair_ids:
        reservation = reservation.complete_pair(pair_id, measured_cost=None)
    return {
        "study_kind": "synthetic_stochastic_qualification_fixture",
        "synthetic_only": True,
        "provider_calls": 0,
        "seed_grid": list(PREDEFINED_SEEDS),
        "regime_grid": list(REGIMES),
        "hypotheses": HYPOTHESES,
        "preregistered_design": {
            "alpha_two_sided": ALPHA,
            "task_pairs_per_seed_regime": CLUSTERS_PER_SEED * TASKS_PER_CLUSTER,
            "independent_unit": "task-family cluster",
            "paired_arms": ["base", "global_reuse", "context_gate"],
            "comparison": "cluster mean of within-task treatment-minus-base differences; normal 95% interval across independent clusters",
            "multiplicity": "H1-H3 are descriptive fixture hypotheses; no confirmatory real-study multiplicity claim",
            "missingness": "fixture generation is complete by construction; real-study missing attempts remain nonlaunched/invalid and require sensitivity bounds",
        },
        "results": rows,
        "power_table": {
            "status": "planning_approximation_not_observed_power",
            "method": "two-sided normal approximation for paired binary difference using the declared discordant-pair-rate assumption",
            "rows": power_table(),
        },
        "cost_accounting": {
            "status": "not_measured_for_synthetic_fixture",
            "provider_calls": {"value": 0, "status": "guaranteed_zero_by_fixture_design"},
            "input_tokens": {"value": None, "status": "unknown_not_applicable"},
            "output_tokens": {"value": None, "status": "unknown_not_applicable"},
            "wall_seconds": {"value": None, "status": "unknown_not_measured"},
            "gpu_seconds": {"value": None, "status": "unknown_not_measured"},
        },
        "reservation": {
            **reservation.as_dict(),
            "measurement_note": "costs are unknown; no real task execution occurred",
            "budget_unit_note": "pair reservations are planning units supplied by the fixture, not measured tokens, time, or money",
        },
        "real_phase_i": {
            **PHASE_I_CONFIG,
            "arms": list(PHASE_I_CONFIG["arms"]),
            "matching_keys": list(PHASE_I_CONFIG["matching_keys"]),
            "scope": "matched train-only viability; stateful-vs-legacy is an agent/harness effect, not a Skill effect",
            "readiness_gate": "data and executable verifier available; fixed model/harness configuration; matched task manifest and pair reserve; server resource contract and new evidence identity approved",
        },
        "claim_boundary": "Synthetic stochastic mechanism fixture only; no real-task, provider, hosted, calibration, or efficacy result.",
    }


__all__ = [
    "HYPOTHESES",
    "PHASE_I_CONFIG",
    "PREDEFINED_SEEDS",
    "REGIMES",
    "power_table",
    "required_fixture_task_execution_budget",
    "run_fixture_study",
]
