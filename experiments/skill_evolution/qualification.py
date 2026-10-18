"""Train/evaluation synthetic qualification through the current ACRE APIs."""

from __future__ import annotations

import json
import math
import random
import time
import tracemalloc
from statistics import mean, stdev
from typing import Any

from core.acre.cegis import BoundaryObservation, StatisticalCEGIS, synthesis_certificate_valid
from core.acre.predicates import PredicateGrammar
from core.acre.regimes import SupportGroup, SupportScope, classify_support_regime
from core.acre.typed_gate import decision_from_synthesis
from core.acre.engine import AcreEngine
from core.acre.router import _semantic_spec_digest
from core.models import RuleSpec, RuleState
from core.sequential_stats import paired_repetition_interval
from core.acre.resource_budget import PairReservation
from .study import PHASE_I_CONFIG, power_table


TRAIN_SEEDS = (101, 211)
EVALUATION_SEEDS = (307, 401)
REGIME_GRID = ("broad_benefit", "selective_transfer", "null_effect")
STRATA = ("supported", "unsupported")
GROUPS_PER_STRATUM_PER_SEED = 60
FAMILY_ALPHA = 0.05
MEASUREMENT_ALPHA = 0.025
REGIME_ALPHA = 0.0125
CEGIS_ALPHA = 0.0125
TRAINING_GROUP_COUNT = (
    len(REGIME_GRID) * len(TRAIN_SEEDS) * len(STRATA) * GROUPS_PER_STRATUM_PER_SEED
)
PER_GROUP_MEASUREMENT_ALPHA = MEASUREMENT_ALPHA / TRAINING_GROUP_COUNT
PLANNED_MEASUREMENT_RADIUS_TARGET = 0.15
MIN_REPETITIONS_FOR_RADIUS = math.ceil(
    2.0 * math.log(2.0 / PER_GROUP_MEASUREMENT_ALPHA)
    / PLANNED_MEASUREMENT_RADIUS_TARGET**2
)
REPETITIONS_PER_GROUP = 1 << (MIN_REPETITIONS_FOR_RADIUS - 1).bit_length()
PLANNED_MEASUREMENT_RADIUS = math.sqrt(
    2.0 * math.log(2.0 / PER_GROUP_MEASUREMENT_ALPHA) / REPETITIONS_PER_GROUP
)
MIN_GROUPS_FOR_GLOBAL = 40
MAX_CANDIDATES = 1000
MAX_CEGIS_SECONDS = 2.0
STRESS_CARDINALITIES = (8, 16, 32, 64)
EVOLUTION_SEARCH_BUDGET = MAX_CANDIDATES * (len(REGIME_GRID) + len(STRESS_CARDINALITIES))
ARMS = ("base", "global", "old_selective", "new_regime", "oracle")
SYNTHETIC_RULE_SPEC = RuleSpec(
    rule_id="RULE-SYNTHETIC-SKILLSTUDY",
    version=1,
    parent=None,
    applicability={"all": []},
    intervention={"kind": "skill", "action": "synthetic_skill_reuse"},
    expected_mechanism="synthetic qualification fixture only",
    evidence_requirements=["synthetic train-only support evidence"],
    scientific_invariants=[],
    abstain_conditions={},
    relations={},
    runtime_cost={"tokens": 1},
    provenance_policy={"required": True},
    domain="skillstudy",
)
SYNTHETIC_SKILL_DIGEST = _semantic_spec_digest(SYNTHETIC_RULE_SPEC)
SYNTHETIC_RULE_STATE = RuleState(
    rule_id=SYNTHETIC_RULE_SPEC.rule_id,
    version=SYNTHETIC_RULE_SPEC.version,
    status="canonical",
    effect={"lower_utility": 1.0},
    retrieval_utility=1.0,
)
SYNTHETIC_ENGINE = AcreEngine(
    rule_specs=(SYNTHETIC_RULE_SPEC,),
    rule_states={SYNTHETIC_RULE_SPEC.rule_id: SYNTHETIC_RULE_STATE},
)
RUNTIME_VERSIONS = {
    "environment": "synthetic-skillstudy-v1",
    "agent": "fixed-agent-v1",
    "skill_digest": SYNTHETIC_SKILL_DIGEST,
}
SUPPORT_SCOPE = SupportScope(
    population_id="skillstudy-synthetic-v1",
    population_predicate={"equals": {"domain": "skillstudy"}},
    versions=RUNTIME_VERSIONS,
    required_strata=STRATA,
)


def required_task_execution_budget() -> float:
    evaluation_groups = (
        len(REGIME_GRID) * len(EVALUATION_SEEDS) * len(STRATA) * GROUPS_PER_STRATUM_PER_SEED
    )
    # One budget unit is one simulated arm-task outcome; each matched scenario
    # is reserved for all five compared arms before any result is generated.
    return float(evaluation_groups * REPETITIONS_PER_GROUP * len(ARMS))


def required_evolution_search_budget() -> float:
    """Candidate-evaluation cap across the three train fits and four stresses."""
    return float(EVOLUTION_SEARCH_BUDGET)


def qualification_config(*, task_execution_budget: float | None = None) -> dict[str, Any]:
    """Return the complete frozen design, without generating study outcomes."""
    return {
        "study_kind": "synthetic_learned_acre_qualification",
        "synthetic_only": True,
        "task_execution_budget": (
            required_task_execution_budget()
            if task_execution_budget is None
            else task_execution_budget
        ),
        "train_seeds": list(TRAIN_SEEDS),
        "independent_evaluation_seeds": list(EVALUATION_SEEDS),
        "regimes": list(REGIME_GRID),
        "strata": list(STRATA),
        "groups_per_stratum_per_seed": GROUPS_PER_STRATUM_PER_SEED,
        "training_group_count_across_all_regimes": TRAINING_GROUP_COUNT,
        "repetitions_per_group": REPETITIONS_PER_GROUP,
        "confidence_allocation": {
            "family_alpha": FAMILY_ALPHA,
            "measurement_alpha": MEASUREMENT_ALPHA,
            "measurement_method": "Bonferroni simultaneous two-sided paired-repetition intervals across every predeclared training group in all three regimes",
            "measurement_comparisons": TRAINING_GROUP_COUNT,
            "per_group_measurement_alpha": PER_GROUP_MEASUREMENT_ALPHA,
            "planned_radius_target": PLANNED_MEASUREMENT_RADIUS_TARGET,
            "planned_worst_case_radius": PLANNED_MEASUREMENT_RADIUS,
            "minimum_repetitions_for_radius_target": MIN_REPETITIONS_FOR_RADIUS,
            "regime_classifier_alpha": REGIME_ALPHA,
            "cegis_certificate_alpha": CEGIS_ALPHA,
            "group_bound_interpretation": "synthetic only; simultaneous group intervals feed selection, while classifier and CEGIS certificate budgets are prospectively separated within the family alpha",
        },
        "minimum_global_groups": MIN_GROUPS_FOR_GLOBAL,
        "max_candidates": MAX_CANDIDATES,
        "evolution_candidate_evaluation_budget": EVOLUTION_SEARCH_BUDGET,
        "max_cegis_seconds": MAX_CEGIS_SECONDS,
        "cegis_feature": "stratum (categorical)",
        "decision_contexts": list(STRATA),
        "support_minimum_net_benefit": 0.0,
        "support_maximum_harm_probability": 0.10,
        "support_minimum_groups": MIN_GROUPS_FOR_GLOBAL,
        "evaluation_arms": list(ARMS),
        "synthetic_rule": {
            "rule_id": SYNTHETIC_RULE_SPEC.rule_id,
            "version": SYNTHETIC_RULE_SPEC.version,
            "state": "canonical synthetic fixture only",
            "skill_digest": SYNTHETIC_SKILL_DIGEST,
            "routing_api": "AcreEngine.route(support_regimes=..., runtime_versions=...)",
            "canonical_promotion_claim": False,
        },
        "predicate_stress_cardinalities": list(STRESS_CARDINALITIES),
        "train_evaluation_overlap": False,
        "multiplicity_claim": "descriptive synthetic comparisons; no confirmatory real-study claim",
        "real_phase_i": {
            **PHASE_I_CONFIG,
            "arms": list(PHASE_I_CONFIG["arms"]),
            "matching_keys": list(PHASE_I_CONFIG["matching_keys"]),
            "families": list(PHASE_I_CONFIG["families"]),
        },
    }


def _probabilities(regime: str, stratum: str) -> tuple[float, float]:
    if regime == "broad_benefit":
        return 0.10, 0.95
    if regime == "selective_transfer":
        return (0.10, 0.95) if stratum == "supported" else (0.90, 0.05)
    return 0.50, 0.50


def _effect_interval(differences: list[float]) -> tuple[float, float, float]:
    effect = mean(differences)
    lower, upper = paired_repetition_interval(differences, PER_GROUP_MEASUREMENT_ALPHA)
    return effect, lower, upper


def _train_regime(regime: str) -> tuple[list[SupportGroup], Any]:
    groups: list[SupportGroup] = []
    for seed in TRAIN_SEEDS:
        for stratum in STRATA:
            base_probability, skill_probability = _probabilities(regime, stratum)
            for index in range(GROUPS_PER_STRATUM_PER_SEED):
                group_id = f"train-{regime}-{seed}-{stratum}-{index:03d}"
                rng = random.Random(seed * 1_000_003 + index * 101 + (0 if stratum == "supported" else 17))
                differences = []
                for _ in range(REPETITIONS_PER_GROUP):
                    draw = rng.random()
                    differences.append(float(draw < skill_probability) - float(draw < base_probability))
                effect, lower, upper = _effect_interval(differences)
                context = {"domain": "skillstudy", "stratum": stratum}
                groups.append(SupportGroup(
                    group_id=group_id,
                    context=context,
                    effect_lower=lower,
                    effect_upper=upper,
                    hard_gates_passed=True,
                    stratum=stratum,
                    versions=RUNTIME_VERSIONS,
                ))

    observations = [
        BoundaryObservation(
            observation_id=group.group_id,
            context=group.context,
            effect=(group.effect_lower + group.effect_upper) / 2.0,
            gate_passed=group.hard_gates_passed,
            effect_lower=group.effect_lower,
            effect_upper=group.effect_upper,
        )
        for group in groups
    ]
    positive = [item for item in observations if item.positive_anchor()]
    negative = [item for item in observations if item.certified_counterexample()]
    grammar = PredicateGrammar.from_dict({
        "schema_version": 1,
        "features": [{"path": "stratum", "type": "categorical"}],
        "max_depth": 2,
        "max_literals": 2,
    })
    synthesis = StatisticalCEGIS(
        grammar,
        delta=CEGIS_ALPHA,
        max_candidates=MAX_CANDIDATES,
        max_wall_time=MAX_CEGIS_SECONDS,
    ).synthesize(
        positive=positive,
        counterexamples=negative,
        parent_predicate=None,
        decision_contexts=[
            {"domain": "skillstudy", "stratum": stratum}
            for stratum in STRATA
        ],
        certificate_scope=SUPPORT_SCOPE.to_dict(),
    )
    return groups, synthesis


def _interval(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    estimate = mean(values) if values else 0.0
    standard_error = stdev(values) / (n ** 0.5) if n > 1 else 0.0
    return {
        "estimate": estimate,
        "lower_95": estimate - 1.96 * standard_error,
        "upper_95": estimate + 1.96 * standard_error,
        "independent_clusters": n,
    }


def _evaluate_regime(regime: str, decisions: dict[str, Any]) -> dict[str, Any]:
    counts = {arm: {"successes": 0, "deployed": 0, "harms": 0, "repairs_retained": 0} for arm in ARMS}
    total = available_repairs = pairwise_oracle_successes = 0
    clusters: dict[str, list[float]] = {arm: [] for arm in ARMS if arm != "base"}
    harms: dict[str, list[float]] = {arm: [] for arm in ARMS if arm != "base"}
    activation: dict[str, list[float]] = {arm: [] for arm in ARMS if arm != "base"}
    stratified = {
        stratum: {
            key: {arm: [] for arm in ARMS if arm != "base"}
            for key in ("contrasts", "negative_transfer_rate", "activation_rate")
        }
        for stratum in STRATA
    }
    engine_selected_by_stratum: dict[str, bool] = {}
    for seed in EVALUATION_SEEDS:
        for stratum in STRATA:
            base_probability, skill_probability = _probabilities(regime, stratum)
            context = {"domain": "skillstudy", "stratum": stratum}
            old_use = decision_from_synthesis(decisions["synthesis"], context).action == "reuse"
            routed = SYNTHETIC_ENGINE.route(
                context,
                support_regimes={SYNTHETIC_RULE_SPEC.rule_id: decisions["regime"]},
                runtime_versions=RUNTIME_VERSIONS,
                candidate_ids=(SYNTHETIC_RULE_SPEC.rule_id,),
            )
            new_use = SYNTHETIC_RULE_SPEC.rule_id in routed.selected_rule_ids
            engine_selected_by_stratum[stratum] = new_use
            truth_use = (
                regime == "broad_benefit"
                or (regime == "selective_transfer" and stratum == "supported")
            )
            for index in range(GROUPS_PER_STRATUM_PER_SEED):
                rng = random.Random(seed * 9_999_991 + index * 503 + (0 if stratum == "supported" else 31))
                group_differences = {arm: [] for arm in clusters}
                group_harms = {arm: [] for arm in harms}
                for _ in range(REPETITIONS_PER_GROUP):
                    draw = rng.random()
                    base_success = float(draw < base_probability)
                    skill_success = float(draw < skill_probability)
                    total += 1
                    available_repairs += int(skill_success > base_success)
                    pairwise_oracle_successes += int(max(base_success, skill_success))
                    counts["base"]["successes"] += int(base_success)
                    policies = {
                        "global": True,
                        "old_selective": old_use,
                        "new_regime": new_use,
                        "oracle": truth_use,
                    }
                    for arm, use_skill in policies.items():
                        success = skill_success if use_skill else base_success
                        counts[arm]["successes"] += int(success)
                        counts[arm]["deployed"] += int(use_skill)
                        counts[arm]["harms"] += int(success < base_success)
                        counts[arm]["repairs_retained"] += int(success > base_success)
                        group_differences[arm].append(success - base_success)
                        group_harms[arm].append(float(base_success > success))
                for arm in clusters:
                    group_effect = mean(group_differences[arm])
                    group_harm = mean(group_harms[arm])
                    group_activation = float({"global": True, "old_selective": old_use, "new_regime": new_use, "oracle": truth_use}[arm])
                    clusters[arm].append(group_effect)
                    harms[arm].append(group_harm)
                    activation[arm].append(group_activation)
                    stratified[stratum]["contrasts"][arm].append(group_effect)
                    stratified[stratum]["negative_transfer_rate"][arm].append(group_harm)
                    stratified[stratum]["activation_rate"][arm].append(group_activation)
    return {
        "regime": regime,
        "independent_unit": "synthetic task-family cluster; evaluation groups are independent of CEGIS/classifier training groups",
        "training": {
            "seeds": list(TRAIN_SEEDS),
            "groups": len(decisions["groups"]),
            "positive_anchors": len(decisions["synthesis"].positive_anchors),
            "certified_counterexamples": len(decisions["synthesis"].certified_counterexamples),
            "cegis_status": decisions["synthesis"].status,
            "cegis_predicate": decisions["synthesis"].predicate,
            "cegis_certificate_valid": synthesis_certificate_valid(decisions["synthesis"]),
            "cegis_candidates_examined": decisions["synthesis"].candidates_examined,
            "cegis_search_exhausted": decisions["synthesis"].search_exhausted,
            "support_regime": decisions["regime"].regime.value,
            "support_reason": decisions["regime"].reason_code,
        },
        "new_regime_routing": {
            "api": "AcreEngine.route",
            "rule_id": SYNTHETIC_RULE_SPEC.rule_id,
            "rule_state": "canonical synthetic fixture only",
            "skill_digest": SYNTHETIC_SKILL_DIGEST,
            "selected_by_stratum": engine_selected_by_stratum,
            "canonical_promotion_claim": False,
        },
        "evaluation": {
            "paired_outcome_count": total,
            "available_repairs": available_repairs,
            "pairwise_oracle_successes": pairwise_oracle_successes,
            "policy_metrics": {
                arm: {**values,
                      "success_rate": values["successes"] / total,
                      "coverage": values["deployed"] / total,
                      "harm_among_deployed": values["harms"] / values["deployed"] if values["deployed"] else None,
                      "harm_among_all": values["harms"] / total,
                      "repair_recall": values["repairs_retained"] / available_repairs if available_repairs else None,
                      "incremental_success_count": values["successes"] - counts["base"]["successes"],
                      "incremental_success_rate": (values["successes"] - counts["base"]["successes"]) / total,
                      "regret_vs_pairwise_oracle": (pairwise_oracle_successes - values["successes"]) / total}
                for arm, values in counts.items()
            },
            "seeds": list(EVALUATION_SEEDS),
            "groups_per_stratum": len(EVALUATION_SEEDS) * GROUPS_PER_STRATUM_PER_SEED,
            "contrasts_vs_base": {arm: _interval(values) for arm, values in clusters.items()},
            "negative_transfer_rate": {arm: _interval(values) for arm, values in harms.items()},
            "activation_rate": {arm: _interval(values) for arm, values in activation.items()},
            "by_stratum": {
                stratum: {
                    metric: {arm: _interval(values) for arm, values in arms.items()}
                    for metric, arms in metrics.items()
                }
                for stratum, metrics in stratified.items()
            },
        },
        "oracle_status": "synthetic ground-truth upper bound; never a learned or deployable policy",
    }


def _predicate_stress() -> list[dict[str, Any]]:
    rows = []
    for cardinality in STRESS_CARDINALITIES:
        contexts = [{"x": f"value-{index:02d}"} for index in range(cardinality)]
        grammar = PredicateGrammar.from_dict({
            "schema_version": 1,
            "features": [{"path": "x", "type": "categorical"}],
            "max_depth": 2,
            "max_literals": 2,
        })
        positive = BoundaryObservation("stress-positive", contexts[0], 0.5, True, 0.25, 0.75)
        negative = BoundaryObservation("stress-negative", contexts[-1], -0.5, True, -0.75, -0.25)
        tracemalloc.start()
        started = time.perf_counter()
        result = StatisticalCEGIS(
            grammar,
            max_candidates=MAX_CANDIDATES,
            max_wall_time=MAX_CEGIS_SECONDS,
        ).synthesize(
            positive=[positive],
            counterexamples=[negative],
            parent_predicate=None,
            decision_contexts=contexts,
            vocabulary_contexts=contexts,
        )
        wall_seconds = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        serialized = json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"))
        rows.append({
            "distinct_values": cardinality,
            "candidate_cap": MAX_CANDIDATES,
            "candidates_examined": result.candidates_examined,
            "search_status": "truncated" if result.search_exhausted else "exact",
            "result_status": result.status,
            "wall_seconds": wall_seconds,
            "python_peak_traced_bytes": peak_bytes,
            "serialized_result_bytes": len(serialized.encode("utf-8")),
            "decision_contexts": len(contexts),
        })
    return rows


def run_qualification_study(*, task_execution_budget: float | None = None) -> dict[str, Any]:
    """Run learned synthetic ACRE train/evaluation and bounded CEGIS stress."""
    required = required_task_execution_budget()
    budget = required if task_execution_budget is None else task_execution_budget
    pair_costs = tuple(
        (
            f"eval-{regime}-{seed}-{stratum}-cluster-{index:03d}",
            REPETITIONS_PER_GROUP * len(ARMS),
        )
        for regime in REGIME_GRID
        for seed in EVALUATION_SEEDS
        for stratum in STRATA
        for index in range(GROUPS_PER_STRATUM_PER_SEED)
    )
    reservation = PairReservation.reserve_all_pairs(
        pair_costs,
        task_execution_budget=budget,
        evolution_search_budget=EVOLUTION_SEARCH_BUDGET,
    )
    if not reservation.launch_allowed:
        return {
            "study_kind": "synthetic_learned_acre_qualification",
            "synthetic_only": True,
            "qualification_status": "NOT_LAUNCHED_INCOMPLETE_PAIR_RESERVATION",
            "reservation": reservation.as_dict(),
            "provider_calls": 0,
        }

    rows = []
    for regime in REGIME_GRID:
        groups, synthesis = _train_regime(regime)
        regime_decision = classify_support_regime(
            groups,
            scope=SUPPORT_SCOPE,
            target_context={"domain": "skillstudy"},
            runtime_versions=RUNTIME_VERSIONS,
            selective=synthesis,
            fixed_time_complete=True,
            delta=REGIME_ALPHA,
            min_groups=MIN_GROUPS_FOR_GLOBAL,
            min_net_benefit=0.0,
            max_harm_probability=0.10,
        )
        rows.append(_evaluate_regime(regime, {"groups": groups, "synthesis": synthesis, "regime": regime_decision}))
    stress_rows = _predicate_stress()
    evolution_spent = sum(
        row["training"]["cegis_candidates_examined"] for row in rows
    ) + sum(row["candidates_examined"] for row in stress_rows)
    reservation = reservation.record_evolution(evolution_spent)
    for pair_id, _ in pair_costs:
        reservation = reservation.complete_pair(pair_id, measured_cost=None)
    return {
        "study_kind": "synthetic_learned_acre_qualification",
        "synthetic_only": True,
        "qualification_status": "SYNTHETIC_FIXTURE_COMPLETE",
        "provider_calls": 0,
        "pre_frozen_config": qualification_config(task_execution_budget=budget),
        "hypotheses": {
            "H4_BROAD_GLOBAL": "Train-only CEGIS and support classification permit broad reuse only when the classifier certifies the frozen population; otherwise the new regime policy abstains.",
            "H5_SELECTIVE_LEARNED": "Train-only CEGIS identifies the supported stratum, and new regime decisions reduce unsupported harm versus global reuse on independent evaluation groups.",
            "H6_NULL_ABSTENTION": "When train evidence is uncertain under a null effect, operational old-selective and new-regime paths abstain; oracle results remain a synthetic upper bound.",
        },
        "results": rows,
        "power_table": {
            "status": "planning_approximation_not_observed_power",
            "method": "two-sided normal approximation for paired binary difference using the declared discordant-pair-rate assumption",
            "rows": power_table(),
        },
        "predicate_stress": {
            "status": "synthetic_bounded_search_diagnostics",
            "candidate_cap": MAX_CANDIDATES,
            "wall_and_memory_note": "measured by this invocation on the execution host; Python traced allocations are not process RSS or GPU memory",
            "rows": stress_rows,
        },
        "reservation": {
            **reservation.as_dict(),
            "unit": "simulated arm-task outcomes",
            "measurement_note": "real token/time/GPU cost unknown; no real task execution occurred",
        },
        "real_phase_i": {
            **PHASE_I_CONFIG,
            "arms": list(PHASE_I_CONFIG["arms"]),
            "families": list(PHASE_I_CONFIG["families"]),
        },
        "claim_boundary": "Synthetic noisy train/evaluation qualification of the current CEGIS and support-regime APIs only; no empirical agent or Skill efficacy claim.",
    }


__all__ = [
    "qualification_config",
    "required_evolution_search_budget",
    "required_task_execution_budget",
    "run_qualification_study",
]
