"""Run a small, family-held-out simulation of cost-sensitive ACRE routing."""

from __future__ import annotations

import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark.boundary.soft_acre import SoftAcrePolicy, fit_soft_acre


STUDY_ID = "EVOSPE_SOFT_ACRE_REGIME_SIM_20261102_03"
SEED = 20261102
CALIBRATION_FAMILIES = 6
FINAL_FAMILIES = 8
CONTEXTS_PER_TYPE = 4
RISK_WEIGHT = 1.0
COMPLEXITY_PENALTY = 0.002
MAX_LITERALS = 3
SWITCHING_MARGIN_SE = 1.0
REGIMES = ("broad", "selective", "selective_interaction", "null")
TASK_TYPES = ("matched", "adjacent", "distractor")
SCALES = ("small", "large")
COMPATIBILITIES = (0, 1)


def _family_rows(regime: str, family_index: int, split: str, rng: random.Random):
    family = f"{regime}-{split}-{family_index:02d}"
    rows: list[dict[str, Any]] = []
    scores: dict[str, dict[str, Any]] = {}
    contexts = [
        {"task_type": task_type, "scientific_gate": True}
        for task_type in TASK_TYPES
    ]
    if regime == "selective_interaction":
        contexts = [
            {
                "task_type": task_type,
                "scale_regime": scale,
                "constraint_compatibility": compatibility,
                "scientific_gate": True,
            }
            for task_type in TASK_TYPES
            for scale in SCALES
            for compatibility in COMPATIBILITIES
        ]
    for context in contexts:
        for repetition in range(CONTEXTS_PER_TYPE):
            identifier = f"{family}-{len(rows)}-{repetition}"
            task_type = context["task_type"]
            if regime == "broad":
                tau = 0.10 + rng.uniform(-0.02, 0.02)
            elif regime == "selective":
                center = {"matched": 0.18, "adjacent": 0.02, "distractor": -0.18}[task_type]
                tau = center + rng.uniform(-0.02, 0.02)
            elif regime == "selective_interaction":
                complete_match = (
                    task_type == "matched"
                    and context["scale_regime"] == "large"
                    and context["constraint_compatibility"] == 1
                )
                partial_match = sum((
                    task_type == "matched",
                    context["scale_regime"] == "large",
                    context["constraint_compatibility"] == 1,
                )) == 2
                tau = (0.25 if complete_match else -0.20 if partial_match else 0.0) + rng.uniform(-0.01, 0.01)
            else:
                tau = 0.0
            rows.append({
                "id": identifier,
                "family": family,
                "context": context,
            })
            scores[identifier] = {"tau": tau, "scientific_gate": True}
    return rows, scores


def _make_split(regime: str, split: str, count: int, rng: random.Random):
    rows: list[dict[str, Any]] = []
    scores: dict[str, dict[str, Any]] = {}
    for family_index in range(count):
        family_rows, family_scores = _family_rows(regime, family_index, split, rng)
        rows.extend(family_rows)
        scores.update(family_scores)
    return rows, scores


def _family_utility(rows, scores, policy) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        selected = policy.decide(row["context"]).action == "reuse"
        score = scores[row["id"]]
        values[row["family"]].append(float(score["tau"]) if selected and score["scientific_gate"] else 0.0)
    return {family: sum(items) / len(items) for family, items in values.items()}


def _deployment_metrics(rows, scores, policy) -> dict[str, float | None]:
    deployed = 0
    harmful = 0
    for row in rows:
        if policy.decide(row["context"]).action == "reuse":
            deployed += 1
            harmful += scores[row["id"]]["tau"] < 0
    return {
        "coverage": deployed / len(rows),
        "negative_transfer_all": harmful / len(rows),
        "negative_transfer_deployed": harmful / deployed if deployed else None,
    }


def _bootstrap_interval(differences: list[float], *, seed: int) -> list[float]:
    rng = random.Random(seed)
    draws = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(10_000)
    )
    return [draws[249], draws[9749]]


def run() -> dict[str, Any]:
    results = []
    for regime_index, regime in enumerate(REGIMES):
        rng = random.Random(SEED + regime_index)
        calibration, calibration_scores = _make_split(regime, "cal", CALIBRATION_FAMILIES, rng)
        feature_names = (
            ("task_type", "scale_regime", "constraint_compatibility")
            if regime == "selective_interaction" else ("task_type",)
        )
        soft = fit_soft_acre(
            calibration,
            calibration_scores,
            feature_names=feature_names,
            risk_weight=RISK_WEIGHT,
            complexity_penalty=COMPLEXITY_PENALTY,
            max_literals=MAX_LITERALS if regime == "selective_interaction" else 1,
            switching_margin_se=SWITCHING_MARGIN_SE,
        )
        simple = fit_soft_acre(
            calibration,
            calibration_scores,
            feature_names=feature_names,
            risk_weight=0.0,
            complexity_penalty=0.0,
            max_literals=1,
            switching_margin_se=0.0,
        )
        evaluation, evaluation_scores = _make_split(regime, "final", FINAL_FAMILIES, rng)
        global_policy = SoftAcrePolicy("global", (), 0.0, 0.0, 0.0, CALIBRATION_FAMILIES)
        soft_family = _family_utility(evaluation, evaluation_scores, soft)
        simple_family = _family_utility(evaluation, evaluation_scores, simple)
        global_family = _family_utility(evaluation, evaluation_scores, global_policy)
        differences = [soft_family[key] - simple_family[key] for key in sorted(soft_family)]
        global_differences = [soft_family[key] - global_family[key] for key in sorted(soft_family)]
        results.append({
            "regime": regime,
            "calibration": {
                "independent_families": soft.calibration_families,
                "contexts": len(calibration),
            },
            "final": {
                "independent_families": len(soft_family),
                "contexts": len(evaluation),
            },
            "soft_acre": {
                "route": soft.route,
                "literals": [[field, value] for field, value in soft.literals],
                "calibration_objective": soft.calibration_utility,
                "switching_standard_error": soft.switching_standard_error,
                "mean_family_utility": sum(soft_family.values()) / len(soft_family),
                **_deployment_metrics(evaluation, evaluation_scores, soft),
            },
            "strongest_single_feature": {
                "route": simple.route,
                "literals": [[field, value] for field, value in simple.literals],
                "mean_family_utility": sum(simple_family.values()) / len(simple_family),
                **_deployment_metrics(evaluation, evaluation_scores, simple),
            },
            "soft_minus_simple": {
                "mean_family_utility": sum(differences) / len(differences),
                "family_bootstrap_95_ci": _bootstrap_interval(differences, seed=SEED + len(results)),
                "positive_families": sum(value > 0 for value in differences),
                "negative_families": sum(value < 0 for value in differences),
                "tied_families": sum(value == 0 for value in differences),
                "family_differences": differences,
            },
            "soft_minus_global": {
                "mean_family_utility": sum(global_differences) / len(global_differences),
                "family_bootstrap_95_ci": _bootstrap_interval(global_differences, seed=SEED + 10 + len(results)),
                "positive_families": sum(value > 0 for value in global_differences),
                "negative_families": sum(value < 0 for value in global_differences),
                "tied_families": sum(value == 0 for value in global_differences),
                "family_differences": global_differences,
            },
        })
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {
        "study_id": STUDY_ID,
        "source_revision": revision,
        "comparison_class": "algorithmic_simulation",
        "evidence_level": "micro",
        "data_status": "synthetic_family_heldout",
        "resource_contract": {"provider_calls": 0, "gpu": "none", "local_runtime": False},
        "seed": SEED,
        "contract": {
            "split": {"calibration_families_per_regime": CALIBRATION_FAMILIES, "final_families_per_regime": FINAL_FAMILIES},
            "repetitions_per_context_cell": CONTEXTS_PER_TYPE,
            "feature_grammar": ["categorical equality", "conjunctions up to three literals in the interaction regime"],
            "risk_weight": RISK_WEIGHT,
            "complexity_penalty": COMPLEXITY_PENALTY,
            "switching_margin_se": SWITCHING_MARGIN_SE,
            "max_literals": MAX_LITERALS,
            "strongest_single_feature": "maximize unpenalized family-balanced utility over abstain, global, and one-literal predicates",
            "hard_scientific_gate": "deployment is disabled when context.scientific_gate is not true",
            "final_access": "final outcomes are used only after calibration policy fitting",
        },
        "results": results,
        "claim_boundary": "Mechanism simulation only; no language-model task, natural deployment distribution, or empirical Skill efficacy was measured.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
