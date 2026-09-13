#!/usr/bin/env python3
"""Run fixed-grid, provider-free sensitivity checks for support certification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from itertools import combinations
from pathlib import Path
import sys
from statistics import median
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.boundary.families import family_cases
from benchmark.boundary.policy_eval import Decision, evaluate_policy
from benchmark.families import PILOT_FAMILIES, family_decision_lattice, family_views
from core.acre.cegis import StatisticalCEGIS
from core.acre.predicates import PredicateGrammar, predicate_complexity
from core.predicates import match_predicate
from scripts.run_support_boundary_analysis import (
    _context_key,
    _grammar_for,
    _split_cases,
    _support_metrics,
)


REGISTERED_CONFIDENCE = 0.95
REGISTERED_EPSILON_TRUE = 0.0
REGISTERED_EPSILON_FALSE = 0.0
REGISTERED_UTILITY_SCALE = None
REGISTERED_REPETITIONS = 1
REGISTERED_DOSE = 1.0


def sensitivity_grid() -> dict[str, tuple[float | int, ...]]:
    """Return the complete pre-registered one-factor grid."""

    return {
        "confidence_level": (0.90, 0.95, 0.99),
        "repetition_fraction": (0.50, 0.75, 1.00),
        "delta_multiplier": (0.5, 1.0, 1.5),
        "utility_scale_multiplier": (0.5, 1.0, 2.0),
        "grammar_depth_offset": (-1, 0, 1),
        "grammar_literal_offset": (-1, 0, 1),
    }


def classify_boundary_observation(
    effect_lower: float,
    effect_upper: float,
    gate_passed: bool,
    epsilon_true: float,
    epsilon_false: float,
) -> str:
    """Apply the core BoundaryObservation rules without changing their order."""

    if not gate_passed or float(effect_upper) <= float(epsilon_false):
        return "counterexample"
    if float(effect_lower) > float(epsilon_true):
        return "positive_anchor"
    return "unresolved"


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _grammar_variant(family: str, *, depth_offset: int = 0, literal_offset: int = 0) -> tuple[PredicateGrammar, dict[str, Any]]:
    grammar = _grammar_for(family)
    default_depth = int(grammar.max_depth)
    default_literals = int(grammar.max_literals)
    depth = default_depth + int(depth_offset)
    literals = default_literals + int(literal_offset)
    if not 1 <= depth <= 3:
        raise ValueError(f"depth offset is not applicable for {family}: {depth}")
    if not 1 <= literals <= 4:
        raise ValueError(f"literal offset is not applicable for {family}: {literals}")
    variant = PredicateGrammar(
        schema_version=grammar.schema_version,
        features=grammar.features,
        max_depth=depth,
        max_literals=literals,
        threshold_universe=grammar.threshold_universe,
        allow_observed_thresholds=grammar.allow_observed_thresholds,
    )
    return variant, {
        "default_depth": default_depth,
        "default_literals": default_literals,
        "depth": depth,
        "literals": literals,
        "depth_offset": int(depth_offset),
        "literal_offset": int(literal_offset),
    }


def _classify_cases(cases: Sequence[Any], *, epsilon_true: float, epsilon_false: float) -> dict[str, str]:
    return {
        item.observation_id: classify_boundary_observation(
            item.effect_lower,
            item.effect_upper,
            item.gate_passed,
            epsilon_true,
            epsilon_false,
        )
        for item in cases
    }


def _prediction_policy(predictions: Mapping[str, bool], evaluation_cases: Sequence[Any]):
    by_context = {
        _context_key(item.context): bool(predictions.get(item.observation_id, False))
        for item in evaluation_cases
    }

    def decide(context: Mapping[str, Any]) -> Decision:
        admitted = by_context.get(_context_key(context), False)
        return Decision(
            "reuse" if admitted else "abstain",
            1.0 if admitted else 0.0,
            0.0,
            "sensitivity_frozen_prediction",
        )

    return decide


def _support_metrics_with_truth(
    predictions: Mapping[str, bool],
    evaluation_cases: Sequence[Any],
    truth: Mapping[str, bool],
) -> dict[str, Any]:
    tp = sum(int(predictions.get(key, False) and value) for key, value in truth.items())
    fp = sum(int(predictions.get(key, False) and not value) for key, value in truth.items())
    fn = sum(int(not predictions.get(key, False) and value) for key, value in truth.items())
    union = tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "support_tp": tp,
        "support_fp": fp,
        "support_fn": fn,
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": (2 * precision * recall / (precision + recall))
        if precision is not None and recall is not None and precision + recall
        else None,
        "support_iou": tp / union if union else None,
        "oracle_supported": tp + fn,
    }


def _latent_truth_crosscheck(family: str, seed: int, cases: Sequence[Any]) -> dict[str, Any]:
    latent = {
        item.instance_id: bool(item.applicable)
        for values in family_views(family, count=324, seed=seed).values()
        for item in values
    }
    registered = {
        item.observation_id: classify_boundary_observation(
            item.effect_lower,
            item.effect_upper,
            item.gate_passed,
            REGISTERED_EPSILON_TRUE,
            REGISTERED_EPSILON_FALSE,
        )
        == "positive_anchor"
        for item in cases
    }
    mismatches = [
        item.observation_id
        for item in cases
        if item.observation_id in latent and latent[item.observation_id] != registered[item.observation_id]
    ]
    return {
        "latent_source": "FamilySpec.applicable",
        "registered_source": "BoundaryObservation effect interval and scientific gate",
        "delta_independent": True,
        "cases_checked": len(cases),
        "mismatch_count": len(mismatches),
        "mismatch_ids": mismatches,
        "registered_truth_used_for_metrics": True,
        "access": "scoring_only",
    }


def _result_status(result: Mapping[str, Any] | None) -> str:
    status = str((result or {}).get("status", "insufficient_evidence"))
    if status in {"unsynthesizable_boundary", "conflict", "inconsistent"}:
        return "inconsistent"
    if status in {"identified", "underidentified"}:
        return status
    return status


def _fit_predictions(
    family: str,
    synthesis: Sequence[Any],
    evaluation: Sequence[Any],
    *,
    dose: float,
    confidence_level: float,
    epsilon_true: float,
    epsilon_false: float,
    grammar: PredicateGrammar,
) -> tuple[dict[str, bool], dict[str, Any] | None, dict[str, str]]:
    labels = _classify_cases(synthesis, epsilon_true=epsilon_true, epsilon_false=epsilon_false)
    anchors = [item for item in synthesis if labels[item.observation_id] == "positive_anchor"]
    counterexamples = [item for item in synthesis if labels[item.observation_id] == "counterexample"]
    status_counts = dict(Counter(labels.values()))
    if dose <= 0.0:
        predictions = {
            item.observation_id: labels.get(item.observation_id) == "positive_anchor"
            for item in evaluation
        }
        return predictions, None, status_counts
    count = min(len(counterexamples), int(round(len(counterexamples) * dose)))
    if not anchors or count == 0:
        return {item.observation_id: False for item in evaluation}, None, status_counts
    result = StatisticalCEGIS(
        grammar,
        epsilon_true=epsilon_true,
        epsilon_false=epsilon_false,
        delta=1.0 - float(confidence_level),
    ).synthesize(
        positive=list(anchors),
        counterexamples=list(counterexamples[:count]),
        parent_predicate=None,
        decision_contexts=[item.context for item in evaluation],
    )
    result_dict = result.to_dict()
    predictions = {
        item.observation_id: bool(result.predicate and match_predicate(result.predicate, item.context))
        for item in evaluation
    }
    return predictions, result_dict, status_counts


def _evaluate_row(
    *,
    family: str,
    seed: int,
    evaluation: Sequence[Any],
    calibration: Sequence[Any],
    predictions: Mapping[str, bool],
    truth: Mapping[str, bool],
    predicate: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    status_counts: Mapping[str, int],
    confidence_level: float,
    delta: float,
    configuration_id: str,
    factor: str,
    parameter: str,
    value: Any,
    dose: float,
    default_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    eval_rows = [
        {"id": item.observation_id, "family": family, "seed": seed, "context": dict(item.context)}
        for item in evaluation
    ]
    calibration_rows = [
        {"id": item.observation_id, "context": dict(item.context), "label": bool(truth.get(item.observation_id, False))}
        for item in calibration
    ]
    scores = {
        item.observation_id: {
            "tau": item.effect,
            "scientific_gate": item.gate_passed,
            "ci_width": item.effect_upper - item.effect_lower,
        }
        for item in evaluation
    }
    metrics = evaluate_policy(
        _prediction_policy(predictions, evaluation),
        calibration_rows,
        eval_rows,
        scores,
        delta=delta,
    )
    support = _support_metrics_with_truth(predictions, evaluation, truth)
    complexity = predicate_complexity(predicate) if predicate else {
        "depth": None,
        "literals": None,
        "description_length": None,
    }
    row = {
        "family": family,
        "seed": seed,
        "configuration_id": configuration_id,
        "factor": factor,
        "parameter": parameter,
        "value": value,
        "counterexample_dose": dose,
        "confidence_level": confidence_level,
        "coverage": metrics["coverage"],
        "unsupported_deployed": metrics["unsupported_deployed"],
        "unsupported_all": metrics["unsupported_all"],
        "support_precision": support["support_precision"],
        "support_recall": support["support_recall"],
        "support_decision_regret_mean": metrics["support_decision_regret_mean"],
        "anchor_count": status_counts.get("positive_anchor", 0),
        "counterexample_count": status_counts.get("counterexample", 0),
        "unresolved_count": status_counts.get("unresolved", 0),
        "predicate_depth": complexity["depth"],
        "predicate_literals": complexity["literals"],
        "predicate_dl": complexity["description_length"],
        "status": _result_status(result),
        "status_counts": dict(status_counts),
        "evaluation_ids": [item.observation_id for item in evaluation],
        "predicate": predicate,
    }
    for metric, source in (
        ("coverage", "coverage"),
        ("unsupported_deployed", "unsupported_deployed"),
        ("support_recall", "support_recall"),
    ):
        default_value = default_metrics.get(source) if default_metrics else None
        current_value = row.get(metric)
        row[f"delta_{metric}"] = (
            current_value - default_value
            if current_value is not None and default_value is not None
            else None
        )
    # Use the publication-facing name requested by the sensitivity protocol;
    # retain delta_support_recall as the unambiguous source-field alias.
    row["delta_recall"] = row["delta_support_recall"]
    return row


def _stability_report(
    family_rules: Mapping[tuple[str, int], Mapping[str, Any] | None],
    *,
    families: Sequence[str],
    generator_seeds: Sequence[int],
) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []
    for family in families:
        grid = family_decision_lattice(family, seed=0, count=264)
        contexts = [row["workload"] for row in grid]
        masks: dict[int, tuple[bool, ...]] = {}
        complexities: dict[int, dict[str, Any]] = {}
        for seed in generator_seeds:
            predicate = family_rules.get((family, int(seed)))
            masks[int(seed)] = tuple(bool(predicate and match_predicate(predicate, context)) for context in contexts)
            complexities[int(seed)] = predicate_complexity(predicate) if predicate else {
                "depth": None,
                "literals": None,
                "description_length": None,
            }
        for seed_a, seed_b in combinations([int(seed) for seed in generator_seeds], 2):
            left, right = masks[seed_a], masks[seed_b]
            intersection = sum(a and b for a, b in zip(left, right))
            union = sum(a or b for a, b in zip(left, right))
            pair_rows.append({
                "family": family,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "canonical_context_count": len(contexts),
                "support_size_a": sum(left),
                "support_size_b": sum(right),
                "jaccard": intersection / union if union else 1.0,
                "predicate_dl_a": complexities[seed_a]["description_length"],
                "predicate_dl_b": complexities[seed_b]["description_length"],
            })
    family_summary: list[dict[str, Any]] = []
    for family in families:
        values = [row["jaccard"] for row in pair_rows if row["family"] == family]
        sizes = [row["support_size_a"] for row in pair_rows if row["family"] == family]
        sizes += [row["support_size_b"] for row in pair_rows if row["family"] == family]
        mean_size = sum(sizes) / len(sizes) if sizes else 0.0
        sd_size = (
            (sum((value - mean_size) ** 2 for value in sizes) / len(sizes)) ** 0.5
            if sizes
            else 0.0
        )
        family_summary.append({
            "family": family,
            "pair_count": len(values),
            "jaccard_median": median(values) if values else None,
            "jaccard_iqr": (
                _quantile(values, 0.75) - _quantile(values, 0.25)
                if values
                else None
            ),
            "support_size_cv": sd_size / mean_size if mean_size else None,
        })
    all_jaccards = [row["jaccard"] for row in pair_rows]
    return {
        "canonical_context_count": 264,
        "canonical_context_source": "family_decision_lattice(family, seed=0, count=264)",
        "family_seed_pair_rows": pair_rows,
        "across_family_summary": {
            "family_count": len(family_summary),
            "jaccard_median": median(all_jaccards) if all_jaccards else None,
            "jaccard_iqr": (
                _quantile(all_jaccards, 0.75) - _quantile(all_jaccards, 0.25)
                if all_jaccards
                else None
            ),
            "family_summaries": family_summary,
        },
    }


def _configuration_registry(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the exact executed configurations without duplicating row data."""
    registry: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = json.dumps(
            {
                "configuration_id": row["configuration_id"],
                "factor": row["factor"],
                "parameter": row["parameter"],
                "value": row["value"],
                "confidence_level": row["confidence_level"],
                "counterexample_dose": row["counterexample_dose"],
                "grammar": row.get("grammar"),
            },
            sort_keys=True,
            default=str,
        )
        registry.setdefault(key, {
            "configuration_id": row["configuration_id"],
            "factor": row["factor"],
            "parameter": row["parameter"],
            "value": row["value"],
            "confidence_level": row["confidence_level"],
            "counterexample_dose": row["counterexample_dose"],
            "grammar": row.get("grammar"),
        })
    return sorted(registry.values(), key=lambda item: json.dumps(item, sort_keys=True, default=str))


def _grid_accounting(
    *,
    grid: Mapping[str, Sequence[float]],
    doses: Sequence[float],
    registry: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = [
        {"configuration_id": "default", "factor": "default", "parameter": "registered", "value": REGISTERED_CONFIDENCE},
        *[
            {"configuration_id": f"confidence_{value:.2f}", "factor": "confidence", "parameter": "confidence_level", "value": value}
            for value in grid["confidence_level"]
        ],
        *[
            {"configuration_id": f"grammar_depth_{int(value):+d}", "factor": "grammar", "parameter": "depth_offset", "value": value}
            for value in grid["grammar_depth_offset"]
        ],
        *[
            {"configuration_id": f"grammar_literals_{int(value):+d}", "factor": "grammar", "parameter": "literal_offset", "value": value}
            for value in grid["grammar_literal_offset"]
        ],
        *[
            {"configuration_id": f"dose_{int(round(value * 100))}", "factor": "counterexample_dose", "parameter": "dose", "value": value}
            for value in doses
        ],
    ]
    executed = {str(item["configuration_id"]) for item in registry}
    return {
        "expected_configurations": expected,
        "executed_configuration_ids": sorted(executed),
        "not_executed": [
            {
                **item,
                "status": "not_applicable",
                "reason": "grammar bound rejects this offset; no independent implementation exists at the requested +1 setting",
            }
            for item in expected
            if item["configuration_id"] not in executed
        ],
    }


def run_support_boundary_sensitivity(
    *,
    surface_count: int = 64,
    generator_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    families: Sequence[str] | None = None,
    doses: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    registered_artifact: Path | None = None,
) -> dict[str, Any]:
    selected_families = tuple(families or PILOT_FAMILIES)
    seeds = tuple(int(seed) for seed in generator_seeds)
    grid = sensitivity_grid()
    metric_rows: list[dict[str, Any]] = []
    dose_rows: list[dict[str, Any]] = []
    family_rules: dict[tuple[str, int], Mapping[str, Any] | None] = {}
    truth_audits: list[dict[str, Any]] = []

    case_cache: dict[tuple[str, int], dict[str, list[Any]]] = {}
    for family in selected_families:
        for seed in seeds:
            case_cache[(family, seed)] = _split_cases(family, seed, surface_count)
            all_cases = [item for pool in case_cache[(family, seed)].values() for item in pool]
            truth_audits.append(_latent_truth_crosscheck(family, seed, all_cases))

    availability = {
        "confidence_level": {
            "status": "available",
            "source": "StatisticalCEGIS.delta = 1 - confidence_level",
            "values": list(grid["confidence_level"]),
            "interval_recomputation": False,
            "reason": "BoundaryCase stores degenerate intervals; confidence changes certificate alpha metadata only.",
        },
        "repetition_fraction": {
            "status": "not_applicable",
            "registered_repetitions": REGISTERED_REPETITIONS,
            "values": list(grid["repetition_fraction"]),
            "reason": "BOUNDARY_REPETITIONS=1 and no raw repetition arrays are persisted.",
        },
        "delta": {
            "status": "not_applicable",
            "registered_value": REGISTERED_EPSILON_TRUE,
            "values": [REGISTERED_EPSILON_TRUE * float(value) for value in grid["delta_multiplier"]],
            "reason": "The registered boundary runner uses epsilon_true=epsilon_false=0.0; no positive delta0 is instantiated.",
        },
        "utility_scale": {
            "status": "not_applicable",
            "registered_value": REGISTERED_UTILITY_SCALE,
            "values": None,
            "reason": "The provider-free BoundaryCase effect is computed from FamilyEnvironment utility differences; core.utility.utility_effect is not called.",
        },
        "grammar": {
            "status": "available_with_bounds",
            "depth_offsets": list(grid["grammar_depth_offset"]),
            "literal_offsets": list(grid["grammar_literal_offset"]),
            "dl_budget": "not_applicable: predicate grammar has no independent description-length cap",
        },
    }

    # The registered effect-sign truth is kept fixed while confidence and grammar
    # factors change.  It is used only after policy decisions have been frozen.
    for family in selected_families:
        for seed in seeds:
            split = case_cache[(family, seed)]
            synthesis, calibration, evaluation = split["synthesis"], split["calibration"], split["evaluation"]
            registered_labels = _classify_cases(
                evaluation,
                epsilon_true=REGISTERED_EPSILON_TRUE,
                epsilon_false=REGISTERED_EPSILON_FALSE,
            )
            truth = {key: value == "positive_anchor" for key, value in registered_labels.items()}
            default_metrics: dict[str, Any] | None = None
            configurations: list[tuple[str, str, str, Any, float, int, int]] = [
                ("default", "default", "registered", REGISTERED_CONFIDENCE, 0, 0, 0),
            ]
            for confidence in grid["confidence_level"]:
                configurations.append((f"confidence_{confidence:.2f}", "confidence", "confidence_level", confidence, 0, 0, 0))
            for offset in grid["grammar_depth_offset"]:
                try:
                    _grammar_variant(family, depth_offset=int(offset))
                except ValueError:
                    continue
                configurations.append((f"grammar_depth_{int(offset):+d}", "grammar", "depth_offset", int(offset), int(offset), 0, 0))
            for offset in grid["grammar_literal_offset"]:
                try:
                    _grammar_variant(family, literal_offset=int(offset))
                except ValueError:
                    continue
                configurations.append((f"grammar_literals_{int(offset):+d}", "grammar", "literal_offset", int(offset), 0, int(offset), 0))

            for configuration_id, factor, parameter, value, depth_offset, literal_offset, _unused in configurations:
                confidence = float(value) if factor == "confidence" else REGISTERED_CONFIDENCE
                grammar, grammar_meta = _grammar_variant(
                    family,
                    depth_offset=depth_offset,
                    literal_offset=literal_offset,
                )
                predictions, result, status_counts = _fit_predictions(
                    family,
                    synthesis,
                    evaluation,
                    dose=REGISTERED_DOSE,
                    confidence_level=confidence,
                    epsilon_true=REGISTERED_EPSILON_TRUE,
                    epsilon_false=REGISTERED_EPSILON_FALSE,
                    grammar=grammar,
                )
                row = _evaluate_row(
                    family=family,
                    seed=seed,
                    evaluation=evaluation,
                    calibration=calibration,
                    predictions=predictions,
                    truth=truth,
                    predicate=(result or {}).get("predicate"),
                    result=result,
                    status_counts=status_counts,
                    confidence_level=confidence,
                    delta=REGISTERED_EPSILON_TRUE,
                    configuration_id=configuration_id,
                    factor=factor,
                    parameter=parameter,
                    value=value,
                    dose=REGISTERED_DOSE,
                    default_metrics=default_metrics,
                )
                row["grammar"] = grammar_meta
                row["certificate_confidence_alpha"] = 1.0 - confidence
                if configuration_id == "default":
                    # The registered row is the reference point for every
                    # subsequent configuration, including dose summaries.
                    # Materialize zero deltas here so the report schema does
                    # not depend on a prior row being present.
                    row["delta_coverage"] = 0.0
                    row["delta_unsupported_deployed"] = 0.0
                    row["delta_support_recall"] = 0.0
                    row["delta_recall"] = 0.0
                metric_rows.append(row)
                if configuration_id == "default":
                    default_metrics = {
                        "coverage": row["coverage"],
                        "unsupported_deployed": row["unsupported_deployed"],
                        "support_recall": row["support_recall"],
                    }
                    family_rules[(family, seed)] = row["predicate"]

            # Dose rows are a fixed sample-complexity summary under the registered
            # configuration and use the same evaluation IDs for every dose.
            for dose in doses:
                grammar, grammar_meta = _grammar_variant(family)
                predictions, result, status_counts = _fit_predictions(
                    family,
                    synthesis,
                    evaluation,
                    dose=float(dose),
                    confidence_level=REGISTERED_CONFIDENCE,
                    epsilon_true=REGISTERED_EPSILON_TRUE,
                    epsilon_false=REGISTERED_EPSILON_FALSE,
                    grammar=grammar,
                )
                dose_rows.append(_evaluate_row(
                    family=family,
                    seed=seed,
                    evaluation=evaluation,
                    calibration=calibration,
                    predictions=predictions,
                    truth=truth,
                    predicate=(result or {}).get("predicate"),
                    result=result,
                    status_counts=status_counts,
                    confidence_level=REGISTERED_CONFIDENCE,
                    delta=REGISTERED_EPSILON_TRUE,
                    configuration_id=f"dose_{int(round(float(dose) * 100))}",
                    factor="counterexample_dose",
                    parameter="dose",
                    value=float(dose),
                    dose=float(dose),
                    default_metrics=default_metrics,
                ) | {"grammar": grammar_meta})

    truth_mismatches = sum(int(item["mismatch_count"]) for item in truth_audits)
    configuration_registry = _configuration_registry(metric_rows + dose_rows)
    grid_accounting = _grid_accounting(grid=grid, doses=doses, registry=configuration_registry)
    report = {
        "schema_version": 1,
        "analysis": "provider_free_context_indexed_support_boundary_sensitivity",
        "provider_calls": 0,
        "surface_count": int(surface_count),
        "generator_seeds": list(seeds),
        "families": list(selected_families),
        "doses": [float(dose) for dose in doses],
        "split": "deterministic sha256 observation split: 50% synthesis, 25% calibration, 25% evaluation",
        "registered_input_artifact": str(registered_artifact) if registered_artifact else None,
        "registered_input_artifact_sha256": _sha256(registered_artifact) if registered_artifact else None,
        "fixed_sensitivity_grid": {key: list(value) for key, value in grid.items()},
        "availability": availability,
        "hidden_truth_usage": "scoring_only",
        "hidden_truth_semantics": {
            "registered_definition": "positive_anchor iff gate_passed and effect_lower > epsilon_true; counterexample iff not gate_passed or effect_upper <= epsilon_false; otherwise unresolved",
            "registered_epsilon_true": REGISTERED_EPSILON_TRUE,
            "registered_epsilon_false": REGISTERED_EPSILON_FALSE,
            "delta_independent": True,
            "latent_family_crosscheck": truth_audits,
            "crosscheck_mismatch_count": truth_mismatches,
            "decision": "registered effect-sign scoring truth retained; latent FamilySpec applicability is audit-only and never passed to learner/router",
            "evaluator_boundary": "scoring truth is joined after frozen decisions are serialized",
        },
        "underidentified_operational_semantics": "abstain -> optional acquire/revalidate when admissible -> otherwise remain unresolved",
        "metric_rows": metric_rows,
        "dose_rows": dose_rows,
        "configuration_registry": configuration_registry,
        "grid_accounting": grid_accounting,
        "stability": _stability_report(family_rules, families=selected_families, generator_seeds=seeds),
        "claim_boundary": "provider-free sensitivity and predicate-stability analysis; no hosted efficacy claim",
        "code_commit": _git_commit(),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-count", type=int, default=64)
    parser.add_argument("--registered-artifact", type=Path, default=ROOT / "results" / "support_boundary_analysis.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "support_boundary_sensitivity.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "support_boundary_sensitivity_manifest.json")
    args = parser.parse_args()
    report = run_support_boundary_sensitivity(
        surface_count=args.surface_count,
        registered_artifact=args.registered_artifact,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "analysis": report["analysis"],
        "artifact": str(args.out),
        "artifact_sha256": _sha256(args.out),
        "registered_input_artifact": str(args.registered_artifact),
        "registered_input_artifact_sha256": report["registered_input_artifact_sha256"],
        "code_commit": report["code_commit"],
        "surface_count": report["surface_count"],
        "generator_seeds": report["generator_seeds"],
        "families": report["families"],
        "fixed_sensitivity_grid": report["fixed_sensitivity_grid"],
        "configuration_registry": report["configuration_registry"],
        "grid_accounting": report["grid_accounting"],
        "provider_calls": 0,
        "claim_boundary": report["claim_boundary"],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "metric_rows": len(report["metric_rows"]), "output": str(args.out), "manifest": str(args.manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
