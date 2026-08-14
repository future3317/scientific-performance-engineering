#!/usr/bin/env python3
"""Reject incomparable benchmark records and calculate before/after deltas."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


COMPARABILITY_PATHS = (
    "identity.repository",
    "identity.base_revision",
    "identity.benchmark_harness_hash",
    "hardware.cpu",
    "hardware.cpu_affinity",
    "hardware.numa_nodes",
    "hardware.gpu",
    "hardware.gpu_uuid",
    "hardware.device_index",
    "hardware.world_size",
    "hardware.storage",
    "hardware.host_contention",
    "software.python",
    "software.pytorch",
    "software.cuda_or_rocm",
    "software.driver",
    "software.torch_geometric",
    "software.e3nn",
    "software.cuequivariance",
    "software.triton",
    "software.torchao",
    "software.transformer_engine",
    "software.nccl",
    "contract.scientific_contract_id",
    "contract.model_outputs",
    "contract.loss_objective",
    "contract.invariance_equivariance_physical_constraints",
    "contract.probability_or_sampler",
    "contract.data_manifest_or_hash",
    "contract.data_order_augmentation",
    "contract.seeds",
    "contract.precision_policy",
    "contract.effective_batch",
    "contract.optimizer_lr_scheduler_clipping",
    "contract.world_size",
    "contract.checkpoint_resume",
    "contract.stochastic_thinning",
    "contract.checkpoint_state_contract",
    "contract.checkpoint_state_contract.dataloader_cursor",
    "contract.checkpoint_state_contract.ema_swa_scheduler",
    "contract.gradient_clipping_contract",
    "contract.quality_gates",
    "contract.numerical_tolerances",
    "work.unit",
    "work.total_units",
    "work.microbatch",
    "work.effective_batch",
    "work.accumulation_steps",
    "work.warmup_steps",
    "work.measured_steps",
    "work.repetitions",
    "work.timing_boundaries",
    "work.includes_data_loading",
    "work.includes_h2d",
    "work.includes_optimizer",
    "work.includes_logging",
    "work.includes_checkpoint",
    "work.includes_validation",
    "work.optimization_objective",
    "work.benchmark_levels",
    "work.logical_update_definition",
    "work.task_composition",
    "work.logical_update_dag",
    "work.sync_census",
    "work.cache_contract",
    "work.h2d_proof",
    "work.timing_bucket_definition",
    "work.cuda_timing_proof",
    "acceptance.primary_metric",
    "acceptance.higher_is_better",
    "acceptance.minimum_improvement_percent",
    "acceptance.noise_floor_percent",
    "acceptance.confidence_level",
    "acceptance.bootstrap_samples",
    "acceptance.minimum_runs",
    "acceptance.required_quality_gates",
    "acceptance.max_unaccounted_ratio",
    "measurements.run_order",
)

SYSTEMS_CANDIDATE_PATHS = frozenset(
    {
        "identity.commit",
        "identity.dirty",
        "identity.command",
        "identity.config",
        "software.compile_backend",
        "software.compile_mode",
        "compiler.fallback",
    }
)

ALGORITHMIC_PATHS = frozenset(
    {
        "contract.loss_objective",
        "contract.probability_or_sampler",
        "contract.data_manifest_or_hash",
        "contract.data_order_augmentation",
        "contract.seeds",
        "contract.effective_batch",
        "contract.world_size",
        "work.effective_batch",
        "work.task_composition",
        "work.logical_update_definition",
    }
)

REQUIRED_PATHS = (
    "identity.repository",
    "identity.base_revision",
    "identity.benchmark_harness_hash",
    "identity.candidate_patch_hash",
    "hardware.cpu",
    "hardware.gpu",
    "hardware.gpu_uuid",
    "hardware.device_index",
    "hardware.world_size",
    "software.python",
    "software.pytorch",
    "software.cuda_or_rocm",
    "software.driver",
    "software.compile_backend",
    "software.compile_mode",
    "compiler.compile_cache_state",
    "contract.scientific_contract_id",
    "contract.data_manifest_or_hash",
    "contract.seeds",
    "contract.effective_batch",
    "contract.world_size",
    "contract.stochastic_thinning",
    "contract.checkpoint_state_contract",
    "contract.gradient_clipping_contract",
    "work.unit",
    "work.total_units",
    "work.effective_batch",
    "work.measured_steps",
    "work.repetitions",
    "work.timing_boundaries",
    "work.optimization_objective",
    "work.benchmark_levels",
    "work.logical_update_definition",
    "work.task_composition",
    "work.timing_bucket_definition",
    "work.cuda_timing_proof",
    "metrics.throughput_units_per_s",
    "metrics.step_ms_p50",
    "metrics.step_ms_p95",
    "metrics.amortized_training_throughput",
    "metrics.time_to_quality_seconds",
    "preflight.compatibility_status",
    "preflight.runtime_topology",
)

REQUIRED_CANDIDATE_PATHS = (
    "candidate.changed_levers",
    "candidate.active_path_evidence",
    "candidate.reference_output",
    "candidate.falsification_test",
)

REQUIRED_ACCEPTANCE_PATHS = (
    "acceptance.primary_metric",
    "acceptance.higher_is_better",
    "acceptance.minimum_improvement_percent",
    "acceptance.noise_floor_percent",
    "acceptance.confidence_level",
    "acceptance.bootstrap_samples",
    "acceptance.minimum_runs",
    "acceptance.required_quality_gates",
    "acceptance.max_unaccounted_ratio",
    "measurements.run_order",
    "measurements.runs",
)

DELTA_METRICS = {
    "throughput_units_per_s": True,
    "step_ms_p50": False,
    "step_ms_p95": False,
    "peak_allocated_mb": False,
    "peak_reserved_mb": False,
    "external_memory_mb": False,
    "host_rss_mb": False,
    "data_wait_ms_p50": False,
    "h2d_ms_p50": False,
    "forward_ms_p50": False,
    "loss_ms_p50": False,
    "backward_ms_p50": False,
    "optimizer_ms_p50": False,
    "communication_ms_p50": False,
    "scaling_efficiency_percent": True,
}


def get_path(record: dict[str, Any], dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def percent_change(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return None
    return (candidate / baseline - 1.0) * 100.0


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def robust_statistics(values: list[float]) -> dict[str, float | None]:
    median = percentile(values, 0.5)
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    deviations = [abs(value - median) for value in values] if median is not None else []
    return {
        "median": median,
        "iqr": q3 - q1 if q1 is not None and q3 is not None else None,
        "mad": percentile(deviations, 0.5),
    }


def bootstrap_ci(
    values: list[float], confidence_level: float, samples: int, seed: int = 0
) -> tuple[float | None, float | None]:
    if not values or samples < 1:
        return None, None
    rng = random.Random(seed)
    medians: list[float] = []
    for _ in range(samples):
        resample = [values[rng.randrange(len(values))] for _ in values]
        median = percentile(resample, 0.5)
        if median is not None:
            medians.append(median)
    alpha = (1.0 - confidence_level) / 2.0
    return percentile(medians, alpha), percentile(medians, 1.0 - alpha)


def record_metric_samples(record: dict[str, Any], metric: str) -> list[float]:
    runs = get_path(record, "measurements.runs")
    if not isinstance(runs, list):
        return []
    values: list[float] = []
    for run in runs:
        if isinstance(run, dict):
            value = finite_number(run.get(metric))
            if value is not None:
                values.append(value)
    return values


def timing_status(record: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    proof = get_path(record, "work.cuda_timing_proof")
    buckets = get_path(record, "work.timing_buckets")
    step_ms = finite_number(get_path(record, "metrics.step_ms_p50"))
    reasons: list[str] = []
    if not isinstance(proof, dict) or any(
        missing(proof.get(key)) for key in ("clock", "stream", "completion_proof")
    ):
        reasons.append("missing CUDA timing completion proof")
    if not isinstance(buckets, dict) or not buckets:
        reasons.append("missing timing bucket proofs")
        return reasons, {}
    bucket_sum = 0.0
    for name, bucket in buckets.items():
        if not isinstance(bucket, dict):
            reasons.append(f"timing bucket {name} is not an object")
            continue
        if any(missing(bucket.get(key)) for key in ("clock", "stream", "completion_proof")):
            reasons.append(f"timing bucket {name} lacks completion proof")
        value = finite_number(bucket.get("p50_ms"))
        if value is None or value < 0:
            reasons.append(f"timing bucket {name} lacks finite p50_ms")
        else:
            bucket_sum += value
    if step_ms is None or step_ms <= 0:
        reasons.append("step_ms_p50 must be positive for timing accounting")
        return reasons, {"bucket_sum_ms": bucket_sum}
    calculated_ratio = abs(step_ms - bucket_sum) / step_ms
    declared_ratio = finite_number(get_path(record, "work.unaccounted_ratio"))
    if declared_ratio is None:
        reasons.append("missing work.unaccounted_ratio")
    elif abs(declared_ratio - calculated_ratio) > 0.01:
        reasons.append("declared unaccounted ratio disagrees with bucket sum")
    return reasons, {
        "step_ms_p50": step_ms,
        "bucket_sum_ms": bucket_sum,
        "calculated_unaccounted_ratio": calculated_ratio,
        "declared_unaccounted_ratio": declared_ratio,
    }


def required_quality_gates(record: dict[str, Any]) -> list[str]:
    value = get_path(record, "acceptance.required_quality_gates")
    return value if isinstance(value, list) and all(isinstance(item, str) for item in value) else []


def compare_records(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    allowed_differences: set[str],
    min_throughput_gain: float | None = None,
    max_p95_regression: float | None = None,
    max_peak_memory_regression: float | None = None,
    required_quality: tuple[str, ...] = (),
) -> dict[str, Any]:
    errors: list[str] = []
    for path in sorted(allowed_differences - SYSTEMS_CANDIDATE_PATHS):
        if path in ALGORITHMIC_PATHS:
            errors.append(f"algorithmic path cannot be allowed as a systems difference: {path}")
        else:
            errors.append(f"unknown or immutable --allow-difference path: {path}")
    for label, record in (("baseline", baseline), ("candidate", candidate)):
        if record.get("schema_version") != 3:
            errors.append(f"{label}: schema_version must be 3")
        for path in REQUIRED_PATHS:
            if missing(get_path(record, path)):
                errors.append(f"{label}: missing required field {path}")

    mismatches: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    algorithmic_mismatches: list[dict[str, Any]] = []
    for path in COMPARABILITY_PATHS:
        before = get_path(baseline, path)
        after = get_path(candidate, path)
        if before != after:
            item = {"path": path, "baseline": before, "candidate": after}
            if path in ALGORITHMIC_PATHS:
                algorithmic_mismatches.append(item)
            else:
                mismatches.append(item)
    for path in sorted(allowed_differences):
        before = get_path(baseline, path)
        after = get_path(candidate, path)
        if before != after:
            item = {"path": path, "baseline": before, "candidate": after}
            allowed.append(item)
            mismatches = [item for item in mismatches if item["path"] != path]

    deltas: dict[str, Any] = {}
    for metric, higher_is_better in DELTA_METRICS.items():
        before = finite_number(get_path(baseline, f"metrics.{metric}"))
        after = finite_number(get_path(candidate, f"metrics.{metric}"))
        if before is None or after is None:
            continue
        change = percent_change(before, after)
        deltas[metric] = {
            "baseline": before,
            "candidate": after,
            "percent_change": change,
            "improvement_percent": change if higher_is_better else (-change if change is not None else None),
        }

    gates: list[dict[str, Any]] = []
    inconclusive_reasons: list[str] = []
    for label, record in (("baseline", baseline), ("candidate", candidate)):
        for path in REQUIRED_ACCEPTANCE_PATHS:
            if missing(get_path(record, path)):
                inconclusive_reasons.append(f"{label}: missing acceptance field {path}")
        if not required_quality_gates(record):
            inconclusive_reasons.append(f"{label}: acceptance.required_quality_gates must be non-empty")
        if get_path(record, "preflight.compatibility_status") != "pass":
            inconclusive_reasons.append(f"{label}: runtime compatibility preflight is not proven pass")
    for path in REQUIRED_CANDIDATE_PATHS:
        if missing(get_path(candidate, path)):
            inconclusive_reasons.append(f"candidate missing evidence: {path}")
    if missing(get_path(candidate, "identity.declared_change_set")):
        inconclusive_reasons.append("candidate identity.declared_change_set is missing")
    baseline_timing_reasons, baseline_timing = timing_status(baseline)
    candidate_timing_reasons, candidate_timing = timing_status(candidate)
    inconclusive_reasons.extend(f"baseline: {reason}" for reason in baseline_timing_reasons)
    inconclusive_reasons.extend(f"candidate: {reason}" for reason in candidate_timing_reasons)
    acceptance = get_path(baseline, "acceptance")
    candidate_acceptance = get_path(candidate, "acceptance")
    if acceptance != candidate_acceptance:
        inconclusive_reasons.append("baseline and candidate acceptance policies differ")
    for record_label, record in (("baseline", baseline), ("candidate", candidate)):
        for path in REQUIRED_CANDIDATE_PATHS:
            if record_label == "candidate" and missing(get_path(record, path)):
                inconclusive_reasons.append(f"candidate missing evidence: {path}")
    if get_path(candidate, "compiler.fallback") is not False:
        inconclusive_reasons.append("candidate compiler fallback is not proven false")
    if get_path(candidate, "software.compile_backend") not in (None, "", "eager") and not get_path(candidate, "candidate.active_path_evidence"):
        inconclusive_reasons.append("compiled candidate lacks active-path evidence")

    primary_metric = get_path(baseline, "acceptance.primary_metric")
    higher_is_better = get_path(baseline, "acceptance.higher_is_better")
    baseline_values = record_metric_samples(baseline, primary_metric) if isinstance(primary_metric, str) else []
    candidate_values = record_metric_samples(candidate, primary_metric) if isinstance(primary_metric, str) else []
    min_runs = get_path(baseline, "acceptance.minimum_runs")
    if not isinstance(min_runs, int) or len(baseline_values) < min_runs or len(candidate_values) < min_runs:
        inconclusive_reasons.append("insufficient raw measurement runs")
    if len(baseline_values) != len(candidate_values) or not baseline_values:
        inconclusive_reasons.append("baseline/candidate raw measurement windows are not paired")
    else:
        improvements = [
            ((after / before) - 1.0) * 100.0 if higher_is_better else ((before / after) - 1.0) * 100.0
            for before, after in zip(baseline_values, candidate_values)
            if before > 0 and after > 0
        ]
        confidence = get_path(baseline, "acceptance.confidence_level")
        bootstrap_samples = get_path(baseline, "acceptance.bootstrap_samples")
        if not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
            inconclusive_reasons.append("invalid acceptance confidence level")
            confidence = 0.95
        if not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
            inconclusive_reasons.append("invalid acceptance bootstrap sample count")
            bootstrap_samples = 1
        ci_low, ci_high = bootstrap_ci(improvements, float(confidence), bootstrap_samples)
        stats = robust_statistics(improvements)
        stats.update({"ci_lower": ci_low, "ci_upper": ci_high})
        deltas["primary_metric_statistics"] = stats
        threshold = finite_number(get_path(baseline, "acceptance.minimum_improvement_percent"))
        noise_floor = finite_number(get_path(baseline, "acceptance.noise_floor_percent"))
        required_margin = max(threshold or 0.0, noise_floor or 0.0)
        gates.append(
            {
                "gate": "primary_metric_confidence_interval",
                "threshold": required_margin,
                "observed": ci_low,
                "passed": ci_low is not None and ci_low >= required_margin,
            }
        )
        if ci_low is None or ci_low < required_margin:
            inconclusive_reasons.append("improvement confidence interval does not clear threshold/noise floor")

    max_unaccounted = finite_number(get_path(baseline, "acceptance.max_unaccounted_ratio"))
    for label, timing in (("baseline", baseline_timing), ("candidate", candidate_timing)):
        ratio = finite_number(timing.get("calculated_unaccounted_ratio"))
        if max_unaccounted is not None and (ratio is None or ratio > max_unaccounted):
            inconclusive_reasons.append(f"{label} unaccounted timing ratio exceeds threshold")

    if min_throughput_gain is not None:
        change = get_path(deltas, "throughput_units_per_s.percent_change")
        gates.append(
            {
                "gate": "minimum_throughput_gain_percent",
                "threshold": min_throughput_gain,
                "observed": change,
                "passed": change is not None and change >= min_throughput_gain,
            }
        )
    if max_p95_regression is not None:
        change = get_path(deltas, "step_ms_p95.percent_change")
        gates.append(
            {
                "gate": "maximum_step_p95_regression_percent",
                "threshold": max_p95_regression,
                "observed": change,
                "passed": change is not None and change <= max_p95_regression,
            }
        )
    if max_peak_memory_regression is not None:
        change = get_path(deltas, "peak_allocated_mb.percent_change")
        gates.append(
            {
                "gate": "maximum_peak_allocated_memory_regression_percent",
                "threshold": max_peak_memory_regression,
                "observed": change,
                "passed": change is not None and change <= max_peak_memory_regression,
            }
        )
    quality_keys = list(dict.fromkeys(required_quality_gates(baseline) + list(required_quality)))
    for key in quality_keys:
        observed = get_path(candidate, f"quality.{key}")
        if observed is None:
            inconclusive_reasons.append(f"missing quality result: {key}")
        gates.append(
            {
                "gate": f"quality.{key}",
                "threshold": True,
                "observed": observed,
                "passed": observed is True,
            }
        )

    comparable = not errors and not mismatches and not algorithmic_mismatches
    if algorithmic_mismatches:
        assessment = "algorithmic_experiment"
    elif not comparable:
        assessment = "incomparable"
    elif inconclusive_reasons:
        assessment = "inconclusive"
    elif not gates:
        assessment = "comparable_unjudged"
    elif all(gate["passed"] for gate in gates):
        assessment = "gates_passed"
    else:
        assessment = "gates_failed"

    return {
        "assessment": assessment,
        "comparable": comparable,
        "errors": errors,
        "mismatches": mismatches,
        "algorithmic_mismatches": algorithmic_mismatches,
        "allowed_differences": allowed,
        "deltas": deltas,
        "gates": gates,
        "inconclusive_reasons": inconclusive_reasons,
        "timing": {"baseline": baseline_timing, "candidate": candidate_timing},
        "warning": "Passing numeric gates does not replace repository correctness or scientific-quality gates.",
    }


def load_record(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def exit_code_for_assessment(assessment: str) -> int:
    """Return a non-zero code when comparison is not an accepted result."""
    if assessment == "gates_passed":
        return 0
    if assessment == "algorithmic_experiment":
        return 4
    if assessment == "comparable_unjudged":
        return 3
    if assessment == "inconclusive":
        return 3
    return 2


def self_test() -> None:
    record: dict[str, Any] = {
        "schema_version": 3,
        "identity": {
            "repository": "demo",
            "commit": "abc",
            "dirty": False,
            "command": ["run"],
            "config": "cfg",
            "base_revision": "base-123",
            "benchmark_harness_hash": "harness-123",
            "candidate_patch_hash": "none",
            "declared_change_set": [],
        },
        "hardware": {
            "cpu": "cpu",
            "cpu_affinity": [0],
            "numa_nodes": 1,
            "gpu": "gpu",
            "gpu_uuid": "GPU-abc",
            "device_index": 0,
            "world_size": 1,
            "storage": "ssd",
        },
        "software": {
            **{key: "same" for key in (
                "python", "pytorch", "cuda_or_rocm", "driver", "torch_geometric", "e3nn",
                "cuequivariance", "triton", "torchao", "transformer_engine", "nccl"
            )},
            "compile_backend": "eager",
            "compile_mode": "none",
        },
        "contract": {
            "scientific_contract_id": "frozen-v1",
            "model_outputs": "y",
            "loss_objective": "loss",
            "invariance_equivariance_physical_constraints": "gate",
            "probability_or_sampler": "fixed",
            "data_manifest_or_hash": "sha256:x",
            "data_order_augmentation": "fixed",
            "seeds": [1],
            "precision_policy": "fp32",
            "effective_batch": 4,
            "optimizer_lr_scheduler_clipping": "fixed",
            "world_size": 1,
            "checkpoint_resume": "fixed",
            "stochastic_thinning": {"enabled": False},
            "checkpoint_state_contract": {
                "boundary": "optimizer",
                "dataloader_cursor": "fixed",
                "ema_swa_scheduler": "fixed",
            },
            "gradient_clipping_contract": {"enabled": False},
            "quality_gates": {"ok": True},
            "numerical_tolerances": {"rtol": 1e-5},
        },
        "work": {
            "unit": "samples",
            "total_units": 100,
            "microbatch": 4,
            "effective_batch": 4,
            "accumulation_steps": 1,
            "warmup_steps": 5,
            "measured_steps": 25,
            "repetitions": 3,
            "timing_boundaries": "step",
            "includes_data_loading": True,
            "includes_h2d": True,
            "includes_optimizer": True,
            "includes_logging": False,
            "includes_checkpoint": False,
            "includes_validation": False,
            "optimization_objective": "throughput",
            "benchmark_levels": ["end-to-end"],
            "logical_update_definition": "one optimizer update",
            "task_composition": {"main": 1},
            "logical_update_dag": [{"stage": stage} for stage in (
                "fetch", "cpu_preprocess", "h2d", "gpu_preprocess", "forward", "loss",
                "autograd_aux", "backward", "grad_transform", "clipping", "communication",
                "optimizer", "scheduler", "ema_swa", "metrics", "checkpoint", "validation"
            )],
            "sync_census": [{"event": "loss.item", "disposition": "removable"}],
            "cache_contract": {
                "cache_state": "warm",
                "dataset_identity": "dataset",
                "sample_identity": "sample",
                "cutoff": "cutoff",
                "pbc_convention": "pbc",
                "augmentation": "fixed",
                "species_mapping": "species",
                "graph_builder_version": "graph-v1",
                "dtype_layout": "fp32-contiguous",
                "basis_version": "basis-v1",
            },
            "h2d_proof": {
                "is_pinned": True,
                "non_blocking": True,
                "copy_stream": "copy",
                "source_lifetime": "until-consume",
                "consumer_dependency": "event",
                "overlap_evidence": "timeline",
            },
            "timing_bucket_definition": {"step": "synchronized"},
            "cuda_timing_proof": {
                "clock": "cuda_event",
                "stream": "current",
                "completion_proof": "paired_events",
            },
            "timing_buckets": {
                "forward": {
                    "clock": "cuda_event",
                    "stream": "current",
                    "completion_proof": "paired_events",
                    "p50_ms": 8.0,
                },
                "loss": {
                    "clock": "cuda_event",
                    "stream": "current",
                    "completion_proof": "paired_events",
                    "p50_ms": 2.0,
                },
            },
            "unaccounted_ratio": 0.01,
        },
        "metrics": {
            "throughput_units_per_s": 100.0,
            "step_ms_p50": 10.0,
            "step_ms_p95": 12.0,
            "peak_allocated_mb": 1000.0,
            "amortized_training_throughput": 90.0,
            "time_to_quality_seconds": 1000.0,
        },
        "compiler": {
            "fallback": False,
            "compile_cache_state": "disabled",
            "cache_fingerprint": "none",
            "cache_hit_miss_evidence": "disabled",
        },
        "candidate": {
            "hypothesis": "vectorize measured loop",
            "measured_bottleneck_share": 0.6,
            "changed_levers": ["vectorize mechanism loop"],
            "expected_metric_movement": "throughput up",
            "semantic_risk": "reduction order",
            "falsification_test": "compare outputs and gradients",
            "reference_output": "reference.npz",
            "amdahl_ceiling": 1.5,
            "explicitly_authorized_algorithmic_changes": [],
            "active_path_evidence": "runtime counter=vectorized",
        },
        "acceptance": {
            "primary_metric": "throughput_units_per_s",
            "higher_is_better": True,
            "minimum_improvement_percent": 5.0,
            "noise_floor_percent": 1.0,
            "confidence_level": 0.95,
            "bootstrap_samples": 200,
            "minimum_runs": 3,
            "required_quality_gates": ["numerics_passed"],
            "max_unaccounted_ratio": 0.05,
        },
        "measurements": {
            "run_order": "A/B/A/B randomized",
            "runs": [
                {"throughput_units_per_s": 100.0},
                {"throughput_units_per_s": 101.0},
                {"throughput_units_per_s": 99.0},
            ],
        },
        "quality": {"numerics_passed": True},
        "preflight": {
            "compatibility_status": "pass",
            "runtime_topology": {"device_rank_mapping": "fixed"},
            "unsupported_combinations": [],
        },
    }
    candidate = copy.deepcopy(record)
    candidate["metrics"]["throughput_units_per_s"] = 110.0
    candidate["metrics"]["step_ms_p95"] = 11.0
    candidate["identity"]["candidate_patch_hash"] = "patch-456"
    candidate["identity"]["declared_change_set"] = ["vectorize mechanism loop"]
    candidate["measurements"]["runs"] = [
        {"throughput_units_per_s": 106.0},
        {"throughput_units_per_s": 107.0},
        {"throughput_units_per_s": 105.0},
    ]
    result = compare_records(record, candidate, set(), 5.0, 0.0, 5.0, ("numerics_passed",))
    assert result["assessment"] == "gates_passed", result
    no_quality = copy.deepcopy(candidate)
    no_quality["quality"] = {}
    assert compare_records(record, no_quality, set())["assessment"] == "inconclusive"
    algorithmic = copy.deepcopy(candidate)
    algorithmic["contract"]["data_manifest_or_hash"] = "sha256:changed"
    assert compare_records(record, algorithmic, {"contract.data_manifest_or_hash"})["assessment"] == "algorithmic_experiment"
    code_only = copy.deepcopy(candidate)
    code_only["identity"]["commit"] = "different-commit"
    assert compare_records(record, code_only, set())["assessment"] == "gates_passed"
    inactive = copy.deepcopy(candidate)
    inactive["candidate"]["active_path_evidence"] = ""
    assert compare_records(record, inactive, set())["assessment"] == "inconclusive"
    noisy = copy.deepcopy(candidate)
    noisy["measurements"]["runs"] = [
        {"throughput_units_per_s": 90.0},
        {"throughput_units_per_s": 105.0},
        {"throughput_units_per_s": 110.0},
    ]
    assert compare_records(record, noisy, set())["assessment"] == "inconclusive"
    wrong_gpu = copy.deepcopy(candidate)
    wrong_gpu["hardware"]["gpu_uuid"] = "GPU-other"
    assert compare_records(record, wrong_gpu, set())["assessment"] == "incomparable"
    bucket_gap = copy.deepcopy(candidate)
    bucket_gap["work"]["unaccounted_ratio"] = 0.2
    assert compare_records(record, bucket_gap, set())["assessment"] == "inconclusive"
    preflight_gap = copy.deepcopy(candidate)
    preflight_gap["preflight"]["compatibility_status"] = "inconclusive"
    assert compare_records(record, preflight_gap, set())["assessment"] == "inconclusive"
    host_drift = copy.deepcopy(candidate)
    host_drift["hardware"]["host_contention"] = {"load_average_1m": 99.0}
    assert compare_records(record, host_drift, set())["assessment"] == "incomparable"
    candidate["contract"]["seeds"] = [2]
    result = compare_records(record, candidate, set())
    assert result["assessment"] == "algorithmic_experiment", result
    result = compare_records(record, record, set())
    assert result["assessment"] == "inconclusive", result
    result = compare_records(record, record, {"contract.typo"})
    assert result["assessment"] == "incomparable", result
    assert exit_code_for_assessment("gates_passed") == 0
    assert exit_code_for_assessment("comparable_unjudged") == 3
    assert exit_code_for_assessment("incomparable") == 2
    assert exit_code_for_assessment("algorithmic_experiment") == 4
    assert exit_code_for_assessment("inconclusive") == 3
    print("self-test: ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", nargs="?", type=Path)
    parser.add_argument("candidate", nargs="?", type=Path)
    parser.add_argument("--allow-difference", action="append", default=[], metavar="DOTTED_PATH")
    parser.add_argument("--min-throughput-gain-percent", type=float)
    parser.add_argument("--max-p95-regression-percent", type=float)
    parser.add_argument("--max-peak-memory-regression-percent", type=float)
    parser.add_argument("--require-quality", action="append", default=[], metavar="KEY")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.baseline is None or args.candidate is None:
        parser.error("baseline and candidate JSON files are required unless --self-test is used")

    result = compare_records(
        load_record(args.baseline),
        load_record(args.candidate),
        set(args.allow_difference),
        args.min_throughput_gain_percent,
        args.max_p95_regression_percent,
        args.max_peak_memory_regression_percent,
        tuple(args.require_quality),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered, end="")
    raise SystemExit(exit_code_for_assessment(result["assessment"]))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
