#!/usr/bin/env python3
"""Reject incomparable benchmark records and calculate before/after deltas."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any


COMPARABILITY_PATHS = (
    "identity.repository",
    "identity.commit",
    "identity.dirty",
    "identity.command",
    "identity.config",
    "hardware.cpu",
    "hardware.cpu_affinity",
    "hardware.numa_nodes",
    "hardware.gpu",
    "hardware.device_index",
    "hardware.world_size",
    "hardware.storage",
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
    "software.compile_backend",
    "software.compile_mode",
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
    "work.timing_bucket_definition",
    "compiler.fallback",
)

REQUIRED_PATHS = (
    "identity.repository",
    "identity.commit",
    "identity.dirty",
    "identity.command",
    "identity.config",
    "hardware.cpu",
    "hardware.gpu",
    "hardware.device_index",
    "hardware.world_size",
    "software.python",
    "software.pytorch",
    "software.cuda_or_rocm",
    "software.driver",
    "software.compile_backend",
    "software.compile_mode",
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
    "compiler.fallback",
    "metrics.throughput_units_per_s",
    "metrics.step_ms_p50",
    "metrics.step_ms_p95",
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
    for path in sorted(allowed_differences - set(COMPARABILITY_PATHS)):
        errors.append(f"unknown --allow-difference path: {path}")
    for label, record in (("baseline", baseline), ("candidate", candidate)):
        if record.get("schema_version") != 3:
            errors.append(f"{label}: schema_version must be 3")
        for path in REQUIRED_PATHS:
            if missing(get_path(record, path)):
                errors.append(f"{label}: missing required field {path}")

    mismatches: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    for path in COMPARABILITY_PATHS:
        before = get_path(baseline, path)
        after = get_path(candidate, path)
        if before != after:
            item = {"path": path, "baseline": before, "candidate": after}
            (allowed if path in allowed_differences else mismatches).append(item)

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
    for key in required_quality:
        observed = get_path(candidate, f"quality.{key}")
        gates.append(
            {
                "gate": f"quality.{key}",
                "threshold": True,
                "observed": observed,
                "passed": observed is True,
            }
        )

    comparable = not errors and not mismatches
    if not comparable:
        assessment = "incomparable"
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
        "allowed_differences": allowed,
        "deltas": deltas,
        "gates": gates,
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
    if assessment == "comparable_unjudged":
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
        },
        "hardware": {
            "cpu": "cpu",
            "cpu_affinity": [0],
            "numa_nodes": 1,
            "gpu": "gpu",
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
            "checkpoint_state_contract": {"boundary": "optimizer"},
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
            "timing_bucket_definition": {"step": "synchronized"},
        },
        "metrics": {
            "throughput_units_per_s": 100.0,
            "step_ms_p50": 10.0,
            "step_ms_p95": 12.0,
            "peak_allocated_mb": 1000.0,
        },
        "compiler": {"fallback": False},
        "quality": {"numerics_passed": True},
    }
    candidate = copy.deepcopy(record)
    candidate["metrics"]["throughput_units_per_s"] = 110.0
    candidate["metrics"]["step_ms_p95"] = 11.0
    result = compare_records(record, candidate, set(), 5.0, 0.0, 5.0, ("numerics_passed",))
    assert result["assessment"] == "gates_passed", result
    candidate["contract"]["seeds"] = [2]
    result = compare_records(record, candidate, set())
    assert result["assessment"] == "incomparable", result
    result = compare_records(record, record, set())
    assert result["assessment"] == "comparable_unjudged", result
    result = compare_records(record, record, {"contract.typo"})
    assert result["assessment"] == "incomparable", result
    assert exit_code_for_assessment("gates_passed") == 0
    assert exit_code_for_assessment("comparable_unjudged") == 3
    assert exit_code_for_assessment("incomparable") == 2
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
