#!/usr/bin/env python3
"""S0-S6 verification pipeline orchestrator (BENCHMARK_DESIGN.md section 6).

:func:`verify_task` drives a task's ``benchmark.py`` in the fixed order:

    S0 sandbox      -> materialize, hash harness files
    S1 static scan  -> AST/regex anti-cheat + canary check
    S2 correctness  -> fresh seeded inputs, fp64-recomputed reference
    S3 scientific   -> task-declared gates
    S4 activation   -> compile/cache/sync evidence (absence = inconclusive note)
    S5 performance  -> paired interleaved measurement, verified-speedup verdict
    S6 verdict      -> result.json per section 8.1

Correctness and scientific gates precede speed: S5 only runs on the artifact
that passed S2-S4 (P2/P3).
"""

from __future__ import annotations

import json
import copy
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from . import anticheat, miniyaml, runner, stats
from .api import execution_class_for_task, metric_type_for_task
from .fingerprint import capture_fingerprint

TASK_YAML_REQUIRED = (
    "schema_version",
    "task_id",
    "track",
    "family",
    "mechanism",
    "kind",
    "lineage",
    "title",
    "requires_cuda",
    "time_budget_s",
    "workspace",
    "measurement",
    "correctness",
    "scientific_gates",
    "diagnosis",
    "oracle",
    "generator_family_id",
    "oracle_fix_pattern_id",
    "scientific_contract_id",
    "workspace_ast_skeleton_hash",
    "difficulty_tier",
)

TRACKS = {"spe_core", "sciml", "evolution"}
KINDS = {"positive", "counterexample", "do_not_apply"}
BENCHMARK_FUNCTIONS = (
    "load_solution",
    "make_fixtures",
    "run_correctness",
    "run_scientific_gates",
    "run_performance",
)


# ---------------------------------------------------------------------------
# task.yaml loading / structural validation
# ---------------------------------------------------------------------------


def load_task_yaml(task_dir: str | Path) -> dict[str, Any]:
    """Load and structurally validate task.yaml (subset parse + required fields)."""
    task_dir = Path(task_dir)
    if not task_dir.is_dir():
        raise FileNotFoundError(f"task directory not found: {task_dir}")
    path = task_dir / "task.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"task.yaml not found in {task_dir}")
    spec = miniyaml.load(str(path))
    if not isinstance(spec, dict):
        raise ValueError(f"{path} must parse to a mapping")
    missing = [key for key in TASK_YAML_REQUIRED if key not in spec]
    if missing:
        raise ValueError(f"{path} missing required keys: {', '.join(missing)}")
    return spec


def validate_task(task_dir: str | Path, check_fixtures: bool = True) -> list[str]:
    """Structural + light self-consistency validation; returns a list of errors.

    Checks: task.yaml parses in the miniyaml subset with all required fields and
    sane values; the named API exists in harness/api.py; the workspace entrypoint
    exists; benchmark.py exposes the five driver functions; oracle/ files exist;
    (optionally) fixtures are deterministic (same seed -> same fixture hash).

    The full oracle-patch calibration (baseline fails the speedup gate, oracle
    patch passes all gates) is an evaluation, not a static check; run it via
    run-task against the oracle patch during task authoring.
    """
    errors: list[str] = []
    task_dir = Path(task_dir)
    if not task_dir.is_dir():
        return [f"task directory not found: {task_dir}"]
    try:
        spec = load_task_yaml(task_dir)
    except (FileNotFoundError, ValueError, miniyaml.MiniYAMLError) as exc:
        return [str(exc)]

    import re

    if not re.match(r"^[A-Z0-9-]+$", str(spec.get("task_id", ""))):
        errors.append("task_id must match ^[A-Z0-9-]+$")
    if spec.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if spec.get("track") not in TRACKS:
        errors.append(f"track must be one of {sorted(TRACKS)}")
    if spec.get("kind") not in KINDS:
        errors.append(f"kind must be one of {sorted(KINDS)}")
    for key in ("generator_family_id", "oracle_fix_pattern_id", "scientific_contract_id"):
        if not isinstance(spec.get(key), str) or not spec[key]:
            errors.append(f"{key} must be a non-empty string")
    if not isinstance(spec.get("workspace_ast_skeleton_hash"), str) or len(spec["workspace_ast_skeleton_hash"]) != 64:
        errors.append("workspace_ast_skeleton_hash must be a 64-character digest")
    if spec.get("difficulty_tier") not in {"easy", "medium", "hard"}:
        errors.append("difficulty_tier must be easy, medium, or hard")
    # Canonical SPE anchors carry explicit family lineage.  Generic harness
    # fixtures used by split-contract tests may omit it and are validated by
    # the population validator when they enter the benchmark population.
    if any(key in spec for key in ("family_id", "anchor_instance_id", "family_parameters", "family_instance_digest")):
        if not isinstance(spec.get("family_id"), str) or not spec["family_id"]:
            errors.append("family_id must be a non-empty string")
        if spec.get("anchor_instance_id") != spec.get("task_id"):
            errors.append("anchor_instance_id must equal task_id for a materialized anchor")
        if not isinstance(spec.get("family_parameters"), dict) or not spec["family_parameters"]:
            errors.append("family_parameters must be a non-empty mapping")
        if not isinstance(spec.get("family_instance_digest"), str) or len(spec["family_instance_digest"]) != 64:
            errors.append("family_instance_digest must be a 64-character digest")
    if not isinstance(spec.get("requires_cuda"), bool):
        errors.append("requires_cuda must be a boolean")

    lineage = spec.get("lineage")
    if not isinstance(lineage, dict) or "source" not in lineage or "mutation_template_id" not in lineage:
        errors.append("lineage needs source and mutation_template_id")

    workspace = spec.get("workspace")
    entrypoint = None
    if not isinstance(workspace, dict) or "entrypoint" not in workspace or "api" not in workspace:
        errors.append("workspace needs entrypoint and api")
    else:
        from . import api as api_registry

        try:
            api_registry.get_api(str(workspace["api"]))
        except KeyError as exc:
            errors.append(str(exc))
        entrypoint = task_dir / "workspace" / str(workspace["entrypoint"])
        if not entrypoint.is_file():
            errors.append(f"workspace entrypoint missing: {entrypoint}")

    measurement = spec.get("measurement")
    if not isinstance(measurement, dict):
        errors.append("measurement must be a mapping")
    else:
        for key in (
            "primary_metric",
            "higher_is_better",
            "warmup_iterations",
            "measured_iterations",
            "repetitions",
            "min_improvement_percent",
            "noise_floor_percent",
            "speedup_tripwire",
        ):
            if key not in measurement:
                errors.append(f"measurement.{key} is required")
        primary_metric = measurement.get("primary_metric")
        if spec.get("track") == "evolution" and primary_metric != "episode_score":
            errors.append("evolution tasks must use measurement.primary_metric=episode_score")
        if spec.get("track") == "evolution":
            effect_range = spec.get("oracle", {}).get("expected_delta_range") if isinstance(spec.get("oracle"), dict) else None
            if not isinstance(effect_range, list) or len(effect_range) != 2 or float(effect_range[0]) >= float(effect_range[1]):
                errors.append("evolution tasks must declare oracle.expected_delta_range")
        if spec.get("track") != "evolution" and primary_metric == "episode_score":
            errors.append("episode_score is reserved for evolution tasks")

    correctness = spec.get("correctness")
    if not isinstance(correctness, dict) or "num_fresh_inputs" not in correctness:
        errors.append("correctness.num_fresh_inputs is required")

    benchmark_py = task_dir / "benchmark.py"
    if not benchmark_py.is_file():
        errors.append(f"benchmark.py missing: {benchmark_py}")
    else:
        try:
            module = runner.import_module_by_path(benchmark_py)
            for name in BENCHMARK_FUNCTIONS:
                if not callable(getattr(module, name, None)):
                    errors.append(f"benchmark.py missing callable {name}()")
            if check_fixtures and not errors:
                device, usable = runner.select_device(bool(spec.get("requires_cuda")))
                if usable:
                    fixtures_a = runner.call_benchmark_fn(module.make_fixtures, seed=0, device=device)
                    fixtures_b = runner.call_benchmark_fn(module.make_fixtures, seed=0, device=device)
                    if runner.fixture_hash(fixtures_a) != runner.fixture_hash(fixtures_b):
                        errors.append("make_fixtures is not deterministic for a fixed seed")
        except Exception as exc:
            errors.append(f"benchmark.py import/fixture check failed: {exc!r}")

    oracle = task_dir / "oracle"
    if not oracle.is_dir():
        errors.append(f"oracle/ directory missing: {oracle}")
    return errors


# ---------------------------------------------------------------------------
# S0-S6 pipeline
# ---------------------------------------------------------------------------


def _scan_solution(solution_dir: Path, canaries: list[str]) -> tuple[bool, list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    for path in sorted(solution_dir.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        findings.extend(anticheat.scan_source(text, filename=path.name))
        findings.extend(anticheat.canary_check(text, canaries))
    return anticheat.has_hard_fail(findings), findings


def _load_expected_mechanisms(task_dir: Path) -> list[str] | None:
    """Read oracle/expected_mechanism.json — harness-side only, never sandboxed."""
    path = task_dir / "oracle" / "expected_mechanism.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        value = payload.get("mechanisms", payload.get("mechanism"))
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value]
    return None


def _fresh_input_correctness(
    benchmark_module: Any, solution_path: Path, spec: dict[str, Any], seed: int, device: str
) -> dict[str, Any]:
    """S2: correctness on num_fresh_inputs fresh seeded draws; checksums recorded."""
    num_inputs = int(spec["correctness"]["num_fresh_inputs"])
    per_input: list[dict[str, Any]] = []
    all_passed = True
    checksums: dict[str, Any] = {}
    for index in range(num_inputs):
        fixtures = runner.call_benchmark_fn(
            benchmark_module.make_fixtures, seed=seed * 100003 + 7000 + index, device=device
        )
        solution = runner.call_benchmark_fn(
            benchmark_module.load_solution, path=str(solution_path), device=device
        )
        outcome = runner.normalize_correctness(
            runner.call_benchmark_fn(
                benchmark_module.run_correctness, solution=solution, fixtures=fixtures
            )
        )
        per_input.append({"input_index": index, "passed": outcome["passed"], "details": outcome["details"]})
        all_passed = all_passed and outcome["passed"]
        raw_checksum = outcome["details"].get("output_checksum")
        if raw_checksum is not None:
            checksums[f"input:{index}"] = raw_checksum
    return {"passed": all_passed, "per_input": per_input, "output_checksums": checksums}


def cache_policy_for_task(spec: Mapping[str, Any]) -> str:
    """Return the cache scope actually established by the task harness."""
    if str(spec.get("family", "")) == "compiler":
        return "arm-repetition-fresh"
    return "verifier-invocation-scoped"


def calibrate_noise_control(
    task_dir: str | Path,
    solution_dir: str | Path,
    out_path: str | Path,
    *,
    task_id: str,
    outer_trial_id: str,
    benchmark_revision: str,
    task_manifest_digest: str,
    task_package_digest: str | None = None,
    population_manifest_digest: str | None = None,
    hardware_fingerprint: dict[str, Any] | None = None,
    compiler_cache_policy: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the one preregistered baseline-vs-baseline calibration.

    This path intentionally performs only the control measurement.  Candidate
    verifiers consume its immutable artifact and never rerun these controls.
    """
    task_dir = Path(task_dir)
    spec = load_task_yaml(task_dir)
    if execution_class_for_task(spec) == "episode":
        raise ValueError("episode_v1 uses bounded-score paired execution; noise control is not applicable")
    compiler_cache_policy = compiler_cache_policy or cache_policy_for_task(spec)
    fingerprint = hardware_fingerprint or capture_fingerprint()
    device, usable = runner.select_device(bool(spec.get("requires_cuda")))
    if not usable:
        raise RuntimeError("noise control requires CUDA but no usable device is available")
    module = runner.import_module_by_path(task_dir / "benchmark.py")
    entrypoint = str(spec["workspace"]["entrypoint"])
    # Noise is a null control for the registered baseline.  The oracle is
    # intentionally never used here; its variability belongs to verifier
    # evidence, not to the denominator-side null distribution.
    baseline_path = task_dir / "workspace" / entrypoint
    if not baseline_path.is_file():
        raise FileNotFoundError(f"baseline entrypoint not found: {baseline_path}")
    measurement_cfg = spec["measurement"]
    calibration_cfg = dict(measurement_cfg)
    calibration_cfg["repetitions"] = 5
    is_kernel = str(spec.get("family", "")) == "compiler" or metric_type_for_task(spec) == "kernel"
    record = runner.run_paired_measurement(
        module,
        baseline_path=baseline_path,
        candidate_path=baseline_path,
        measurement_cfg=calibration_cfg,
        seed=seed,
        device=device,
        l2_thrash_between=is_kernel,
        reuse_fixture_per_repetition=str(spec.get("family_id", "")) == "h2d_pipeline",
    )
    control_a = [float(value) for value in record.get("baseline_runs", []) if value is not None]
    control_b = [float(value) for value in record.get("candidate_runs", []) if value is not None]
    if len(control_a) != 5 or len(control_b) != 5:
        raise RuntimeError("noise control requires five complete repetitions per arm")
    higher_is_better = bool(measurement_cfg.get("higher_is_better", False))
    floor = stats.estimate_noise_floor(control_a, control_b, higher_is_better)
    artifact = {
        "task_id": task_id,
        "outer_trial_id": outer_trial_id,
        "benchmark_revision": benchmark_revision,
        "task_manifest_digest": task_manifest_digest,
        "task_package_digest": task_package_digest or task_manifest_digest,
        "population_manifest_digest": population_manifest_digest or task_manifest_digest,
        "control_implementation": "baseline",
        "hardware_fingerprint": fingerprint,
        "software_fingerprint": fingerprint,
        "compile_threads": int(measurement_cfg.get("compile_threads", 0)),
        "compiler_cache_policy": compiler_cache_policy,
        "expected_speedup_range": spec.get("oracle", {}).get("expected_speedup_range"),
        "primary_metric": measurement_cfg.get("primary_metric"),
        "higher_is_better": higher_is_better,
        "control_a_runs": control_a,
        "control_b_runs": control_b,
        "observed_noise_floor_percent": floor["noise_floor_percent_observed"],
        "declared_noise_floor_percent": float(measurement_cfg.get("noise_floor_percent", 2.0)),
        "calibration_record": {"timing": record.get("timing", []), "fixture_hashes": record.get("fixture_hashes", {})},
    }
    return stats.write_noise_control(out_path, artifact)


def verify_task(
    task_dir: str | Path,
    solution_dir: str | Path,
    out_path: str | Path | None = None,
    predicted_mechanism: list[str] | None = None,
    seed: int = 0,
    condition: str = "standalone",
    context_mode: str = "reset",
    noise_control_path: str | Path | None = None,
    noise_control_required: bool = False,
    noise_control_expected: Mapping[str, Any] | None = None,
    outer_trial_id: str | None = None,
    task_package_digest: str | None = None,
    population_manifest_digest: str | None = None,
) -> dict[str, Any]:
    """Run the S0-S6 pipeline; return (and optionally write) the result dict."""
    started = time.perf_counter()
    task_dir = Path(task_dir)
    solution_dir = Path(solution_dir)
    errors: list[str] = []
    expected_identity = dict(noise_control_expected or {})
    task_package_digest = task_package_digest or expected_identity.get("task_package_digest")
    population_manifest_digest = population_manifest_digest or expected_identity.get("population_manifest_digest")

    if context_mode not in {"reset", "carry"}:
        raise ValueError("context_mode must be reset or carry")
    spec = load_task_yaml(task_dir)  # raises cleanly on missing task
    if not solution_dir.is_dir():
        raise FileNotFoundError(f"solution directory not found: {solution_dir}")

    # Evolution episodes have a bounded-score contract, not an atomic
    # latency/speedup contract.  Keep them on their own verifier path so one
    # outer trial executes exactly one baseline and one candidate episode.
    if execution_class_for_task(spec) == "episode":
        return _verify_episode_task(
            task_dir, solution_dir, out_path=out_path, seed=seed,
            condition=condition, context_mode=context_mode, outer_trial_id=outer_trial_id,
            task_package_digest=task_package_digest,
            population_manifest_digest=population_manifest_digest,
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "task_id": spec["task_id"],
        "outer_trial_id": outer_trial_id,
        "task_package_digest": task_package_digest,
        "population_manifest_digest": population_manifest_digest,
        "seed": int(seed),
        "measurement_class": "atomic_performance",
        "condition": condition,
        "context_mode": context_mode,
        "verdict": "error",
        "validity": "invalid", "execution_validity": "invalid", "efficacy_eligible": False,
        "protocol_failure": False,
        "correctness_pass": False,
        "scientific_gates": {},
        "verified_speedup": {
            "median_speedup": None,
            "ci_low": None,
            "ci_high": None,
            "verified": False,
            "inconclusive": True,
            "reason": "pipeline did not reach S5",
        },
        "task": {
            "track": spec["track"],
            "family": spec["family"],
            "kind": spec["kind"],
            "expected_speedup_range": spec.get("oracle", {}).get("expected_speedup_range"),
        },
        "diagnosis": {"enabled": bool(spec["diagnosis"].get("enabled")), "predicted": predicted_mechanism, "expected": None, "diagnosis_correct": None},
        "cost": {"wall_time_s": 0.0, "tokens": None, "tool_calls": None, "retries": 0},
        "anticheat": {"hard_fail": False, "findings": [], "tripwired": False, "canary_tripped": False},
        "fingerprint": capture_fingerprint(),
        "calibration_status": "not_evaluated",
        "measurement": {},
        "stage_times_s": {},
        "harness_hash": {},
        "errors": errors,
    }
    # Formal cells must use the immutable, same-host calibration artifact.  A
    # missing or incompatible artifact is a resource block, never a fallback
    # to the task's declared floor.
    noise_control: dict[str, Any] | None = None
    if noise_control_path is not None or noise_control_required:
        expected = dict(noise_control_expected or {})
        expected.setdefault("task_id", str(spec.get("task_id")))
        expected.setdefault("hardware_fingerprint", result["fingerprint"])
        expected.setdefault("software_fingerprint", result["fingerprint"])
        expected.setdefault("primary_metric", spec["measurement"].get("primary_metric"))
        expected.setdefault("higher_is_better", bool(spec["measurement"].get("higher_is_better", False)))
        expected.setdefault("compile_threads", int(spec["measurement"].get("compile_threads", 0)))
        expected.setdefault("compiler_cache_policy", cache_policy_for_task(spec))
        expected.setdefault("expected_speedup_range", spec.get("oracle", {}).get("expected_speedup_range"))
        try:
            if noise_control_path is None:
                raise ValueError("noise control artifact is required")
            noise_control = stats.read_noise_control(noise_control_path, expected)
            result["noise_control_digest"] = noise_control["artifact_digest"]
            expected_range = spec.get("oracle", {}).get("expected_speedup_range")
            effective_floor = float(noise_control["effective_noise_floor_percent"])
            if isinstance(expected_range, (list, tuple)) and len(expected_range) == 2:
                oracle_upper = float(expected_range[1])
                if 1.0 + effective_floor / 100.0 >= oracle_upper:
                    result["calibration_status"] = "blocked"
                    result["calibration_block_reason"] = (
                        "effective noise floor reaches or exceeds oracle speedup cap"
                    )
                else:
                    result["calibration_status"] = "eligible"
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result.update({
                "verdict": "inconclusive",
                "validity": "invalid",
                "execution_validity": "resource_blocked",
                "protocol_failure": False,
                "noise_control_error": str(exc),
            })
            errors.append(f"noise control unavailable or incompatible: {exc}")
            return _finalize(result, started, out_path)
    stage_started = time.perf_counter()

    def mark_stage(name: str) -> None:
        nonlocal stage_started
        result["stage_times_s"][name] = round(time.perf_counter() - stage_started, 6)
        stage_started = time.perf_counter()

    # --- S0: sandbox + harness hashing --------------------------------------
    harness_hash_s0 = runner.hash_harness_files()
    result["harness_hash"] = harness_hash_s0
    sandbox = runner.materialize_sandbox(task_dir)
    result["sandbox_dir"] = str(sandbox)
    mark_stage("S0")

    # --- S1: static scan ------------------------------------------------------
    oracle_dir = task_dir / "oracle"
    canaries = anticheat.load_canaries(oracle_dir)
    hard_fail, findings = _scan_solution(solution_dir, canaries)
    result["anticheat"]["findings"] = findings
    result["anticheat"]["hard_fail"] = hard_fail
    result["anticheat"]["canary_tripped"] = any(f["rule"] == "canary" for f in findings)
    if hard_fail:
        result["verdict"] = "fail"
        mark_stage("S1")
        return _finalize(result, started, out_path)
    mark_stage("S1")

    # --- S2/S3: correctness + scientific gates --------------------------------
    device, usable = runner.select_device(bool(spec.get("requires_cuda")))
    if not usable:
        result["verdict"] = "inconclusive"
        result["execution_validity"] = "resource_blocked"
        result["verified_speedup"]["reason"] = "task requires CUDA; host has none"
        mark_stage("S2")
        return _finalize(result, started, out_path)

    benchmark_module = runner.import_module_by_path(task_dir / "benchmark.py")
    entrypoint_name = str(spec["workspace"]["entrypoint"])
    baseline_path = task_dir / "workspace" / entrypoint_name
    candidate_path = solution_dir / entrypoint_name
    if not candidate_path.is_file():
        candidates = sorted(solution_dir.rglob(entrypoint_name))
        if candidates:
            candidate_path = candidates[0]
        else:
            result["verdict"] = "error"
            errors.append(f"candidate entrypoint {entrypoint_name} not found under {solution_dir}")
            mark_stage("S2")
            return _finalize(result, started, out_path)

    runner.set_global_seeds(seed)
    try:
        correctness = _fresh_input_correctness(benchmark_module, candidate_path, spec, seed, device)
    except Exception as exc:
        result["verdict"] = "error"
        errors.append(f"S2 correctness raised: {exc!r}")
        mark_stage("S2")
        return _finalize(result, started, out_path)
    result["correctness_pass"] = correctness["passed"]
    result["correctness_details"] = correctness

    if not correctness["passed"]:
        result["verdict"] = "fail"
        mark_stage("S2")
        return _finalize(result, started, out_path)
    mark_stage("S2")

    try:
        fixtures = runner.call_benchmark_fn(benchmark_module.make_fixtures, seed=seed * 100003 + 9000, device=device)
        solution = runner.call_benchmark_fn(benchmark_module.load_solution, path=str(candidate_path), device=device)
        gates = runner.normalize_gates(
            runner.call_benchmark_fn(benchmark_module.run_scientific_gates, solution=solution, fixtures=fixtures)
        )
    except Exception as exc:
        result["verdict"] = "error"
        errors.append(f"S3 scientific gates raised: {exc!r}")
        mark_stage("S3")
        return _finalize(result, started, out_path)
    result["scientific_gates"] = {name: gate["passed"] for name, gate in gates.items()}
    result["scientific_gate_details"] = {name: gate["details"] for name, gate in gates.items()}
    if not all(result["scientific_gates"].values()):
        result["verdict"] = "fail"
        mark_stage("S3")
        return _finalize(result, started, out_path)
    mark_stage("S3")

    # --- S4: activation evidence (absence = inconclusive note, not failure) ---
    activation_fn = getattr(benchmark_module, "run_activation_evidence", None)
    if callable(activation_fn):
        try:
            baseline_solution = runner.call_benchmark_fn(benchmark_module.load_solution, path=str(baseline_path), device=device)
            observed = runner.call_benchmark_fn(
                activation_fn,
                solution=solution,
                baseline_solution=baseline_solution,
                fixtures=fixtures,
            )
            if not isinstance(observed, dict):
                raise ValueError("run_activation_evidence must return candidate_metrics and baseline_metrics")
            candidate_metrics = observed.get("candidate_metrics", observed.get("metrics"))
            baseline_metrics = observed.get("baseline_metrics")
            if not isinstance(candidate_metrics, dict) or not isinstance(baseline_metrics, dict):
                raise ValueError("activation evidence must include contrastive candidate and baseline traces")
            from benchmark.families import FAMILY_SPECS, resolve_family_id
            from benchmark.families.activation import classify_activation
            family_spec = FAMILY_SPECS[resolve_family_id(str(spec.get("family_id", spec.get("family", ""))))]
            result["activation"] = classify_activation(
                family_spec.family_id, family_spec.action_specs, candidate_metrics, baseline_metrics,
            )
        except Exception as exc:
            result["activation"] = {"status": "unavailable", "error": repr(exc)}
    else:
        # Worker-provided activation hooks are not evidence.  A task without
        # benchmark-owned instrumentation is explicitly unavailable for
        # formal causal attribution.
        result["activation"] = {"status": "not_declared", "required": True}
    mark_stage("S4")

    # --- S5: paired interleaved performance -----------------------------------
    measurement_cfg = spec["measurement"]
    kernel_task = str(spec.get("family", "")) == "compiler" or metric_type_for_task(spec) == "kernel"
    reuse_fixture_per_repetition = str(spec.get("family_id", "")) == "h2d_pipeline"
    try:
        record = runner.run_paired_measurement(
            benchmark_module,
            baseline_path=baseline_path,
            candidate_path=candidate_path,
            measurement_cfg=measurement_cfg,
            seed=seed,
            device=device,
            l2_thrash_between=kernel_task,
            reuse_fixture_per_repetition=reuse_fixture_per_repetition,
        )
    except Exception as exc:
        result["verdict"] = "error"
        result["protocol_failure"] = True
        result["validity"] = "invalid"
        errors.append(f"S5 performance raised: {exc!r}")
        result["failure_detail"] = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "missing_path": str(getattr(exc, "filename", "") or "") or None,
            "traceback": traceback.format_exc(),
            "stage": "S5",
        }
        mark_stage("S5")
        return _finalize(result, started, out_path)

    # Work-unit counters must match the baseline exactly (section 7).
    work_ok = True
    work_diffs: list[str] = []
    for rep in range(int(measurement_cfg.get("repetitions", 5))):
        ok, diffs = runner.compare_work_units(
            record["work_units"].get(f"baseline:{rep}", {}),
            record["work_units"].get(f"candidate:{rep}", {}),
        )
        work_ok = work_ok and ok
        work_diffs.extend(diffs)
    if not work_ok:
        result["verdict"] = "fail"
        errors.extend(f"work-unit mismatch: {d}" for d in work_diffs)
        result["measurement"] = record
        mark_stage("S5")
        return _finalize(result, started, out_path)

    higher_is_better = bool(measurement_cfg.get("higher_is_better", False))
    declared_floor = float(measurement_cfg.get("noise_floor_percent", 2.0))
    observed_floor = noise_control.get("observed_noise_floor_percent") if noise_control else None
    effective_floor = float(noise_control["effective_noise_floor_percent"]) if noise_control else stats.effective_noise_floor(declared_floor, observed_floor)
    verdict = stats.robust_speedup_verdict(
        [v for v in record["baseline_runs"] if v is not None],
        [v for v in record["candidate_runs"] if v is not None],
        higher_is_better,
        float(measurement_cfg.get("min_improvement_percent", 5.0)),
        effective_floor,
    )
    record["noise_floor_percent_declared"] = declared_floor
    record["noise_floor_percent_observed"] = observed_floor
    record["noise_floor_percent_effective"] = effective_floor
    if noise_control:
        record["noise_control_digest"] = noise_control["artifact_digest"]
    record["primary_metric"] = measurement_cfg.get("primary_metric")
    record["higher_is_better"] = higher_is_better
    result["measurement"] = record
    result["verified_speedup"] = verdict

    if measurement_cfg.get("primary_metric") == "time_to_quality_s":
        # Per section 6.2: failures-to-reach score as the cap with reached=false.
        # run_performance reports the flag via its raw dict (recorded in timing).
        reached_flags = [
            entry["timing"].get("reached")
            for entry in record.get("timing", [])
            if entry.get("arm") == "candidate" and isinstance(entry.get("timing"), dict)
        ]
        result["time_to_quality_s"] = stats.median(
            [v for v in record["candidate_runs"] if v is not None]
        )
        result["time_to_quality_reached"] = (
            all(bool(flag) for flag in reached_flags) if reached_flags else None
        )

    # S5 ends after the paired result and tripwire have been recorded; S6
    # diagnosis and manifest checks must have their own timing bucket.
    mark_stage("S5")

    # Tripwire: verified speedup above tripwire is flagged, not auto-passed.
    tripped, trip_message = anticheat.tripwire_check(
        verdict["median_speedup"], float(measurement_cfg.get("speedup_tripwire", 20.0))
    )
    result["anticheat"]["tripwired"] = tripped
    if tripped:
        result["anticheat"]["findings"].append(
            {"severity": "warning", "rule": "speedup_tripwire", "message": trip_message, "location": None}
        )

    # --- S6: verdict + diagnosis ----------------------------------------------
    expected = _load_expected_mechanisms(task_dir)
    result["diagnosis"]["expected"] = expected
    if result["diagnosis"]["enabled"] and predicted_mechanism is not None and expected is not None:
        result["diagnosis"]["diagnosis_correct"] = set(predicted_mechanism) == set(expected)

    harness_hash_s6 = runner.hash_harness_files()
    same, diffs = anticheat.manifests_equal(harness_hash_s0, harness_hash_s6)
    if not same:
        result["verdict"] = "fail"
        errors.append(f"harness files mutated during evaluation: {diffs}")
        mark_stage("S6")
        return _finalize(result, started, out_path)

    result["verdict"] = "pass" if verdict["verified"] else "inconclusive"
    result["validity"] = "valid"
    result["execution_validity"] = "valid"
    result["efficacy_eligible"] = True
    mark_stage("S6")
    return _finalize(result, started, out_path)


def _episode_arm_budget(spec: Mapping[str, Any], remaining_s: float | None = None) -> float:
    """Return the arm cap, never exceeding the shared outer deadline."""
    budget = float(spec["time_budget_s"])
    if budget <= 0.0:
        raise ValueError("episode time_budget_s must be positive")
    if remaining_s is None:
        return budget
    return max(0.0, min(budget, float(remaining_s)))


def _verify_episode_task(
    task_dir: Path,
    solution_dir: Path,
    *,
    out_path: str | Path | None,
    seed: int,
    condition: str,
    context_mode: str,
    outer_trial_id: str | None,
    task_package_digest: str | None,
    population_manifest_digest: str | None,
) -> dict[str, Any]:
    """Verify one evolution episode with one paired C/D execution.

    ``episode_v1`` reports a bounded score.  Repetitions in its task manifest
    identify independent outer trials; they are not inner performance arms.
    """
    started = time.perf_counter()
    spec = load_task_yaml(task_dir)
    result: dict[str, Any] = {
        "schema_version": 1, "task_id": spec["task_id"], "metric_class": "evolution", "condition": condition,
        "outer_trial_id": outer_trial_id, "seed": int(seed), "measurement_class": "episode_bounded_score",
        "task_package_digest": task_package_digest,
        "population_manifest_digest": population_manifest_digest,
        "context_mode": context_mode, "verdict": "error", "validity": "invalid",
        "execution_validity": "invalid", "protocol_failure": False,
        "correctness_pass": False, "scientific_gates": {},
        "task": {"track": spec["track"], "family": spec["family"], "kind": spec["kind"],
                 "metric_class": "evolution", "expected_score_range": spec.get("oracle", {}).get("expected_score_range", [0.0, 1.0]),
                 "expected_delta_range": spec.get("oracle", {}).get("expected_delta_range")},
        "cost": {"wall_time_s": 0.0, "tokens": None, "tool_calls": None, "retries": 0},
        "anticheat": {"hard_fail": False, "findings": [], "tripwired": False, "canary_tripped": False},
        "fingerprint": capture_fingerprint(), "calibration_status": "not_evaluated",
        "stage_times_s": {}, "harness_hash": {}, "errors": [],
    }
    try:
        harness_hash_s0 = runner.hash_harness_files()
        result["harness_hash"] = harness_hash_s0
        sandbox = runner.materialize_sandbox(task_dir)
        result["sandbox_dir"] = str(sandbox)
        oracle_dir = task_dir / "oracle"
        hard_fail, findings = _scan_solution(solution_dir, anticheat.load_canaries(oracle_dir))
        result["anticheat"].update({"hard_fail": hard_fail, "findings": findings})
        if hard_fail:
            result["verdict"] = "fail"
            return _finalize(result, started, out_path)
        device, usable = runner.select_device(bool(spec.get("requires_cuda")))
        if not usable:
            result["verdict"] = "inconclusive"
            result["execution_validity"] = "resource_blocked"
            result["errors"].append("episode requires CUDA; host has none")
            return _finalize(result, started, out_path)
        module = runner.import_module_by_path(task_dir / "benchmark.py")
        entrypoint = str(spec["workspace"]["entrypoint"])
        baseline_path = task_dir / "workspace" / entrypoint
        candidate_path = solution_dir / entrypoint
        if not candidate_path.is_file():
            candidates = sorted(solution_dir.rglob(entrypoint))
            if candidates:
                candidate_path = candidates[0]
            else:
                raise FileNotFoundError(f"candidate entrypoint {entrypoint} not found")
        fixtures = runner.call_benchmark_fn(module.make_fixtures, seed=seed, device=device)
        baseline_probe = runner.call_benchmark_fn(
            module.run_performance,
            solution=runner.call_benchmark_fn(module.load_solution, path=str(baseline_path), device=device),
            fixtures=copy.deepcopy(fixtures), warmup=0, iterations=1, device=device,
        )
        candidate_probe = runner.call_benchmark_fn(
            module.run_performance,
            solution=runner.call_benchmark_fn(module.load_solution, path=str(candidate_path), device=device),
            fixtures=copy.deepcopy(fixtures), warmup=0, iterations=1, device=device,
        )
        def action_from_probe(probe: Any) -> dict[str, Any]:
            if not isinstance(probe, dict) or not isinstance(probe.get("action"), dict):
                raise ValueError("episode candidate must return a declarative action mapping")
            action = dict(probe["action"])
            if str(action.get("condition", "")).upper() not in {"C", "C_STRESS", "D"}:
                raise ValueError("episode action condition must be C, C_STRESS, or D")
            return action

        outer_budget = float(spec["time_budget_s"])
        deadline = started + outer_budget
        arm_cleanups: list[dict[str, Any]] = []

        def execute_action(probe: Any, label: str) -> dict[str, Any]:
            action = action_from_probe(probe)
            arm_budget = _episode_arm_budget(spec, deadline - time.perf_counter())
            if arm_budget <= 0.0:
                raise TimeoutError("evolution outer deadline exhausted before episode arm")
            episode_path = next(task_dir.glob("episodes/*.yaml"), None)
            if episode_path is None:
                raise FileNotFoundError(f"episode manifest missing in {task_dir}")
            with tempfile.TemporaryDirectory(prefix=f"acre-episode-{label}-") as temp:
                out_dir = Path(temp)
                snippet = (
                    "import json; from benchmark.harness import evolution; "
                    f"r=evolution.run_episode({str(episode_path)!r}, {str(action['condition']).upper()!r}, "
                    f"{str(out_dir)!r}, core_repo={str(Path(__file__).resolve().parents[2])!r}, "
                    f"snapshot_dir={str(Path(__file__).resolve().parents[2])!r}, context_mode={context_mode!r}, "
                    f"seed={int(seed)!r}, max_wall_time_s={arm_budget!r}); "
                    f"open({str(out_dir / 'harness_episode.json')!r}, 'w', encoding='utf-8').write(json.dumps(r, default=str))"
                )
                completed = runner.run_python_subprocess(
                    snippet=snippet, timeout=arm_budget, cwd=Path(__file__).resolve().parents[2]
                )
                cleanup = completed.get("cleanup", {})
                arm_cleanups.append(cleanup)
                if cleanup.get("residual_detected"):
                    result["executor_cleanup"] = list(arm_cleanups)
                    raise runner.ResourceBlockedError(
                        f"episode arm {label} left a residual process group: {cleanup}"
                    )
                if completed["timed_out"]:
                    raise TimeoutError(f"episode arm {label} exceeded {arm_budget:g}s")
                if completed["exit_code"] != 0:
                    raise RuntimeError(f"episode arm {label} failed: {completed['stderr'] or completed['stdout']}")
                return json.loads((out_dir / "harness_episode.json").read_text(encoding="utf-8"))

        if time.perf_counter() >= deadline:
            raise TimeoutError("evolution outer deadline exhausted before baseline arm")
        base_raw = execute_action(baseline_probe, "baseline")
        cand_raw = execute_action(candidate_probe, "candidate")
        if time.perf_counter() > deadline:
            result["verdict"] = "inconclusive"
            result["validity"] = "valid"
            result["execution_validity"] = "resource_blocked"
            result["errors"].append("episode verifier exceeded the outer task budget")
            return _finalize(result, started, out_path)
        result["executor_cleanup"] = list(arm_cleanups)
        score_fn = getattr(module, "score_harness_episode", None)
        gates_fn = getattr(module, "gates_harness_episode", None)
        if not callable(score_fn) or not callable(gates_fn):
            raise ValueError("episode benchmark must expose harness-owned score_harness_episode/gates_harness_episode")
        base_score = score_fn(base_raw)
        cand_score = score_fn(cand_raw)
        base_gates = runner.normalize_gates(gates_fn(base_raw))
        gates = runner.normalize_gates(gates_fn(cand_raw))
        if not isinstance(base_score, (int, float)) or not isinstance(cand_score, (int, float)):
            raise ValueError("harness episode scorer must return a numeric bounded score")
        base_gates = {name: bool(value["passed"]) for name, value in base_gates.items()}
        gates = {name: bool(value["passed"]) for name, value in gates.items()}
        result["correctness_pass"] = True
        result["scientific_gates"] = {str(k): bool(v) for k, v in gates.items()}
        result["baseline_scientific_gates"] = {str(k): bool(v) for k, v in base_gates.items()}
        result["scientific_gate_details"] = cand_raw.get("episode_gate_details", {})
        result["episode_measurement"] = {
            "metric_class": "evolution",
            "seed": seed,
            "required_outer_trials": int(spec["measurement"].get("repetitions", 1)),
            "baseline_score": float(base_score), "candidate_score": float(cand_score),
            "absolute_score_delta": float(cand_score) - float(base_score),
            "paired_seed_effect": {
                "seed": seed, "baseline_score": float(base_score),
                "candidate_score": float(cand_score),
                "delta": float(cand_score) - float(base_score),
            },
            "baseline_wall_time_s": base_raw.get("wall_time_s"),
            "candidate_wall_time_s": cand_raw.get("wall_time_s"),
            "baseline_result": base_raw,
            "candidate_result": cand_raw,
            "baseline_gates": result["baseline_scientific_gates"],
            "candidate_gates": result["scientific_gates"],
        }
        result["execution_validity"] = "valid"
        result["validity"] = "valid"
        result["efficacy_eligible"] = True
        gates_ok = all(result["scientific_gates"].values()) and all(result["baseline_scientific_gates"].values())
        result["task_score"] = float(cand_score) if gates_ok else 0.0
        result["calibration_status"] = "eligible" if gates_ok else "blocked"
        result["verdict"] = "pass" if gates_ok else "fail"
        expected_delta = spec.get("oracle", {}).get("expected_delta_range")
        delta = float(cand_score) - float(base_score)
        if gates_ok and isinstance(expected_delta, list) and len(expected_delta) == 2:
            if not (float(expected_delta[0]) <= delta <= float(expected_delta[1])):
                result["calibration_status"] = "blocked"
                result["verdict"] = "fail"
                result["efficacy_eligible"] = False
                result["errors"].append("episode score delta is outside the preregistered effect range")
        harness_hash_s6 = runner.hash_harness_files()
        same, diffs = anticheat.manifests_equal(harness_hash_s0, harness_hash_s6)
        if not same:
            result["protocol_failure"] = True
            result["validity"] = "invalid"
            result["execution_validity"] = "invalid"
            result["errors"].append(f"harness files mutated during episode evaluation: {diffs}")
    except runner.ResourceBlockedError as exc:
        result["verdict"] = "inconclusive"
        result["validity"] = "valid"
        result["execution_validity"] = "resource_blocked"
        result["efficacy_eligible"] = False
        result["calibration_status"] = "blocked"
        result["failure_stage"] = "executor_cleanup"
        result["protocol_failure"] = False
        result["executor_cleanup"] = list(arm_cleanups) if "arm_cleanups" in locals() else []
        result["errors"].append(str(exc))
    except TimeoutError as exc:
        result["verdict"] = "inconclusive"
        result["validity"] = "valid"
        result["execution_validity"] = "resource_blocked"
        result["efficacy_eligible"] = False
        result["calibration_status"] = "blocked"
        result["timeout"] = True
        result["failure_stage"] = "evolution_outer"
        result["protocol_failure"] = False
        result["executor_cleanup"] = list(arm_cleanups) if "arm_cleanups" in locals() else []
        result["errors"].append(str(exc))
    except Exception as exc:
        result["verdict"] = "error"
        result["validity"] = "invalid"
        result["execution_validity"] = "invalid"
        result["protocol_failure"] = True
        result["errors"].append(f"episode verifier raised: {exc!r}")
        result["failure_detail"] = {"exception_type": type(exc).__name__, "exception_message": str(exc), "traceback": traceback.format_exc()}
    return _finalize(result, started, out_path)


def _finalize(result: dict[str, Any], started: float, out_path: str | Path | None) -> dict[str, Any]:
    result["cost"]["wall_time_s"] = round(time.perf_counter() - started, 3)
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return result
