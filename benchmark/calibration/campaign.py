#!/usr/bin/env python3
"""Run the real oracle calibration for the explicit active-30 population.

This is a calibration runner, not a second verifier or a formal campaign.  It
calls the existing task verifier and shared noise-control path, records raw
results, and then projects those results through the population validator.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from benchmark.harness import miniyaml, runner, stats, verifier
from benchmark.harness.api import execution_class_for_task
from benchmark.harness.fingerprint import capture_fingerprint, fingerprints_compatible, selected_gpu_preflight
from benchmark.calibration.bundle import classify_result, calibration_envelope, validate_calibration_envelope
from benchmark.calibration.execution import CellExecutor, executor_digest
from benchmark.calibration.identity import canonical_cell_identity, task_package_digest, taskset_digest
from benchmark.calibration.protocol import load_calibration_protocol, outer_trial_count
from benchmark.calibration.report import rebuild_calibration_views
from benchmark.provenance import benchmark_revision, digest_mapping, file_digest
from benchmark.calibration.state import derive_cell_state, serialize_cell_state


def _bounded_verifier_result(
    *, task_id: str, outer_trial_id: str, result_path: Path, timeout_s: float,
    module: str, args: tuple[str, ...], cwd: Path, measurement_class: str = "atomic_performance",
) -> dict[str, Any]:
    """Run one verifier cell in a killable process and persist timeout evidence."""
    completed = CellExecutor(cwd).run_module(module=module, args=args, timeout_s=float(timeout_s))
    cleanup = completed.get("cleanup", {})
    if cleanup.get("residual_detected"):
        result = _resource_blocked_result(
            task_id=task_id, outer_trial_id=outer_trial_id,
            failure_stage="executor_cleanup", timeout_s=timeout_s,
            wall_time_s=float(completed["wall_time_s"]),
            error="verifier left a residual process group",
            measurement_class=measurement_class, timeout=False,
        )
        result["executor_cleanup"] = cleanup
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result
    if not completed["timed_out"] and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["executor_cleanup"] = cleanup
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result
    if not completed["timed_out"]:
        raise RuntimeError(
            f"verifier exited without a result for {task_id}/{outer_trial_id}: "
            f"{completed['stderr'] or completed['stdout']}"
        )
    result = _resource_blocked_result(
        task_id=task_id, outer_trial_id=outer_trial_id,
        failure_stage="verifier", timeout_s=timeout_s,
        wall_time_s=float(completed["wall_time_s"]),
        error=completed["stderr"] or f"verifier timed out after {timeout_s:g}s",
        measurement_class=measurement_class,
    )
    result["executor_cleanup"] = cleanup
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def _resource_blocked_result(
    *, task_id: str, outer_trial_id: str, failure_stage: str,
    timeout_s: float, wall_time_s: float, error: str,
    measurement_class: str = "atomic_performance", timeout: bool = True,
) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "task_id": task_id,
        "outer_trial_id": outer_trial_id,
        "measurement_class": measurement_class,
        "condition": "standalone",
        "context_mode": "reset",
        "verdict": "inconclusive",
        "validity": "valid",
        "execution_validity": "resource_blocked",
        "efficacy_eligible": False,
        "protocol_failure": False,
        "correctness_pass": False,
        "scientific_gates": {},
        "calibration_status": "blocked",
        "calibration_block_reason": error or f"{failure_stage} exceeded the task time budget",
        "timeout": timeout,
        "failure_stage": failure_stage,
        "cost": {"wall_time_s": wall_time_s, "tokens": None, "tool_calls": None, "retries": 0},
        "anticheat": {"hard_fail": False, "findings": [], "tripwired": False, "status": "pass"},
        "errors": [error or f"{failure_stage} timed out after {timeout_s:g}s"],
        "fingerprint": capture_fingerprint(),
    }
    return serialize_cell_state(result)


def _write_resource_blocked_cell(
    *, out: Path, task_id: str, outer_id: str, task_spec: dict[str, Any],
    task_digest: str, population_digest: str, revision: str,
    harness_digest: str, runner_digest: str, protocol_digest: str,
    fingerprint: dict[str, Any], task_manifest_digest: str, error: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Persist a complete, reusable infrastructure-failure cell."""
    measurement_class = "evolution" if execution_class_for_task(task_spec) == "episode" else "atomic_performance"
    raw_path = out / "raw" / outer_id / f"{task_id}.json"
    noise_path = out / "noise-control" / outer_id / f"{task_id}.json"
    envelope_path = out / "envelopes" / outer_id / f"{task_id}.json"
    result = _resource_blocked_result(
        task_id=task_id, outer_trial_id=outer_id, failure_stage="resource_preflight",
        timeout_s=0.0, wall_time_s=0.0, error=error,
        measurement_class=measurement_class, timeout=False,
    )
    result["calibration_status"] = "blocked"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    measurement = task_spec.get("measurement", {})
    noise = stats.write_noise_control(noise_path, {
        "schema_version": 1, "metric_class": "resource_blocked",
        "task_id": task_id, "outer_trial_id": outer_id,
        "benchmark_revision": revision, "task_manifest_digest": task_manifest_digest,
        "task_package_digest": task_digest, "population_manifest_digest": population_digest,
        "hardware_fingerprint": fingerprint, "software_fingerprint": fingerprint,
        "compile_threads": int(measurement.get("compile_threads", 2)),
        "compiler_cache_policy": verifier.cache_policy_for_task(task_spec),
        "primary_metric": measurement.get("primary_metric", "step_ms_p50"),
        "higher_is_better": bool(measurement.get("higher_is_better", False)),
        "control_a_runs": [1.0] * 5, "control_b_runs": [1.0] * 5,
        "observed_noise_floor_percent": 0.0,
        "declared_noise_floor_percent": float(measurement.get("noise_floor_percent", 2.0)),
        "expected_speedup_range": list(task_spec.get("oracle", {}).get("expected_speedup_range", [0.0, 1.0])),
        "execution_validity": "resource_blocked", "failure_stage": "resource_preflight",
        "error": error,
    })
    envelope = calibration_envelope(
        producer_revision=revision, task_package_digest=task_digest,
        population_manifest_digest=population_digest, harness_digest_value=harness_digest,
        calibration_runner_digest=runner_digest, noise_digest=str(noise["artifact_digest"]),
        raw_result_digest=file_digest(raw_path), fingerprint=fingerprint,
        task_id=task_id, outer_trial_id=outer_id, seed=int(outer_id.split("-")[-1]),
        measurement_class=measurement_class, calibration_protocol_digest=protocol_digest,
    )
    envelope_path.parent.mkdir(parents=True, exist_ok=True)
    envelope_path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result, noise, envelope


def _bounded_noise_control(
    *, task_id: str, outer_trial_id: str, noise_path: Path, timeout_s: float,
    args: tuple[str, ...], cwd: Path,
) -> tuple[dict[str, Any], bool]:
    """Run one noise-control cell in a killable process."""
    completed = CellExecutor(cwd).run_noise_control(args=args, timeout_s=float(timeout_s))
    cleanup = completed.get("cleanup", {})
    if not completed["timed_out"] and noise_path.is_file() and not cleanup.get("residual_detected"):
        noise = json.loads(noise_path.read_text(encoding="utf-8"))
        noise["executor_cleanup"] = cleanup
        stats.write_noise_control(noise_path, noise)
        return stats.read_noise_control(noise_path), False
    if not completed["timed_out"] and cleanup.get("residual_detected") and noise_path.is_file():
        noise = stats.read_noise_control(noise_path)
        noise.update({
            "execution_validity": "resource_blocked",
            "failure_stage": "executor_cleanup",
            "timeout": False,
            "error": completed["stderr"] or "noise control left a residual process group",
        })
        noise["executor_cleanup"] = cleanup
        stats.write_noise_control(noise_path, noise)
        return stats.read_noise_control(noise_path), True
    if not completed["timed_out"]:
        raise RuntimeError(
            f"noise control exited without an artifact for {task_id}/{outer_trial_id}: "
            f"{completed['stderr'] or completed['stdout']}"
        )
    return {
        "schema_version": 1,
        "task_id": task_id,
        "outer_trial_id": outer_trial_id,
        "timeout": True,
        "failure_stage": "noise_control",
        "error": completed["stderr"] or f"noise control timed out after {timeout_s:g}s",
        "wall_time_s": float(completed["wall_time_s"]),
        "executor_cleanup": cleanup,
    }, True


def _bounded_post_validation(task_dir: Path, cwd: Path) -> dict[str, Any]:
    """Run the cheap structural validation with an OS-level bound."""
    return CellExecutor(cwd).run_atomic(
        args=("validate-task", str(task_dir), "--no-fixture-check"), timeout_s=120.0,
    )


def _patch_strip_level(patch_path: Path, solution_dir: Path) -> int:
    """Return the strip level matching the copied workspace layout.

    Authoring patches in the active population use three valid header forms:
    ``solution.py``, ``a/solution.py`` and ``a/workspace/solution.py``.  The
    calibration workspace always contains the entrypoint at its root, so the
    strip level must be derived from the actual source path rather than fixed
    globally.
    """
    for line in patch_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("--- "):
            continue
        header = line[4:].split("\t", 1)[0].strip()
        if header == "/dev/null":
            continue
        parts = Path(header).parts
        for strip in range(len(parts)):
            relative = Path(*parts[strip:])
            if relative != Path(".") and (solution_dir / relative).is_file():
                return strip
        break
    raise RuntimeError(f"reference patch source is not in copied workspace: {patch_path}")


def _copy_oracle(task_dir: Path, solution_dir: Path) -> None:
    spec = miniyaml.load(str(task_dir / "task.yaml"))
    entrypoint = str(spec["workspace"]["entrypoint"])
    solution_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{solution_dir.name}.", dir=solution_dir.parent) as temp:
        staged = Path(temp) / "solution"
        shutil.copytree(
            task_dir / "workspace",
            staged,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        oracle = task_dir / "oracle" / "solution_oracle.py"
        if oracle.is_file():
            target = staged / entrypoint
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(oracle, target)
        else:
            patch_path = (task_dir / "oracle" / "reference_patch.diff").resolve()
            if not patch_path.is_file():
                raise FileNotFoundError(f"oracle solution and reference patch missing for {task_dir}")
            strip_level = _patch_strip_level(patch_path, staged)
            completed = subprocess.run(
                ["patch", "--batch", "--forward", "--fuzz=0", f"-p{strip_level}", "-i", str(patch_path)],
                cwd=staged, text=True, capture_output=True, check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"reference patch failed for {task_dir.name}: {completed.stderr or completed.stdout}")
        if solution_dir.exists():
            shutil.rmtree(solution_dir)
        os.replace(staged, solution_dir)


def _quarantine_cell_files(out: Path, task_id: str, outer_id: str, paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    artifact_key = digest_mapping({path.name: file_digest(path) for path in existing})
    destination = out / "quarantine" / outer_id / task_id / artifact_key
    destination.mkdir(parents=True, exist_ok=True)
    for path in existing:
        target = destination / path.name
        shutil.move(str(path), str(target))


def _cell_envelope_compatible(path: Path, *, static: dict[str, Any], noise_path: Path, result_path: Path) -> bool:
    if not path.is_file() or not noise_path.is_file() or not result_path.is_file():
        return False
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        noise = json.loads(noise_path.read_text(encoding="utf-8"))
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        identity = canonical_cell_identity(
            task_id=str(static["task_id"]), outer_trial_id=str(static["outer_trial_id"]),
            seed=int(static["seed"]), measurement_family=str(static["measurement_class"]),
            task_package_digest=str(static["task_package_digest"]),
            population_manifest_digest=str(static["population_manifest_digest"]),
        )
        for key in ("task_id", "outer_trial_id", "seed", "task_package_digest", "population_manifest_digest"):
            if raw.get(key) != identity[key]:
                return False
        if raw.get("measurement_class") != identity["raw_measurement_class"]:
            return False
        if validate_calibration_envelope(envelope, static):
            return False
        if any(envelope.get(key) != value for key, value in static.items() if key != "fingerprint"):
            return False
        compatible, _ = fingerprints_compatible(envelope.get("fingerprint", {}), static.get("fingerprint", {}))
        if not compatible or envelope.get("noise_digest") != noise.get("artifact_digest"):
            return False
        if static.get("measurement_class") == "atomic_performance":
            stats.read_noise_control(noise_path, {
                "task_id": static.get("task_id"),
                "outer_trial_id": static.get("outer_trial_id"),
                "benchmark_revision": static.get("producer_revision"),
                "task_package_digest": static.get("task_package_digest"),
                "population_manifest_digest": static.get("population_manifest_digest"),
                "control_implementation": "baseline",
            })
        return envelope.get("raw_result_digest") == file_digest(result_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False


def _percent(value: Any) -> float | None:
    return (float(value) - 1.0) * 100.0 if isinstance(value, (int, float)) else None


def _calibration_record(task_dir: Path, task_id: str, revision: str, digest: str, results: list[dict[str, Any]], noises: list[dict[str, Any]], envelopes: list[dict[str, Any]] | None = None, protocol_digest: str | None = None, population_manifest_digest: str | None = None, artifact_paths: dict[str, list[str]] | None = None) -> dict[str, Any]:
    first = results[0] if results else {}
    declared_episode = False
    try:
        declared_episode = execution_class_for_task(miniyaml.load(str(task_dir / "task.yaml"))) == "episode"
    except (OSError, KeyError, TypeError, ValueError):
        pass
    episode = declared_episode or bool(results and isinstance(first.get("episode_measurement"), dict))
    gates = {
        "correctness_pass": all(bool(item.get("correctness_pass")) for item in results),
        "scientific_gates": all(all(bool(value) for value in item.get("scientific_gates", {}).values()) for item in results),
        "execution_valid": all(item.get("execution_validity", "valid") not in {"invalid", "resource_blocked"} for item in results),
    }
    verified_values = [item.get("verified_speedup", {}) for item in results if isinstance(item.get("verified_speedup"), dict)]
    verified = {}
    if verified_values:
        lows = [float(item["ci_low"]) for item in verified_values if isinstance(item.get("ci_low"), (int, float))]
        highs = [float(item["ci_high"]) for item in verified_values if isinstance(item.get("ci_high"), (int, float))]
        medians = [float(item["median_speedup"]) for item in verified_values if isinstance(item.get("median_speedup"), (int, float))]
        verified = {
            "median_speedup": sum(medians) / len(medians) if medians else None,
            "ci_low": min(lows) if lows else None,
            "ci_high": max(highs) if highs else None,
            "verified": all(bool(item.get("verified")) for item in verified_values),
            "inconclusive": any(bool(item.get("inconclusive")) for item in verified_values),
        }
    episode_effects = [float(item["episode_measurement"]["absolute_score_delta"]) for item in results if isinstance(item.get("episode_measurement"), dict) and isinstance(item["episode_measurement"].get("absolute_score_delta"), (int, float))]
    anti_findings = [finding for item in results for finding in (item.get("anticheat", {}).get("findings", []) if isinstance(item.get("anticheat"), dict) else [])]
    anti = {"hard_fail": any(bool(item.get("anticheat", {}).get("hard_fail")) for item in results), "tripwired": any(bool(item.get("anticheat", {}).get("tripwired")) for item in results), "findings": anti_findings}
    calibration_status = "eligible" if all(derive_cell_state(item) == "eligible" for item in results) else "blocked"
    return {
        "task_id": task_id,
        "task_digest": digest,
        "revision": revision,
        "calibration_protocol_digest": protocol_digest,
        "population_manifest_digest": population_manifest_digest,
        "environment": first.get("fingerprint") or capture_fingerprint(),
        "outer_trials": results,
        "evidence_envelopes": list(envelopes or []),
        "artifact_paths": dict(artifact_paths or {}),
        "calibration_status": calibration_status,
        "noise_control": {"artifacts": noises, "effective_noise_floor_percent": max((float(item.get("effective_noise_floor_percent", 0.0)) for item in noises), default=0.0)},
        "control_noise_percent": [float(item.get("observed_noise_floor_percent")) for item in noises if isinstance(item.get("observed_noise_floor_percent"), (int, float))],
        "oracle_ci": verified,
        "oracle_ci_low_percent": _percent(verified.get("ci_low")),
        "oracle_ci_high_percent": _percent(verified.get("ci_high")),
        "semantic_gates": gates,
        "semantic_gate_pass_rate": sum(bool(value) for value in gates.values()) / max(1, len(gates)),
        "anti_cheat": {"status": "pass" if not anti.get("hard_fail") and not anti.get("tripwired") else "fail", "findings": anti.get("findings", [])},
        "metric_class": "evolution" if episode else "atomic_performance",
        "episode_effect": {
            "outer_trial_deltas": episode_effects,
            "mean_absolute_score_delta": sum(episode_effects) / len(episode_effects) if episode_effects else None,
            "min_absolute_score_delta": min(episode_effects) if episode_effects else None,
            "max_absolute_score_delta": max(episode_effects) if episode_effects else None,
        } if episode else None,
    }


def run_calibration_campaign(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    tasks_root = repo_root / "benchmark" / "tasks"
    active_path = repo_root / "benchmark" / "pilot_population.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    task_ids = [str(item) for item in active["task_ids"]]
    if args.task_id:
        task_ids = [task_id for task_id in task_ids if task_id == args.task_id]
        if not task_ids:
            raise ValueError(f"task is not in active population: {args.task_id}")
    out = Path(args.out).resolve()
    if args.task_id and out.exists() and any(out.iterdir()):
        raise ValueError("--task-id requires an independent empty output directory")
    raw_root = out / "raw"
    noise_root = out / "noise-control"
    raw_root.mkdir(parents=True, exist_ok=True)
    noise_root.mkdir(parents=True, exist_ok=True)
    protocol, protocol_digest = load_calibration_protocol(repo_root)
    observed_topology = runner.configure_thread_topology(protocol["thread_topology"])
    if observed_topology != {
        key: str(value) if key not in {"torch_num_threads", "torch_num_interop_threads"} else int(value)
        for key, value in protocol["thread_topology"].items()
    }:
        raise RuntimeError(f"declared thread topology was not applied: {observed_topology}")
    revision = benchmark_revision(repo_root)
    digest = taskset_digest(tasks_root, [str(item) for item in active["task_ids"]])
    fingerprint = capture_fingerprint()
    if fingerprint.get("thread_topology") != observed_topology:
        raise RuntimeError("captured fingerprint does not match declared thread topology")
    population_digest = file_digest(active_path)
    harness_digest = executor_digest(repo_root)
    runner_digest = file_digest(Path(__file__))
    if int(args.outer_trials) != int(protocol["atomic_outer_trials"]):
        raise ValueError(
            f"atomic calibration requires --outer-trials={protocol['atomic_outer_trials']} "
            f"from the frozen protocol, got {args.outer_trials}"
        )
    empirical: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    campaign_started_monotonic = time.perf_counter()
    campaign_started_utc = time.time()
    for task_id in task_ids:
        task_dir = tasks_root / task_id
        spec = miniyaml.load(str(task_dir / "task.yaml"))
        task_digest = task_package_digest(task_dir)
        task_results: list[dict[str, Any]] = []
        task_noises: list[dict[str, Any]] = []
        task_envelopes: list[dict[str, Any]] = []
        for outer in range(outer_trial_count(spec, protocol)):
            outer_id = f"outer-{outer:03d}"
            cell_started = time.perf_counter()
            timing = {
                "task_id": task_id, "outer_trial_id": outer_id,
                "oracle_materialization_s": 0.0, "noise_control_s": 0.0,
                "verifier_s": 0.0, "post_validation_s": 0.0,
                "reused": False, "quarantined": False, "timeout": False,
                "cleanup_status": "not_recorded",
            }
            print(json.dumps({"event": "start_outer_trial", "task_id": task_id, "outer_trial_id": outer_id}, ensure_ascii=False), flush=True)
            preflight = selected_gpu_preflight() if fingerprint.get("cuda_available") else {"status": "clean", "foreign_pids": []}
            foreign_pids = list(preflight.get("foreign_pids", []))
            if preflight.get("status") != "clean":
                result, noise, envelope = _write_resource_blocked_cell(
                    out=out, task_id=task_id, outer_id=outer_id, task_spec=spec,
                    task_digest=task_digest, population_digest=population_digest,
                    revision=revision, harness_digest=harness_digest,
                    runner_digest=runner_digest, protocol_digest=protocol_digest,
                    fingerprint=fingerprint, task_manifest_digest=digest,
                    error=(
                        f"selected GPU {fingerprint.get('gpu_uuid')} preflight {preflight.get('status')}: "
                        f"{preflight.get('reason')}"
                    ),
                )
                result["resource_preflight"] = {"gpu_uuid": fingerprint.get("gpu_uuid"), **preflight}
                result = serialize_cell_state(result)
                raw_path = out / "raw" / outer_id / f"{task_id}.json"
                raw_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                envelope = calibration_envelope(
                    producer_revision=revision, task_package_digest=task_digest,
                    population_manifest_digest=population_digest, harness_digest_value=harness_digest,
                    calibration_runner_digest=runner_digest, noise_digest=str(noise["artifact_digest"]),
                    raw_result_digest=file_digest(raw_path), fingerprint=fingerprint,
                    task_id=task_id, outer_trial_id=outer_id, seed=outer,
                    measurement_class="evolution" if execution_class_for_task(spec) == "episode" else "atomic_performance",
                    calibration_protocol_digest=protocol_digest,
                )
                envelope_path = out / "envelopes" / outer_id / f"{task_id}.json"
                envelope_path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                task_results.append(result)
                task_noises.append(noise)
                task_envelopes.append(envelope)
                timing.update({"timeout": False, "resource_blocked": True, "total_s": 0.0})
                timing_rows.append(timing)
                continue
            solution_dir = out / "solutions" / task_id / outer_id
            materialize_started = time.perf_counter()
            _copy_oracle(task_dir, solution_dir)
            timing["oracle_materialization_s"] = round(time.perf_counter() - materialize_started, 3)
            noise_path = noise_root / outer_id / f"{task_id}.json"
            noise_path.parent.mkdir(parents=True, exist_ok=True)
            result_path = raw_root / outer_id / f"{task_id}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            envelope_path = out / "envelopes" / outer_id / f"{task_id}.json"
            envelope_path.parent.mkdir(parents=True, exist_ok=True)
            static_envelope = {
                "schema_version": 1, "producer_revision": revision,
                "task_id": task_id, "outer_trial_id": outer_id, "seed": outer,
                "measurement_class": "evolution" if execution_class_for_task(spec) == "episode" else "atomic_performance",
                "task_package_digest": task_digest, "population_manifest_digest": population_digest,
                "harness_digest": harness_digest, "calibration_runner_digest": runner_digest,
                "calibration_protocol_digest": protocol_digest,
                "fingerprint": fingerprint,
            }
            cell_compatible = _cell_envelope_compatible(envelope_path, static=static_envelope, noise_path=noise_path, result_path=result_path)
            if cell_compatible:
                noise = json.loads(noise_path.read_text(encoding="utf-8"))
                result = json.loads(result_path.read_text(encoding="utf-8"))
                classification = classify_result(result)
                if classification == "blocked_requires_revision":
                    raise RuntimeError(f"calibration cell {task_id}/{outer_id} has a protocol failure and is blocked_requires_revision")
                if classification != "reusable":
                    had_artifacts = any(path.exists() for path in (noise_path, result_path, envelope_path))
                    _quarantine_cell_files(out, task_id, outer_id, [noise_path, result_path, envelope_path])
                    timing["quarantined"] = had_artifacts
                    classification = "rerun"
                else:
                    timing["reused"] = True
                    task_envelopes.append(json.loads(envelope_path.read_text(encoding="utf-8")))
            else:
                had_artifacts = any(path.exists() for path in (noise_path, result_path, envelope_path))
                _quarantine_cell_files(out, task_id, outer_id, [noise_path, result_path, envelope_path])
                timing["quarantined"] = had_artifacts
                classification = "rerun"
            if classification == "rerun":
                if execution_class_for_task(spec) == "episode":
                    noise = {
                        "schema_version": 1, "metric_class": "evolution",
                        "task_id": task_id, "outer_trial_id": outer_id,
                        "benchmark_revision": revision, "task_package_digest": task_digest,
                        "population_manifest_digest": population_digest,
                        "fingerprint": fingerprint,
                    }
                    noise["artifact_digest"] = digest_mapping({key: value for key, value in noise.items() if key != "artifact_digest"})
                    noise_path.write_text(json.dumps(noise, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                else:
                    print(json.dumps({"event": "start_noise_control", "task_id": task_id, "outer_trial_id": outer_id}, ensure_ascii=False), flush=True)
                    noise_started = time.perf_counter()
                    noise, noise_timed_out = _bounded_noise_control(
                        task_id=task_id, outer_trial_id=outer_id, noise_path=noise_path,
                        timeout_s=float(spec.get("time_budget_s", 600.0)),
                        args=(
                            "calibrate-noise-control", str(task_dir), "--solution", str(solution_dir),
                            "--out", str(noise_path), "--task-id", task_id,
                            "--outer-trial-id", outer_id, "--benchmark-revision", revision,
                            "--task-manifest-digest", digest,
                            "--task-package-digest", task_digest,
                            "--population-manifest-digest", population_digest,
                            "--compiler-cache-policy", verifier.cache_policy_for_task(spec),
                            "--seed", str(outer),
                        ),
                        cwd=repo_root,
                    )
                    timing["noise_control_s"] = round(time.perf_counter() - noise_started, 3)
                    if noise_timed_out:
                        cleanup = noise.get("executor_cleanup") if isinstance(noise, dict) else None
                        if isinstance(cleanup, dict):
                            timing["cleanup_status"] = "quiescent" if cleanup.get("quiescent") else "residual_process_group"
                        timing.update({"timeout": bool(noise.get("timeout", False)), "total_s": round(time.perf_counter() - cell_started, 3)})
                        timing_rows.append(timing)
                        result = _resource_blocked_result(
                            task_id=task_id, outer_trial_id=outer_id,
                            failure_stage=str(noise.get("failure_stage", "noise_control")),
                            timeout_s=float(spec.get("time_budget_s", 600.0)),
                            wall_time_s=float(noise.get("wall_time_s", 0.0)),
                            error=str(noise.get("error", "noise control timed out")),
                            measurement_class="atomic_performance",
                            timeout=bool(noise.get("timeout", False)),
                        )
                        result_path.write_text(
                            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8",
                        )
                        if isinstance(noise.get("artifact_digest"), str):
                            envelope = calibration_envelope(
                                producer_revision=revision, task_package_digest=task_digest,
                                population_manifest_digest=population_digest,
                                harness_digest_value=harness_digest,
                                calibration_runner_digest=runner_digest,
                                noise_digest=str(noise["artifact_digest"]),
                                raw_result_digest=file_digest(result_path),
                                fingerprint=fingerprint,
                                task_id=task_id, outer_trial_id=outer_id, seed=outer,
                                measurement_class="atomic_performance",
                                calibration_protocol_digest=protocol_digest,
                            )
                            envelope_path.write_text(
                                json.dumps(envelope, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8",
                            )
                            task_envelopes.append(envelope)
                        task_noises.append(noise)
                        task_results.append(result)
                        print(json.dumps({"task_id": task_id, "outer_trial_id": outer_id, "verdict": result.get("verdict"), "calibration_status": result.get("calibration_status"), "wall_time_s": result.get("cost", {}).get("wall_time_s")}, ensure_ascii=False), flush=True)
                        continue
                    noise = stats.read_noise_control(noise_path, {
                        "task_id": task_id,
                        "outer_trial_id": outer_id,
                        "benchmark_revision": revision,
                        "task_manifest_digest": digest,
                        "task_package_digest": task_digest,
                        "population_manifest_digest": population_digest,
                        "control_implementation": "baseline",
                    })
                print(json.dumps({"event": "start_verifier", "task_id": task_id, "outer_trial_id": outer_id}, ensure_ascii=False), flush=True)
                verifier_started = time.perf_counter()
                if execution_class_for_task(spec) == "episode":
                    result = verifier.verify_task(
                        task_dir, solution_dir, out_path=result_path, seed=outer,
                        condition="standalone", context_mode="reset",
                        outer_trial_id=outer_id,
                        task_package_digest=task_digest,
                        population_manifest_digest=population_digest,
                        noise_control_path=noise_path, noise_control_required=True,
                        noise_control_expected={
                            "task_id": task_id, "outer_trial_id": outer_id,
                            "benchmark_revision": revision, "task_manifest_digest": digest,
                            "task_package_digest": task_digest, "population_manifest_digest": population_digest,
                            "hardware_fingerprint": fingerprint, "software_fingerprint": fingerprint,
                        },
                    )
                else:
                    result = _bounded_verifier_result(
                        task_id=task_id, outer_trial_id=outer_id, result_path=result_path,
                        timeout_s=float(spec.get("time_budget_s", 600.0)),
                        module="benchmark.harness.cli",
                        args=(
                            "run-task", str(task_dir), "--solution", str(solution_dir),
                            "--out", str(result_path), "--seed", str(outer),
                            "--condition", "standalone", "--context-mode", "reset",
                            "--noise-control", str(noise_path), "--noise-control-required",
                            "--outer-trial-id", outer_id, "--benchmark-revision", revision,
                            "--task-manifest-digest", digest,
                            "--task-package-digest", task_digest,
                            "--population-manifest-digest", population_digest,
                        ),
                        cwd=repo_root,
                        measurement_class="atomic_performance",
                    )
                timing["verifier_s"] = round(time.perf_counter() - verifier_started, 3)
                cleanup = result.get("executor_cleanup") if isinstance(result, dict) else None
                if isinstance(cleanup, dict):
                    timing["cleanup_status"] = "quiescent" if cleanup.get("quiescent") else "residual_process_group"
                post_validation_started = time.perf_counter()
                post_validation = _bounded_post_validation(task_dir, repo_root)
                timing["post_validation_s"] = round(time.perf_counter() - post_validation_started, 3)
                if post_validation["timed_out"]:
                    result["protocol_failure"] = True
                    result["validity"] = "invalid"
                    result["execution_validity"] = "invalid"
                    result.setdefault("errors", []).append("post-run validate-task timed out")
                    timing["timeout"] = True
                elif post_validation["exit_code"] != 0:
                    result["protocol_failure"] = True
                    result["validity"] = "invalid"
                    result["execution_validity"] = "invalid"
                    result.setdefault("errors", []).append("post-run validate-task failed")
                result = serialize_cell_state(result)
                result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
                envelope = calibration_envelope(
                    producer_revision=revision, task_package_digest=task_digest,
                    population_manifest_digest=population_digest, harness_digest_value=harness_digest,
                    calibration_runner_digest=runner_digest, noise_digest=str(noise["artifact_digest"]),
                    raw_result_digest=file_digest(result_path), fingerprint=fingerprint,
                        task_id=task_id, outer_trial_id=outer_id, seed=outer,
                        measurement_class="evolution" if execution_class_for_task(spec) == "episode" else "atomic_performance",
                        calibration_protocol_digest=protocol_digest,
                )
                envelope_path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                task_envelopes.append(envelope)
            task_noises.append(noise)
            task_results.append(result)
            timing.update({"total_s": round(time.perf_counter() - cell_started, 3)})
            timing_rows.append(timing)
            print(json.dumps({"task_id": task_id, "outer_trial_id": outer_id, "verdict": result.get("verdict"), "calibration_status": result.get("calibration_status"), "wall_time_s": result.get("cost", {}).get("wall_time_s")}, ensure_ascii=False), flush=True)
        empirical.append(_calibration_record(
            task_dir, task_id, revision, task_digest, task_results, task_noises,
            task_envelopes, protocol_digest, population_digest,
            {
                "raw": [str((raw_root / f"outer-{index:03d}" / f"{task_id}.json").relative_to(out)) for index in range(len(task_results))],
                "noise": [str((noise_root / f"outer-{index:03d}" / f"{task_id}.json").relative_to(out)) for index in range(len(task_noises))],
                "envelopes": [str((out / "envelopes" / f"outer-{index:03d}" / f"{task_id}.json").relative_to(out)) for index in range(len(task_envelopes))],
            },
        ))
    report_generation_started = time.perf_counter()
    empirical_path = out / "empirical.json"
    empirical_path.write_text(json.dumps({"schema_version": 1, "calibration_protocol_digest": protocol_digest, "tasks": empirical}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report, calibration, errors = rebuild_calibration_views(
        tasks_root=tasks_root, empirical_path=empirical_path, manifest_path=active_path,
    )
    report_path = out / "population_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "pilot_calibration.json").write_text(json.dumps(calibration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"task_count": len(empirical), "errors": errors, "calibration_gate": calibration.get("calibration_gate"), "eligible": [item["task_id"] for item in calibration["tasks"] if item.get("eligibility")], "blocked": [item["task_id"] for item in calibration["tasks"] if not item.get("eligibility")]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    campaign_finished_monotonic = time.perf_counter()
    timing_report = {
        "schema_version": 1,
        "protocol_digest": protocol_digest,
        "campaign_started_monotonic": campaign_started_monotonic,
        "campaign_finished_monotonic": campaign_finished_monotonic,
        "campaign_started_unix_s": campaign_started_utc,
        "campaign_total_wall_s": campaign_finished_monotonic - campaign_started_monotonic,
        "report_generation_s": time.perf_counter() - report_generation_started,
        "cells": timing_rows,
        "cell_wall_sum_s": sum(float(item.get("total_s", 0.0)) for item in timing_rows),
    }
    (out / "timing_report.json").write_text(json.dumps(timing_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if not errors else 1
