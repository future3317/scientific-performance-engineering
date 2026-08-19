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
import sys
import os
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.formal import attest
from benchmark.harness import miniyaml, stats, verifier
from benchmark.harness.fingerprint import capture_fingerprint, fingerprints_compatible
from benchmark.taskgen.validate_population import build_pilot_calibration, build_report


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
        shutil.copytree(task_dir / "workspace", staged)
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
    destination = out / "quarantine" / outer_id / task_id
    destination.mkdir(parents=True, exist_ok=True)
    for path in existing:
        target = destination / path.name
        if target.exists():
            target = destination / f"{path.stem}.previous{path.suffix}"
        shutil.move(str(path), str(target))


def _cell_envelope_compatible(path: Path, *, static: dict[str, Any], noise_path: Path, result_path: Path) -> bool:
    if not path.is_file() or not noise_path.is_file() or not result_path.is_file():
        return False
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        noise = json.loads(noise_path.read_text(encoding="utf-8"))
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        for key in ("task_id", "outer_trial_id", "seed", "measurement_class"):
            if raw.get(key) != static.get(key):
                return False
        if attest.validate_calibration_envelope(envelope, static):
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
            })
        return envelope.get("raw_result_digest") == attest.file_digest(result_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False


def _percent(value: Any) -> float | None:
    return (float(value) - 1.0) * 100.0 if isinstance(value, (int, float)) else None


def _calibration_record(task_dir: Path, task_id: str, revision: str, digest: str, results: list[dict[str, Any]], noises: list[dict[str, Any]], envelopes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    first = results[0] if results else {}
    episode = bool(results and isinstance(first.get("episode_measurement"), dict))
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
    calibration_status = "eligible" if all(item.get("calibration_status") == "eligible" for item in results) else "blocked"
    return {
        "task_id": task_id,
        "task_digest": digest,
        "revision": revision,
        "environment": first.get("fingerprint") or capture_fingerprint(),
        "outer_trials": results,
        "evidence_envelopes": list(envelopes or []),
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


def run(args: argparse.Namespace) -> int:
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
    raw_root = out / "raw"
    noise_root = out / "noise-control"
    raw_root.mkdir(parents=True, exist_ok=True)
    noise_root.mkdir(parents=True, exist_ok=True)
    revision = attest.benchmark_revision(repo_root)
    digest = attest.task_manifest_digest(tasks_root, [str(item) for item in active["task_ids"]])
    fingerprint = capture_fingerprint()
    population_digest = attest.file_digest(active_path)
    harness_digest = attest.harness_digest(repo_root)
    runner_digest = attest.file_digest(Path(__file__))
    empirical: list[dict[str, Any]] = []
    for task_id in task_ids:
        task_dir = tasks_root / task_id
        spec = miniyaml.load(str(task_dir / "task.yaml"))
        task_digest = attest.task_package_digest(task_dir)
        task_results: list[dict[str, Any]] = []
        task_noises: list[dict[str, Any]] = []
        task_envelopes: list[dict[str, Any]] = []
        for outer in range(int(args.outer_trials)):
            outer_id = f"outer-{outer:03d}"
            print(json.dumps({"event": "start_outer_trial", "task_id": task_id, "outer_trial_id": outer_id}, ensure_ascii=False), flush=True)
            solution_dir = out / "solutions" / task_id / outer_id
            _copy_oracle(task_dir, solution_dir)
            noise_path = noise_root / outer_id / f"{task_id}.json"
            noise_path.parent.mkdir(parents=True, exist_ok=True)
            result_path = raw_root / outer_id / f"{task_id}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            envelope_path = out / "envelopes" / outer_id / f"{task_id}.json"
            envelope_path.parent.mkdir(parents=True, exist_ok=True)
            static_envelope = {
                "schema_version": 1, "producer_revision": revision,
                "task_id": task_id, "outer_trial_id": outer_id, "seed": outer,
                "measurement_class": "evolution" if spec.get("workspace", {}).get("api") == "episode_v1" else "atomic_performance",
                "task_package_digest": task_digest, "population_manifest_digest": population_digest,
                "harness_digest": harness_digest, "calibration_runner_digest": runner_digest,
                "fingerprint": fingerprint,
            }
            if _cell_envelope_compatible(envelope_path, static=static_envelope, noise_path=noise_path, result_path=result_path):
                noise = json.loads(noise_path.read_text(encoding="utf-8"))
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("protocol_failure") or (result.get("validity") == "invalid" and result.get("execution_validity") == "invalid"):
                    raise RuntimeError(f"calibration cell {task_id}/{outer_id} has a protocol failure and is blocked_requires_revision")
                task_envelopes.append(json.loads(envelope_path.read_text(encoding="utf-8")))
            else:
                _quarantine_cell_files(out, task_id, outer_id, [noise_path, result_path, envelope_path])
                if spec.get("workspace", {}).get("api") == "episode_v1":
                    noise = {
                        "schema_version": 1, "metric_class": "evolution",
                        "task_id": task_id, "outer_trial_id": outer_id,
                        "benchmark_revision": revision, "task_package_digest": task_digest,
                        "population_manifest_digest": population_digest,
                        "fingerprint": fingerprint,
                    }
                    noise["artifact_digest"] = attest.digest_mapping({key: value for key, value in noise.items() if key != "artifact_digest"})
                    noise_path.write_text(json.dumps(noise, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                else:
                    print(json.dumps({"event": "start_noise_control", "task_id": task_id, "outer_trial_id": outer_id}, ensure_ascii=False), flush=True)
                    noise = verifier.calibrate_noise_control(
                        task_dir, solution_dir, noise_path,
                        task_id=task_id, outer_trial_id=outer_id,
                        benchmark_revision=revision, task_manifest_digest=digest,
                        task_package_digest=task_digest, population_manifest_digest=population_digest,
                        hardware_fingerprint=fingerprint,
                        compiler_cache_policy=verifier.cache_policy_for_task(spec), seed=outer,
                    )
                task_noises.append(noise)
                print(json.dumps({"event": "start_verifier", "task_id": task_id, "outer_trial_id": outer_id}, ensure_ascii=False), flush=True)
                result = verifier.verify_task(
                    task_dir, solution_dir, out_path=result_path, seed=outer,
                    condition="standalone", context_mode="reset",
                    outer_trial_id=outer_id,
                    noise_control_path=noise_path, noise_control_required=True,
                    noise_control_expected={
                        "task_id": task_id, "outer_trial_id": outer_id,
                        "benchmark_revision": revision, "task_manifest_digest": digest,
                        "task_package_digest": task_digest, "population_manifest_digest": population_digest,
                        "hardware_fingerprint": fingerprint, "software_fingerprint": fingerprint,
                    },
                )
                envelope = attest.calibration_envelope(
                    producer_revision=revision, task_package_digest=task_digest,
                    population_manifest_digest=population_digest, harness_digest_value=harness_digest,
                    calibration_runner_digest=runner_digest, noise_digest=str(noise["artifact_digest"]),
                    raw_result_digest=attest.file_digest(result_path), fingerprint=fingerprint,
                    task_id=task_id, outer_trial_id=outer_id, seed=outer,
                    measurement_class="evolution" if spec.get("workspace", {}).get("api") == "episode_v1" else "atomic_performance",
                )
                envelope_path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                task_envelopes.append(envelope)
            if not task_noises or task_noises[-1] is not noise:
                task_noises.append(noise)
            task_results.append(result)
            print(json.dumps({"task_id": task_id, "outer_trial_id": outer_id, "verdict": result.get("verdict"), "calibration_status": result.get("calibration_status"), "wall_time_s": result.get("cost", {}).get("wall_time_s")}, ensure_ascii=False), flush=True)
        empirical.append(_calibration_record(task_dir, task_id, revision, task_digest, task_results, task_noises, task_envelopes))
    empirical_path = out / "empirical.json"
    empirical_path.write_text(json.dumps({"schema_version": 1, "tasks": empirical}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report, errors = build_report(tasks_root, empirical_path, active_path)
    report_path = out / "population_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    calibration = build_pilot_calibration(report, tasks_root)
    (out / "pilot_calibration.json").write_text(json.dumps(calibration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"task_count": len(empirical), "errors": errors, "calibration_gate": calibration.get("calibration_gate"), "eligible": [item["task_id"] for item in calibration["tasks"] if item.get("eligibility")], "blocked": [item["task_id"] for item in calibration["tasks"] if not item.get("eligibility")]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--outer-trials", type=int, default=3)
    parser.add_argument("--task-id", default=None)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
