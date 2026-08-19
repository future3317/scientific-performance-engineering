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
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.formal import attest
from benchmark.harness import miniyaml, verifier
from benchmark.harness.fingerprint import capture_fingerprint
from benchmark.taskgen.validate_population import build_pilot_calibration, build_report


def _copy_oracle(task_dir: Path, solution_dir: Path) -> None:
    spec = miniyaml.load(str(task_dir / "task.yaml"))
    entrypoint = str(spec["workspace"]["entrypoint"])
    solution_dir.mkdir(parents=True, exist_ok=True)
    oracle = task_dir / "oracle" / "solution_oracle.py"
    if oracle.is_file():
        shutil.copy2(oracle, solution_dir / entrypoint)
        return
    patch_path = task_dir / "oracle" / "reference_patch.diff"
    if not patch_path.is_file():
        raise FileNotFoundError(f"oracle solution and reference patch missing for {task_dir}")
    shutil.copytree(task_dir / "workspace", solution_dir, dirs_exist_ok=True)
    completed = subprocess.run(
        ["patch", "-p2", "-i", str(patch_path)],
        cwd=solution_dir, text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"reference patch failed for {task_dir.name}: {completed.stderr or completed.stdout}")


def _percent(value: Any) -> float | None:
    return (float(value) - 1.0) * 100.0 if isinstance(value, (int, float)) else None


def _calibration_record(task_dir: Path, task_id: str, revision: str, digest: str, results: list[dict[str, Any]], noises: list[dict[str, Any]]) -> dict[str, Any]:
    first = results[0] if results else {}
    gates = {
        "correctness_pass": all(bool(item.get("correctness_pass")) for item in results),
        "scientific_gates": all(all(bool(value) for value in item.get("scientific_gates", {}).values()) for item in results),
        "execution_valid": all(item.get("execution_validity", "valid") not in {"invalid", "resource_blocked"} for item in results),
    }
    verified = first.get("verified_speedup", {}) if isinstance(first.get("verified_speedup"), dict) else {}
    anti = first.get("anticheat", {}) if isinstance(first.get("anticheat"), dict) else {}
    return {
        "task_id": task_id,
        "task_digest": digest,
        "revision": revision,
        "environment": first.get("fingerprint") or capture_fingerprint(),
        "outer_trials": results,
        "noise_control": {"artifacts": noises, "effective_noise_floor_percent": max((float(item.get("effective_noise_floor_percent", 0.0)) for item in noises), default=0.0)},
        "oracle_ci": verified,
        "oracle_ci_low_percent": _percent(verified.get("ci_low")),
        "oracle_ci_high_percent": _percent(verified.get("ci_high")),
        "semantic_gates": gates,
        "semantic_gate_pass_rate": sum(bool(value) for value in gates.values()) / max(1, len(gates)),
        "anti_cheat": {"status": "pass" if not anti.get("hard_fail") and not anti.get("tripwired") else "fail", "findings": anti.get("findings", [])},
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
    empirical: list[dict[str, Any]] = []
    for task_id in task_ids:
        task_dir = tasks_root / task_id
        spec = miniyaml.load(str(task_dir / "task.yaml"))
        task_digest = attest.task_package_digest(task_dir)
        task_results: list[dict[str, Any]] = []
        task_noises: list[dict[str, Any]] = []
        for outer in range(int(args.outer_trials)):
            outer_id = f"outer-{outer:03d}"
            solution_dir = out / "solutions" / task_id / outer_id
            _copy_oracle(task_dir, solution_dir)
            noise_path = noise_root / outer_id / f"{task_id}.json"
            noise_path.parent.mkdir(parents=True, exist_ok=True)
            if noise_path.is_file():
                noise = json.loads(noise_path.read_text(encoding="utf-8"))
            else:
                noise = verifier.calibrate_noise_control(
                    task_dir, solution_dir, noise_path,
                    task_id=task_id, outer_trial_id=outer_id,
                    benchmark_revision=revision, task_manifest_digest=digest,
                    hardware_fingerprint=fingerprint,
                    compiler_cache_policy=verifier.cache_policy_for_task(spec), seed=outer,
                )
            task_noises.append(noise)
            result_path = raw_root / outer_id / f"{task_id}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                result = verifier.verify_task(
                    task_dir, solution_dir, out_path=result_path, seed=outer,
                    condition="standalone", context_mode="reset",
                    noise_control_path=noise_path, noise_control_required=True,
                    noise_control_expected={
                        "task_id": task_id, "outer_trial_id": outer_id,
                        "benchmark_revision": revision, "task_manifest_digest": digest,
                        "hardware_fingerprint": fingerprint, "software_fingerprint": fingerprint,
                    },
                )
            task_results.append(result)
            print(json.dumps({"task_id": task_id, "outer_trial_id": outer_id, "verdict": result.get("verdict"), "calibration_status": result.get("calibration_status"), "wall_time_s": result.get("cost", {}).get("wall_time_s")}, ensure_ascii=False), flush=True)
        empirical.append(_calibration_record(task_dir, task_id, revision, task_digest, task_results, task_noises))
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
