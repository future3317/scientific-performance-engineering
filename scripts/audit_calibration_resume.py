#!/usr/bin/env python3
"""Classify active calibration cells before a resume.

An artifact is reusable only when its cell envelope binds the current task
package, population manifest, executable harness, runner, noise artifact and
raw result.  Legacy cells are reported as rerun_required; this command never
rewrites their provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.formal import attest
from benchmark.harness import miniyaml, runner, stats
from benchmark.harness.api import execution_class_for_task
from benchmark.harness.fingerprint import fingerprints_compatible
from benchmark.calibration.bundle import classify_result
from benchmark.calibration.identity import canonical_cell_identity
from benchmark.calibration.protocol import load_calibration_protocol, outer_trial_count


def audit(repo_root: Path, out: Path, outer_trials: int = 3) -> dict[str, object]:
    active_path = repo_root / "benchmark" / "pilot_population.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    task_root = repo_root / "benchmark" / "tasks"
    revision = attest.benchmark_revision(repo_root)
    population_digest = attest.file_digest(active_path)
    static_harness = attest.harness_digest(repo_root)
    runner_digest = attest.file_digest(repo_root / "benchmark" / "calibration" / "campaign.py")
    protocol, protocol_digest = load_calibration_protocol(repo_root)
    runner.configure_thread_topology(protocol["thread_topology"])
    from benchmark.harness.fingerprint import capture_fingerprint
    fingerprint = capture_fingerprint()
    reusable: list[dict[str, str]] = []
    rerun: list[dict[str, str]] = []
    blocked: list[dict[str, str]] = []
    for task_id in active["task_ids"]:
        task_id = str(task_id)
        task_dir = task_root / task_id
        task_digest = attest.task_package_digest(task_dir)
        task_spec = miniyaml.load(str(task_dir / "task.yaml"))
        if outer_trials != outer_trial_count(task_spec, protocol) and execution_class_for_task(task_spec) != "episode":
            blocked.append({"task_id": task_id, "reason": "protocol_outer_trial_count_mismatch"})
            continue
        for outer in range(outer_trial_count(task_spec, protocol)):
            outer_id = f"outer-{outer:03d}"
            noise = out / "noise-control" / outer_id / f"{task_id}.json"
            raw = out / "raw" / outer_id / f"{task_id}.json"
            envelope = out / "envelopes" / outer_id / f"{task_id}.json"
            reason = ""
            try:
                payload = json.loads(envelope.read_text(encoding="utf-8"))
                noise_payload = json.loads(noise.read_text(encoding="utf-8"))
                raw_payload = json.loads(raw.read_text(encoding="utf-8"))
                identity = canonical_cell_identity(
                    task_id=task_id, outer_trial_id=outer_id, seed=outer,
                    measurement_family=("evolution" if execution_class_for_task(task_spec) == "episode" else "atomic_performance"),
                    task_package_digest=task_digest,
                    population_manifest_digest=population_digest,
                )
                expected = {
                    "schema_version": 1, "producer_revision": revision,
                    "task_id": identity["task_id"], "outer_trial_id": identity["outer_trial_id"], "seed": identity["seed"],
                    "measurement_class": identity["envelope_measurement_class"],
                    "task_package_digest": task_digest,
                    "population_manifest_digest": population_digest,
                    "harness_digest": static_harness,
                    "calibration_runner_digest": runner_digest,
                    "calibration_protocol_digest": protocol_digest,
                    "fingerprint": fingerprint,
                }
                if attest.validate_calibration_envelope(payload, expected):
                    reason = "provenance_mismatch"
                elif (
                    any(raw_payload.get(key) != identity[key] for key in ("task_id", "outer_trial_id", "seed"))
                    or raw_payload.get("measurement_class") != identity["raw_measurement_class"]
                ):
                    reason = "raw_identity_mismatch"
                else:
                    compatible, _ = fingerprints_compatible(payload.get("fingerprint", {}), fingerprint)
                    if not compatible:
                        reason = "fingerprint_mismatch"
                    elif payload.get("noise_digest") != noise_payload.get("artifact_digest"):
                        reason = "noise_digest_mismatch"
                    else:
                        noise_expected = {
                            "task_id": task_id, "outer_trial_id": outer_id,
                            "benchmark_revision": revision, "task_package_digest": task_digest,
                            "population_manifest_digest": population_digest,
                        }
                    if expected["measurement_class"] == "atomic_performance":
                            try:
                                stats.read_noise_control(noise, noise_expected)
                            except ValueError:
                                reason = "noise_artifact_invalid"
                    if not reason and payload.get("raw_result_digest") != attest.file_digest(raw):
                        reason = "raw_result_digest_mismatch"
                    elif not reason and (noise_payload.get("task_package_digest") != task_digest or noise_payload.get("population_manifest_digest") != population_digest):
                        reason = "noise_identity_mismatch"
                    if not reason:
                        classification = classify_result(raw_payload)
                        if classification == "blocked_requires_revision":
                            reason = "blocked_requires_revision"
                        elif classification == "rerun":
                            reason = "rerun_required"
            except (OSError, json.JSONDecodeError, KeyError):
                reason = "legacy_or_incomplete_envelope"
            cell = {"task_id": task_id, "outer_trial_id": outer_id}
            if reason == "blocked_requires_revision":
                cell["reason"] = reason
                blocked.append(cell)
            elif reason:
                cell["reason"] = reason
                rerun.append(cell)
            else:
                reusable.append(cell)
    report = {
        "schema_version": 1, "producer_revision": revision,
        "reusable": reusable, "rerun_required": rerun, "blocked_requires_revision": blocked,
        "counts": {"reusable": len(reusable), "rerun_required": len(rerun), "blocked_requires_revision": len(blocked)},
    }
    destination = out / "calibration_resume_audit.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--outer-trials", type=int, default=3)
    args = parser.parse_args()
    report = audit(args.repo_root.resolve(), args.out.resolve(), args.outer_trials)
    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
