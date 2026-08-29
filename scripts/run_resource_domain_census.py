#!/usr/bin/env python3
"""Run the baseline-only resource feasibility census for the active population."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.calibration.execution import CellExecutor
from benchmark.calibration.identity import measurement_session_id, task_package_digest
from benchmark.calibration.protocol import load_calibration_protocol
from benchmark.formal.resource_domain import (
    build_resource_domain_manifest,
    classify_baseline_probe,
)
from benchmark.harness import miniyaml, runner, verifier
from benchmark.harness.api import execution_class_for_task
from benchmark.harness.fingerprint import capture_fingerprint, wait_for_selected_gpu_admission
from benchmark.provenance import benchmark_revision, file_digest


def _runtime_contract() -> dict[str, Any]:
    fingerprint = capture_fingerprint()
    runtime = {
        key: value for key, value in fingerprint.items()
        if key not in {"timestamp_utc", "env"}
    }
    try:
        import torch
        if torch.cuda.is_available():
            runtime["total_device_memory"] = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        pass
    return runtime


def _empty_probe(task_id: str, task_digest: str, session_id: str, **values: Any) -> dict[str, Any]:
    probe = {
        "task_id": task_id,
        "task_package_digest": task_digest,
        "execution_class": "atomic_performance",
        "baseline_only": True,
        "clean_admission": False,
        "foreign_processes": [],
        "protocol_valid": False,
        "baseline_completed": False,
        "resource_exhausted": False,
        "total_device_memory": None,
        "preflight_free_memory": None,
        "peak_allocated": None,
        "peak_reserved": None,
        "reserved_unallocated": None,
        "failed_allocation_bytes": None,
        "exception_class": None,
        "failure_signature": None,
        "confirmation_attempt_count": 0,
        "capacity_failure_reproduced": False,
        "cleanup_quiescent": False,
        "baseline_probe_session_id": session_id,
    }
    probe.update(values)
    return probe


def _foreign_processes(*snapshots: dict[str, Any]) -> list[Any]:
    observed: dict[str, Any] = {}
    for snapshot in snapshots:
        for process in snapshot.get("foreign_processes", []):
            key = str(process.get("pid")) if isinstance(process, dict) else repr(process)
            observed[key] = process
    return list(observed.values())


def _run_attempt(
    *, repo_root: Path, task_dir: Path, task_id: str, task_digest: str,
    output_path: Path, session_id: str, seed: int,
    session_start: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_spec = miniyaml.load(str(task_dir / "task.yaml"))
    started = time.perf_counter()
    solution_path = task_dir / "workspace" / str(task_spec["workspace"]["entrypoint"])
    completed = CellExecutor(repo_root).run_module(
        module="benchmark.formal.resource_domain",
        args=("--probe-task", str(task_dir), "--solution", str(solution_path), "--out", str(output_path), "--seed", str(seed)),
        timeout_s=float(task_spec.get("time_budget_s", 600.0)),
    )
    cleanup = completed.get("cleanup", {})
    if output_path.is_file():
        probe = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        probe = _empty_probe(
            task_id, task_digest, session_id,
            failure_signature=(completed.get("stderr") or completed.get("stdout") or "probe produced no artifact")[-4000:],
            exception_class="ResourceProbeProcessError",
        )
    try:
        from benchmark.harness.fingerprint import selected_gpu_preflight
        session_end = selected_gpu_preflight()
    except Exception:
        session_end = {}
    probe.update({
        "task_id": task_id,
        "task_package_digest": task_digest,
        "baseline_probe_session_id": session_id,
        "clean_admission": True,
        "foreign_processes": _foreign_processes(session_start or {}, session_end),
        "protocol_valid": bool(
            not runner.cleanup_failed(cleanup)
            and (
                completed.get("exit_code") == 0
                or completed.get("timed_out")
            )
        ),
        "cleanup_quiescent": not runner.cleanup_failed(cleanup),
        "executor_cleanup": cleanup,
        "timeout_stage": "baseline_probe" if completed.get("timed_out") else None,
        "timeout": bool(completed.get("timed_out")),
        "timeout_s": float(task_spec.get("time_budget_s", 600.0)),
        "wall_time_s": float(completed.get("wall_time_s", time.perf_counter() - started)),
        "stdout": str(completed.get("stdout", "")),
        "stderr": str(completed.get("stderr", "")),
        "session_start": session_start,
        "session_end": session_end,
        "baseline_artifact_completed": output_path.is_file(),
        "verifier_began": False,
        "timeout_reproduced": False,
    })
    if completed.get("timed_out"):
        probe["failure_signature"] = "resource probe timed out"
    return probe


def run_resource_domain_census(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "probes").mkdir(exist_ok=True)
    active_path = repo_root / "benchmark" / "pilot_population.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    protocol, protocol_digest = load_calibration_protocol(repo_root)
    observed_topology = runner.configure_thread_topology(protocol["thread_topology"])
    revision = benchmark_revision(repo_root)
    population_digest = file_digest(active_path)
    runtime = _runtime_contract()
    probes: list[dict[str, Any]] = []
    for task_id_value in active["task_ids"]:
        task_id = str(task_id_value)
        task_dir = repo_root / "benchmark" / "tasks" / task_id
        task_spec = miniyaml.load(str(task_dir / "task.yaml"))
        task_digest = task_package_digest(task_dir)
        session_id = measurement_session_id(
            task_id=task_id, outer_trial_id="resource-probe", task_package_digest=task_digest,
            population_manifest_digest=population_digest, benchmark_revision=revision,
            calibration_protocol_digest=protocol_digest,
        )
        if execution_class_for_task(task_spec) == "episode":
            probe = _empty_probe(
                task_id, task_digest, session_id, execution_class="evolution",
                clean_admission=True, protocol_valid=True, baseline_completed=True,
                cleanup_quiescent=True, not_applicable=True,
            )
            probes.append(probe)
            continue
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, 3):
            preflight = wait_for_selected_gpu_admission(
                timeout_s=float(protocol["resource_admission"]["timeout_s"]),
                poll_interval_s=float(protocol["resource_admission"]["poll_interval_s"]),
            )
            if preflight.get("status") not in {"clean", "busy"}:
                probes.append(_empty_probe(
                    task_id, task_digest, session_id,
                    foreign_processes=list(preflight.get("foreign_pids", [])),
                    failure_signature=f"resource admission {preflight.get('status')}: {preflight.get('reason')}",
                ))
                break
            attempt_path = out / "probes" / task_id / f"attempt-{attempt:02d}.json"
            attempt_path.parent.mkdir(parents=True, exist_ok=True)
            current = _run_attempt(
                repo_root=repo_root, task_dir=task_dir, task_id=task_id,
                task_digest=task_digest, output_path=attempt_path,
                session_id=f"{session_id}-attempt-{attempt:02d}", seed=attempt - 1,
                session_start=preflight,
            )
            current["confirmation_attempt_count"] = attempt
            attempts.append(current)
            status = classify_baseline_probe(current)
            needs_confirmation = (
                status == "unresolved"
                and (
                    bool(current.get("resource_exhausted"))
                    or (
                        bool(current.get("timeout"))
                        and current.get("timeout_stage") == "baseline_probe"
                    )
                )
            )
            if attempt == 2:
                current["capacity_failure_reproduced"] = bool(
                    attempts[0].get("failure_signature") == current.get("failure_signature")
                    and attempts[0].get("resource_exhausted")
                    and current.get("resource_exhausted")
                )
                current["timeout_reproduced"] = bool(
                    attempts[0].get("failure_signature") == current.get("failure_signature")
                    and attempts[0].get("timeout")
                    and current.get("timeout")
                    and attempts[0].get("timeout_stage") == current.get("timeout_stage")
                    and attempts[0].get("timeout_s") == current.get("timeout_s")
                )
                status = classify_baseline_probe(current)
            if status != "unresolved" or not needs_confirmation or attempt == 2:
                probes.append(current)
                break
        else:
            probes.append(attempts[-1])
    manifest_path = out / "resource_domain_eligibility.json"
    manifest = build_resource_domain_manifest(
        repo_root=repo_root, task_probes=probes, runtime_contract=runtime, out_path=manifest_path,
    )
    (out / "census_metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "benchmark_revision": revision,
        "population_manifest_digest": population_digest,
        "resource_protocol_digest": file_digest(repo_root / "benchmark" / "formal" / "resource_domain.py"),
        "calibration_protocol_digest": protocol_digest,
        "thread_topology": observed_topology,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime_contract": runtime,
        "task_count": len(probes),
        "in_domain_count": len(manifest["in_domain_task_ids"]),
        "out_of_domain_count": len(manifest["out_of_domain_task_ids"]),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "frozen", "task_count": len(manifest["task_ids"]),
        "in_domain": len(manifest["in_domain_task_ids"]),
        "out_of_domain_capacity": len(manifest["out_of_domain_task_ids"]),
        "artifact_digest": manifest["artifact_digest"],
    }))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    return run_resource_domain_census(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
