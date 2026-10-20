"""One guarded server run from read-only v3 audit through v4 closeout."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

from .audit_v3 import write_failure_table
from .cache_lifecycle import cache_usage
from .contracts import (
    build_frozen_manifest, family_split, file_digest, read_config,
    registered_family_ids, resolve_server_path, split_family_counts,
    write_or_verify_manifest,
)
from .matched_sessions import (
    load_pair_observation, run_pair, select_study_gpu, wait_for_selected_gpu,
)
from .outcomes import atomic_json, atomic_write_text, canonical_digest, recover_attempt
from .process_supervisor import execute_arm


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validated_results_root(value: str | Path) -> Path:
    """Resolve the one supported result-root boundary before creating anything."""
    root = Path(value).expanduser().resolve()
    allowed = (Path.home() / "RESULTS" / "evospe-rt-v4").resolve()
    if root == allowed or root.parent != allowed:
        raise ValueError(f"v4 results root must be one study directory under {allowed}")
    return root


def _git(*arguments: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=cwd, capture_output=True, text=True,
        timeout=10, check=True,
    )
    return result.stdout.strip()


def _assert_server_contract(config: Mapping[str, Any], root: Path) -> tuple[Path, str]:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("v4 evidence runs are supported only on the configured dbcloud Linux server")
    repo = _repo_root()
    expected_repo = (Path.home() / "CODE" / "scientific-performance-engineering").resolve()
    if repo != expected_repo or Path.cwd().resolve() != expected_repo:
        raise RuntimeError(f"run from the canonical server repository: {expected_repo}")
    expected_python = Path.home() / "miniconda3" / "envs" / "equivcompiler" / "bin" / "python"
    if Path(sys.executable) != expected_python:
        raise RuntimeError(f"use the project interpreter exactly: {expected_python}")
    commit = _git("rev-parse", "HEAD", cwd=repo)
    if len(commit) != 40:
        raise RuntimeError("could not establish the exact source commit")
    dirty = _git("status", "--porcelain", "--untracked-files=all", cwd=repo)
    if dirty:
        raise RuntimeError("refusing evidence run from a dirty canonical repository")
    allowed_root = (Path.home() / "RESULTS" / "evospe-rt-v4").resolve()
    if not root.is_relative_to(allowed_root):
        raise RuntimeError(f"v4 output must stay under {allowed_root}")
    torchbench = resolve_server_path(str(config["torchbench_root"]))
    overlay = resolve_server_path(str(config["workload_dependency_overlay"]))
    if not torchbench.is_dir() or not overlay.is_dir():
        raise FileNotFoundError("the frozen TorchBench source or dependency overlay is unavailable")
    torchbench_commit = _git("rev-parse", "HEAD", cwd=torchbench)
    if torchbench_commit != str(config["torchbench_commit"]):
        raise RuntimeError("TorchBench checkout differs from the preregistered commit")
    return repo, commit


def _validate_split_freeze(config: Mapping[str, Any], repo: Path) -> str:
    path = repo / "experiments" / "real_intervention_v4" / "config" / "SPLIT_FREEZE.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    split = config["split"]
    confirmatory = frozen["confirmatory"]
    exploratory = frozen["underpowered_exploratory"]
    expected_confirmatory = {
        "minimum_total_families": int(split["minimum_total_families_for_confirmatory"]),
        "minimum_development_families": int(split["minimum_development_families"]),
        "minimum_calibration_families": int(split["minimum_calibration_families"]),
        "minimum_calibration_support_families": int(
            config["acre"]["minimum_calibration_support_families"]
        ),
        "minimum_calibration_non_support_families": int(
            config["acre"]["minimum_calibration_non_support_families"]
        ),
        "minimum_final_families": int(split["minimum_final_families"]),
        "development_allocation": "remaining families after the frozen calibration and final minima",
    }
    expected_exploratory = {
        "development_family_cap": int(split["underpowered_development_family_cap"]),
        "calibration_family_cap": int(split["underpowered_calibration_family_cap"]),
        "residual_assignment": "final; never opened unless every confirmatory gate passes",
        "final_opened": False,
    }
    if (
        frozen.get("schema_version") != 2
        or int(frozen.get("seed", -1)) != int(config["seed"])
        or frozen.get("algorithm") != "sorted_unique_registered_family_lineage_ids_then_python_random.Random(seed).shuffle"
        or frozen.get("assignment_order") != ["development", "calibration", "final"]
        or frozen.get("split_population") != (
            "all registered independent family lineages are assigned before qualification; "
            "qualification failures remain in their assigned split and denominator"
        )
        or confirmatory != expected_confirmatory
        or exploratory != expected_exploratory
        or frozen.get("primary_final_gate") != (
            "Open the independent primary final after the frozen family minima, terminal development/calibration pairs, "
            "passing 12-pair shakedown, and pre-final policy freeze. ACRE certification is required for ACRE "
            "deployment/scoring only, not for the primary effect or simple comparator final."
        )
        or not frozen.get("eligibility_roster_rule")
    ):
        raise ValueError("PRIMARY_LAYOUT_STUDY.json differs from the committed SPLIT_FREEZE.json")
    return file_digest(path)


def _atomic_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple))
                         else value for key, value in row.items()})
    atomic_write_text(path, buffer.getvalue())


def _write_cpu_suite(root: Path, *, source_commit: str, config_digest: str) -> bool:
    command = [sys.executable, "-m", "pytest", "-q", "experiments/real_intervention_v4/tests"]
    result_path = root / "PRE_EXPERIMENT_TESTS.json"
    log_path = root / "PRE_EXPERIMENT_TESTS.log"
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if (prior.get("source_commit") != source_commit
                or prior.get("config_digest") != config_digest
                or prior.get("command") != command
                or prior.get("suite") != "experiments/real_intervention_v4/tests"
                or not log_path.is_file()):
            raise ValueError("pre-experiment CPU suite identity or log differs; use a new v4 study root")
        return prior.get("exit_code") == 0
    if log_path.exists():
        raise ValueError("pre-experiment CPU log has no result record; refusing to rerun it")
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    started = time.time()
    try:
        result = subprocess.run(
            command, cwd=_repo_root(), env=env, capture_output=True, text=True,
            timeout=900,
        )
        output = result.stdout + result.stderr
        exit_code = result.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        exit_code = 124
        output += "\nCPU_FAULT_SUITE_TIMEOUT after 900 seconds\n"
    atomic_write_text(log_path, output)
    atomic_json(result_path, {
        "command": command, "cwd": str(_repo_root()), "cuda_visible_devices": "",
        "source_commit": source_commit, "config_digest": config_digest,
        "started_unix": started, "ended_unix": time.time(),
        "elapsed_seconds": time.time() - started, "exit_code": exit_code,
        "suite": "experiments/real_intervention_v4/tests",
    })
    return exit_code == 0


def _source_environment(commit: str, config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "torchvision", "triton", "timm", "numpy", "scipy", "PyYAML", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    torchbench = resolve_server_path(str(config["torchbench_root"]))
    overlay = resolve_server_path(str(config["workload_dependency_overlay"]))
    try:
        gpu_inventory = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=3, check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        gpu_inventory = [f"INVENTORY_ERROR:{type(exc).__name__}:{exc}"]
    return {
        "source_commit": commit, "python_executable": sys.executable,
        "python_version": sys.version, "platform": platform.platform(),
        "package_versions": versions,
        "torchbench_commit": _git("rev-parse", "HEAD", cwd=torchbench),
        "torchbench_root": str(torchbench), "dependency_overlay": str(overlay),
        "gpu_inventory_before_selection": gpu_inventory,
        "config_digest": canonical_digest(dict(config)),
        "config_filename": "PRIMARY_LAYOUT_STUDY.json",
        "run_root": str(root), "recorded_utc": _utc_now(),
    }


def _write_v3_audit(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    roots = [resolve_server_path(item) for item in config["v3_failure_roots"]]
    pair_path = roots[0] / "PAIRS.jsonl"
    if not pair_path.is_file():
        raise FileNotFoundError(f"authoritative latest v3 PAIRS.jsonl is missing: {pair_path}")
    summary = write_failure_table(pair_path, roots, root)
    if not summary["closure_check"]["matches_expected"]:
        raise RuntimeError("v3 read-only failure table did not close exactly 90 original pairs")
    return summary


def _qualification_spec(candidate: Mapping[str, Any], batch: int, phase: str,
                        config: Mapping[str, Any], root: Path, gpu_uuid: str,
                        repo: Path) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    safe = "".join(char if char.isalnum() or char in "-_." else "-" for char in candidate_id)
    phase_id = {"s0-reference": "S0REF", "s0-replicate": "S0AA",
                "s1-correctness": "S1CORR"}[phase]
    attempt_id = f"qual-{safe}-B{batch}-{phase_id}"
    attempt_dir = root / "raw" / attempt_id
    cache_root = (attempt_dir / "cache").resolve()
    cache_environment = {
        "TORCHINDUCTOR_CACHE_DIR": str((cache_root / "inductor").resolve()),
        "TRITON_CACHE_DIR": str((cache_root / "triton").resolve()),
        "CUDA_CACHE_PATH": str((cache_root / "cuda").resolve()),
        "TMPDIR": str((cache_root / "tmp").resolve()),
        "XDG_CACHE_HOME": str((cache_root / "xdg").resolve()),
    }
    reference_key = f"{safe}-B{batch}"
    reference_path = (root / "references" / f"{reference_key}.pt").resolve()
    qualification_spec = {
        "candidate": dict(candidate), "batch": int(batch), "seed": int(config["seed"]),
        "source_revisions": dict((candidate.get("source_identity") or {}).get("source_revisions", {})),
        "shape": list(candidate.get("input_shape", config["shape"])), "phase": phase,
        "torchbench_root": str(resolve_server_path(str(config["torchbench_root"]))),
        "reference_root": str(reference_path.parent), "reference_path": str(reference_path),
        "metadata_path": str(reference_path.with_suffix(".s0.json")),
        "correctness": dict(config["correctness"]), "gpu_uuid": gpu_uuid,
        "cache_scope_path": str(cache_root), "cache_environment": cache_environment,
    }
    identity = {
        "study_id": config["study_id"], "phase": phase.upper().replace("-", "_"),
        "candidate_id": candidate_id, "batch": int(batch),
        "source_commit": _git("rev-parse", "HEAD", cwd=repo),
        "config_digest": canonical_digest(dict(config)), "gpu_uuid": gpu_uuid,
        "qualification_digest": canonical_digest(qualification_spec),
    }
    return {**qualification_spec, "identity": identity, "attempt_dir": str(attempt_dir),
            "attempt_id": attempt_id}


def _qualification_phase_row(candidate: Mapping[str, Any], batch: int, phase: str,
                              details: Mapping[str, Any], *, recovered: Mapping[str, Any],
                              execution: Mapping[str, Any] | None = None) -> dict[str, Any]:
    worker_result = recovered.get("worker_result")
    if isinstance(worker_result, Mapping):
        row = dict(worker_result)
    else:
        arm_execution = recovered.get("arm_execution", "UNKNOWN")
        row = {
            "candidate_id": candidate["candidate_id"], "source": candidate.get("source"),
            "model": candidate.get("model"), "family_id": candidate.get("family_id"),
            "batch": batch, "phase": phase,
            "status": "INCOMPLETE" if arm_execution in {"UNKNOWN", "NOT_STARTED"} else "ERROR",
            "eligibility": f"QUALIFICATION_{arm_execution}",
            "arm_execution": arm_execution,
            "scientific_correctness": "UNKNOWN" if arm_execution == "UNKNOWN" else "UNASSESSED",
            "first_failure_stage": recovered.get("first_failure_stage"),
            "reason": recovered.get("reason") or recovered.get("root_cause"),
        }
    row.update({
        "phase": phase, "phase_identity": details["identity"],
        "supervisor_receipt_sha256": recovered.get("raw_receipt_sha256"),
        "receipt_verified": recovered.get("receipt_verified") is True,
        "supervisor_execution": recovered.get("arm_execution"),
    })
    for key in ("root_cause", "error_type", "error", "cache_error_type", "cache_error"):
        if recovered.get(key) is not None:
            row[key] = recovered[key]
    if execution is not None:
        row["parent_wall_seconds"] = execution.get("parent_wall_seconds")
        row["elapsed_ceiling_seconds"] = execution.get("elapsed_ceiling_seconds")
        row["within_deadline_plus_grace"] = execution.get("within_deadline_plus_grace")
    return row


def _run_qualification_phase(candidate: Mapping[str, Any], batch: int, phase: str, *,
                             config: Mapping[str, Any], root: Path, gpu_uuid: str,
                             wall_deadline: float, poll_seconds: float,
                             repo: Path) -> dict[str, Any]:
    details = _qualification_spec(candidate, batch, phase, config, root, gpu_uuid, repo)
    attempt_id = str(details["attempt_id"])
    attempt_dir = Path(details["attempt_dir"])
    spec_path = attempt_dir / "spec.json"
    result_path = attempt_dir / "result.json"
    phase_record = root / "qualification" / "phases" / f"{attempt_id}.json"
    phase_record.parent.mkdir(parents=True, exist_ok=True)
    wrapper = {key: value for key, value in details.items()
               if key not in {"identity", "attempt_dir", "attempt_id"}}
    wrapper["qualification_phase"] = phase
    command = [sys.executable, "-m", "experiments.real_intervention_v4.qualification",
               "--spec", str(spec_path), "--output", str(result_path), "--phase", phase]
    overlay = str(resolve_server_path(str(config["workload_dependency_overlay"])))
    environment = {key: str(value) for key, value in details["cache_environment"].items()}
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (overlay, str(repo), os.environ.get("PYTHONPATH", "")))
    )
    frozen_identity = {
        **details["identity"], "attempt_id": attempt_id, "gpu_uuid": gpu_uuid,
        "spec_digest": canonical_digest(wrapper),
    }

    prior = None
    if phase_record.is_file():
        prior = json.loads(phase_record.read_text(encoding="utf-8"))
        if prior.get("phase_identity") != details["identity"]:
            raise ValueError(f"qualification phase identity changed for {attempt_id}")
        if prior.get("arm_execution") == "NOT_STARTED":
            return prior
        if not attempt_dir.is_dir():
            prior.update({
                "status": "INCOMPLETE", "eligibility": "QUALIFICATION_RECEIPT_UNRESOLVED",
                "arm_execution": "UNKNOWN", "scientific_correctness": "UNKNOWN",
                "first_failure_stage": "ATTEMPT_DIRECTORY_MISSING",
            })
            atomic_json(phase_record, prior)
            return prior
        if not spec_path.is_file():
            prior.update({
                "status": "INCOMPLETE", "eligibility": "QUALIFICATION_RECEIPT_UNRESOLVED",
                "arm_execution": "UNKNOWN", "scientific_correctness": "UNKNOWN",
                "first_failure_stage": "ATTEMPT_SPEC_MISSING",
            })
            atomic_json(phase_record, prior)
            return prior
    if attempt_dir.is_dir() and spec_path.is_file():
        prior_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if prior_spec != wrapper:
            raise ValueError(f"qualification phase spec changed for {attempt_id}")
        recovered = recover_attempt(attempt_dir, frozen_identity)
        if recovered.get("arm_execution") != "NOT_STARTED":
            row = _qualification_phase_row(candidate, batch, phase, details, recovered=recovered)
            if prior is not None:
                if (prior.get("supervisor_receipt_sha256")
                        != recovered.get("raw_receipt_sha256")
                        or recovered.get("receipt_verified") is not True):
                    prior.update({
                        "status": "INCOMPLETE", "eligibility": "QUALIFICATION_RECEIPT_UNRESOLVED",
                        "arm_execution": "UNKNOWN", "scientific_correctness": "UNKNOWN",
                        "first_failure_stage": "RECEIPT_RECOVERY",
                        "reason": recovered.get("reason") or "verified receipt differs from the frozen phase record",
                    })
                    atomic_json(phase_record, prior)
                return prior
            if prior is None:
                atomic_json(phase_record, row)
            return row
        if any(attempt_dir.iterdir()):
            row = _qualification_phase_row(candidate, batch, phase, details, recovered={
                "arm_execution": "UNKNOWN", "first_failure_stage": "ATTEMPT_WITHOUT_INTENT",
                "reason": "prior attempt has artifacts without a recoverable lifecycle intent",
            })
            atomic_json(phase_record, row)
            return row
    elif attempt_dir.exists() and any(attempt_dir.iterdir()):
        row = _qualification_phase_row(candidate, batch, phase, details, recovered={
            "arm_execution": "UNKNOWN", "first_failure_stage": "CACHE_INITIALIZATION_RECOVERY",
            "reason": "prior cache artifacts exist without a frozen phase spec; automatic retry is forbidden",
        })
        atomic_json(phase_record, row)
        return row

    if time.monotonic() >= wall_deadline or not wait_for_selected_gpu(
        root, gpu_uuid, deadline=wall_deadline, poll_seconds=poll_seconds,
        idle_memory_limit_mib=int(config["gpu_selection"]["idle_memory_limit_mib"]),
    ):
        row = _qualification_phase_row(candidate, batch, phase, details, recovered={
            "arm_execution": "NOT_STARTED", "first_failure_stage": "STUDY_WALL_CAP",
            "reason": "study wall cap expired before this qualification phase started",
        })
        atomic_json(phase_record, row)
        return row

    attempt_dir.mkdir(parents=True, exist_ok=True)
    execution = execute_arm(
        results_root=root, attempt_id=attempt_id, command=command,
        spec=wrapper, identity=details["identity"], gpu_uuid=gpu_uuid,
        timeout_seconds=float(config["arm_timeout_seconds"]),
        cleanup_grace_seconds=float(config["cleanup_grace_seconds"]),
        term_grace_seconds=float(config["term_grace_seconds"]),
        heartbeat_seconds=float(config["watchdog_heartbeat_seconds"]),
        sample_seconds=float(config["gpu_sample_seconds"]), environment=environment,
        cache_setup={"mode": "EMPTY"},
        study_deadline_monotonic=wall_deadline,
    )
    recovered = recover_attempt(attempt_dir, frozen_identity)
    row = _qualification_phase_row(candidate, batch, phase, details,
                                   recovered=recovered, execution=execution)
    atomic_json(phase_record, row)
    return row


def _run_qualification(candidate: Mapping[str, Any], batch: int, *, config: Mapping[str, Any],
                       root: Path, gpu_uuid: str, wall_deadline: float,
                       poll_seconds: float, repo: Path) -> dict[str, Any]:
    from .qualification import aggregate_qualification

    reference = _run_qualification_phase(
        candidate, batch, "s0-reference", config=config, root=root, gpu_uuid=gpu_uuid,
        wall_deadline=wall_deadline, poll_seconds=poll_seconds, repo=repo,
    )
    replicate = None
    correctness = None
    within_process_aa = reference.get("s0_within_process_aa") or {}
    if reference.get("arm_execution") == "COMPLETE" and within_process_aa.get("passed") is True:
        replicate = _run_qualification_phase(
            candidate, batch, "s0-replicate", config=config, root=root, gpu_uuid=gpu_uuid,
            wall_deadline=wall_deadline, poll_seconds=poll_seconds, repo=repo,
        )
        if replicate.get("eligibility") == "S0_AA_PASS":
            correctness = _run_qualification_phase(
                candidate, batch, "s1-correctness", config=config, root=root,
                gpu_uuid=gpu_uuid, wall_deadline=wall_deadline,
                poll_seconds=poll_seconds, repo=repo,
            )
    row = aggregate_qualification(
        candidate, batch, s0_reference=reference,
        s0_replicate=replicate, s1_correctness=correctness,
    )
    row["phase_receipts"] = {
        "s0_reference": reference.get("supervisor_receipt_sha256"),
        "s0_replicate": replicate.get("supervisor_receipt_sha256") if replicate else None,
        "s1_correctness": correctness.get("supervisor_receipt_sha256") if correctness else None,
    }
    safe = "".join(char if char.isalnum() or char in "-_." else "-"
                   for char in str(candidate["candidate_id"]))
    row_path = root / "qualification" / f"qual-{safe}-B{batch}.json"
    if row_path.is_file():
        prior_row = json.loads(row_path.read_text(encoding="utf-8"))
        if (prior_row.get("candidate_id") != row.get("candidate_id")
                or prior_row.get("batch") != row.get("batch")
                or prior_row.get("family_id") != row.get("family_id")):
            raise ValueError("qualification aggregate identity changed during resume")
    atomic_json(row_path, row)
    return row


def _write_qualification_matrix(universe: list[dict[str, Any]],
                                qualifications: list[dict[str, Any]], root: Path) -> None:
    by_id_batch = {(str(row.get("candidate_id")), int(row.get("batch", -1))): row
                   for row in qualifications}
    rows = []
    for candidate in universe:
        for batch in (1, 8):
            result = by_id_batch.get((str(candidate["candidate_id"]), batch), {})
            rows.append({
                "candidate_id": candidate.get("candidate_id"), "source": candidate.get("source"),
                "model": candidate.get("model"), "family_id": candidate.get("family_id"),
                "registered_availability": candidate.get("availability"), "batch": batch,
                "status": result.get("status", "NOT_RUN"),
                "eligibility": result.get("eligibility", "NOT_RUN"),
                "reason": result.get("reason"),
                "root_cause": result.get("root_cause"),
                "qualification_complete": result.get("qualification_complete"),
                "failure_phase": result.get("failure_phase"),
                "phase_summary": result.get("phase_summary"),
                "phase_receipts": result.get("phase_receipts"),
                "conv2d_count": result.get("conv2d_count"),
                "actual_input_shapes": result.get("actual_input_shapes"),
                "model_summary": result.get("model_summary"),
                "kernel_exposure": result.get("kernel_exposure"),
                "runtime_identity": result.get("runtime_identity"),
                "supervisor_receipt_sha256": result.get("supervisor_receipt_sha256"),
                "performance_timing_recorded": result.get("performance_timing_recorded", False),
            })
    fields = list(rows[0]) if rows else ["candidate_id", "source", "model", "family_id", "batch"]
    _atomic_csv(root / "QUALIFICATION_MATRIX.csv", rows, fields)
    registry = []
    for candidate in universe:
        candidate_rows = [by_id_batch.get((str(candidate["candidate_id"]), batch), {})
                          for batch in (1, 8)]
        registry.append({
            "candidate_id": candidate.get("candidate_id"), "source": candidate.get("source"),
            "model": candidate.get("model"), "family_id": candidate.get("family_id"),
            "registered_availability": candidate.get("availability"),
            "baseline_loads_and_channels_last_correct": all(
                row.get("eligibility") == "ELIGIBLE_CHANNELS_LAST_CORRECTNESS_ONLY"
                for row in candidate_rows
            ),
            "batch_1_eligibility": candidate_rows[0].get("eligibility", "NOT_RUN"),
            "batch_8_eligibility": candidate_rows[1].get("eligibility", "NOT_RUN"),
        })
    _atomic_csv(root / "MODEL_REGISTRY.csv", registry, list(registry[0]) if registry else ["candidate_id"])


def _qualification_rows(universe: list[dict[str, Any]], config: Mapping[str, Any],
                        root: Path, gpu_uuid: str, wall_deadline: float,
                        poll_seconds: float, repo: Path) -> list[dict[str, Any]]:
    rows = []
    _update_resource_progress(
        root, config, pair_id="QUALIFICATION_BASELINE", split="qualification",
        pair_index=0, attempt_storage_delta=0, force_full_scan=True,
    )
    qualification_index = 1
    known = {(str(candidate["candidate_id"]), batch): root / "qualification" /
             f"qual-{''.join(char if char.isalnum() or char in '-_.' else '-' for char in str(candidate['candidate_id']))}-B{batch}.json"
             for candidate in universe for batch in (1, 8)}
    for candidate in universe:
        for batch in (1, 8):
            path = known[(str(candidate["candidate_id"]), batch)]
            safe = "".join(char if char.isalnum() or char in "-_." else "-"
                           for char in str(candidate["candidate_id"]))
            attempt_ids = [
                f"qual-{safe}-B{batch}-{phase_id}"
                for phase_id in ("S0REF", "S0AA", "S1CORR")
            ]
            has_history = path.is_file() or any(
                (root / "raw" / attempt_id).exists()
                or (root / "qualification" / "phases" / f"{attempt_id}.json").is_file()
                for attempt_id in attempt_ids
            )
            if time.monotonic() >= wall_deadline and not has_history:
                break
            row = _run_qualification(
                candidate, batch, config=config, root=root, gpu_uuid=gpu_uuid,
                wall_deadline=wall_deadline, poll_seconds=poll_seconds, repo=repo,
            )
            rows.append(row)
            _write_qualification_matrix(universe, rows, root)
            _update_resource_progress(
                root, config,
                pair_id=f"QUALIFICATION:{candidate['candidate_id']}:B{batch}",
                split="qualification", pair_index=qualification_index,
                attempt_storage_delta=None, force_full_scan=True,
            )
            qualification_index += 1
        if time.monotonic() >= wall_deadline:
            break
    return rows


def _load_or_build_manifest(config: Mapping[str, Any], config_path: Path, repo: Path,
                            root: Path, gpu_uuid: str, environment: Mapping[str, Any],
                            universe: list[dict[str, Any]], qualifications: list[dict[str, Any]],
                            frozen_family_split: Mapping[str, str]) -> dict[str, Any]:
    manifest_path = root / "FROZEN_MANIFEST.json"
    manifest = build_frozen_manifest(
        config=config, source_commit=_git("rev-parse", "HEAD", cwd=repo),
        config_path=config_path, selected_gpu_uuid=gpu_uuid,
        runtime_identity=dict(environment), universe=universe,
        qualifications=qualifications, frozen_family_split=frozen_family_split,
    )
    return write_or_verify_manifest(manifest_path, manifest)


def _freeze_registered_family_split(*, root: Path, config: Mapping[str, Any],
                                    source_commit: str,
                                    universe: list[dict[str, Any]]) -> dict[str, str]:
    family_ids = registered_family_ids(universe)
    assignments = family_split(
        family_ids, seed=int(config["seed"]), split_config=config["split"],
    )
    roster = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "intervention_id": config["intervention_id"],
        "source_commit": source_commit,
        "config_digest": canonical_digest(dict(config)),
        "seed": int(config["seed"]),
        "family_assignment_algorithm": (
            "sorted_unique_registered_family_lineage_ids_then_python_random.Random(seed).shuffle"
        ),
        "registered_candidate_count": len(universe),
        "registered_candidate_roster": [
            {"candidate_id": row["candidate_id"], "family_id": row.get("family_id"),
             "source": row.get("source"), "model": row.get("model"),
             "assigned_split": assignments.get(str(row.get("family_id")))}
            for row in universe
        ],
        "registered_family_ids": family_ids,
        "family_split": assignments,
        "split_family_counts": split_family_counts(assignments),
        "qualification_outcomes_used": False,
        "treatment_timing_used": False,
        "final_opened": False,
    }
    write_or_verify_manifest(root / "FAMILY_SPLIT_FREEZE.json", roster)
    return assignments


def _pair_record(root: Path, pair: Mapping[str, Any], manifest_digest: str,
                 gpu_uuid: str, config: Mapping[str, Any], purpose: str,
                 source_commit: str | None = None) -> dict[str, Any]:
    directory = root / ("shakedown" if purpose == "ENGINEERING_SHAKEDOWN" else "pairs")
    path = directory / f"{pair['pair_id']}.json"
    if purpose in {"PRIMARY", "ENGINEERING_SHAKEDOWN"}:
        row = load_pair_observation(
            planned_pair=pair, manifest_digest=manifest_digest,
            source_commit=str(source_commit or config.get("source_commit", "")), config=config,
            results_root=root, gpu_uuid=gpu_uuid, purpose=purpose,
        )
        return row
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1, "study_id": config["study_id"],
        "intervention_id": config["intervention_id"], "purpose": purpose,
        "manifest_digest": manifest_digest, **dict(pair), "selected_gpu_uuid": gpu_uuid,
        "S0": {"arm_execution": "NOT_STARTED", "scientific_correctness": "UNASSESSED"},
        "S1": {"arm_execution": "NOT_STARTED", "scientific_correctness": "UNASSESSED"},
        "classification": {
            "arm_execution": "NOT_STARTED", "scientific_correctness": "UNASSESSED",
            "paired_comparability": "INCOMPLETE", "policy_evidence": "MISSING",
            "first_failure_stage": "ARM_NOT_STARTED", "root_cause": None,
        },
        "terminal_pair": False,
    }


def _write_primary_pairs(root: Path, manifest: Mapping[str, Any], config: Mapping[str, Any],
                         gpu_uuid: str) -> list[dict[str, Any]]:
    rows = [_pair_record(root, pair, str(manifest["manifest_digest"]), gpu_uuid,
                         config, "PRIMARY", str(manifest.get("source_commit", "")))
            for pair in manifest["planned_pairs"]]
    text = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    atomic_write_text(root / "PAIRS.jsonl", text)
    return rows


def _update_resource_progress(root: Path, config: Mapping[str, Any], *,
                              pair_id: str, split: str, pair_index: int,
                              attempt_storage_delta: int | None,
                              force_full_scan: bool = False) -> dict[str, Any]:
    progress_path = root / "RESOURCE_PROGRESS.json"
    prior = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    interval = int(config.get("resource_accounting", {}).get("full_scan_every_pairs", 16))
    full_scan = (force_full_scan or pair_index % interval == 0
                 or attempt_storage_delta is None
                 or prior.get("last_complete_root_scan_bytes") is None)
    if full_scan:
        usage = cache_usage(root)
        if not usage["complete"]:
            atomic_json(progress_path, {
                "updated_utc": _utc_now(), "status": "UNKNOWN",
                "complete": False, "result_storage": usage,
                "pair_id": pair_id, "split": split,
            })
            raise RuntimeError("RESULT_ROOT_STORAGE_UNKNOWN; refusing further arms without a complete disk-cap measurement")
        measured_bytes = int(usage["bytes"])
        estimate = measured_bytes
        since_scan = 0
    else:
        measured_bytes = int(prior["last_complete_root_scan_bytes"])
        since_scan = int(prior.get("unreconciled_pair_attempt_growth_bytes", 0)) + int(attempt_storage_delta)
        estimate = max(0, measured_bytes + since_scan)
        usage = {
            "bytes": None, "observed_bytes_lower_bound": estimate,
            "complete": False, "status": "PER_PAIR_DELTA_ESTIMATE",
        }
    record = {
        "updated_utc": _utc_now(), "status": "COMPLETE" if full_scan else "ESTIMATED",
        "complete": full_scan, "result_storage": usage,
        "estimated_root_bytes": estimate,
        "last_complete_root_scan_bytes": measured_bytes,
        "unreconciled_pair_attempt_growth_bytes": since_scan,
        "pair_id": pair_id, "split": split,
        "accounting": (
            "full quiescent scans before and after performance-blind qualification batches, "
            "then periodic result-root scans plus per-pair raw-attempt-directory byte deltas"
        ),
    }
    atomic_json(progress_path, record)
    if estimate > int(config["disk_cap_bytes"]):
        atomic_json(root / "DISK_CAP_EXCEEDED.json", {
            "estimated_or_measured_bytes": estimate,
            "measurement_status": record["status"],
            "limit_bytes": int(config["disk_cap_bytes"]),
            "pair_id": pair_id,
        })
        raise RuntimeError("RESULT_ROOT_DISK_CAP_EXCEEDED")
    return record


def _write_shakedown(root: Path, pairs: list[Mapping[str, Any]], manifest: Mapping[str, Any],
                     config: Mapping[str, Any], gpu_uuid: str) -> dict[str, Any]:
    rows = [_pair_record(root, pair, str(manifest["manifest_digest"]), gpu_uuid,
                         config, "ENGINEERING_SHAKEDOWN",
                         str(manifest.get("source_commit", ""))) for pair in pairs]
    successes = sum(
        row.get("classification", {}).get("arm_execution") == "COMPLETE"
        and row.get("classification", {}).get("scientific_correctness") == "PASS"
        and row.get("classification", {}).get("paired_comparability") == "MATCHED"
        for row in rows
    )
    failures = [{
        "pair_id": row["pair_id"],
        "first_failure_stage": row.get("classification", {}).get("first_failure_stage") or "NONE",
        "root_cause": row.get("classification", {}).get("root_cause"),
        "arm_execution": row.get("classification", {}).get("arm_execution"),
        "scientific_correctness": row.get("classification", {}).get("scientific_correctness"),
        "paired_comparability": row.get("classification", {}).get("paired_comparability"),
    } for row in rows]
    closed = sum(row.get("terminal_pair") is True for row in rows)
    denominator = int(config["engineering_shakedown"]["minimum_model_level_pairs"])
    summary = {
        "study_id": config["study_id"], "purpose": "ENGINEERING_SHAKEDOWN",
        "planned_opportunities": len(rows), "required_minimum": denominator,
        "matched_and_correct": successes,
        "closed_opportunities": closed,
        "matched_and_correct_fraction": successes / len(rows) if rows else 0.0,
        "required_fraction": float(config["engineering_shakedown"]["minimum_matched_and_correct_fraction"]),
        "first_failure_stages": failures,
        "all_opportunities_terminal": closed == len(rows),
        "all_first_failure_stages_recorded": all(
            item["first_failure_stage"] != "" for item in failures
        ),
        "passes": (
            len(rows) >= denominator
            and closed == len(rows)
            and all(item["first_failure_stage"] != "" for item in failures)
            and successes / len(rows) >= float(
                config["engineering_shakedown"]["minimum_matched_and_correct_fraction"]
            )
        ),
        "included_in_primary_effects": False,
    }
    atomic_json(root / "ENGINEERING_SHAKEDOWN.json", summary)
    atomic_write_text(root / "SHAKEDOWN.jsonl", "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    ))
    return summary


def _write_power_feasibility(root: Path, manifest: Mapping[str, Any],
                             qualifications: list[Mapping[str, Any]],
                             config: Mapping[str, Any], source_commit: str) -> dict[str, Any]:
    planned_by_split = {
        split: sum(pair.get("split") == split for pair in manifest["planned_pairs"])
        for split in ("development", "calibration", "final")
    }
    eligible = dict(manifest.get("eligible_family_counts_by_split", {}))
    split = config["split"]
    acre = config["acre"]
    from core.acre.experiments import ReplaySequentialCertificate
    from core.sequential_stats import minimum_all_successes

    replay_gate = ReplaySequentialCertificate(minimum_groups=1)
    promotion_families_required = minimum_all_successes(
        replay_gate.p_min, replay_gate.statistical_budget.replay_minimum_delta,
    )
    support_families_required = max(
        int(acre["minimum_calibration_support_families"]),
        promotion_families_required + 1,
    )
    calibration_families_required_for_acre = (
        support_families_required
        + int(acre["minimum_calibration_non_support_families"])
    )
    minimums = {
        "development": int(split["minimum_development_families"]),
        "calibration": int(split["minimum_calibration_families"]),
        "final": int(split["minimum_final_families"]),
    }
    primary_checks = {
        "registered_family_minimum": len(manifest.get("registered_families", []))
        >= int(split["minimum_total_families_for_confirmatory"]),
        **{
            f"eligible_{name}_family_minimum": int(eligible.get(name, 0)) >= minimum
            for name, minimum in minimums.items()
        },
    }
    acre_checks = {
        "calibration_can_supply_declared_support_and_non_support_minima": (
            int(eligible.get("calibration", 0))
            >= int(acre["minimum_calibration_support_families"])
            + int(acre["minimum_calibration_non_support_families"])
        ),
        "calibration_can_clear_canonical_promotion_probability_gate": (
            int(eligible.get("calibration", 0)) >= calibration_families_required_for_acre
        ),
    }
    qualifying_performance_free = all(
        row.get("performance_timing_recorded") is False for row in qualifications
    )
    if not qualifying_performance_free:
        raise ValueError("power feasibility must be frozen before performance-effect timing")
    record = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "intervention_id": config["intervention_id"],
        "source_commit": source_commit,
        "config_digest": canonical_digest(dict(config)),
        "manifest_digest": manifest["manifest_digest"],
        "registered_families": len(manifest.get("registered_families", [])),
        "eligible_families_by_split": eligible,
        "minimum_independent_families_by_split": minimums,
        "planned_primary_pairs_by_split": planned_by_split,
        "planned_primary_pairs_total": sum(planned_by_split.values()),
        "planned_primary_arm_invocations": 2 * sum(planned_by_split.values()),
        "qualification_rows": len(qualifications),
        "qualification_recorded_performance_timing": False,
        "confirmatory_family_count_feasible": all(primary_checks.values()),
        "primary_final_checks": primary_checks,
        "acre_certificate_sample_feasible": all(acre_checks.values()),
        "acre_certificate_checks": acre_checks,
        "canonical_acre_promotion_gate": {
            "p_min": replay_gate.p_min,
            "delta_mix": replay_gate.statistical_budget.replay_minimum_delta,
            "minimum_all_success_promotion_families": promotion_families_required,
            "independent_support_holdout_families": 1,
            "minimum_non_support_families": int(acre["minimum_calibration_non_support_families"]),
            "minimum_calibration_families_for_certificate": calibration_families_required_for_acre,
        },
        "effect_variance_available_before_treatment": False,
        "statistical_power": "NOT_ESTIMABLE_BEFORE_TREATMENT_EFFECTS",
        "preregistered_material_log_benefit": config["effect_contract"]["minimum_material_log_benefit"],
        "target_family_ci_half_width_log_benefit": config["effect_contract"]["target_95pct_family_ci_half_width_log_benefit"],
        "final_outcomes_read": False,
        "frozen_before_engineering_shakedown_and_primary_effects": True,
    }
    return write_or_verify_manifest(root / "POWER_FEASIBILITY.json", record)


def _run_formal_splits(manifest: Mapping[str, Any], *, config: Mapping[str, Any], root: Path,
                       gpu_uuid: str, wall_deadline: float) -> None:
    pair_index = 0
    for split in ("development", "calibration"):
        for pair in manifest["planned_pairs"]:
            if pair["split"] != split:
                continue
            if time.monotonic() >= wall_deadline:
                return
            record = run_pair(
                planned_pair=pair, manifest_digest=str(manifest["manifest_digest"]),
                source_commit=str(manifest["source_commit"]), config=config,
                results_root=root, gpu_uuid=gpu_uuid, wall_deadline=wall_deadline,
                poll_seconds=float(config["gpu_selection"]["scan_interval_seconds"]),
                purpose="PRIMARY",
            )
            _update_resource_progress(
                root, config, pair_id=str(pair["pair_id"]), split=split,
                pair_index=pair_index,
                attempt_storage_delta=record.get("result_storage_delta_bytes_this_invocation"),
            )
            pair_index += 1


def _valid_families(rows: list[Mapping[str, Any]], split: str) -> set[str]:
    return {
        str(row["family_id"]) for row in rows
        if row.get("split") == split and row.get("classification", {}).get("policy_evidence") == "ELIGIBLE"
    }


def _gate(manifest: Mapping[str, Any], rows: list[Mapping[str, Any]], shakedown: Mapping[str, Any],
          config: Mapping[str, Any], dev_cal_terminal: bool,
          policy_freeze: Mapping[str, Any] | None) -> dict[str, Any]:
    dev = _valid_families(rows, "development")
    cal = _valid_families(rows, "calibration")
    registered_split_counts = manifest.get("split_family_counts", {})
    eligible_split_counts = manifest.get("eligible_family_counts_by_split", {})
    minimums = config["split"]
    bridge = (policy_freeze or {}).get("acre_bridge", {})
    support_evidence = bridge.get("support_evidence") or {}
    support_families = set(support_evidence.get("support_families") or ())
    non_support_families = set(support_evidence.get("non_support_families") or ())
    acre_fitted = (
        bridge.get("status") == "FITTED"
        and bridge.get("canonical_support_certificate_created") is True
        and support_evidence.get("certificate_scope_bound") is True
    )
    primary_checks = {
        "registered_family_roster_minimum": len(manifest.get("registered_families", [])) >= int(minimums["minimum_total_families_for_confirmatory"]),
        "eligible_family_universe_minimum": len(manifest.get("eligible_families", [])) >= int(minimums["minimum_total_families_for_confirmatory"]),
        "development_split_minimum": int(eligible_split_counts.get("development", 0)) >= int(minimums["minimum_development_families"]),
        "calibration_split_minimum": int(eligible_split_counts.get("calibration", 0)) >= int(minimums["minimum_calibration_families"]),
        "final_split_minimum": int(eligible_split_counts.get("final", 0)) >= int(minimums["minimum_final_families"]),
        "valid_development_families_minimum": len(dev) >= int(config["acre"]["minimum_development_families"]),
        "valid_calibration_families_minimum": len(cal) >= int(config["acre"]["minimum_calibration_deploy_families"]),
        "development_calibration_closed": dev_cal_terminal,
        "policy_freeze_written_before_final": bool(policy_freeze),
        "engineering_shakedown_passed": bool(shakedown.get("passes")),
    }
    acre_checks = {
        "calibration_support_families_minimum": len(support_families) >= int(config["acre"]["minimum_calibration_support_families"]),
        "calibration_non_support_families_minimum": len(non_support_families) >= int(config["acre"]["minimum_calibration_non_support_families"]),
        "canonical_acre_support_certificate": acre_fitted,
    }
    checks = {**primary_checks, **acre_checks}
    return {
        "status": "QUALIFIED_FOR_FINAL" if all(primary_checks.values()) else "UNDERPOWERED_OR_INCOMPLETE",
        "final_opened": all(primary_checks.values()), "checks": checks,
        "required_primary_final_checks": list(primary_checks),
        "acre_certificate_qualified": all(acre_checks.values()),
        "acre_certificate_required_for_primary_final": False,
        "final_scope": "primary layout effect and frozen comparator scores; ACRE rows remain ACRE_NOT_FITTED without its certificate",
        "registered_family_count": len(manifest.get("registered_families", [])),
        "eligible_families": len(manifest.get("eligible_families", [])),
        "valid_development_families": len(dev), "valid_calibration_families": len(cal),
        "calibration_support_families": len(support_families),
        "calibration_non_support_families": len(non_support_families),
        "acre_fitted_with_scope_bound_certificate": acre_fitted,
        "registered_split_family_counts": dict(registered_split_counts),
        "eligible_family_counts_by_split": dict(eligible_split_counts),
        "reason": None if all(primary_checks.values()) else "one or more preregistered primary final-open gates failed",
    }


def _acre_candidate_engine(root: Path, config: Mapping[str, Any]):
    from core.acre import AcreEngine
    from core.models import RuleSpec, RuleState, identifier_digest

    contract_digest = canonical_digest({
        "intervention_id": config["intervention_id"],
        "effect_contract": dict(config["effect_contract"]),
    })
    store = root / "acre_store"
    registry_path = store / "registry" / "rules.json"
    if registry_path.is_file():
        engine = AcreEngine.from_store(store)
        specs = [spec for spec in engine.rule_specs if spec.rule_id == "CHANNELS_LAST"]
        if len(specs) != 1 or specs[0].intervention.get("contract_digest") != contract_digest:
            raise ValueError("saved ACRE candidate store differs from the frozen intervention")
        return engine, contract_digest

    spec = RuleSpec(
        rule_id="CHANNELS_LAST", version=1, parent=None,
        applicability={"all": []},
        intervention={"layout": "channels_last", "contract_digest": contract_digest,
                      "study_intervention_id": str(config["intervention_id"])},
        expected_mechanism="channels-last memory layout",
        evidence_requirements=["independent paired receipts"],
        scientific_invariants=["numerical correctness"],
        abstain_conditions={}, relations={}, runtime_cost={"tokens": 1.0},
        provenance_policy={"required": True},
    )
    payload = spec.to_dict()
    rule_dir = store / "rules" / identifier_digest(spec.rule_id)
    spec_path = rule_dir / "v0001.json"
    state = RuleState(spec.rule_id, spec.version, status="candidate")
    atomic_json(spec_path, payload)
    atomic_json(rule_dir / "v0001.state.json", state.to_dict())
    registry = {
        "schema_version": 1,
        "rules": [{
            "rule_id": spec.rule_id,
            "path": spec_path.relative_to(store).as_posix(),
            "status": "candidate", "version": spec.version,
            "semantic_spec_digest": canonical_digest(payload),
            "artifact_digest": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        }],
    }
    atomic_json(registry_path, registry)
    return AcreEngine.from_store(store), contract_digest


def _freeze_policies(rows: list[Mapping[str, Any]], manifest: Mapping[str, Any],
                     config: Mapping[str, Any], root: Path, source_commit: str,
                     gpu_uuid: str) -> dict[str, Any]:
    from .effect_dataset import summarize_effects
    from .policy_bridge import _context_features, build_development_candidate, validate_calibration_and_route
    from .policy_eval import freeze_policy_set

    summarized = summarize_effects(rows, seed=int(config["bootstrap_seed"]),
                                   replicates=int(config["bootstrap_repetitions"]))["pairs"]
    development = [row for row in summarized if row.get("split") == "development"]
    calibration = [row for row in summarized if row.get("split") == "calibration"]
    path = root / "POLICY_FREEZE.json"
    if path.is_file():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if (prior.get("source_commit") != source_commit
                or prior.get("manifest_digest") != manifest["manifest_digest"]
                or prior.get("development_pair_ids") != [row["pair_id"] for row in development]
                or prior.get("calibration_pair_ids") != [row["pair_id"] for row in calibration]
                or prior.get("final_outcomes_read") is not False):
            raise ValueError("saved policy freeze differs from the closed development/calibration evidence")
        return prior

    feature_names = (
        "horizon", "batch", "conv2d_module_count", "input_channels",
        "input_height", "input_width",
    )
    runtime = manifest.get("runtime_identity") or {}
    driver = runtime.get("driver") if isinstance(runtime.get("driver"), Mapping) else {}
    runtime_versions = {
        "torch": str(runtime.get("torch_version") or ""),
        "gpu_uuid": str(gpu_uuid), "source_commit": str(source_commit),
        "torch_cuda_version": str(runtime.get("torch_cuda_version") or ""),
        "driver_version": str(driver.get("driver_version") or ""),
    }
    packages = runtime.get("packages")
    if isinstance(packages, Mapping):
        runtime_versions.update({str(key): str(value) for key, value in packages.items()
                                 if value is not None})
    if not all(runtime_versions[key] for key in (
        "torch", "gpu_uuid", "source_commit", "torch_cuda_version", "driver_version",
    )):
        runtime_versions = {}

    bridge_result = None
    candidate = None
    bridge_reason = "frozen runtime identity is incomplete"
    route_decisions: dict[str, dict[str, Any]] = {
        "ORIGINAL_ACRE": {}, "REGIME_AWARE_ACRE": {},
    }
    acre_status = "ACRE_NOT_FITTED"
    if runtime_versions:
        from core.acre.regimes import SupportScope

        engine, contract_digest = _acre_candidate_engine(root, config)
        scope = SupportScope(
            "v4-calibration", {"equals": {"domain": "runtime"}},
            {**runtime_versions, "skill_digest": contract_digest},
        )
        planned_development = next(
            pair for pair in manifest["planned_pairs"] if pair.get("split") == "development"
        )
        seed_pair = development[0] if development else {"context": planned_development["context"]}
        candidate_features = _context_features(seed_pair, feature_names)
        candidate = build_development_candidate(
            engine=engine, rule_id="CHANNELS_LAST", development_pairs=development,
            candidate_features=candidate_features, pre_execution_features=feature_names,
            family_id="channels_last", support_scope=scope,
            runtime_versions=runtime_versions, hardware={"gpu_uuid": gpu_uuid},
            software={"torch": runtime_versions["torch"]},
            source_id=f"{config['study_id']}:development", timestamp=_utc_now(),
        )
        planned_calibration_ids = [
            str(pair["pair_id"]) for pair in manifest["planned_pairs"]
            if pair.get("split") == "calibration"
        ]
        bridge_result = validate_calibration_and_route(
            engine=engine, candidate=candidate, calibration_pairs=calibration,
            candidate_features=candidate_features, pre_execution_features=feature_names,
            support_scope=scope, runtime_versions=runtime_versions,
            hardware={"gpu_uuid": gpu_uuid},
            software={"torch": runtime_versions["torch"]},
            source_id=f"{config['study_id']}:calibration", timestamp=_utc_now(),
            min_calibration_families=int(config["acre"]["minimum_calibration_deploy_families"]),
            planned_calibration_pair_ids=planned_calibration_ids,
            minimum_support_families=int(config["acre"]["minimum_calibration_support_families"]),
            minimum_non_support_families=int(config["acre"]["minimum_calibration_non_support_families"]),
        )
        acre_status = "FITTED" if bridge_result.status == "FITTED" else "ACRE_NOT_FITTED"
        bridge_reason = bridge_result.reason
        route_engine = bridge_result.route_engine
        rule_version = route_engine.rule_states["CHANNELS_LAST"].version
        for planned in manifest["planned_pairs"]:
            if planned.get("split") != "final":
                continue
            context_id = str(planned["context_id"])
            if context_id in route_decisions["ORIGINAL_ACRE"]:
                continue
            workload = _context_features({"context": planned["context"]}, feature_names)
            route_context = {
                "domain": "runtime", "workload": workload,
                "hardware": {"gpu_uuid": gpu_uuid},
                "software": {"torch": runtime_versions["torch"]},
                "evidence": {"source": "planned_final_preexecution",
                             "rule_versions": {"CHANNELS_LAST": rule_version}},
            }
            original = route_engine.route(
                route_context, candidate_ids=["CHANNELS_LAST"], runtime_versions=runtime_versions,
            )
            regime = route_engine.route(
                route_context, candidate_ids=["CHANNELS_LAST"], runtime_versions=runtime_versions,
                support_regimes={"CHANNELS_LAST": bridge_result.regime if acre_status == "FITTED" else None},
            )
            route_decisions["ORIGINAL_ACRE"][context_id] = {
                "selected_rule_ids": list(original.selected_rule_ids),
                "rejected_reasons": {key: list(value) for key, value in original.rejected_reasons.items()},
            }
            route_decisions["REGIME_AWARE_ACRE"][context_id] = {
                "selected_rule_ids": list(regime.selected_rule_ids),
                "rejected_reasons": {key: list(value) for key, value in regime.rejected_reasons.items()},
            }
    else:
        bridge_reason = "frozen runtime identity is incomplete; canonical route was not attempted"

    bridge_status = {
        "status": acre_status, "intervention_id": config["intervention_id"],
        "selected_gpu_uuid": gpu_uuid,
        "reason": bridge_reason,
        "development_pairs_available": len(development),
        "calibration_pairs_available": len(calibration),
        "development_families": len(candidate.development_families) if candidate else 0,
        "calibration_families": len(bridge_result.calibration_families) if bridge_result else 0,
        "candidate_status": candidate.status if candidate else "NOT_ATTEMPTED",
        "candidate_synthesis_certificate_created": bool(candidate and candidate.candidate_spec),
        "canonical_support_certificate_created": bool(
            bridge_result
            and bridge_result.status == "FITTED"
            and bridge_result.regime.certificate_is_scope_bound()
        ),
        "support_evidence": ({
            "regime": bridge_result.regime.regime.value,
            "action": bridge_result.regime.action,
            "reason_code": bridge_result.regime.reason_code,
            "independent_families": bridge_result.regime.independent_groups,
            "support_families": sorted({
                certificate.case_id for certificate in bridge_result.validation_certificates
                if certificate.kind.value == "replication"
            }),
            "non_support_families": sorted({
                certificate.case_id for certificate in bridge_result.validation_certificates
                if certificate.kind.value == "boundary"
            }),
            "net_benefit_lcb": bridge_result.regime.net_benefit_lcb,
            "harm_ucb": bridge_result.regime.harm_ucb,
            "certificate_scope_bound": bridge_result.regime.certificate_is_scope_bound(),
            "certificate_digest": bridge_result.regime.certificate_digest,
            "validation_certificate_count": len(bridge_result.validation_certificates),
            "governance_errors": [list(item) for item in bridge_result.governance_errors],
        } if bridge_result else None),
        "engine_route_invoked": bridge_result is not None,
        "planned_final_context_count": len(route_decisions["ORIGINAL_ACRE"]),
        "final_outcomes_read": False,
        "interpretation": "not fitted is not a learned abstention and is not efficacy evidence",
    }
    atomic_json(root / "ACRE_STATUS.json", bridge_status)
    frozen = freeze_policy_set(
        development, calibration_pairs=calibration, acre_status=acre_status,
        acre_rule_id="CHANNELS_LAST", acre_route_decisions=route_decisions,
    )
    record = {
        **frozen, "study_id": config["study_id"],
        "intervention_id": config["intervention_id"],
        "source_commit": source_commit,
        "manifest_digest": manifest["manifest_digest"],
        "development_pair_ids": [row["pair_id"] for row in development],
        "calibration_pair_ids": [row["pair_id"] for row in calibration],
        "acre_bridge": bridge_status,
        "final_outcomes_read": False,
        "preexecution_feature_schema": list(feature_names),
        "effect_contract": dict(config["effect_contract"]),
        "frozen_utc": _utc_now(),
    }
    atomic_json(path, record)
    return record


def _shakedown_roster(planned_pairs: list[Mapping[str, Any]], minimum: int) -> list[Mapping[str, Any]]:
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for pair in planned_pairs:
        if pair.get("split") != "development":
            continue
        by_family.setdefault(str(pair["family_id"]), []).append(pair)
    selected = []
    family_order = sorted(by_family)
    depth = 0
    while len(selected) < minimum:
        added = False
        for family in family_order:
            if depth < len(by_family[family]):
                selected.append(by_family[family][depth])
                added = True
                if len(selected) == minimum:
                    break
        if not added:
            break
        depth += 1
    return selected


def _dev_cal_closed(manifest: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> bool:
    by_id = {str(row["pair_id"]): row for row in rows}
    planned = [pair for pair in manifest["planned_pairs"]
               if pair["split"] in {"development", "calibration"}]
    return all(
        pair["pair_id"] in by_id
        and by_id[pair["pair_id"]].get("terminal_pair") is True
        for pair in planned
    )


def run(config_path: Path, results_root: Path, *, resume_safe: bool) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = read_config(config_path)
    root = _validated_results_root(results_root)
    repo, source_commit = _assert_server_contract(config, root)
    expected_config_path = (repo / "experiments" / "real_intervention_v4" / "config" /
                            "PRIMARY_LAYOUT_STUDY.json").resolve()
    if config_path != expected_config_path:
        raise ValueError(f"v4 primary runs must use the committed config at {expected_config_path}")
    split_freeze_digest = _validate_split_freeze(config, repo)
    if root.name != str(config["study_id"]):
        raise ValueError("results-root basename must equal the frozen study_id")
    root.mkdir(parents=True, exist_ok=True)
    config_digest = canonical_digest(config)
    start_path = root / "RUN_START.json"
    if start_path.is_file() and not resume_safe:
        raise ValueError("an existing v4 result root requires --resume-safe")
    if not start_path.is_file() and any(path.name != ".run.lock" for path in root.iterdir()):
        raise ValueError("a non-empty v4 result root without RUN_START identity cannot be adopted")
    from fcntl import LOCK_EX, LOCK_NB, flock

    lock = (root / ".run.lock").open("a+")
    try:
        flock(lock.fileno(), LOCK_EX | LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("another v4 runner already owns this result root") from exc

    try:
        if start_path.is_file():
            start = json.loads(start_path.read_text(encoding="utf-8"))
            if (start.get("study_id") != config["study_id"]
                    or start.get("config_digest") != config_digest
                    or start.get("source_commit") != source_commit
                    or start.get("split_freeze_digest") != split_freeze_digest):
                raise ValueError("resume source/config identity drift; use a new v4 study identity")
        else:
            start = {
                "study_id": config["study_id"], "started_unix": time.time(),
                "started_utc": _utc_now(), "source_commit": source_commit,
                "config_digest": config_digest, "split_freeze_digest": split_freeze_digest,
                "resume_safe": bool(resume_safe),
            }
            atomic_json(start_path, start)
        v3_summary = _write_v3_audit(config, root)
        atomic_json(root / "V3_AUDIT_SUMMARY.json", v3_summary)
        if not _write_cpu_suite(root, source_commit=source_commit, config_digest=config_digest):
            atomic_json(root / "FATAL.json", {
                "stage": "PRE_EXPERIMENT_CPU_FAULT_INJECTION",
                "reason": "v4 CPU/fault-injection suite failed or exceeded its limit",
                "source_commit": source_commit,
            })
            completion = {
                "status": "PLATFORM_BLOCKED", "stage": "PRE_EXPERIMENT_CPU_FAULT_INJECTION",
                "source_commit": source_commit, "study_id": config["study_id"],
                "final_opened": False, "ended_utc": _utc_now(), "exit_code": 1,
            }
            atomic_json(root / "RUN_COMPLETE.json", completion)
            atomic_json(root / "RUN_EXIT.json", {
                "exit_code": completion["exit_code"], "status": completion["status"],
            })
            return completion

        wall_deadline = time.monotonic() + max(
            0.0, float(config["wall_cap_seconds"]) - (time.time() - float(start["started_unix"])),
        )
        identity_path = root / "RUN_IDENTITY.json"
        if identity_path.is_file():
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if (identity.get("source_commit") != source_commit
                    or identity.get("config_digest") != config_digest
                    or identity.get("split_freeze_digest") != split_freeze_digest):
                raise ValueError("v4 run identity drift; do not reuse this result root")
            gpu_uuid = str(identity["selected_gpu_uuid"])
        else:
            preferred = tuple(str(item) for item in config.get("gpu_selection", {}).get("preferred_uuids", []))
            gpu_uuid = select_study_gpu(
                root, preferred_uuids=preferred, deadline=wall_deadline,
                poll_seconds=float(config["gpu_selection"]["scan_interval_seconds"]),
            )
            if not gpu_uuid:
                blocked = {
                    "stage": "GPU_SELECTION", "reason": "no idle physical GPU became available",
                    "wall_cap_seconds": config["wall_cap_seconds"],
                }
                atomic_json(root / "PLATFORM_BLOCKED.json", blocked)
                completion = {
                    **blocked, "status": "PLATFORM_BLOCKED", "source_commit": source_commit,
                    "study_id": config["study_id"], "final_opened": False,
                    "ended_utc": _utc_now(), "exit_code": 2,
                }
                atomic_json(root / "RUN_COMPLETE.json", completion)
                atomic_json(root / "RUN_EXIT.json", {"exit_code": completion["exit_code"], "status": completion["status"]})
                return completion
            identity = _source_environment(source_commit, config, root)
            identity.update({
                "study_id": config["study_id"], "selected_gpu_uuid": gpu_uuid,
                "config_file_sha256": file_digest(config_path),
                "split_freeze_digest": split_freeze_digest,
            })
            identity["source_environment_digest"] = canonical_digest(identity)
            atomic_json(identity_path, identity)
            atomic_json(root / "GPU_SELECTION.json", {
                "selected_gpu_uuid": gpu_uuid,
                "all_gpu_scans": "gpu_scans/*.json",
                "selection_rule": "first idle physical UUID from full inventory; fixed for this study",
            })

        config_copy = root / "PRIMARY_LAYOUT_STUDY.json"
        if config_copy.is_file():
            if json.loads(config_copy.read_text(encoding="utf-8")) != config:
                raise ValueError("saved primary configuration differs from this run")
        else:
            atomic_json(config_copy, config)

        from .source_registry import candidate_universe
        universe = candidate_universe(
            resolve_server_path(str(config["torchbench_root"])),
            source_revisions={
                "repository_commit": source_commit,
                "torchbench_commit": str(config["torchbench_commit"]),
            },
            supplemental_sources=dict(config.get("supplemental_sources", {})),
        )
        atomic_json(root / "SOURCE_UNIVERSE.json", {
            "source_commit": source_commit, "torchbench_commit": config["torchbench_commit"],
            "candidate_count": len(universe), "candidates": universe,
        })
        frozen_family_split = _freeze_registered_family_split(
            root=root, config=config, source_commit=source_commit, universe=universe,
        )
        qualifications = _qualification_rows(
            universe, config, root, gpu_uuid, wall_deadline,
            float(config["gpu_selection"]["scan_interval_seconds"]), repo,
        )
        _write_qualification_matrix(universe, qualifications, root)
        expected_qualifications = {
            (str(candidate["candidate_id"]), int(batch))
            for candidate in universe for batch in config["batches"]
        }
        observed_qualifications = {
            (str(row.get("candidate_id")), int(row.get("batch", -1)))
            for row in qualifications
        }
        missing_qualifications = sorted(expected_qualifications - observed_qualifications)
        incomplete_qualifications = [
            {"candidate_id": row.get("candidate_id"), "batch": row.get("batch"),
             "eligibility": row.get("eligibility"), "arm_execution": row.get("arm_execution"),
             "first_failure_stage": row.get("first_failure_stage")}
            for row in qualifications if row.get("qualification_complete") is not True
        ]
        if missing_qualifications or incomplete_qualifications:
            atomic_json(root / "QUALIFICATION_INCOMPLETE.json", {
                "status": "PLATFORM_BLOCKED_BEFORE_TIMING",
                "missing_candidate_batch_rows": missing_qualifications,
                "incomplete_candidate_batch_rows": incomplete_qualifications,
                "reason": "performance-blind qualification did not reach a terminal result for every registered candidate and batch",
                "timing_started": False,
            })
            completion = {
                "status": "PLATFORM_BLOCKED", "stage": "PERFORMANCE_BLIND_QUALIFICATION",
                "source_commit": source_commit, "study_id": config["study_id"],
                "final_opened": False, "ended_utc": _utc_now(), "exit_code": 2,
            }
            atomic_json(root / "RUN_COMPLETE.json", completion)
            atomic_json(root / "RUN_EXIT.json", {
                "exit_code": completion["exit_code"], "status": "PLATFORM_BLOCKED",
            })
            return completion
        qualification_runtime = next((row.get("runtime_identity") for row in qualifications
                                      if row.get("runtime_identity")), {})
        if isinstance(qualification_runtime, Mapping):
            qualification_runtime = {key: value for key, value in qualification_runtime.items()
                                     if key not in {"pid", "thread_settings"}}
        manifest = _load_or_build_manifest(
            config, config_path, repo, root, gpu_uuid,
            qualification_runtime if isinstance(qualification_runtime, Mapping) else {},
            universe, qualifications, frozen_family_split,
        )
        _write_power_feasibility(root, manifest, qualifications, config, source_commit)

        shakedown_pairs = _shakedown_roster(
            list(manifest["planned_pairs"]),
            int(config["engineering_shakedown"]["minimum_model_level_pairs"]),
        )
        shakedown_roster = {
            "study_id": config["study_id"], "manifest_digest": manifest["manifest_digest"],
            "purpose": "ENGINEERING_SHAKEDOWN", "source_split": "development",
            "pair_ids": [str(pair["pair_id"]) for pair in shakedown_pairs],
            "family_ids": sorted({str(pair["family_id"]) for pair in shakedown_pairs}),
            "included_in_primary_effects": False,
        }
        roster_path = root / "SHAKEDOWN_ROSTER.json"
        if roster_path.is_file() and json.loads(roster_path.read_text(encoding="utf-8")) != shakedown_roster:
            raise ValueError("engineering shakedown roster changed after it was frozen")
        atomic_json(roster_path, shakedown_roster)
        _update_resource_progress(
            root, config, pair_id="PRE_TIMING_BASELINE", split="PRE_TIMING",
            pair_index=0, attempt_storage_delta=0, force_full_scan=True,
        )
        if len(shakedown_pairs) < int(config["engineering_shakedown"]["minimum_model_level_pairs"]):
            shakedown = {
                "passes": False, "planned_opportunities": len(shakedown_pairs),
                "required_minimum": config["engineering_shakedown"]["minimum_model_level_pairs"],
                "reason": "qualified cohort cannot provide the frozen 12-pair shakedown",
            }
            atomic_json(root / "ENGINEERING_SHAKEDOWN.json", shakedown)
        else:
            for pair_index, pair in enumerate(shakedown_pairs):
                if time.monotonic() >= wall_deadline:
                    break
                record = run_pair(
                    planned_pair=pair, manifest_digest=str(manifest["manifest_digest"]),
                    source_commit=source_commit, config=config, results_root=root,
                    gpu_uuid=gpu_uuid, wall_deadline=wall_deadline,
                    poll_seconds=float(config["gpu_selection"]["scan_interval_seconds"]),
                    purpose="ENGINEERING_SHAKEDOWN",
                )
                _update_resource_progress(
                    root, config, pair_id=str(pair["pair_id"]), split="ENGINEERING_SHAKEDOWN",
                    pair_index=pair_index,
                    attempt_storage_delta=record.get("result_storage_delta_bytes_this_invocation"),
                )
            shakedown = _write_shakedown(root, shakedown_pairs, manifest, config, gpu_uuid)
        if not shakedown.get("passes"):
            completion = {
                "status": "ENGINEERING_SHakedown_FAILED", "final_opened": False,
                "source_commit": source_commit, "manifest_digest": manifest["manifest_digest"],
                "reason": "engineering acceptance gate failed; formal efficacy timing was not started",
                "ended_utc": _utc_now(), "exit_code": 3,
            }
            from .generate_report import write_final_outputs
            write_final_outputs(
                root=root, config=config, manifest=manifest, universe=universe,
                qualifications=qualifications,
                gate={"status": "ENGINEERING_SHAKEDOWN_FAILED", "final_opened": False,
                      "checks": {"engineering_shakedown_passed": False}},
                shakedown=shakedown, source_commit=source_commit,
                selected_gpu_uuid=gpu_uuid,
            )
            atomic_json(root / "RUN_COMPLETE.json", completion)
            atomic_json(root / "RUN_EXIT.json", {"exit_code": completion["exit_code"], "status": completion["status"]})
            return completion

        _run_formal_splits(manifest, config=config, root=root, gpu_uuid=gpu_uuid,
                           wall_deadline=wall_deadline)
        primary_rows = _write_primary_pairs(root, manifest, config, gpu_uuid)
        dev_cal_terminal = _dev_cal_closed(manifest, primary_rows)
        policy_freeze = (_freeze_policies(
            primary_rows, manifest, config, root, source_commit, gpu_uuid=gpu_uuid,
        )
                         if dev_cal_terminal else None)
        gate = _gate(manifest, primary_rows, shakedown, config,
                     dev_cal_terminal, policy_freeze)
        atomic_json(root / "FINAL_GATE.json", gate)
        if gate["final_opened"]:
            atomic_json(root / "FINAL_OPENED.json", {
                "source_commit": source_commit, "manifest_digest": manifest["manifest_digest"],
                "policy_freeze_must_precede_final": True, "opened_utc": _utc_now(),
            })
            final_pair_index = 0
            for pair in manifest["planned_pairs"]:
                if pair["split"] != "final" or time.monotonic() >= wall_deadline:
                    continue
                record = run_pair(
                    planned_pair=pair, manifest_digest=str(manifest["manifest_digest"]),
                    source_commit=source_commit, config=config, results_root=root,
                    gpu_uuid=gpu_uuid, wall_deadline=wall_deadline,
                    poll_seconds=float(config["gpu_selection"]["scan_interval_seconds"]),
                    purpose="PRIMARY",
                )
                _update_resource_progress(
                    root, config, pair_id=str(pair["pair_id"]), split="final",
                    pair_index=final_pair_index,
                    attempt_storage_delta=record.get("result_storage_delta_bytes_this_invocation"),
                )
                final_pair_index += 1
            _write_primary_pairs(root, manifest, config, gpu_uuid)

        from .generate_report import write_final_outputs
        closeout = write_final_outputs(
            root=root, config=config, manifest=manifest, universe=universe,
            qualifications=qualifications, gate=gate, shakedown=shakedown,
            source_commit=source_commit, selected_gpu_uuid=gpu_uuid,
        )
        if closeout.get("confirmatory_analysis_complete"):
            status = "FINAL_ANALYSIS_COMPLETE"
        elif closeout.get("closeout_complete") and not gate["final_opened"]:
            status = "UNDERPOWERED_EXPLORATORY_CLOSEOUT"
        else:
            pair_rows = _write_primary_pairs(root, manifest, config, gpu_uuid)
            unknown = any(
                row.get("classification", {}).get("arm_execution") == "UNKNOWN"
                for row in pair_rows
            )
            status = "INCOMPLETE_TERMINAL_UNKNOWN" if unknown else "INCOMPLETE_WALL_CAP"
        completion = {
            "status": status, "source_commit": source_commit,
            "study_id": config["study_id"], "selected_gpu_uuid": gpu_uuid,
            "manifest_digest": manifest["manifest_digest"],
            "final_opened": bool(gate["final_opened"]),
            "closeout": closeout, "ended_utc": _utc_now(),
            "exit_code": 0 if closeout.get("closeout_complete") else 1,
        }
        atomic_json(root / "RUN_COMPLETE.json", completion)
        atomic_json(root / "RUN_EXIT.json", {
            "exit_code": completion["exit_code"], "status": status,
        })
        return completion
    finally:
        lock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("experiments/real_intervention_v4/config/PRIMARY_LAYOUT_STUDY.json"))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--resume-safe", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.config, args.results_root, resume_safe=args.resume_safe)
    except BaseException as exc:
        try:
            root = _validated_results_root(args.results_root)
            expected_repo = (Path.home() / "CODE" / "scientific-performance-engineering").resolve()
            if (sys.platform.startswith("linux") and _repo_root() == expected_repo
                    and Path.cwd().resolve() == expected_repo):
                root.mkdir(parents=True, exist_ok=True)
                atomic_json(root / "FATAL.json", {
                    "stage": "RUN_ALL_EXCEPTION", "exception_type": type(exc).__name__,
                    "message": str(exc), "traceback": __import__("traceback").format_exc(),
                    "recorded_utc": _utc_now(),
                })
                atomic_json(root / "RUN_COMPLETE.json", {
                    "status": "FATAL", "stage": "RUN_ALL_EXCEPTION",
                    "exception_type": type(exc).__name__, "ended_utc": _utc_now(),
                    "final_opened": (root / "FINAL_OPENED.json").is_file(),
                })
                atomic_json(root / "RUN_EXIT.json", {"exit_code": 1, "status": "FATAL"})
        except Exception:
            pass
        raise
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(int(result.get("exit_code", 0)))


if __name__ == "__main__":
    main()
