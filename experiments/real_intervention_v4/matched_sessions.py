"""Run fixed-UUID, randomized eager/channels-last pairs with durable receipts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

from .cache_lifecycle import cache_usage
from .exposure_monitor import choose_idle_gpu, compare_arm_exposure, scan_all_gpus
from .outcomes import atomic_json, canonical_digest, classify_pair, recover_attempt
from .process_supervisor import _prelaunch_artifacts_allowed, execute_arm


def _save_scan(root: Path, scan: list[dict[str, Any]]) -> None:
    folder = root / "gpu_scans"
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / f"{time.time_ns()}.json", {
        "sample_unix": time.time(), "all_gpus": scan,
    })


def select_study_gpu(results_root: Path, preferred_uuids: tuple[str, ...] = (), *,
                     deadline: float, poll_seconds: float) -> str | None:
    while time.monotonic() < deadline:
        scan = scan_all_gpus()
        _save_scan(results_root, scan)
        selected = choose_idle_gpu(scan, preferred_uuids=preferred_uuids)
        if selected is not None:
            return str(selected["uuid"])
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    return None


def wait_for_selected_gpu(results_root: Path, gpu_uuid: str, *, deadline: float,
                          poll_seconds: float, idle_memory_limit_mib: int) -> bool:
    """Rescan every device before each worker; never switch UUID mid-study."""
    while time.monotonic() < deadline:
        scan = scan_all_gpus()
        _save_scan(results_root, scan)
        row = next((item for item in scan if str(item["uuid"]) == gpu_uuid), None)
        if row is None:
            raise RuntimeError(f"selected physical GPU disappeared from inventory: {gpu_uuid}")
        try:
            memory_used = float(str(row["memory_used_mb"]).replace("MiB", "").strip())
            utilization = float(str(row["utilization_percent"]).replace("%", "").strip())
        except (TypeError, ValueError):
            memory_used, utilization = float("inf"), float("inf")
        if (not row.get("compute_processes") and memory_used <= idle_memory_limit_mib
                and utilization == 0):
            return True
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    return False


def _qualification_for(pair: Mapping[str, Any]) -> dict[str, Any]:
    context = pair["context"]
    value = context.get("qualification")
    if not isinstance(value, Mapping):
        by_batch = context.get("qualification_by_batch")
        value = by_batch.get(str(context["batch"])) if isinstance(by_batch, Mapping) else None
    if not isinstance(value, Mapping):
        raise ValueError("frozen context lacks its baseline-only model qualification")
    return dict(value)


def _arm_identity(*, pair: Mapping[str, Any], arm: str, source_commit: str,
                  config_digest: str, gpu_uuid: str) -> dict[str, Any]:
    context_digest = canonical_digest(pair["context"])
    return {
        "study_id": pair["study_id"], "pair_id": pair["pair_id"], "arm": arm,
        "intervention_id": pair["intervention_id"], "source_commit": source_commit,
        "config_digest": config_digest, "context_digest": context_digest,
        "family_id": pair["family_id"], "split": pair["split"],
        "gpu_uuid": gpu_uuid,
    }


def _attempt_directory_bytes(path: Path) -> int | None:
    if not path.exists():
        return 0
    usage = cache_usage(path)
    return usage["bytes"] if usage["complete"] else None


def _run_arm(*, pair: Mapping[str, Any], arm: str, results_root: Path,
             source_commit: str, config_digest: str, gpu_uuid: str,
             config: Mapping[str, Any], wall_deadline: float) -> dict[str, Any]:
    if time.monotonic() >= wall_deadline:
        return {"arm_execution": "NOT_STARTED", "scientific_correctness": "UNASSESSED",
                "first_failure_stage": "STUDY_WALL_CAP"}
    attempt_id = f"{pair['attempt_prefix']}-{arm}"
    attempt_dir = results_root / "raw" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=True)
    context = dict(pair["context"])
    qualification = _qualification_for(pair)
    cache_root = attempt_dir / "cache"
    expected_cache_environment = {
        "TORCHINDUCTOR_CACHE_DIR": str((cache_root / "inductor").resolve()),
        "TRITON_CACHE_DIR": str((cache_root / "triton").resolve()),
        "CUDA_CACHE_PATH": str((cache_root / "cuda").resolve()),
        "TMPDIR": str((cache_root / "tmp").resolve()),
        "XDG_CACHE_HOME": str((cache_root / "xdg").resolve()),
    }
    spec = {
        **context,
        "study_id": pair["study_id"], "pair_id": pair["pair_id"],
        "intervention_id": pair["intervention_id"], "arm": arm,
        "gpu_uuid": gpu_uuid, "source_commit": source_commit,
        "config_digest": config_digest,
        "context_digest": canonical_digest(context),
        "eligible_channels_last": qualification.get("eligibility") == "ELIGIBLE_CHANNELS_LAST_CORRECTNESS_ONLY",
        "qualification": qualification,
        "cache_scope_path": str(cache_root.resolve()),
        "expected_weights_sha256": context.get("expected_weights_sha256"),
        "expected_inputs_sha256": context.get("expected_inputs_sha256"),
        "reference_path": context.get("reference_path"),
        "reference_sha256": context.get("reference_sha256"),
        "correctness": dict(config["correctness"]),
    }
    spec_path = attempt_dir / "spec.json"
    output_path = attempt_dir / "result.json"
    command = [sys.executable, "-m", "experiments.real_intervention_v4.worker",
               "--spec", str(spec_path), "--output", str(output_path)]
    identity = _arm_identity(pair=pair, arm=arm, source_commit=source_commit,
                              config_digest=config_digest, gpu_uuid=gpu_uuid)
    if time.monotonic() >= wall_deadline:
        return {"arm_execution": "NOT_STARTED", "scientific_correctness": "UNASSESSED",
                "first_failure_stage": "STUDY_WALL_CAP", "identity": identity}
    result = execute_arm(
        results_root=results_root, attempt_id=attempt_id, command=command,
        spec=spec, identity=identity, gpu_uuid=gpu_uuid,
        timeout_seconds=float(config["arm_timeout_seconds"]),
        cleanup_grace_seconds=float(config["cleanup_grace_seconds"]),
        term_grace_seconds=float(config["term_grace_seconds"]),
        heartbeat_seconds=float(config["watchdog_heartbeat_seconds"]),
        sample_seconds=float(config["gpu_sample_seconds"]),
        cache_setup={"mode": "EMPTY"},
        study_deadline_monotonic=wall_deadline,
    )
    if result.get("reason") == "attempt cache initialization is in progress":
        return {
            **result, "arm_execution": "UNKNOWN",
            "scientific_correctness": "UNASSESSED",
            "first_failure_stage": "ATTEMPT_CACHE_LOCKED",
            "root_cause": "ANOTHER_PAIR_RUN_OWNS_THIS_ATTEMPT",
        }
    recovered = recover_attempt(attempt_dir, {
        **identity, "attempt_id": attempt_id, "spec_digest": canonical_digest(spec),
    })
    if recovered.get("receipt_verified") is True:
        result["raw_receipt_sha256"] = recovered["raw_receipt_sha256"]
        result["receipt_verified"] = True
        result["events"] = recovered["events"]
    elif recovered.get("arm_execution") == "UNKNOWN":
        return {**result, **recovered}
    if time.monotonic() > wall_deadline and result.get("arm_execution") == "NOT_STARTED":
        return {"arm_execution": "NOT_STARTED", "scientific_correctness": "UNASSESSED",
                "first_failure_stage": "STUDY_WALL_CAP", "identity": result.get("identity")}
    return result


def load_pair_observation(*, planned_pair: Mapping[str, Any], manifest_digest: str,
                          source_commit: str, config: Mapping[str, Any],
                          results_root: Path, gpu_uuid: str,
                          purpose: str = "PRIMARY") -> dict[str, Any]:
    """Rebuild one pair from frozen plan and raw lifecycle receipts, without launching work."""
    pair_id = str(planned_pair["pair_id"])
    if Path(pair_id).name != pair_id:
        raise ValueError("pair_id must be a path-safe manifest identifier")
    pair_root = results_root / ("shakedown" if purpose == "ENGINEERING_SHAKEDOWN" else "pairs")
    record_path = pair_root / f"{pair_id}.json"
    planned_digest = canonical_digest({
        "manifest_digest": manifest_digest, "pair": dict(planned_pair),
        "purpose": purpose, "selected_gpu_uuid": gpu_uuid,
    })
    prior: dict[str, Any] | None = None
    if record_path.is_file():
        try:
            cached = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            cached = None
        expected_record = {
            "study_id": config["study_id"], "intervention_id": config["intervention_id"],
            "purpose": purpose, "manifest_digest": manifest_digest,
            "planned_digest": planned_digest, "selected_gpu_uuid": gpu_uuid,
            "pair_id": pair_id, "family_id": planned_pair.get("family_id"),
            "split": planned_pair.get("split"),
        }
        if (isinstance(cached, dict)
                and all(cached.get(key) == value for key, value in expected_record.items())):
            prior = cached

    pair = {
        **dict(planned_pair), "study_id": config["study_id"],
        "intervention_id": config["intervention_id"],
        "attempt_prefix": f"{purpose.lower()}-{pair_id}",
    }
    arm_results: dict[str, dict[str, Any]] = {}
    for arm in ("S0", "S1"):
        attempt_id = f"{pair['attempt_prefix']}-{arm}"
        attempt_dir = results_root / "raw" / attempt_id
        if not attempt_dir.exists():
            verified = {"arm_execution": "NOT_STARTED", "scientific_correctness": "UNASSESSED"}
        else:
            spec_path = attempt_dir / "spec.json"
            event_paths = list((attempt_dir / "events").glob("*.json")) if (attempt_dir / "events").is_dir() else []
            artifacts_exist = any(attempt_dir.iterdir())
            if not spec_path.is_file():
                cache_only_before_intent = (
                    not event_paths and _prelaunch_artifacts_allowed(
                        attempt_dir,
                        {"cache_scope_path": str((attempt_dir / "cache").resolve())},
                    )
                )
                if cache_only_before_intent or not artifacts_exist:
                    verified = {"arm_execution": "NOT_STARTED",
                                "scientific_correctness": "UNASSESSED"}
                else:
                    verified = {
                        "arm_execution": "UNKNOWN",
                        "scientific_correctness": "UNKNOWN",
                        "first_failure_stage": "ATTEMPT_SPEC_MISSING",
                        "root_cause": "ATTEMPT_ARTIFACTS_WITHOUT_FROZEN_SPEC",
                    }
            else:
                verified = None
                try:
                    spec = json.loads(spec_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    verified = {
                        "arm_execution": "UNKNOWN",
                        "scientific_correctness": "UNKNOWN",
                        "first_failure_stage": "ATTEMPT_SPEC_CORRUPT",
                        "root_cause": f"FROZEN_ATTEMPT_SPEC_UNREADABLE:{type(exc).__name__}",
                    }
                    spec = None
                if not isinstance(spec, dict):
                    verified = verified or {
                        "arm_execution": "UNKNOWN",
                        "scientific_correctness": "UNKNOWN",
                        "first_failure_stage": "ATTEMPT_SPEC_INVALID",
                        "root_cause": "FROZEN_ATTEMPT_SPEC_NOT_OBJECT",
                    }
                else:
                    identity = _arm_identity(
                        pair=pair, arm=arm, source_commit=source_commit,
                        config_digest=canonical_digest(config), gpu_uuid=gpu_uuid,
                    )
                    verified = recover_attempt(attempt_dir, {
                        **identity, "attempt_id": attempt_id,
                        "spec_digest": canonical_digest(spec),
                    })
                    if verified.get("arm_execution") == "NOT_STARTED" and artifacts_exist:
                        verified = {
                            **verified, "arm_execution": "UNKNOWN",
                            "scientific_correctness": "UNKNOWN",
                            "first_failure_stage": "ATTEMPT_ARTIFACTS_WITHOUT_INTENT",
                            "root_cause": "PRELAUNCH_ATTEMPT_CANNOT_BE_REPLAYED",
                        }
                    elif verified.get("arm_execution") == "NOT_STARTED" and event_paths:
                        verified = {
                            "arm_execution": "UNKNOWN", "scientific_correctness": "UNKNOWN",
                            "first_failure_stage": "ATTEMPT_EVENTS_UNREADABLE",
                            "root_cause": "LIFECYCLE_EVENTS_WERE_NOT_RECOVERED",
                        }
        arm_results[arm] = verified

    exposure = compare_arm_exposure(arm_results["S0"], arm_results["S1"])
    mechanism = None
    worker = arm_results["S1"].get("worker_result")
    if isinstance(worker, Mapping):
        mechanism = bool((worker.get("mechanism_exposure") or {}).get("active"))
    classification = classify_pair(
        arm_results["S0"], arm_results["S1"], intervention_applicable=True,
        exposure_match=exposure.get("matched"), expected_gpu_uuid=gpu_uuid,
        mechanism_active=mechanism,
    ).to_dict()
    terminal = all(arm_results[name].get("arm_execution") in {"COMPLETE", "FAILED", "TIMEOUT"}
                   for name in ("S0", "S1"))
    observation = {
        "schema_version": 1, "study_id": config["study_id"],
        "intervention_id": config["intervention_id"], "purpose": purpose,
        "manifest_digest": manifest_digest, "planned_digest": planned_digest,
        **dict(planned_pair), "selected_gpu_uuid": gpu_uuid,
        "S0": arm_results["S0"], "S1": arm_results["S1"],
        "exposure": exposure, "classification": classification,
        "terminal_pair": terminal,
    }
    if prior is not None:
        # Pair JSON is a derived cache; arm outcomes and validity are rebuilt from raw evidence.
        for key in ("session_wall_seconds", "child_cpu_seconds", "result_storage"):
            if key in prior:
                observation[key] = prior[key]
    atomic_json(record_path, observation)
    return observation


def run_pair(*, planned_pair: Mapping[str, Any], manifest_digest: str,
             source_commit: str, config: Mapping[str, Any], results_root: Path,
             gpu_uuid: str, wall_deadline: float, poll_seconds: float = 15.0,
             purpose: str = "PRIMARY") -> dict[str, Any]:
    """Execute both arms in frozen order; a started UNKNOWN arm is never replayed."""
    if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
        raise ValueError("run_pair requires the frozen physical GPU UUID before either arm starts")
    pair_id = str(planned_pair["pair_id"])
    pair_root = results_root / ("shakedown" if purpose == "ENGINEERING_SHAKEDOWN" else "pairs")
    pair_root.mkdir(parents=True, exist_ok=True)
    record_path = pair_root / f"{pair_id}.json"
    attempt_prefix = f"{purpose.lower()}-{pair_id}"
    pair = {
        **dict(planned_pair), "study_id": config["study_id"],
        "intervention_id": config["intervention_id"], "attempt_prefix": attempt_prefix,
    }
    if not isinstance(planned_pair.get("pair_id"), str) or Path(str(planned_pair["pair_id"])).name != str(planned_pair["pair_id"]):
        raise ValueError("pair_id must be a path-safe manifest identifier")
    planned_digest = canonical_digest({
        "manifest_digest": manifest_digest, "pair": dict(planned_pair),
        "purpose": purpose, "selected_gpu_uuid": gpu_uuid,
    })
    started = time.monotonic()
    attempt_dirs = {
        arm: results_root / "raw" / f"{attempt_prefix}-{arm}"
        for arm in ("S0", "S1")
    }
    storage_before = {arm: _attempt_directory_bytes(path) for arm, path in attempt_dirs.items()}
    arm_results: dict[str, dict[str, Any]] = {}
    for arm in planned_pair["order"]:
        if time.monotonic() >= wall_deadline:
            arm_results[arm] = {"arm_execution": "NOT_STARTED",
                                "scientific_correctness": "UNASSESSED",
                                "first_failure_stage": "STUDY_WALL_CAP"}
            continue
        attempt_id = f"{attempt_prefix}-{arm}"
        attempt_dir = results_root / "raw" / attempt_id
        if attempt_dir.exists():
            identity = _arm_identity(pair=pair, arm=arm, source_commit=source_commit,
                                     config_digest=canonical_digest(config),
                                     gpu_uuid=gpu_uuid)
            spec_path = attempt_dir / "spec.json"
            try:
                spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.is_file() else {}
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                arm_results[arm] = {
                    "arm_execution": "UNKNOWN", "scientific_correctness": "UNKNOWN",
                    "first_failure_stage": "ATTEMPT_SPEC_CORRUPT",
                    "root_cause": f"FROZEN_ATTEMPT_SPEC_UNREADABLE:{type(exc).__name__}",
                }
                break
            if not isinstance(spec, dict):
                arm_results[arm] = {
                    "arm_execution": "UNKNOWN", "scientific_correctness": "UNKNOWN",
                    "first_failure_stage": "ATTEMPT_SPEC_INVALID",
                    "root_cause": "FROZEN_ATTEMPT_SPEC_NOT_OBJECT",
                }
                break
            old = recover_attempt(attempt_dir, {
                **identity, "attempt_id": attempt_id,
                "spec_digest": canonical_digest(spec) if spec else "",
            })
            if old.get("arm_execution") == "UNKNOWN":
                arm_results[arm] = old
                break
            if old.get("arm_execution") != "NOT_STARTED":
                arm_results[arm] = old
                continue
        if not wait_for_selected_gpu(
            results_root, gpu_uuid, deadline=wall_deadline, poll_seconds=poll_seconds,
            idle_memory_limit_mib=int(config["gpu_selection"]["idle_memory_limit_mib"]),
        ):
            arm_results[arm] = {"arm_execution": "NOT_STARTED",
                                "scientific_correctness": "UNASSESSED",
                                "first_failure_stage": "SELECTED_GPU_NOT_IDLE_BEFORE_START"}
            continue
        if time.monotonic() >= wall_deadline:
            arm_results[arm] = {"arm_execution": "NOT_STARTED",
                                "scientific_correctness": "UNASSESSED",
                                "first_failure_stage": "STUDY_WALL_CAP"}
            continue
        arm_results[arm] = _run_arm(
            pair=pair, arm=arm, results_root=results_root,
            source_commit=source_commit, config_digest=str(canonical_digest(config)),
            gpu_uuid=gpu_uuid, config=config, wall_deadline=wall_deadline,
        )
        if arm_results[arm].get("arm_execution") == "UNKNOWN":
            break

    for arm in ("S0", "S1"):
        arm_results.setdefault(arm, {"arm_execution": "NOT_STARTED",
                                     "scientific_correctness": "UNASSESSED"})
    exposure = compare_arm_exposure(arm_results["S0"], arm_results["S1"])
    mechanism = None
    s1_worker = arm_results["S1"].get("worker_result")
    if isinstance(s1_worker, Mapping):
        mechanism = bool((s1_worker.get("mechanism_exposure") or {}).get("active"))
    classification = classify_pair(
        arm_results["S0"], arm_results["S1"],
        intervention_applicable=True,
        exposure_match=exposure.get("matched"), expected_gpu_uuid=gpu_uuid,
        mechanism_active=mechanism,
    ).to_dict()
    terminal = all(arm_results[name].get("arm_execution") in {"COMPLETE", "FAILED", "TIMEOUT"}
                   for name in ("S0", "S1"))
    storage_after = {arm: _attempt_directory_bytes(path) for arm, path in attempt_dirs.items()}
    storage_delta = None
    if all(storage_before[arm] is not None and storage_after[arm] is not None
           for arm in ("S0", "S1")):
        storage_delta = sum(
            int(storage_after[arm]) - int(storage_before[arm])
            for arm in ("S0", "S1")
        )
    record = {
        "schema_version": 1, "study_id": config["study_id"],
        "intervention_id": config["intervention_id"], "purpose": purpose,
        "manifest_digest": manifest_digest, "planned_digest": planned_digest,
        **dict(planned_pair), "selected_gpu_uuid": gpu_uuid,
        "S0": arm_results["S0"], "S1": arm_results["S1"],
        "exposure": exposure, "classification": classification,
        "terminal_pair": terminal,
        "session_wall_seconds": time.monotonic() - started,
        "child_cpu_seconds": sum(
            float((item.get("worker_result") or {}).get("child_cpu_seconds", 0.0))
            for item in arm_results.values()
            if isinstance(item.get("worker_result"), Mapping)
        ),
        "result_storage": {
            "scope": "the two raw attempt directories for this pair",
            "per_arm_attempt_bytes": storage_after,
            "delta_bytes_from_this_invocation": storage_delta,
            "complete": storage_delta is not None,
        },
        "result_storage_delta_bytes_this_invocation": storage_delta,
    }
    atomic_json(record_path, record)
    return record
