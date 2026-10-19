"""Derive a read-only failure taxonomy from the frozen v3 pair and raw files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterator


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_file(roots: list[Path], relative: Path) -> Path | None:
    return next((root / relative for root in roots if (root / relative).is_file()), None)


def _nested_dicts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _nested_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_dicts(child)


def _inherited_attempts(roots: list[Path], pair_id: str, arm_name: str) -> list[dict[str, Any]]:
    matches = []
    for root in roots:
        path = root / "INHERITED_ATTEMPTS.json"
        if not path.is_file():
            continue
        document = _read(path)
        for node in _nested_dicts(document):
            node_pair = str(node.get("pair_id") or "")
            node_arm = str(node.get("arm") or node.get("arm_name") or "")
            if node_pair != pair_id or node_arm != arm_name:
                continue
            matches.append({"path": str(path), "record": node})
    return matches


def _is_reboot_unknown(record: dict[str, Any]) -> bool:
    if str(record.get("state") or "").strip().upper() != "UNKNOWN":
        return False
    interruption = record.get("interruption")
    reason = record.get("interruption_reason") or record.get("reason")
    if isinstance(interruption, dict):
        reason = reason or interruption.get("reason")
    normalized = str(reason or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized == "server_reboot"


def _telemetry(arm: dict[str, Any], study_parent_pids: set[str]) -> dict[str, Any]:
    samples = [arm.get("gpu_telemetry_before"), *arm.get("gpu_telemetry_samples", []),
               arm.get("gpu_telemetry_after")]
    study_owned: set[tuple[str, str]] = set()
    other_account: set[tuple[str, str]] = set()
    same_account_unattributed: set[tuple[str, str]] = set()
    owner_unverified: set[tuple[str, str]] = set()
    utilization: list[float] = []
    process_sample_count = 0
    complete_process_samples = 0
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        snapshot = sample.get("snapshot", sample)
        if not isinstance(snapshot, dict):
            continue
        processes = snapshot.get("compute_processes")
        if isinstance(processes, list):
            complete_process_samples += 1
            process_sample_count += int(bool(processes))
        raw_util = snapshot.get("utilization_percent")
        try:
            if raw_util is not None and str(raw_util).strip() not in {"", "N/A"}:
                utilization.append(float(raw_util))
        except (TypeError, ValueError):
            pass
        for process in processes or []:
            if not isinstance(process, dict):
                continue
            name = str(process.get("name") or "unknown")
            pid = str(process.get("pid") or "unknown")
            entry = (name, pid)
            origin = str(process.get("origin", ""))
            if origin == "STUDY_PROCESS_GROUP" or pid in study_parent_pids:
                study_owned.add(entry)
            elif name.startswith("/home/workspace/mjc/"):
                other_account.add(entry)
            elif name.startswith("/home/workspace/lrh/"):
                # A shared environment path identifies an account, not the
                # process tree that produced this historical measurement.
                same_account_unattributed.add(entry)
            else:
                owner_unverified.add(entry)
    return {
        "study_owned_processes": sorted({f"{name} pid={pid}" for name, pid in study_owned}),
        "other_account_processes": sorted({f"{name} pid={pid}" for name, pid in other_account}),
        "same_account_unattributed_processes": sorted(
            {f"{name} pid={pid}" for name, pid in same_account_unattributed}
        ),
        "owner_unverified_processes": sorted({f"{name} pid={pid}" for name, pid in owner_unverified}),
        "telemetry_sample_count": len([sample for sample in samples if isinstance(sample, dict)]),
        "compute_process_sample_count": complete_process_samples,
        "nonempty_compute_process_sample_count": process_sample_count,
        "gpu_utilization_percent_min": min(utilization) if utilization else None,
        "gpu_utilization_percent_max": max(utilization) if utilization else None,
        "gpu_utilization_samples": len(utilization),
    }


def _cache_identity(arm: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in (arm.get("cache_paths") or {}).items():
        path = Path(value)
        try:
            info = path.stat()
            volume = os.statvfs(path)
        except OSError as exc:
            result[key] = {"path": str(path), "current_stat": None,
                           "current_stat_error": f"{type(exc).__name__}: {exc}",
                           "failure_time_owner_mode": "NOT_RECORDED"}
        else:
            result[key] = {"path": str(path), "current_owner_uid": info.st_uid,
                           "current_mode_octal": oct(stat.S_IMODE(info.st_mode)),
                           "current_is_directory": path.is_dir(),
                           "current_volume_free_bytes": volume.f_bavail * volume.f_frsize,
                           "current_volume_free_inodes": volume.f_favail,
                           "failure_time_owner_mode": "NOT_RECORDED"}
    return result


def _attempt(arm: dict[str, Any], pair_id: str, arm_name: str,
             roots: list[Path]) -> dict[str, Any]:
    relative = Path("raw") / f"{pair_id}-{arm_name}"
    intent_path = _first_file(roots, relative / "intent.json")
    receipt_path = _first_file(roots, relative / "receipt.json")
    intent = _read(intent_path) if intent_path else None
    receipt = _read(receipt_path) if receipt_path else None
    receipt_record = receipt if isinstance(receipt, dict) else {}
    worker = receipt_record or arm
    runtime = worker.get("runtime_identity") or {}
    return {
        "intent_path": str(intent_path) if intent_path else None,
        "intent_started_unix": intent.get("started_unix") if isinstance(intent, dict) else None,
        "intent_parent_pid": intent.get("pid") if isinstance(intent, dict) else None,
        "worker_pid_from_intent": "NOT_RECORDED_IN_V3_INTENT",
        "intent_source": intent.get("source") if isinstance(intent, dict) else None,
        "receipt_path": str(receipt_path) if receipt_path else None,
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest() if receipt_path else None,
        "receipt_file_mtime_unix_not_event_time": receipt_path.stat().st_mtime if receipt_path else None,
        "receipt_returncode": receipt_record.get("returncode"),
        "receipt_timed_out": receipt_record.get("timed_out"),
        "receipt_parent_wall_seconds": receipt_record.get("parent_wall_s"),
        "estimated_parent_end_unix_from_intent_plus_wall": (
            float(intent["started_unix"]) + float(receipt_record["parent_wall_s"])
            if isinstance(intent, dict) and receipt_record.get("parent_wall_s") is not None else None),
        "receipt_worker_pid": "NOT_RECORDED_IN_V3_RECEIPT",
        "receipt_execution_status": receipt_record.get("status"),
        "receipt_error_type": receipt_record.get("error_type"),
        "receipt_error": receipt_record.get("error"),
        "receipt_traceback": receipt_record.get("traceback"),
        "receipt_source": receipt_record.get("source"),
        "receipt_source_location": receipt_record.get("source_location"),
        "receipt_first_failure_stage": receipt_record.get("first_failure_stage"),
        "runtime_gpu_uuid": runtime.get("device_uuid"),
        "inputs_sha256": worker.get("inputs_sha256"),
        "weights_sha256": worker.get("weights_sha256"),
        "scientific_correctness": (receipt_record.get("correctness") or {}).get("passed"),
        "correctness_reason": (receipt_record.get("correctness") or {}).get("reason"),
        "cache_root": worker.get("cache_root"),
        "cache_paths": _cache_identity(worker),
        "telemetry": _telemetry(
            worker,
            {str(intent["pid"])} if isinstance(intent, dict) and intent.get("pid") is not None else set(),
        ),
        "receipt_terminal": receipt_path is not None,
    }


def _fatal_cache_scan(roots: list[Path], pair_id: str,
                      arm_name: str) -> tuple[Path, str] | None:
    attempt_ids = {f"{pair_id}-{arm_name}", f"{pair_id}_{arm_name}"}
    candidates = []
    for root in roots:
        for relative in (Path("prior_execution/FATAL_CAUSE.json"), Path("FATAL_CAUSE.json")):
            path = root / relative
            if path.is_file() and path not in candidates:
                candidates.append(path)
    for path in candidates:
        data = _read(path)
        text = json.dumps(data, sort_keys=True)
        if not all(token in text for token in ("run_state.py", "rglob", "FileNotFoundError")):
            continue
        associated = False
        for node in _nested_dicts(data):
            if (str(node.get("pair_id") or "") == pair_id
                    and str(node.get("arm") or node.get("arm_name") or "") == arm_name):
                associated = True
                break
            if str(node.get("attempt_id") or "") in attempt_ids:
                associated = True
                break
            # v3 FATAL_CAUSE stored the affected cache path rather than the
            # pair id. Attribute it only when that path contains this exact
            # arm directory; a generic traceback is insufficient evidence.
            if any(
                isinstance(value, str)
                and attempt_ids.intersection(Path(value).parts)
                for value in node.values()
            ):
                associated = True
                break
        if associated:
            return path, text
    return None


def _missing_attempt_cause(pair: dict[str, Any], arm_name: str,
                           attempts: dict[str, dict[str, Any]],
                           roots: list[Path]) -> tuple[str, str, str]:
    attempt = attempts[arm_name]
    if attempt["intent_path"] is None:
        return ("UNKNOWN_ROOT_CAUSE", "MISSING_INTENT_AND_TERMINAL_RECEIPT",
                "No raw intent/receipt found in supplied historical roots")

    inherited = _inherited_attempts(roots, str(pair["pair_id"]), arm_name)
    fatal = _fatal_cache_scan(roots, str(pair["pair_id"]), arm_name)
    if inherited and fatal:
        fatal_path, fatal_evidence = fatal
        return ("UNKNOWN_ROOT_CAUSE", "MONITOR_CACHE_SCAN_EXCEPTION",
                f"Observed monitor/cache-scan exception associated with this arm at {fatal_path}: "
                f"{fatal_evidence}; arm-specific inherited record: "
                f"{json.dumps(inherited[0]['record'], sort_keys=True)}. Its causal relation to the "
                "missing worker receipt is not established; terminal outcome remains unknown")

    reboot_record = next((entry for entry in inherited
                          if _is_reboot_unknown(entry["record"])), None)
    if reboot_record:
        return ("SERVER_REBOOT_WITHOUT_TERMINAL_RECEIPT", "INTERRUPTED_ATTEMPT_UNKNOWN",
                f"{reboot_record['path']}: arm-specific inherited record identifies "
                f"{pair['pair_id']} {arm_name} as UNKNOWN after server_reboot: "
                f"{json.dumps(reboot_record['record'], sort_keys=True)}; do not replay")
    return ("UNKNOWN_ROOT_CAUSE", "STARTED_WITHOUT_TERMINAL_RECEIPT",
            f"{attempt['intent_path']}: intent exists but no receipt; do not replay")


def _receipt_correctness(attempts: dict[str, dict[str, Any]]) -> str:
    values = [attempts[name]["scientific_correctness"] for name in ("S0", "S1")]
    if any(value is False for value in values):
        return "FAIL"
    if all(value is True for value in values):
        return "PASS"
    return "UNKNOWN"


def _unresolved_observation(attempts: dict[str, dict[str, Any]]) -> tuple[str, str]:
    arms = [attempts[name] for name in ("S0", "S1")]
    telemetry = [arm["telemetry"] for arm in arms]
    ids = [arm["runtime_gpu_uuid"] for arm in arms]
    inputs = [arm["inputs_sha256"] for arm in arms]
    weights = [arm["weights_sha256"] for arm in arms]
    if (not all(ids) or ids[0] != ids[1]
            or not all(inputs) or inputs[0] != inputs[1]
            or not all(weights) or weights[0] != weights[1]):
        return "UNRESOLVED_TELEMETRY_OR_PAIR_IDENTITY", "Identity mismatch/missing prevents a paired telemetry interpretation"
    if any(item["compute_process_sample_count"] == 0 or
           item["compute_process_sample_count"] != item["telemetry_sample_count"]
           for item in telemetry):
        return "UNRESOLVED_TELEMETRY_OR_PERFORMANCE_CAUSALITY", "At least one arm lacks compute-process telemetry"
    other = [bool(item["other_account_processes"]) for item in telemetry]
    if any(other):
        side = "BOTH_ARMS" if all(other) else "S0_ONLY" if other[0] else "S1_ONLY"
        return f"CO_TENANT_OBSERVED_{side}", (
            "A process path names another account; its performance impact/causality is not identified"
        )
    same_account = [bool(item["same_account_unattributed_processes"]) for item in telemetry]
    if any(same_account):
        side = "BOTH_ARMS" if all(same_account) else "S0_ONLY" if same_account[0] else "S1_ONLY"
        return f"SAME_ACCOUNT_PROCESS_OWNERSHIP_UNKNOWN_{side}", (
            "The process path names the study account, but v3 did not record enough PID-tree evidence "
            "to identify it as this study or a separate job"
        )
    if any(item["owner_unverified_processes"] for item in telemetry):
        return "COMPUTE_PROCESS_OWNER_UNVERIFIED", (
            "A compute process was observed but its account/ownership cannot be established"
        )
    if all(item["study_owned_processes"] for item in telemetry):
        return "STUDY_PROCESS_OBSERVED", (
            "Only a PID explicitly tied to the study or arm process group was observed"
        )
    return "CLASSIFIER_STATUS_NOT_EXPLAINED_BY_RECORDED_TELEMETRY", (
        "Pair was marked unresolved, but available telemetry does not establish the classifier cause"
    )


def derive_failure_table(pair_path: Path, roots: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = [_read_json_line(line) for line in pair_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pair_ids = [str(pair.get("pair_id") or "") for pair in pairs]
    if any(not pair_id.strip() for pair_id in pair_ids):
        raise ValueError("PAIRS.jsonl requires a non-empty pair_id on every row")
    duplicate_ids = sorted({pair_id for pair_id in pair_ids if pair_ids.count(pair_id) > 1})
    if duplicate_ids:
        raise ValueError("PAIRS.jsonl requires unique pair_id values; duplicates: "
                         + ", ".join(duplicate_ids))

    rows: list[dict[str, Any]] = []
    stage_counts: dict[str, int] = {}
    by_original_status: dict[str, int] = {}
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        attempts = {name: _attempt(pair.get(name) or {}, pair_id, name, roots) for name in ("S0", "S1")}
        original = str(pair.get("status", "UNKNOWN"))
        by_original_status[original] = by_original_status.get(original, 0) + 1
        missing = [name for name, attempt in attempts.items() if not attempt["receipt_terminal"]]
        first_stage = "NONE"
        cause = "NONE"
        comparability = "NOT_ASSESSED"
        scientific = "UNASSESSED"
        execution = "UNKNOWN" if missing else "COMPLETE"
        observation_class = "NOT_AN_UNRESOLVED_PAIR"
        performance_causality = "NOT_ASSESSED"
        evidence = ""
        unknown_terminal_arms: list[dict[str, Any]] = []
        if missing:
            missing_causes = []
            for name in missing:
                arm_cause, arm_stage, arm_evidence = _missing_attempt_cause(
                    pair, name, attempts, roots)
                missing_causes.append((arm_cause, arm_stage, arm_evidence))
                unknown_terminal_arms.append({
                    "pair_id": pair_id, "arm": name, "arm_execution": "UNKNOWN",
                    "state": "UNKNOWN_TERMINAL_RECEIPT", "first_invalidity_stage": arm_stage,
                    "derived_root_cause": arm_cause, "evidence": arm_evidence,
                })
            cause, first_stage, evidence = missing_causes[0]
            comparability = "INCOMPLETE"
            policy = "MISSING"
            observation_class = "UNKNOWN_TERMINAL_ARM"
        elif original == "VALID_PAIR":
            scientific = _receipt_correctness(attempts)
            if scientific == "PASS":
                cause = "NO_FAILURE"
                comparability = "MATCHED_AS_RECORDED"
                policy = "MISSING_HISTORICAL_EVIDENCE_NOT_REUSED"
                evidence = "Both terminal receipts explicitly report correctness.passed=true"
            elif scientific == "FAIL":
                cause = "TERMINAL_RECEIPT_REPORTS_CORRECTNESS_FAILURE"
                first_stage = "RECEIPT_CORRECTNESS"
                comparability = "INCOMPLETE"
                policy = "UNSAFE"
                evidence = json.dumps({name: attempts[name]["correctness_reason"]
                                       for name in ("S0", "S1")}, sort_keys=True)
            else:
                cause = "RECEIPT_CORRECTNESS_NOT_ESTABLISHED"
                first_stage = "CORRECTNESS_EVIDENCE"
                comparability = "INCOMPLETE"
                policy = "MISSING"
                evidence = "VALID_PAIR outer status is not numerical correctness evidence"
        elif original == "SESSION_NOISE_UNRESOLVED":
            scientific = _receipt_correctness(attempts)
            observation_class, evidence = _unresolved_observation(attempts)
            if observation_class.startswith("CO_TENANT_OBSERVED"):
                comparability = "OBSERVED_CO_TENANCY_NOT_CAUSAL"
            elif observation_class.startswith("SAME_ACCOUNT_PROCESS_OWNERSHIP_UNKNOWN"):
                comparability = "SAME_ACCOUNT_OWNERSHIP_UNRESOLVED"
            else:
                comparability = "UNRESOLVED"
            performance_causality = "UNRESOLVED"
            cause = observation_class
            first_stage = "PAIR_COMPARABILITY_TELEMETRY"
            policy = "MISSING_HISTORICAL_EVIDENCE_NOT_REUSED"
        elif (original.startswith("INTERVENTION_")
              and attempts["S1"]["receipt_execution_status"] == "error"):
            execution = "FAILED"
            scientific = "UNASSESSED"
            comparability = "INCOMPLETE"
            s1_attempt = attempts["S1"]
            error_type = str(s1_attempt["receipt_error_type"] or "")
            diagnostic = "\n".join(str(s1_attempt.get(key) or "") for key in (
                "receipt_error", "receipt_traceback", "receipt_source",
                "receipt_source_location", "receipt_first_failure_stage"))
            diagnostic = f"{error_type}\n{diagnostic}"
            cache_context = any(token in diagnostic.lower() for token in ("cache", "triton", "inductor"))
            if "PermissionError" in diagnostic and cache_context:
                first_stage = "S1_TRITON_CACHE_DIRECTORY_CREATE"
                cause = "TRITON_CACHE_PERMISSION_DENIED"
            elif "FileNotFoundError" in diagnostic and cache_context:
                first_stage = "S1_TRITON_CACHE_TEMPORARY_ARTIFACT"
                cause = "TRITON_CACHE_PATH_DISAPPEARED_DURING_COMPILATION"
            else:
                first_stage = "S1_EXECUTION"
                cause = f"UNCLASSIFIED_INTERVENTION_ERROR:{error_type}"
            policy = "MISSING"
            evidence = json.dumps({key: s1_attempt.get(key) for key in (
                "receipt_path", "receipt_error_type", "receipt_error", "receipt_traceback",
                "receipt_source", "receipt_source_location", "receipt_first_failure_stage")},
                sort_keys=True)
        elif original == "BASELINE_INVALID":
            s0_attempt = attempts["S0"]
            if s0_attempt["receipt_timed_out"] is True:
                execution = "FAILED"
                scientific = "UNASSESSED"
                comparability = "INCOMPLETE"
                first_stage = "S0_EXECUTION_TIMEOUT"
                cause = "S0_RECEIPT_CONFIRMS_TIMEOUT"
                policy = "MISSING"
                evidence = json.dumps({key: s0_attempt.get(key) for key in (
                    "receipt_path", "receipt_timed_out", "receipt_parent_wall_seconds",
                    "receipt_returncode", "receipt_execution_status")}, sort_keys=True)
            else:
                scientific = _receipt_correctness(attempts)
                first_stage = "UNCONFIRMED_OUTER_STATUS"
                cause = "BASELINE_INVALIDITY_NOT_CONFIRMED_BY_TIMEOUT_RECEIPT"
                comparability = "INCOMPLETE"
                policy = "MISSING"
                evidence = json.dumps({key: s0_attempt.get(key) for key in (
                    "receipt_path", "receipt_timed_out", "receipt_parent_wall_seconds",
                    "receipt_returncode", "receipt_execution_status")}, sort_keys=True)
        elif original == "INTERVENTION_NUMERIC_MISMATCH":
            scientific = _receipt_correctness(attempts)
            s1_attempt = attempts["S1"]
            if s1_attempt["scientific_correctness"] is False:
                comparability = "MATCHED_RUNTIME_ONLY"
                first_stage = "S1_NUMERICAL_COMPARISON"
                cause = str(s1_attempt["correctness_reason"] or "RECEIPT_CONFIRMED_NUMERICAL_MISMATCH")
                policy = "UNSAFE"
                evidence = json.dumps({key: s1_attempt.get(key) for key in (
                    "receipt_path", "scientific_correctness", "correctness_reason")}, sort_keys=True)
            else:
                comparability = "UNRESOLVED"
                first_stage = "NUMERICAL_COMPARISON_EVIDENCE"
                cause = "NUMERIC_MISMATCH_NOT_CONFIRMED_BY_RECEIPT"
                policy = "MISSING"
                evidence = json.dumps({key: s1_attempt.get(key) for key in (
                    "receipt_path", "scientific_correctness", "correctness_reason")}, sort_keys=True)
        else:
            first_stage = "UNCLASSIFIED_STATUS"
            cause = "UNKNOWN_ROOT_CAUSE"
            policy = "MISSING"
            evidence = "No inference from status alone"
        stage_counts[first_stage] = stage_counts.get(first_stage, 0) + 1
        rows.append({
            "pair_id": pair_id, "source": pair.get("source"), "model": pair.get("model"),
            "family": pair.get("family"), "split": pair.get("split"),
            "context_id": pair.get("context_id"), "horizon": pair.get("horizon"),
            "session": pair.get("session"), "arm_order": json.dumps(pair.get("order")),
            "manifest_gpu_uuid": pair.get("runtime_gpu_uuid"), "original_v3_status": original,
            "missing_terminal_arms": json.dumps(missing),
            "unknown_terminal_arms": json.dumps(unknown_terminal_arms, sort_keys=True),
            "arm_execution": execution, "scientific_correctness": scientific,
            "paired_comparability": comparability, "policy_evidence": policy,
            "observation_class": observation_class,
            "performance_causality": performance_causality,
            "first_invalidity_stage": first_stage, "derived_root_cause": cause,
            "evidence_reference": evidence,
            "S0_record_json": json.dumps(attempts["S0"], sort_keys=True),
            "S1_record_json": json.dumps(attempts["S1"], sort_keys=True),
        })
    unresolved = [row for row in rows if row["original_v3_status"] == "SESSION_NOISE_UNRESOLVED"]
    unresolved_counts: dict[str, int] = {}
    for row in unresolved:
        category = row["observation_class"]
        unresolved_counts[category] = unresolved_counts.get(category, 0) + 1
    five_class = {
        "valid": by_original_status.get("VALID_PAIR", 0),
        "numeric_mismatch": by_original_status.get("INTERVENTION_NUMERIC_MISMATCH", 0),
        "infrastructure_error": sum(count for status, count in by_original_status.items()
                                     if status.startswith("INTERVENTION_") and status != "INTERVENTION_NUMERIC_MISMATCH"),
        "baseline_invalid": by_original_status.get("BASELINE_INVALID", 0),
        "unresolved": by_original_status.get("SESSION_NOISE_UNRESOLVED", 0),
    }
    expected_five_class = {"valid": 19, "numeric_mismatch": 11, "infrastructure_error": 6,
                           "baseline_invalid": 1, "unresolved": 53}
    terminal_arms = sum(
        attempt["receipt_terminal"]
        for row in rows
        for attempt in (json.loads(row["S0_record_json"]), json.loads(row["S1_record_json"]))
    )
    unknown_arms = [arm for row in rows for arm in json.loads(row["unknown_terminal_arms"])]
    pairs_with_two_terminal_receipts = sum(
        not json.loads(row["missing_terminal_arms"]) for row in rows)
    expected_receipts = {"terminal_arms": 178, "unknown_arms": 2,
                         "pairs_with_two_terminal_receipts": 88}
    observed_receipts = {"terminal_arms": terminal_arms, "unknown_arms": len(unknown_arms),
                         "pairs_with_two_terminal_receipts": pairs_with_two_terminal_receipts}
    receipt_closure = {**observed_receipts, "expected": expected_receipts,
                       "matches_expected": observed_receipts == expected_receipts}
    pair_id_closure = {"observed_rows": len(rows), "unique_pair_ids": len(set(pair_ids)),
                       "expected_rows": 90,
                       "matches_expected": len(rows) == 90 and len(set(pair_ids)) == 90}
    pair_first_invalidity = {
        row["pair_id"]: {"stage": row["first_invalidity_stage"],
                         "cause": row["derived_root_cause"],
                         "evidence": row["evidence_reference"]}
        for row in rows
    }
    summary = {
        "total_pairs": len(rows),
        "pair_id_closure": pair_id_closure,
        "original_status_counts": by_original_status,
        "first_invalidity_stage_counts": stage_counts,
        "closure_check": {"observed": len(rows), "expected": 90,
                          "matches_expected": pair_id_closure["matches_expected"]
                          and five_class == expected_five_class
                          and receipt_closure["matches_expected"]},
        "raw_terminal_receipt_counts": receipt_closure,
        "unknown_arm_state_counts": {"UNKNOWN_TERMINAL_RECEIPT": len(unknown_arms)},
        "unknown_arm_states": unknown_arms,
        "five_class_closure": {**five_class, "sum": sum(five_class.values()),
                               "expected": expected_five_class,
                               "matches_expected": five_class == expected_five_class},
        "unresolved_observation_counts": unresolved_counts,
        "pair_first_invalidity": pair_first_invalidity,
        "unresolved_performance_causality": "NOT_IDENTIFIED_FROM_CO_TENANCY_OBSERVATION",
        "historical_statuses_modified": False,
        "v3_effects_recomputed_or_promoted": False,
    }
    return rows, summary


def _read_json_line(line: str) -> dict[str, Any]:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("PAIRS.jsonl must contain object rows")
    return value


def write_failure_table(pair_path: Path, roots: list[Path], output_root: Path) -> dict[str, Any]:
    rows, summary = derive_failure_table(pair_path, roots)
    if not rows:
        raise ValueError("PAIRS.jsonl contains no pair rows")
    output_root.mkdir(parents=True, exist_ok=True)
    table_path = output_root / "UNRESOLVED_CAUSE_TABLE.csv"
    with table_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "FIRST_INVALIDITY_STAGE.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, action="append", required=True,
                        help="Historical result root, repeated in priority order")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = write_failure_table(args.pairs, args.search_root, args.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["closure_check"]["matches_expected"]:
        raise SystemExit("v3 failure taxonomy did not close exactly 90 pairs")


if __name__ == "__main__":
    main()
