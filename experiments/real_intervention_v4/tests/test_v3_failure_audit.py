from __future__ import annotations

import csv
import json

import pytest

from experiments.real_intervention_v4.audit_v3 import (
    _fatal_cache_scan, derive_failure_table, write_failure_table,
)


def _write_arm(root, pair_id, name, *, status="valid", correctness=True,
               timed_out=False, returncode=0, terminal=True, external=False,
               telemetry=True):
    folder = root / "raw" / f"{pair_id}-{name}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "intent.json").write_text(
        json.dumps({"started_unix": 10.0, "pid": 1}), encoding="utf-8")
    (folder / "spec.json").write_text(json.dumps({"arm": name}), encoding="utf-8")
    if not terminal:
        return None

    processes = [{"name": "/home/workspace/lrh/equivcompiler/bin/python",
                  "pid": "11", "memory_mb": "100"}]
    if external:
        processes.append({"name": "/home/workspace/mjc/python", "pid": "22",
                          "memory_mb": "200"})
    receipt = {
        "status": status, "returncode": returncode, "timed_out": timed_out,
        "runtime_identity": {"device_uuid": "gpu-1"},
        "inputs_sha256": "input", "weights_sha256": "weights",
        "gpu_telemetry_before": ({"gpu_uuid": "gpu-1", "compute_processes": processes,
                                   "utilization_percent": "1"} if telemetry else {}),
        "gpu_telemetry_after": ({"gpu_uuid": "gpu-1", "compute_processes": processes,
                                  "utilization_percent": "2"} if telemetry else {}),
        "gpu_telemetry_samples": [], "cache_paths": {},
    }
    if correctness is not None:
        receipt["correctness"] = {"passed": correctness, "reason": "fixture"}
    if status == "error":
        receipt.update({
            "error_type": "FileNotFoundError",
            "error": "cache artifact disappeared",
            "traceback": "FileNotFoundError while opening Triton cache artifact",
            "source": "torch/_inductor/codecache.py:214",
            "source_location": "Triton cache artifact open",
        })
    (folder / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return receipt


def _pair(pair_id, status, s0, s1, *, model="fixture"):
    return {"pair_id": pair_id, "source": "torchvision", "model": model,
            "family": model, "split": "development", "context_id": f"{model}-H1",
            "horizon": 1, "session": 0, "order": ["S0", "S1"],
            "runtime_gpu_uuid": "gpu-1", "status": status,
            "S0": s0 or {"status": "unknown"}, "S1": s1 or {"status": "unknown"}}


def _write_pair_file(path, pairs):
    path.write_text("".join(json.dumps(pair) + "\n" for pair in pairs), encoding="utf-8")


def test_expected_pair_receipt_and_failure_closures_are_explicit(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    pairs = []

    for index in range(19):
        pair_id = f"valid-{index}"
        s0 = _write_arm(root, pair_id, "S0")
        s1 = _write_arm(root, pair_id, "S1")
        pairs.append(_pair(pair_id, "VALID_PAIR", s0, s1))

    for index in range(11):
        pair_id = f"numeric-{index}"
        s0 = _write_arm(root, pair_id, "S0")
        s1 = _write_arm(root, pair_id, "S1", status="numeric_mismatch", correctness=False)
        pairs.append(_pair(pair_id, "INTERVENTION_NUMERIC_MISMATCH", s0, s1))

    for index in range(6):
        pair_id = f"infra-{index}"
        s0 = _write_arm(root, pair_id, "S0")
        s1 = _write_arm(root, pair_id, "S1", status="error", correctness=None)
        pairs.append(_pair(pair_id, "INTERVENTION_ERROR", s0, s1))

    pair_id = "baseline-timeout"
    s0 = _write_arm(root, pair_id, "S0", status="error", correctness=None,
                    timed_out=True, returncode=-15)
    s1 = _write_arm(root, pair_id, "S1")
    pairs.append(_pair(pair_id, "BASELINE_INVALID", s0, s1))

    unknown_pair_ids = ["unknown-monitor", "unknown-reboot"]
    for index in range(53):
        pair_id = unknown_pair_ids[index] if index < 2 else f"unresolved-{index}"
        s0 = _write_arm(root, pair_id, "S0")
        s1 = _write_arm(root, pair_id, "S1", terminal=index >= 2)
        pairs.append(_pair(pair_id, "SESSION_NOISE_UNRESOLVED", s0, s1))

    inherited = {
        "reason": "server_reboot",  # Global prose must not classify every arm as reboot-unknown.
        "attempts": [
            {"pair_id": unknown_pair_ids[0], "arm": "S1", "state": "UNKNOWN",
             "reason": "monitor_interruption"},
            {"pair_id": unknown_pair_ids[1], "arm": "S1", "state": "UNKNOWN",
             "interruption": {"reason": "server_reboot"}},
        ],
    }
    (root / "INHERITED_ATTEMPTS.json").write_text(json.dumps(inherited), encoding="utf-8")
    affected_cache = f"/RESULTS/caches/{unknown_pair_ids[0]}-S1/triton/temp"
    fatal = {
        "source": "run_state.py:85", "operation": "timeout cache-size scan",
        "exception": "FileNotFoundError", "error": f"missing {affected_cache}",
        "traceback": "Path.rglob traversed a Triton temporary directory; FileNotFoundError",
    }
    (root / "prior_execution").mkdir()
    (root / "prior_execution" / "FATAL_CAUSE.json").write_text(
        json.dumps(fatal), encoding="utf-8")

    pair_path = tmp_path / "PAIRS.jsonl"
    _write_pair_file(pair_path, pairs)
    source_before = pair_path.read_bytes()
    rows, summary = derive_failure_table(pair_path, [root])
    by_id = {row["pair_id"]: row for row in rows}

    assert summary["pair_id_closure"] == {
        "observed_rows": 90, "unique_pair_ids": 90,
        "expected_rows": 90, "matches_expected": True,
    }
    assert summary["five_class_closure"]["valid"] == 19
    assert summary["five_class_closure"]["numeric_mismatch"] == 11
    assert summary["five_class_closure"]["infrastructure_error"] == 6
    assert summary["five_class_closure"]["baseline_invalid"] == 1
    assert summary["five_class_closure"]["unresolved"] == 53
    assert summary["raw_terminal_receipt_counts"] == {
        "terminal_arms": 178, "unknown_arms": 2,
        "pairs_with_two_terminal_receipts": 88,
        "expected": {"terminal_arms": 178, "unknown_arms": 2,
                     "pairs_with_two_terminal_receipts": 88},
        "matches_expected": True,
    }
    assert summary["unknown_arm_state_counts"] == {"UNKNOWN_TERMINAL_RECEIPT": 2}
    assert {(item["pair_id"], item["arm"]) for item in summary["unknown_arm_states"]} == {
        (unknown_pair_ids[0], "S1"), (unknown_pair_ids[1], "S1")}
    assert by_id[unknown_pair_ids[0]]["derived_root_cause"] == "UNKNOWN_ROOT_CAUSE"
    assert by_id[unknown_pair_ids[0]]["first_invalidity_stage"] == "MONITOR_CACHE_SCAN_EXCEPTION"
    assert "causal relation" in by_id[unknown_pair_ids[0]]["evidence_reference"]
    assert by_id[unknown_pair_ids[1]]["derived_root_cause"] == \
        "SERVER_REBOOT_WITHOUT_TERMINAL_RECEIPT"
    assert by_id["numeric-0"]["scientific_correctness"] == "FAIL"
    assert by_id["infra-0"]["first_invalidity_stage"] == "S1_TRITON_CACHE_TEMPORARY_ARTIFACT"
    cache_evidence = json.loads(by_id["infra-0"]["evidence_reference"])
    assert "FileNotFoundError" in cache_evidence["receipt_traceback"]
    assert cache_evidence["receipt_source"] == "torch/_inductor/codecache.py:214"
    assert summary["pair_first_invalidity"][unknown_pair_ids[1]] == {
        "stage": "INTERRUPTED_ATTEMPT_UNKNOWN",
        "cause": "SERVER_REBOOT_WITHOUT_TERMINAL_RECEIPT",
        "evidence": by_id[unknown_pair_ids[1]]["evidence_reference"],
    }
    assert summary["closure_check"]["matches_expected"] is True
    assert pair_path.read_bytes() == source_before

    output_root = tmp_path / "derived"
    written_summary = write_failure_table(pair_path, [root], output_root)
    with (output_root / "UNRESOLVED_CAUSE_TABLE.csv").open(
            encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    saved_summary = json.loads(
        (output_root / "FIRST_INVALIDITY_STAGE.json").read_text(encoding="utf-8"))
    assert len(csv_rows) == 90
    assert written_summary == saved_summary == summary
    assert set(saved_summary["pair_first_invalidity"]) == {pair["pair_id"] for pair in pairs}
    assert pair_path.read_bytes() == source_before


def test_duplicate_pair_ids_are_rejected_before_arm_classification(tmp_path):
    pair = _pair("duplicate", "VALID_PAIR", {}, {})
    pair_path = tmp_path / "PAIRS.jsonl"
    _write_pair_file(pair_path, [pair, pair])

    with pytest.raises(ValueError, match="unique pair_id"):
        derive_failure_table(pair_path, [])


def test_outer_status_does_not_establish_numerical_pass_or_timeout(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    pairs = []

    valid_id = "valid-without-correctness"
    s0 = _write_arm(root, valid_id, "S0", correctness=None)
    s1 = _write_arm(root, valid_id, "S1", correctness=None)
    pairs.append(_pair(valid_id, "VALID_PAIR", s0, s1))

    numeric_id = "numeric-without-correctness"
    s0 = _write_arm(root, numeric_id, "S0")
    s1 = _write_arm(root, numeric_id, "S1", status="valid", correctness=None)
    pairs.append(_pair(numeric_id, "INTERVENTION_NUMERIC_MISMATCH", s0, s1))

    baseline_id = "baseline-not-timed-out"
    s0 = _write_arm(root, baseline_id, "S0", status="error", correctness=None,
                    timed_out=False, returncode=-15)
    s1 = _write_arm(root, baseline_id, "S1")
    pairs.append(_pair(baseline_id, "BASELINE_INVALID", s0, s1))

    pair_path = tmp_path / "PAIRS.jsonl"
    _write_pair_file(pair_path, pairs)
    rows, _ = derive_failure_table(pair_path, [root])
    by_id = {row["pair_id"]: row for row in rows}

    assert by_id[valid_id]["scientific_correctness"] == "UNKNOWN"
    assert by_id[numeric_id]["scientific_correctness"] == "UNKNOWN"
    assert by_id[numeric_id]["derived_root_cause"] == "NUMERIC_MISMATCH_NOT_CONFIRMED_BY_RECEIPT"
    assert by_id[baseline_id]["first_invalidity_stage"] == "UNCONFIRMED_OUTER_STATUS"
    assert by_id[baseline_id]["derived_root_cause"] == \
        "BASELINE_INVALIDITY_NOT_CONFIRMED_BY_TIMEOUT_RECEIPT"


def test_reboot_requires_pair_arm_unknown_record_not_global_text(tmp_path):
    root = tmp_path / "results"
    pair_id = "only-this-arm"
    _write_arm(root, pair_id, "S0")
    _write_arm(root, pair_id, "S1", terminal=False)
    pair_path = tmp_path / "PAIRS.jsonl"
    _write_pair_file(pair_path, [_pair(pair_id, "SESSION_NOISE_UNRESOLVED", {}, None)])
    inherited = {
        "reason": "server_reboot",
        "attempts": [{"pair_id": pair_id, "arm": "S1", "state": "UNKNOWN",
                      "reason": "monitor_interruption"}],
    }
    (root / "INHERITED_ATTEMPTS.json").write_text(json.dumps(inherited), encoding="utf-8")

    rows, _ = derive_failure_table(pair_path, [root])

    assert rows[0]["derived_root_cause"] == "UNKNOWN_ROOT_CAUSE"
    assert rows[0]["first_invalidity_stage"] == "STARTED_WITHOUT_TERMINAL_RECEIPT"


def test_unresolved_telemetry_keeps_ownership_and_causality_distinct(tmp_path):
    root = tmp_path / "results"
    pairs = []

    own_id = "same-account"
    own_s0 = _write_arm(root, own_id, "S0")
    own_s1 = _write_arm(root, own_id, "S1")
    pairs.append(_pair(own_id, "SESSION_NOISE_UNRESOLVED", own_s0, own_s1))

    external_id = "other-account"
    external_s0 = _write_arm(root, external_id, "S0")
    external_s1 = _write_arm(root, external_id, "S1", external=True)
    pairs.append(_pair(external_id, "SESSION_NOISE_UNRESOLVED", external_s0, external_s1))

    telemetry_id = "missing-process-samples"
    telemetry_s0 = _write_arm(root, telemetry_id, "S0", telemetry=False)
    telemetry_s1 = _write_arm(root, telemetry_id, "S1", telemetry=False)
    pairs.append(_pair(telemetry_id, "SESSION_NOISE_UNRESOLVED", telemetry_s0, telemetry_s1))

    pair_path = tmp_path / "PAIRS.jsonl"
    _write_pair_file(pair_path, pairs)
    rows, _ = derive_failure_table(pair_path, [root])
    by_id = {row["pair_id"]: row for row in rows}

    assert by_id[own_id]["observation_class"] == \
        "SAME_ACCOUNT_PROCESS_OWNERSHIP_UNKNOWN_BOTH_ARMS"
    assert by_id[external_id]["observation_class"] == "CO_TENANT_OBSERVED_S1_ONLY"
    assert "causality is not identified" in by_id[external_id]["evidence_reference"]
    assert by_id[telemetry_id]["observation_class"] == \
        "UNRESOLVED_TELEMETRY_OR_PERFORMANCE_CAUSALITY"


def test_unresolved_pair_with_missing_input_or_weight_identity_stays_unresolved(tmp_path):
    root = tmp_path / "results"
    pair_id = "missing-arm-identity"
    s0 = _write_arm(root, pair_id, "S0")
    s1 = _write_arm(root, pair_id, "S1", external=True)
    s1["inputs_sha256"] = None
    s1_path = root / "raw" / f"{pair_id}-S1" / "receipt.json"
    s1_path.write_text(json.dumps(s1), encoding="utf-8")
    pair_path = tmp_path / "PAIRS.jsonl"
    _write_pair_file(pair_path, [_pair(pair_id, "SESSION_NOISE_UNRESOLVED", s0, s1)])

    rows, _ = derive_failure_table(pair_path, [root])

    assert rows[0]["observation_class"] == "UNRESOLVED_TELEMETRY_OR_PAIR_IDENTITY"
    assert rows[0]["paired_comparability"] == "UNRESOLVED"


def test_unresolved_pair_with_missing_weight_identity_stays_unresolved(tmp_path):
    root = tmp_path / "results"
    pair_id = "missing-weight-identity"
    s0 = _write_arm(root, pair_id, "S0")
    s1 = _write_arm(root, pair_id, "S1", external=True)
    s1["weights_sha256"] = None
    (root / "raw" / f"{pair_id}-S1" / "receipt.json").write_text(
        json.dumps(s1), encoding="utf-8")
    pair_path = tmp_path / "PAIRS.jsonl"
    _write_pair_file(pair_path, [_pair(pair_id, "SESSION_NOISE_UNRESOLVED", s0, s1)])

    rows, _ = derive_failure_table(pair_path, [root])

    assert rows[0]["observation_class"] == "UNRESOLVED_TELEMETRY_OR_PAIR_IDENTITY"
    assert rows[0]["paired_comparability"] == "UNRESOLVED"


def test_global_fatal_traceback_is_not_attributed_without_arm_identity(tmp_path):
    root = tmp_path / "results"
    (root / "prior_execution").mkdir(parents=True)
    fatal = {
        "source": "run_state.py:85", "operation": "timeout cache-size scan",
        "exception": "FileNotFoundError",
        "traceback": "Path.rglob traversed a Triton temporary directory; FileNotFoundError",
    }
    (root / "prior_execution" / "FATAL_CAUSE.json").write_text(
        json.dumps(fatal), encoding="utf-8")

    assert _fatal_cache_scan([root], "unrelated-H1-session-0", "S1") is None
