import json

from experiments.real_intervention_v5.certificate_feasibility import (
    feasibility,
    independent_group_count,
)
from experiments.real_intervention_v5.old_studies_readonly_audit import audit_v4


def test_all_success_e_value_crosses_mixture_threshold_at_36():
    result = feasibility(36, 36)
    bound = result["canonical_certificate_lower_bound"]
    p_min = bound["p_min"]
    n = bound["minimum_all_success_independent_support_groups"]

    assert n == 36
    assert bound["all_success_mixture_e_value_at_n_minus_1"] == 1 / (
        n * p_min ** (n - 1)
    )
    assert bound["all_success_mixture_e_value_at_n"] == 1 / (
        (n + 1) * p_min**n
    )
    assert bound["mixture_e_value_threshold"] == 1 / bound["delta_mix"]
    assert bound["all_success_mixture_e_value_at_n_minus_1"] < bound[
        "mixture_e_value_threshold"
    ]
    assert bound["all_success_mixture_e_value_at_n"] >= bound[
        "mixture_e_value_threshold"
    ]
    assert bound["all_success_crosses_mixture_threshold_at_n"] is True
    assert result["acre_canonical_certificate_fitted"] is False


def test_ten_36_and_42_groups_are_necessary_bounds_not_a_fitted_certificate():
    ten = feasibility(10, 10)
    thirty_six = feasibility(36, 36)
    forty_two = feasibility(42, 42)

    assert ten["acre_canonical_certificate_feasible"] is False
    assert thirty_six["acre_canonical_certificate_feasible"] is False
    assert forty_two["required_independent_groups_lower_bound"] == 42
    assert forty_two["acre_canonical_certificate_feasible"] is True
    assert forty_two["acre_canonical_certificate_feasibility_is_necessary_only"] is True
    assert forty_two["acre_canonical_certificate_fitted"] is False


def test_repeated_sessions_do_not_add_independent_families():
    rows = [
        {"family_id": f"family-{family}", "session": session}
        for family in range(42)
        for session in range(100)
    ]

    assert independent_group_count(rows) == 42


def test_v4_audit_reads_actual_frozen_root_filenames_and_reports_missing_sources(tmp_path):
    from experiments.real_intervention_v5.old_studies_readonly_audit import audit_v4

    (tmp_path / "METRICS.json").write_text(
        json.dumps({"cohort": {}, "by_split": {}}), encoding="utf-8"
    )
    (tmp_path / "PAIRS.jsonl").write_text("", encoding="utf-8")

    result = audit_v4(tmp_path)

    assert result["read_only"] is True
    assert result["counts"]["registered_families"]["status"] == "not_verified"
    assert "manifest" in result["errors"]


def _started_arm_fixture(root, pair_id, arm_name, *, inputs_sha256="inputs-digest"):
    attempt_id = f"primary-{pair_id}-{arm_name}"
    identity = {
        "pair_id": pair_id,
        "arm": arm_name,
        "attempt_id": attempt_id,
        "source_commit": "commit-a",
        "gpu_uuid": "GPU-a",
    }
    worker = {
        "arm_execution": "COMPLETE",
        "scientific_correctness": "PASS",
        "weights_sha256": "weights-digest",
        "inputs_sha256": inputs_sha256,
        "reference_sha256": "reference-digest",
        "identity": identity,
    }
    events = [
        {"kind": "intent", "identity": identity},
        {"kind": "started", "identity": identity},
        {"kind": "heartbeat", "identity": identity},
        {
            "kind": "terminal", "identity": identity, "arm_execution": "COMPLETE",
            "process_returncode": 0, "sequence": 7, "receipt_sha256": "receipt-digest",
        },
    ]
    raw_dir = root / "raw" / attempt_id
    (raw_dir / "events").mkdir(parents=True)
    for index, event in enumerate(events):
        sequence = 7 if event["kind"] == "terminal" else index
        (raw_dir / "events" / f"{sequence:04d}-{event['kind']}.json").write_text(
            json.dumps(event), encoding="utf-8"
        )
    receipt = {
        "identity": identity,
        "source_commit": "commit-a",
        "runtime_gpu_uuid": "GPU-a",
        "worker_result": worker,
    }
    (raw_dir / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    (raw_dir / "result.json").write_text(json.dumps(worker), encoding="utf-8")
    return {
        "identity": identity,
        "source_commit": "commit-a",
        "runtime_gpu_uuid": "GPU-a",
        "events": events,
        "worker_result": worker,
        **{field: worker[field] for field in (
            "arm_execution", "scientific_correctness", "weights_sha256",
            "inputs_sha256", "reference_sha256",
        )},
    }


def test_started_arm_raw_receipts_match_events_and_worker_digests_without_relabeling(tmp_path):
    pair_id = "pair-a"
    pair = {
        "pair_id": pair_id,
        "selected_gpu_uuid": "GPU-a",
        "classification": {"arm_execution": "COMPLETE", "paired_comparability": "MATCHED"},
        "effect_status": "unknown",
    }
    pair["S0"] = _started_arm_fixture(tmp_path, pair_id, "S0")
    pair["S1"] = _started_arm_fixture(tmp_path, pair_id, "S1")
    pairs_path = tmp_path / "PAIRS.jsonl"
    original = json.dumps(pair) + "\n"
    pairs_path.write_text(original, encoding="utf-8")

    result = audit_v4(tmp_path)

    verification = result["raw_receipt_verification"]
    assert verification["status"] == "verified"
    assert verification["started_arms"] == verification["verified_arms"] == 2
    assert verification["summary"]["not_verified_arms"] == 0
    assert verification["source_commit_counts"] == {"commit-a": 2}
    assert verification["gpu_uuid_counts"] == {"GPU-a": 2}
    assert result["counts"]["started_pairs_from_rows"]["value"] == 1
    assert result["counts"]["closed_pairs_from_rows"]["value"] == 1
    assert result["counts"]["valid_pairs_from_rows"]["value"] == 1
    assert result["denominator_checks"]["pair_outcome_categories_close_to_pair_rows"] is True
    assert pairs_path.read_text(encoding="utf-8") == original


def test_started_arm_missing_raw_terminal_is_not_verified(tmp_path):
    pair_id = "pair-a"
    pair = {
        "pair_id": pair_id,
        "selected_gpu_uuid": "GPU-a",
        "classification": {"arm_execution": "COMPLETE", "paired_comparability": "MATCHED"},
        "effect_status": "unknown",
    }
    pair["S0"] = _started_arm_fixture(tmp_path, pair_id, "S0")
    (tmp_path / "PAIRS.jsonl").write_text(json.dumps(pair) + "\n", encoding="utf-8")
    (tmp_path / "raw" / f"primary-{pair_id}-S0" / "events" / "0007-terminal.json").unlink()

    result = audit_v4(tmp_path)

    arm = result["raw_receipt_verification"]["arms"][0]
    assert arm["status"] == "not_verified"
    assert arm["checks"]["raw_terminal_event_exists"] is False
    assert pair["effect_status"] == "unknown"


def test_started_arm_worker_digest_mismatch_is_not_verified(tmp_path):
    pair_id = "pair-a"
    arm = _started_arm_fixture(tmp_path, pair_id, "S0")
    arm["worker_result"]["weights_sha256"] = "different-weights"
    arm["weights_sha256"] = "different-weights"
    (tmp_path / "PAIRS.jsonl").write_text(
        json.dumps({"pair_id": pair_id, "selected_gpu_uuid": "GPU-a", "S0": arm}) + "\n",
        encoding="utf-8",
    )

    result = audit_v4(tmp_path)

    verification = result["raw_receipt_verification"]
    assert verification["status"] == "not_verified"
    assert verification["arms"][0]["worker_field_checks"]["weights_sha256"] is False
