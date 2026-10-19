from __future__ import annotations

import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from experiments.real_intervention_v4.outcomes import (
    ArmExecution,
    Comparability,
    Correctness,
    PolicyEvidence,
    atomic_json,
    classify_pair,
    record_event,
    recover_attempt,
)


def _runtime_identity(*, gpu_uuid="GPU-test", torch_version="2.6.0"):
    return {
        "device_uuid": gpu_uuid,
        "device_name": "RTX-test",
        "device_capability": [8, 9],
        "torch_version": torch_version,
        "torch_cuda_version": "12.6",
        "cudnn_version": 90100,
        "packages": {"torch": torch_version, "torchvision": "0.21.0", "triton": "3.2.0"},
        "driver": {
            "gpu_uuid": gpu_uuid, "driver_version": "535.1", "device_name": "RTX-test",
            "pci_bus_id": "0000:01:00.0", "memory_total_mb": "24576", "power_limit_w": "450",
        },
        "python_version": "3.11.0",
        "pid": 10,
        "thread_settings": {"OMP_NUM_THREADS": "1"},
    }


def _arm(**changes):
    value = {
        "arm_execution": ArmExecution.COMPLETE.value,
        "scientific_correctness": Correctness.PASS.value,
        "runtime_gpu_uuid": "GPU-test",
        "inputs_sha256": "inputs",
        "weights_sha256": "weights",
        "source_commit": "source",
        "config_digest": "config",
        "context_digest": "context",
        "runtime_identity": _runtime_identity(),
        "raw_receipt_sha256": "a" * 64,
        "receipt_verified": True,
    }
    value.update(changes)
    return value


def test_pair_validity_requires_execution_correctness_identity_and_exposure():
    result = classify_pair(_arm(), _arm(), intervention_applicable=True,
                           exposure_match=True, expected_gpu_uuid="GPU-test")
    assert result.arm_execution == ArmExecution.COMPLETE.value
    assert result.scientific_correctness == Correctness.PASS.value
    assert result.paired_comparability == Comparability.MATCHED.value
    assert result.policy_evidence == PolicyEvidence.ELIGIBLE.value

    unknown = classify_pair(_arm(), _arm(), intervention_applicable=True,
                            exposure_match=None, expected_gpu_uuid="GPU-test")
    assert unknown.paired_comparability == Comparability.UNKNOWN.value
    assert unknown.policy_evidence == PolicyEvidence.MISSING.value

    co_tenant_mismatch = classify_pair(_arm(), _arm(), intervention_applicable=True,
                                       exposure_match=False, expected_gpu_uuid="GPU-test")
    assert co_tenant_mismatch.paired_comparability == Comparability.CONFOUNDED.value

    missing_receipt = classify_pair(_arm(raw_receipt_sha256=None), _arm(),
                                    intervention_applicable=True, exposure_match=True,
                                    expected_gpu_uuid="GPU-test")
    assert missing_receipt.policy_evidence == PolicyEvidence.MISSING.value
    assert missing_receipt.first_failure_stage == "PAIR_RECEIPT_INTEGRITY"


def test_failures_are_not_converted_to_candidate_harm_or_zero_effect():
    infra = classify_pair(_arm(arm_execution=ArmExecution.FAILED.value), _arm(),
                          intervention_applicable=True, exposure_match=True)
    assert infra.policy_evidence == PolicyEvidence.MISSING.value
    assert infra.scientific_correctness == Correctness.UNASSESSED.value

    unsafe = classify_pair(_arm(), _arm(scientific_correctness=Correctness.FAIL.value),
                           intervention_applicable=True, exposure_match=True)
    assert unsafe.policy_evidence == PolicyEvidence.UNSAFE.value
    assert unsafe.scientific_correctness == Correctness.FAIL.value


def test_pair_preserves_verified_arm_failure_stage_and_root_cause():
    unknown = classify_pair(
        _arm(),
        _arm(arm_execution=ArmExecution.UNKNOWN.value,
             scientific_correctness=Correctness.UNKNOWN.value,
             first_failure_stage="HOST_SUPERVISOR_EXIT",
             root_cause="SUPERVISOR_EXIT_23_WITHOUT_TERMINAL"),
        intervention_applicable=True, exposure_match=True,
    )
    assert unknown.first_failure_stage == "HOST_SUPERVISOR_EXIT"
    assert unknown.root_cause == "SUPERVISOR_EXIT_23_WITHOUT_TERMINAL"

    timeout = classify_pair(
        _arm(),
        _arm(arm_execution=ArmExecution.TIMEOUT.value,
             first_failure_stage="HOST_SUPERVISOR_HARD_DEADLINE",
             root_cause="SUPERVISOR_DID_NOT_CLOSE_WITHIN_ARM_CEILING"),
        intervention_applicable=True, exposure_match=True,
    )
    assert timeout.first_failure_stage == "HOST_SUPERVISOR_HARD_DEADLINE"
    assert timeout.root_cause == "SUPERVISOR_DID_NOT_CLOSE_WITHIN_ARM_CEILING"


def test_correct_noop_pair_is_not_eligible_intervention_evidence():
    result = classify_pair(_arm(), _arm(), intervention_applicable=True,
                           exposure_match=True, expected_gpu_uuid="GPU-test",
                           mechanism_active=False)

    assert result.arm_execution == ArmExecution.COMPLETE.value
    assert result.scientific_correctness == Correctness.PASS.value
    assert result.paired_comparability == Comparability.MATCHED.value
    assert result.policy_evidence == PolicyEvidence.MISSING.value
    assert result.root_cause == "NO_MECHANISTIC_EXPOSURE"


def test_unrecognized_execution_and_correctness_states_are_missing_not_eligible():
    bad_execution = classify_pair(
        _arm(arm_execution="COMPLETE_WITH_WARNINGS"), _arm(),
        intervention_applicable=True, exposure_match=True,
    )
    assert bad_execution.arm_execution == ArmExecution.UNKNOWN.value
    assert bad_execution.arm_states["S0"]["arm_execution"] == ArmExecution.UNKNOWN.value
    assert bad_execution.policy_evidence == PolicyEvidence.MISSING.value

    bad_correctness = classify_pair(
        _arm(scientific_correctness="PROBABLY_OK"), _arm(),
        intervention_applicable=True, exposure_match=True,
    )
    assert bad_correctness.scientific_correctness == Correctness.UNASSESSED.value
    assert bad_correctness.arm_states["S0"]["scientific_correctness"] == Correctness.UNKNOWN.value
    assert bad_correctness.policy_evidence == PolicyEvidence.MISSING.value


def test_known_correctness_failure_is_not_hidden_by_other_arm_execution_failure():
    result = classify_pair(
        _arm(arm_execution=ArmExecution.FAILED.value),
        _arm(scientific_correctness=Correctness.FAIL.value),
        intervention_applicable=True, exposure_match=True,
    )
    assert result.arm_execution == ArmExecution.FAILED.value
    assert result.scientific_correctness == Correctness.FAIL.value
    assert result.arm_states["S1"]["scientific_correctness"] == Correctness.FAIL.value
    assert result.policy_evidence == PolicyEvidence.UNSAFE.value


def test_s0_correctness_failure_is_baseline_invalid_not_candidate_harm():
    result = classify_pair(
        _arm(scientific_correctness=Correctness.FAIL.value), _arm(),
        intervention_applicable=True, exposure_match=True,
    )

    assert result.scientific_correctness == Correctness.FAIL.value
    assert result.paired_comparability == Comparability.INCOMPLETE.value
    assert result.policy_evidence == PolicyEvidence.MISSING.value
    assert result.first_failure_stage == "BASELINE_SCIENTIFIC_CORRECTNESS"
    assert result.root_cause == "BASELINE_CORRECTNESS_FAILURE"


def test_s1_correctness_failure_remains_candidate_unsafe_when_s0_also_fails():
    result = classify_pair(
        _arm(scientific_correctness=Correctness.FAIL.value),
        _arm(scientific_correctness=Correctness.FAIL.value),
        intervention_applicable=True, exposure_match=True,
    )

    assert result.scientific_correctness == Correctness.FAIL.value
    assert result.policy_evidence == PolicyEvidence.UNSAFE.value
    assert result.root_cause == "KNOWN_CANDIDATE_CORRECTNESS_FAILURE"


@pytest.mark.parametrize("field", ["runtime_gpu_uuid", "source_commit", "config_digest"])
def test_pair_identity_drift_is_confounded(field):
    left = _arm()
    right = _arm(**{field: "different"})
    result = classify_pair(left, right, intervention_applicable=True, exposure_match=True)
    assert result.paired_comparability == Comparability.CONFOUNDED.value
    assert result.policy_evidence == PolicyEvidence.MISSING.value


def test_pair_must_match_the_preselected_physical_gpu():
    result = classify_pair(_arm(), _arm(), intervention_applicable=True,
                           exposure_match=True, expected_gpu_uuid="GPU-other")
    assert result.paired_comparability == Comparability.CONFOUNDED.value
    assert result.root_cause == "RUNTIME_GPU_UUID_DIFFERS_FROM_SELECTED_DEVICE"


def test_pair_requires_identical_observed_runtime_identity():
    left = _arm()
    right = _arm()
    right["runtime_identity"] = _runtime_identity(torch_version="2.7.0")

    result = classify_pair(
        left, right, intervention_applicable=True,
        exposure_match=True, expected_gpu_uuid="GPU-test",
    )

    assert result.paired_comparability == Comparability.CONFOUNDED.value
    assert result.policy_evidence == PolicyEvidence.MISSING.value
    assert result.first_failure_stage == "PAIR_RUNTIME_IDENTITY"
    assert result.root_cause == "CROSS_PROCESS_RUNTIME_IDENTITY_MISMATCH"


def test_pair_missing_runtime_identity_is_unknown_not_matched():
    result = classify_pair(
        _arm(runtime_identity=None), _arm(), intervention_applicable=True,
        exposure_match=True, expected_gpu_uuid="GPU-test",
    )

    assert result.paired_comparability == Comparability.UNKNOWN.value
    assert result.policy_evidence == PolicyEvidence.MISSING.value
    assert result.first_failure_stage == "PAIR_RUNTIME_IDENTITY"
    assert result.root_cause == "REQUIRED_RUNTIME_IDENTITY_MISSING"


def test_pair_without_a_preselected_gpu_uuid_cannot_be_matched():
    result = classify_pair(_arm(), _arm(), intervention_applicable=True,
                           exposure_match=True, expected_gpu_uuid=None)

    assert result.paired_comparability == Comparability.UNKNOWN.value
    assert result.policy_evidence == PolicyEvidence.MISSING.value
    assert result.first_failure_stage == "PAIR_IDENTITY"
    assert result.root_cause == "SELECTED_GPU_UUID_MISSING"


def test_blank_preselected_gpu_uuid_cannot_qualify_a_pair():
    result = classify_pair(_arm(), _arm(), intervention_applicable=True,
                           exposure_match=True, expected_gpu_uuid="  ")

    assert result.paired_comparability == Comparability.UNKNOWN.value
    assert result.policy_evidence == PolicyEvidence.MISSING.value
    assert result.root_cause == "SELECTED_GPU_UUID_MISSING"


def test_digest_shape_without_recovered_receipt_is_not_eligible():
    result = classify_pair(
        _arm(receipt_verified=False), _arm(receipt_verified=False),
        intervention_applicable=True, exposure_match=True, expected_gpu_uuid="GPU-test",
    )

    assert result.arm_execution == ArmExecution.COMPLETE.value
    assert result.scientific_correctness == Correctness.UNKNOWN.value
    assert result.paired_comparability == Comparability.UNKNOWN.value
    assert result.policy_evidence == PolicyEvidence.MISSING.value
    assert result.first_failure_stage == "PAIR_RECEIPT_INTEGRITY"


def test_started_attempt_without_terminal_is_unknown_and_not_recoverable(tmp_path):
    attempt = tmp_path / "attempt"
    identity = {"source_commit": "source", "config_digest": "cfg", "gpu_uuid": "GPU-test"}
    record_event(attempt, "intent", {"identity": identity})
    record_event(attempt, "started", {"identity": identity, "pid": 100})
    recovered = recover_attempt(attempt, identity)
    assert recovered["arm_execution"] == ArmExecution.UNKNOWN.value

    recovered_with_drift = recover_attempt(
        attempt, {**identity, "config_digest": "changed"}
    )
    assert recovered_with_drift["arm_execution"] == ArmExecution.UNKNOWN.value
    assert "identity differs" in recovered_with_drift["reason"]


def test_intermediate_lifecycle_identity_drift_is_not_recovered(tmp_path):
    attempt = tmp_path / "attempt"
    identity = {"source_commit": "source", "config_digest": "cfg", "gpu_uuid": "GPU-test"}
    record_event(attempt, "intent", {"identity": identity})
    record_event(attempt, "started", {"identity": identity, "pid": 100})
    record_event(attempt, "heartbeat", {"identity": identity, "elapsed_seconds": 1.0})

    heartbeat = attempt / "events" / "0002-heartbeat.json"
    event = json.loads(heartbeat.read_text(encoding="utf-8"))
    event["identity"] = {**identity, "gpu_uuid": "GPU-other"}
    heartbeat.write_text(json.dumps(event), encoding="utf-8")

    recovered = recover_attempt(attempt, identity)
    assert recovered["arm_execution"] == ArmExecution.UNKNOWN.value
    assert "identity differs" in recovered["reason"]


@pytest.mark.parametrize("drift_kind", ["intent", "started", "heartbeat", "terminal"])
def test_recovery_validates_identity_on_every_lifecycle_event(tmp_path, drift_kind):
    attempt = tmp_path / drift_kind
    identity = {"source_commit": "source", "config_digest": "cfg", "gpu_uuid": "GPU-test"}
    record_event(attempt, "intent", {"identity": identity})
    if drift_kind in {"started", "heartbeat", "terminal"}:
        record_event(attempt, "started", {"identity": identity, "pid": 100})
    if drift_kind in {"heartbeat", "terminal"}:
        record_event(attempt, "heartbeat", {"identity": identity, "elapsed_seconds": 1.0})
    if drift_kind == "terminal":
        record_event(attempt, "terminal", {"identity": identity, "arm_execution": "FAILED"})

    event_path = attempt / "events" / next(
        path.name for path in (attempt / "events").glob("*.json")
        if path.name.endswith(f"-{drift_kind}.json")
    )
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["identity"] = {**identity, "gpu_uuid": "GPU-other"}
    event_path.write_text(json.dumps(event), encoding="utf-8")

    recovered = recover_attempt(attempt, identity)
    assert recovered["arm_execution"] == ArmExecution.UNKNOWN.value
    assert "identity differs" in recovered["reason"]


def test_concurrent_lifecycle_append_cannot_duplicate_a_start_event(tmp_path):
    attempt = tmp_path / "attempt"
    identity = {"source_commit": "source", "config_digest": "cfg", "gpu_uuid": "GPU-test"}
    record_event(attempt, "intent", {"identity": identity})

    barrier = Barrier(2)

    def append_start():
        barrier.wait()
        try:
            record_event(attempt, "started", {"identity": identity, "pid": 100})
        except ValueError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: append_start(), range(2)))

    assert sum(results) == 1
    starts = list((attempt / "events").glob("*-started.json"))
    assert len(starts) == 1


def test_atomic_json_exclusive_does_not_replace_an_existing_attempt_spec(tmp_path):
    spec_path = tmp_path / "spec.json"
    atomic_json(spec_path, {"source_commit": "first"}, exclusive=True)

    with pytest.raises(FileExistsError):
        atomic_json(spec_path, {"source_commit": "second"}, exclusive=True)

    assert json.loads(spec_path.read_text(encoding="utf-8")) == {"source_commit": "first"}


def test_illegal_state_transition_and_corrupt_event_fail_loudly(tmp_path):
    attempt = tmp_path / "attempt"
    with pytest.raises(ValueError, match="illegal lifecycle transition"):
        record_event(attempt, "started", {"identity": {}})

    event = attempt / "events" / "0000-intent.json"
    event.parent.mkdir(parents=True)
    event.write_text("{broken", encoding="utf-8")
    recovered = recover_attempt(attempt, {})
    assert recovered["arm_execution"] == ArmExecution.UNKNOWN.value
    assert "corrupt lifecycle event" in recovered["reason"]


def test_digest_mismatch_is_not_silently_reused(tmp_path):
    attempt = tmp_path / "attempt"
    identity = {"source_commit": "source", "config_digest": "cfg", "gpu_uuid": "GPU-test"}
    receipt = {"identity": identity, "arm_execution": "COMPLETE"}
    atomic_json(attempt / "receipt.json", receipt)
    import hashlib

    digest = hashlib.sha256((attempt / "receipt.json").read_bytes()).hexdigest()
    record_event(attempt, "intent", {"identity": identity})
    record_event(attempt, "started", {"identity": identity, "pid": 100})
    record_event(attempt, "terminal", {"identity": identity, "receipt_sha256": digest})
    (attempt / "receipt.json").write_text(json.dumps({**receipt, "arm_execution": "FAILED"}), encoding="utf-8")
    recovered = recover_attempt(attempt, identity)
    assert recovered["arm_execution"] == ArmExecution.UNKNOWN.value
    assert recovered["reason"] == "terminal receipt digest mismatch"


def test_terminal_execution_state_must_match_hashed_receipt(tmp_path):
    attempt = tmp_path / "attempt"
    identity = {"source_commit": "source", "config_digest": "cfg", "gpu_uuid": "GPU-test"}
    receipt = {"identity": identity, "arm_execution": "FAILED", "process_returncode": 1}
    atomic_json(attempt / "receipt.json", receipt)
    import hashlib

    digest = hashlib.sha256((attempt / "receipt.json").read_bytes()).hexdigest()
    record_event(attempt, "intent", {"identity": identity})
    record_event(attempt, "started", {"identity": identity, "pid": 100})
    record_event(attempt, "terminal", {"identity": identity, "arm_execution": "COMPLETE",
                                        "process_returncode": 1, "receipt_sha256": digest})
    recovered = recover_attempt(attempt, identity)
    assert recovered["arm_execution"] == ArmExecution.UNKNOWN.value
    assert recovered["reason"] == "terminal event execution status differs from receipt"


def test_complete_receipt_requires_a_digest_bound_worker_result(tmp_path):
    attempt = tmp_path / "attempt"
    identity = {"source_commit": "source", "config_digest": "cfg", "gpu_uuid": "GPU-test"}
    receipt = {
        "identity": identity, "arm_execution": "COMPLETE",
        "scientific_correctness": "PASS", "process_returncode": 0,
    }
    atomic_json(attempt / "receipt.json", receipt)
    digest = hashlib.sha256((attempt / "receipt.json").read_bytes()).hexdigest()
    record_event(attempt, "intent", {"identity": identity})
    record_event(attempt, "started", {"identity": identity, "pid": 100})
    record_event(attempt, "terminal", {
        "identity": identity, "arm_execution": "COMPLETE", "process_returncode": 0,
        "receipt_sha256": digest,
    })

    recovered = recover_attempt(attempt, identity)
    assert recovered["arm_execution"] == ArmExecution.UNKNOWN.value
    assert recovered["reason"] == "complete receipt lacks a clean, digest-bound worker result"


def test_digest_bound_worker_result_must_contain_json_object(tmp_path):
    attempt = tmp_path / "attempt"
    identity = {"source_commit": "source", "config_digest": "cfg", "gpu_uuid": "GPU-test"}
    result_bytes = b"[]"
    (attempt / "result.json").parent.mkdir(parents=True, exist_ok=True)
    (attempt / "result.json").write_bytes(result_bytes)
    receipt = {
        "identity": identity, "arm_execution": "FAILED", "process_returncode": 1,
        "worker_result_sha256": hashlib.sha256(result_bytes).hexdigest(),
    }
    atomic_json(attempt / "receipt.json", receipt)
    digest = hashlib.sha256((attempt / "receipt.json").read_bytes()).hexdigest()
    record_event(attempt, "intent", {"identity": identity})
    record_event(attempt, "started", {"identity": identity, "pid": 100})
    record_event(attempt, "terminal", {
        "identity": identity, "arm_execution": "FAILED", "process_returncode": 1,
        "receipt_sha256": digest,
    })

    recovered = recover_attempt(attempt, identity)
    assert recovered["arm_execution"] == ArmExecution.UNKNOWN.value
    assert recovered["reason"] == "receipt-bound worker result is not a JSON object"


def test_recovery_verifies_worker_result_bytes_and_receipt_summary(tmp_path):
    attempt = tmp_path / "attempt"
    identity = {"source_commit": "source", "config_digest": "cfg", "gpu_uuid": "GPU-test"}
    worker_result = {
        "status": "valid", "arm_execution": "COMPLETE", "scientific_correctness": "PASS",
        "correctness": {"passed": True}, "timing": {"cold_seconds": 1.0},
        "runtime_identity": {"device_uuid": "GPU-test"},
        "inputs_sha256": "inputs", "weights_sha256": "weights",
        "cache_scope_path": "/cache/attempt",
    }
    atomic_json(attempt / "result.json", worker_result)
    result_digest = hashlib.sha256((attempt / "result.json").read_bytes()).hexdigest()
    receipt = {
        "identity": identity, "arm_execution": "COMPLETE", "scientific_correctness": "PASS",
        "process_returncode": 0, "timed_out": False, "orphan_cleanup": False,
        "worker_result": worker_result, "worker_result_sha256": result_digest,
        "correctness": worker_result["correctness"], "timing": worker_result["timing"],
        "runtime_identity": worker_result["runtime_identity"],
        "runtime_gpu_uuid": "GPU-test", "inputs_sha256": "inputs", "weights_sha256": "weights",
        "cache_scope_path": "/cache/attempt",
    }
    atomic_json(attempt / "receipt.json", receipt)
    receipt_digest = hashlib.sha256((attempt / "receipt.json").read_bytes()).hexdigest()
    record_event(attempt, "intent", {"identity": identity})
    record_event(attempt, "started", {"identity": identity, "pid": 100})
    record_event(attempt, "terminal", {
        "identity": identity, "arm_execution": "COMPLETE", "process_returncode": 0,
        "receipt_sha256": receipt_digest,
    })

    recovered = recover_attempt(attempt, identity)
    assert recovered["arm_execution"] == ArmExecution.COMPLETE.value
    assert recovered["receipt_verified"] is True

    (attempt / "result.json").write_text('{"status":"numeric_mismatch"}', encoding="utf-8")
    changed_result = recover_attempt(attempt, identity)
    assert changed_result["arm_execution"] == ArmExecution.UNKNOWN.value
    assert changed_result["reason"] == "worker result digest differs from terminal receipt"


def test_event_identity_is_required_and_immutable_when_recording(tmp_path):
    attempt = tmp_path / "attempt"
    identity = {"source_commit": "source", "config_digest": "cfg", "gpu_uuid": "GPU-test"}
    record_event(attempt, "intent", {"identity": identity})

    with pytest.raises(ValueError, match="identity differs"):
        record_event(attempt, "started", {
            "identity": {**identity, "gpu_uuid": "GPU-other"}, "pid": 100,
        })
