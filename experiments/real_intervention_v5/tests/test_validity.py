from __future__ import annotations

from experiments.real_intervention_v5.validity import classify_pair


def test_ba_scientific_status_uses_semantic_s0_not_list_position():
    reference = {"execution_status": "COMPLETE", "scientific_status": "PASS", "worker_result": {"arm": "S0"}}
    candidate = {"execution_status": "COMPLETE", "scientific_status": "NUMERICAL_UNSAFE", "worker_result": {"arm": "S1"}}
    raw = [candidate.copy(), reference.copy()]
    result = classify_pair(raw, {"status": "MATCHED"})
    assert result["scientific_status"] == "NUMERICAL_UNSAFE"
    assert result["analysis_status"] == "EXCLUDED_PRESPECIFIED"
    assert raw[0]["worker_result"]["arm"] == "S1"
    reference["scientific_status"] = "NUMERICAL_UNSAFE"
    candidate["scientific_status"] = "PASS"
    result = classify_pair([candidate, reference], {"status": "MATCHED"})
    assert result["scientific_status"] == "UNVERIFIED"
    assert result["first_failing_gate"] == "BASELINE_CORRECTNESS"


def test_compile_ba_uses_eager_as_baseline():
    def arm(role, science="PASS"):
        return {"execution_status": "COMPLETE", "scientific_status": science, "worker_result": {"arm": role}}
    result = classify_pair([arm("CACHE_PREFILL", "NUMERICAL_UNSAFE"), arm("COMPILE_PERSISTENT_CACHE"), arm("COMPILE_COLD"), arm("EAGER")], {"status": "MATCHED"})
    assert result["scientific_status"] == "NUMERICAL_UNSAFE"
    result = classify_pair([arm("CACHE_PREFILL"), arm("COMPILE_PERSISTENT_CACHE"), arm("COMPILE_COLD"), arm("EAGER", "NUMERICAL_UNSAFE")], {"status": "MATCHED"})
    assert result["scientific_status"] == "UNVERIFIED"


def arms(*, baseline="PASS", candidate="PASS", execution=("COMPLETE", "COMPLETE"),
         identities=None):
    identities = identities or [
        {"session_id": "pair-1", "arm_index": index, "gpu_uuid": "GPU-a"}
        for index in range(2)
    ]
    return [
        {"execution_status": execution[index],
         "scientific_status": (baseline, candidate)[index],
         "arm_identity": identities[index]}
        for index in range(2)
    ]


def matched():
    return {"status": "MATCHED", "telemetry_complete": True,
            "co_tenant_confirmed": False, "gpu_uuids": ["GPU-a"]}


def test_numerically_unsafe_candidate_is_excluded_even_if_fast():
    pair = arms(candidate="NUMERICAL_UNSAFE")
    pair[1]["worker_result"] = {"timing": {"total_s": 0.01}}

    result = classify_pair(pair, matched())

    assert result["scientific_status"] == "NUMERICAL_UNSAFE"
    assert result["analysis_status"] == "EXCLUDED_PRESPECIFIED"
    assert result["first_failing_gate"] == "SCIENTIFIC_CORRECTNESS"


def test_correct_arms_with_incomplete_exposure_remain_noncomparable():
    result = classify_pair(
        arms(), {"status": "INCOMPLETE", "telemetry_complete": False},
    )

    assert result["scientific_status"] == "PASS"
    assert result["exposure_status"] == "INCOMPLETE"
    assert result["analysis_status"] == "MISSING"
    assert result["first_failing_gate"] == "GPU_EXPOSURE"


def test_confirmed_cotenant_is_separate_from_telemetry_completeness():
    result = classify_pair(arms(), {
        "status": "CONTENDED", "telemetry_complete": True,
        "co_tenant_confirmed": True, "co_tenant_pids": [4242],
    })

    assert result["platform_status"] == "HEALTHY"
    assert result["exposure_status"] == "CONTENDED"
    assert result["analysis_status"] == "EXCLUDED_PRESPECIFIED"


def test_baseline_correctness_failure_is_not_candidate_harm():
    result = classify_pair(arms(baseline="NUMERICAL_UNSAFE"), matched())

    assert result["scientific_status"] == "UNVERIFIED"
    assert result["analysis_status"] == "MISSING"
    assert result["first_failing_gate"] == "BASELINE_CORRECTNESS"
    assert result["error_code"] is None


def test_inconsistent_arm_outputs_are_numerically_unsafe():
    pair = arms()
    pair[1]["outputs_match"] = False

    result = classify_pair(pair, matched())

    assert result["scientific_status"] == "NUMERICAL_UNSAFE"
    assert result["analysis_status"] == "EXCLUDED_PRESPECIFIED"


def test_different_gpu_uuids_are_noncomparable():
    identities = [
        {"session_id": "pair-1", "arm_index": 0, "gpu_uuid": "GPU-a"},
        {"session_id": "pair-1", "arm_index": 1, "gpu_uuid": "GPU-b"},
    ]
    result = classify_pair(arms(identities=identities), matched())

    assert result["analysis_status"] == "NONCOMPARABLE"
    assert result["first_failing_gate"] == "ARM_IDENTITY"


def test_interrupted_second_arm_is_unknown_and_missing():
    result = classify_pair(arms(execution=("COMPLETE", "STARTED")), matched())

    assert result["execution_status"] == "UNKNOWN"
    assert result["analysis_status"] == "MISSING"
    assert result["first_failing_gate"] == "EXECUTION"


def test_unattributed_activity_is_unknown_not_confirmed_contention():
    result = classify_pair(arms(), {
        "status": "UNKNOWN", "telemetry_complete": True,
        "co_tenant_confirmed": False, "unattributed_activity": True,
    })

    assert result["exposure_status"] == "UNKNOWN"
    assert result["analysis_status"] == "NONCOMPARABLE"
