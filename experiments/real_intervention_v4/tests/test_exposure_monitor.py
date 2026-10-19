from __future__ import annotations

from experiments.real_intervention_v4.exposure_monitor import compare_arm_exposure


def _arm(*, external: bool = False, uuid: str = "GPU-1", utilization: str = "0"):
    processes = [{"pid": 11, "origin": "STUDY_PROCESS_GROUP", "name": "python"}]
    if external:
        processes.append({"pid": 22, "origin": "EXTERNAL_OR_UNRESOLVED", "name": "python"})
    return {"telemetry": [{
        "gpu_uuid": uuid,
        "compute_processes": processes,
        "utilization_percent": utilization,
    }]}


def test_complete_uncontended_same_uuid_is_matched():
    result = compare_arm_exposure(_arm(), _arm())

    assert result["matched"] is True
    assert result["reason"] == "SAME_UUID_COMPLETE_UNCONTENDED_TELEMETRY"


def test_any_observed_cotenant_keeps_raw_data_but_is_not_matched():
    result = compare_arm_exposure(_arm(), _arm(external=True))

    assert result["matched"] is False
    assert result["reason"] == "CO_TENANT_OBSERVED_NOT_MATCHED"
    assert result["co_tenant_observed"] == {"S0": False, "S1": True}


def test_unattributed_gpu_activity_is_unknown_not_clean():
    result = compare_arm_exposure(
        {"telemetry": [{"gpu_uuid": "GPU-1", "compute_processes": [],
                        "utilization_percent": "12"}]},
        _arm(),
    )

    assert result["matched"] is None
    assert result["reason"] == "GPU_ACTIVITY_WITHOUT_PROCESS_ATTRIBUTION"


def test_physical_uuid_change_is_not_matched():
    result = compare_arm_exposure(_arm(uuid="GPU-1"), _arm(uuid="GPU-2"))

    assert result["matched"] is False
    assert result["reason"] == "PHYSICAL_GPU_UUID_MISMATCH"


def test_incomplete_exposure_is_unknown_not_matched():
    result = compare_arm_exposure(_arm(), {"telemetry": []})

    assert result["matched"] is None
    assert result["reason"] == "TELEMETRY_INCOMPLETE"
