from benchmark.calibration.bundle import classify_result


def test_resume_policy_preserves_valid_ineligible_evidence():
    raw = {
        "validity": "valid",
        "execution_validity": "valid",
        "efficacy_eligible": False,
        "calibration_status": "blocked",
        "protocol_failure": False,
        "timeout": False,
    }
    assert classify_result(raw) == "reusable"


def test_resume_policy_reruns_resource_blocked_evidence():
    raw = {
        "validity": "valid",
        "execution_validity": "resource_blocked",
        "efficacy_eligible": False,
        "calibration_status": "blocked",
        "protocol_failure": False,
        "timeout": False,
    }
    assert classify_result(raw) == "rerun"


def test_resume_policy_requires_revision_for_invalid_evidence():
    raw = {
        "validity": "invalid",
        "execution_validity": "invalid",
        "efficacy_eligible": False,
        "calibration_status": "blocked",
        "protocol_failure": True,
        "timeout": False,
    }
    assert classify_result(raw) == "blocked_requires_revision"


def test_resume_policy_reuses_eligible_evidence():
    raw = {
        "validity": "valid",
        "execution_validity": "valid",
        "efficacy_eligible": True,
        "calibration_status": "eligible",
        "protocol_failure": False,
        "timeout": False,
    }
    assert classify_result(raw) == "reusable"
