from benchmark.formal.failure_funnel import parse_attempt


def trial(**changes):
    return dict(task_id="task", condition="D", outer_trial_id="outer-000",
                execution_validity="valid", **changes)


def test_nonlaunch_is_missing_not_zero():
    row = parse_attempt({**trial(), "execution_validity": "resource_blocked"}, None, {}, source="x", source_digest="x")
    assert row.stage == "nonlaunched"
    assert row.verified_positive is None and row.correctness_pass is None


def test_empty_trace_does_not_zero_observed_usage():
    row = parse_attempt(trial(), None, {"usage": {"tool_calls": 14}, "host_controller": {"outcome": "turn_limit", "tool_trace": []}}, source="x", source_digest="x")
    assert row.tool_calls == 14 and row.tool_trace_count == 0
    assert row.failure_code == "turn_limit" and not row.verifier_invoked


def test_zero_score_is_not_evidence_of_correctness_failure():
    row = parse_attempt(trial(), {"correctness_pass": True, "measurement": {"samples": [1]}, "verified_speedup": {"verified": False}}, {}, source="x", source_digest="x")
    assert row.stage == "measurement_only" and row.correctness_pass


def test_actual_api_error_is_separate_from_scientific_failure():
    row = parse_attempt(trial(), {"correctness_pass": False, "errors": ["S2 correctness raised: missing build_model"]}, {}, source="x", source_digest="x")
    assert row.failure_code == "api_contract_violation"
