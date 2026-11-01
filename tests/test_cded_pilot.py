import pytest

from experiments.scientific_skill_evolution.cded_pilot import (
    LIKELIHOODS,
    MAX_CALLS,
    MAX_INPUT,
    MAX_OUTPUT,
    MAX_STEPS,
    PRICING,
    WORST_CASE_USD,
    _fixed_order,
    _generic_length_control,
    _length_match,
    _posterior,
    _simple_calibration,
    _validate_approval,
)
from core.acre.resource_budget import ProviderBudgetExceeded


def _receipt(cluster, arm, success):
    return {
        "cluster_id": cluster,
        "arm": arm,
        "status": "VERIFIED_SUCCESS" if success else "VERIFIED_FAILURE",
        "success": success,
        "environment_won": success,
        "environment_done": True,
    }


def test_development_posterior_selects_supported_item_and_requires_complete_cells():
    outcomes = {
        "development-a": {"parent_00": False, "item_a_10": True, "item_b_01": False, "LENGTH": False},
        "development-b": {"parent_00": False, "item_a_10": True, "item_b_01": False, "LENGTH": False},
    }
    rows = [_receipt(cluster, arm, result) for cluster, cells in outcomes.items()
            for arm, result in cells.items()]

    posterior = _posterior(rows)

    assert set(posterior) == set(LIKELIHOODS)
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert posterior["A_ONLY"] > posterior["B_ONLY"]

    with pytest.raises(ValueError, match="two complete four-arm development clusters"):
        _posterior(rows[:-1])


def test_fixed_order_never_keeps_an_item_with_a_family_regression():
    rows = [
        _receipt("a", "parent_00", True), _receipt("a", "item_a_10", False),
        _receipt("a", "item_b_01", True), _receipt("a", "LENGTH", True),
        _receipt("b", "parent_00", False), _receipt("b", "item_a_10", True),
        _receipt("b", "item_b_01", True), _receipt("b", "LENGTH", False),
    ]

    action, details = _fixed_order(rows)

    assert action == "retain_b"
    assert details["evaluated_order"] == ["item_a_10", "item_b_01"]
    assert details["selected_item"] == "item_b_01"


def test_calibration_freezes_strongest_simple_and_adaptive_gate():
    rows = []
    outcomes = {
        "clean": {"NO_SKILL": False, "LENGTH": True, "STATIC_CDED": False,
                  "ADAPTIVE_CDED": True, "FIXED_ORDER_CDED": False},
        "heat": {"NO_SKILL": False, "LENGTH": False, "STATIC_CDED": True,
                 "ADAPTIVE_CDED": True, "FIXED_ORDER_CDED": False},
    }
    for cluster, arms in outcomes.items():
        rows.extend(_receipt(cluster, arm, result) for arm, result in arms.items())

    baseline, decision = _simple_calibration(rows)

    assert baseline == "LENGTH"
    assert decision["adaptive_gate_passed"] is True
    assert decision["frozen_route"] == "ADAPTIVE_CDED"


def test_length_controls_are_character_matched_and_total_worst_case_fits_one_usd():
    target = "semantic candidate text" * 20
    length = _generic_length_control(target)
    padded = _length_match("candidate", target)
    worst_case = MAX_CALLS * (MAX_INPUT * 0.30 + MAX_OUTPUT * 1.32) / 1_000_000

    assert len(length) == len(target)
    assert len(padded) == len(target)
    assert MAX_STEPS == 16
    assert MAX_CALLS == 305
    assert MAX_CALLS < 2 + 24 * MAX_STEPS
    assert worst_case < 0.977854084
    assert str(WORST_CASE_USD) == "0.9744384"
    assert MAX_INPUT == 6144
    assert MAX_OUTPUT == 1024
    assert PRICING["output_per_million"] == "1.32"


def test_approval_must_match_every_frozen_usage_cap():
    manifest = {
        "max_provider_calls": 305, "max_input_tokens": 1873920,
        "max_completion_tokens": 312320, "max_total_usd": "0.977854084",
        "max_wall_seconds": 86400, "max_failed_retries": 0,
        "max_steps": 16, "max_completion_per_request": 1024,
        "max_usd_per_episode": "0.052", "max_calls_per_episode": 16,
        "max_episode_wall_seconds": 1800,
    }
    approval = {
        "max_provider_calls": 305, "max_input_tokens": 1873920,
        "max_completion_tokens": 312320, "max_total_usd": "0.977854084",
        "max_wall_seconds": 86400, "max_failed_retries": 0,
        "per_episode_max_steps": 16, "per_request_max_completion_tokens": 1024,
        "max_usd_per_episode": "0.052", "max_calls_per_episode": 16,
        "max_episode_wall_seconds": 1800,
    }

    _validate_approval(manifest, approval)
    approval["max_provider_calls"] = 1000
    with pytest.raises(ProviderBudgetExceeded, match="approval_usage_envelope_mismatch"):
        _validate_approval(manifest, approval)
