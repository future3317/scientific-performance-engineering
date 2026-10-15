from datetime import datetime, timedelta, timezone

import pytest

from core.acre.resource_budget import ApprovedProviderBudget, ProviderBudgetExceeded, manifest_digest


def inputs():
    manifest = {"study_id": "phase2", "run_id": "b1", "stage": "train_viability",
                "split_manifest_sha256": "split", "controller_commit": "commit"}
    approval = {"provider_enabled": True, "study_id": "phase2", "run_id": "b1",
                "controller_commit": "commit", "model": "deepseek-flash",
                "approved_manifest_sha256": manifest_digest(manifest),
                "approved_split_manifest_sha256": "split", "approved_by": "test-only",
                "approval_reference": "SYNTHETIC_FIXTURE", "approved_at_utc": "2026-10-16T00:00:00Z",
                "expiration_utc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                "permitted_stages": ["train_viability"], "api_base_allowlist": ["https://api.deepseek.com"],
                "max_provider_calls": 3, "max_input_tokens": 1000, "max_completion_tokens": 1000,
                "per_episode_max_steps": 60, "per_request_max_completion_tokens": 200,
                "max_total_usd": "0.01", "max_failed_retries": 0, "max_wall_seconds": 300}
    pricing = {"model": "deepseek-flash", "currency": "USD", "pricing_source_url": "https://example.test/pricing",
               "checked_at_utc": "2026-10-16T00:00:00Z", "price_period": "fixture",
               "estimate_quality": "SYNTHETIC_FIXTURE", "input_hit_per_million": "1",
               "input_miss_per_million": "2", "output_per_million": "3"}
    return manifest, approval, pricing


def budget(manifest, approval, pricing):
    return ApprovedProviderBudget(manifest, approval, pricing, controller_commit="commit",
                                  stage="train_viability", model="deepseek-flash",
                                  base_url="https://api.deepseek.com")


@pytest.mark.parametrize("change", ["disabled", "expired", "wrong_stage", "wrong_revision", "wrong_split", "changed_manifest"])
def test_unapproved_scope_rejected_before_network(change):
    m, a, p = inputs()
    if change == "disabled":
        a["provider_enabled"] = False
    elif change == "expired":
        a["expiration_utc"] = "2000-01-01T00:00:00Z"
    elif change == "wrong_stage":
        a["permitted_stages"] = ["skill_proposal"]
    elif change == "wrong_revision":
        a["controller_commit"] = "other"
    elif change == "wrong_split":
        a["approved_split_manifest_sha256"] = "other"
    else:
        m["added_episode"] = "other"
    with pytest.raises(ProviderBudgetExceeded):
        budget(m, a, p)


def test_money_reserve_before_call_and_actual_cache_charge():
    b = budget(*inputs())
    b.reserve(100, 200)
    assert str(b.pending_usd) == "0.0008"
    b.record_usage(100, 50, cache_hit_tokens=75, cache_miss_tokens=25)
    assert str(b.usd_used) == "0.000275"
    assert b.input_used == 100 and b.output_used == 50
    assert b.as_dict()["billing_status"] == "ESTIMATE_NOT_BILL"


def test_missing_cache_counters_charge_conservative_rate_not_zero():
    b = budget(*inputs())
    b.reserve(100, 200)
    b.record_usage(100, 50)
    assert str(b.usd_used) == "0.00035"


def test_full_pair_reservation_and_usd_cap_block_before_first_arm():
    m, a, p = inputs()
    a["max_total_usd"] = "0.0001"
    b = budget(m, a, p)
    with pytest.raises(ProviderBudgetExceeded, match="usd_cap"):
        b.ensure_capacity(2, 200, 400)
    assert b.calls_reserved == 0


def test_unknown_charge_keeps_reserve_and_stops_further_requests():
    b = budget(*inputs())
    b.reserve(100, 200)
    b.mark_unknown("timeout")
    assert b.as_dict()["billing_status"] == "BILLING_UNKNOWN"
    assert b.pending_usd > 0
    with pytest.raises(ProviderBudgetExceeded):
        b.reserve(1, 1)


def test_output_over_reservation_stops_even_when_total_cap_fits():
    b = budget(*inputs())
    b.reserve(100, 10)
    b.record_usage(100, 11)
    assert b.stopped


def test_per_request_output_limit_and_wall_limit(monkeypatch):
    b = budget(*inputs())
    with pytest.raises(ProviderBudgetExceeded, match="per_request"):
        b.reserve(10, 201)
    monkeypatch.setattr("core.acre.resource_budget.time.monotonic", lambda: b.started + 301)
    with pytest.raises(ProviderBudgetExceeded, match="wall_cap"):
        b.reserve(10, 1)
