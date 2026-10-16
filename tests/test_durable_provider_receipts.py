from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from core.acre.resource_budget import ApprovedProviderBudget, manifest_digest
from experiments.external_real.alfworld_stateful_runner import DeepSeekStatefulClient
from experiments.skill_evolution.receipt_journal import DurableProviderJournal


class SimulatedProcessKill(BaseException):
    pass


def _budget():
    manifest = {"study_id": "phase5", "run_id": "receipt-test", "stage": "train_inner",
                "controller_commit": "test-commit", "split_manifest_sha256": "split"}
    approval = {
        "provider_enabled": True, "execute": True, "study_id": "phase5", "run_id": "receipt-test",
        "controller_commit": "test-commit", "model": "deepseek-flash",
        "approved_manifest_sha256": manifest_digest(manifest), "approved_split_manifest_sha256": "split",
        "approved_by": "test", "approval_reference": "TEST_ONLY",
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "expiration_utc": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
        "permitted_stages": ["train_inner"], "api_base_allowlist": ["https://api.deepseek.com"],
        "max_provider_calls": 20, "max_input_tokens": 100_000,
        "max_completion_tokens": 100_000, "per_episode_max_steps": 60,
        "per_request_max_completion_tokens": 1000, "max_failed_retries": 0,
        "max_wall_seconds": 3600, "max_total_usd": "1",
    }
    pricing = {
        "model": "deepseek-flash", "currency": "USD", "pricing_source_url": "fixture",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "price_period": "fixture",
        "estimate_quality": "SYNTHETIC_FIXTURE", "input_hit_per_million": "0.1",
        "input_miss_per_million": "1", "output_per_million": "2",
    }
    return ApprovedProviderBudget(manifest, approval, pricing, controller_commit="test-commit",
                                  stage="train_inner", model="deepseek-flash",
                                  base_url="https://api.deepseek.com")


def _response(content="Action: look"):
    message = SimpleNamespace(content=content, reasoning_content="PRIVATE REASONING MUST NOT BE SAVED")
    usage = SimpleNamespace(
        prompt_tokens=31, completion_tokens=9, prompt_cache_hit_tokens=4,
        prompt_cache_miss_tokens=27,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=6),
    )
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=usage, model="deepseek-flash-v1",
                           id="response-1", system_fingerprint="fingerprint")


def _client(journal, budget, provider, persist_budget=None):
    client = DeepSeekStatefulClient.__new__(DeepSeekStatefulClient)
    client.client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=provider)))
    client.base_url = "https://api.deepseek.com"
    client.model = "deepseek-flash"
    client.temperature = 0.0
    client.budget = budget
    client.max_completion_tokens = 100
    client.max_input_per_request = 10_000
    client.persist_budget = persist_budget
    client.response_identity = None
    client.response_ids = set()
    client.receipt_journal = journal
    client._episode_context = None
    client._episode_turn = 0
    client.set_episode_context("episode-1", "SKILL", "skill-v1")
    return client


def _messages(text="prompt"):
    return [{"role": "system", "content": "system"}, {"role": "user", "content": text}]


def test_raw_receipt_survives_kill_before_summary_and_replays_without_call_or_charge(tmp_path):
    journal = DurableProviderJournal(tmp_path / "server-spool")
    provider_calls = []

    def provider(**_request):
        provider_calls.append(1)
        return _response()

    callback_count = []

    def fail_after_accounting(_snapshot):
        callback_count.append(1)
        if len(callback_count) == 1:
            raise SimulatedProcessKill("killed after raw receipt, before summary")

    first_budget = _budget()
    first = _client(journal, first_budget, provider, fail_after_accounting)
    with pytest.raises(SimulatedProcessKill, match="before summary"):
        first.complete_text(_messages())

    receipt_path = next((tmp_path / "server-spool" / "receipts").glob("*.json"))
    receipt_text = receipt_path.read_text(encoding="utf-8")
    assert "Action: look" in receipt_text
    assert "PRIVATE REASONING" not in receipt_text

    resumed_budget = _budget()
    assert journal.restore_budget(resumed_budget) == 1
    assert journal.restore_budget(resumed_budget) == 0
    expected_usd = resumed_budget.usd_used
    assert sum(event.get("event") == "usage_ledger" for event in journal.events()) == 1
    replay = _client(journal, resumed_budget, provider)
    result = replay.complete_text(_messages())

    assert result["raw_output"] == "Action: look"
    assert provider_calls == [1]
    assert resumed_budget.calls_reserved == 1
    assert resumed_budget.usd_used == expected_usd


def test_changed_prompt_rejects_saved_response_before_provider(tmp_path):
    journal = DurableProviderJournal(tmp_path / "server-spool")
    calls = []

    def provider(**_request):
        calls.append(1)
        return _response()

    budget = _budget()
    client = _client(journal, budget, provider)
    client.complete_text(_messages())
    client.set_episode_context("episode-1", "SKILL", "skill-v1")

    with pytest.raises(RuntimeError, match="identity changed"):
        client.complete_text(_messages("changed prompt"))
    assert calls == [1]


def test_unresolved_intent_blocks_duplicate_provider_call_and_budget_restore(tmp_path):
    journal = DurableProviderJournal(tmp_path / "server-spool")
    calls = []

    def timeout(**_request):
        calls.append(1)
        raise TimeoutError("provider outcome unknown")

    failed = _client(journal, _budget(), timeout)
    with pytest.raises(TimeoutError):
        failed.complete_text(_messages())

    resumed_budget = _budget()
    with pytest.raises(RuntimeError, match="unresolved provider intent"):
        journal.restore_budget(resumed_budget)
    resumed = _client(journal, resumed_budget, lambda **_request: _response())
    with pytest.raises(RuntimeError, match="unresolved durable intent"):
        resumed.complete_text(_messages())
    assert calls == [1]


def test_same_spool_cannot_have_two_live_invocations_or_change_manifest(tmp_path):
    spool=tmp_path/"server-spool"
    first=DurableProviderJournal(spool,run_identity="manifest-one")
    with pytest.raises(RuntimeError,match="another invocation"):
        DurableProviderJournal(spool,run_identity="manifest-one")
    first.close()
    with pytest.raises(RuntimeError,match="different frozen manifest"):
        DurableProviderJournal(spool,run_identity="manifest-two")


def test_recovered_unknown_usage_never_releases_action(tmp_path):
    journal=DurableProviderJournal(tmp_path/"server-spool")
    def provider(**_):
        response=_response();response.usage.prompt_tokens=None;return response
    first=_client(journal,_budget(),provider)
    from core.acre.resource_budget import ProviderBudgetExceeded
    with pytest.raises(ProviderBudgetExceeded):
        first.complete_text(_messages())
    resumed_budget=_budget()
    journal.restore_budget(resumed_budget)
    resumed=_client(journal,resumed_budget,lambda **_:pytest.fail("duplicate provider call"))
    with pytest.raises(ProviderBudgetExceeded):
        resumed.complete_text(_messages())

