from __future__ import annotations

import json

from scripts.run_semantic_router_frontier import SemanticRouter, router_cache_key


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, prompt: str, model: str) -> dict[str, object]:
        self.calls += 1
        return {"apply": True, "reason": "fixture", "_usage": {"total_tokens": 7}}


def test_cache_key_scopes_provider_and_router_revision() -> None:
    context = {"task": "t1", "model": "m1"}
    base = router_cache_key(context, "skill", "deepseek-v4-flash")
    assert base != router_cache_key(context, "skill", "deepseek-v4-flash", provider="deepseek")
    assert base != router_cache_key(context, "skill", "deepseek-v4-flash", router_version="v2")


def test_semantic_router_records_calls_and_reuses_scoped_cache(tmp_path) -> None:
    cache = tmp_path / "router.json"
    first_client = _FakeClient()
    first = SemanticRouter(provider="deepseek", model="deepseek-v4-flash", client=first_client, cache_path=cache)
    decision = first.decide({"task": "t1"}, "fixed skill")
    assert decision.action == "reuse"
    assert first.provider_calls == 1
    assert first.usage_receipts == [{"total_tokens": 7}]

    second_client = _FakeClient()
    second = SemanticRouter(provider="deepseek", model="deepseek-v4-flash", client=second_client, cache_path=cache)
    assert second.decide({"task": "t1"}, "fixed skill").action == "reuse"
    assert second.provider_calls == 0
    assert second_client.calls == 0
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert next(iter(saved["entries"].values()))["provider"] == "deepseek"
