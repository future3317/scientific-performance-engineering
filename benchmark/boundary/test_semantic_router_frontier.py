from __future__ import annotations

import json

import pytest

from benchmark.boundary.policy_eval import Decision
from scripts.run_semantic_router_frontier import (
    SemanticRouter,
    build_router_prompt,
    router_cache_key,
)


def test_router_prompt_contains_only_public_context_and_skill_text():
    context = {"workload": {"graph_size": 128}}
    prompt = build_router_prompt(context, "reuse cached backbone", "router-v1")
    assert "graph_size" in prompt
    assert "reuse cached backbone" in prompt
    assert "tau" not in prompt
    assert "scientific_gate" not in prompt
    assert "oracle" not in prompt.lower()


def test_router_cache_key_is_canonical():
    left = {"b": 2, "a": 1}
    right = {"a": 1, "b": 2}
    assert router_cache_key(left, "skill", "model") == router_cache_key(right, "skill", "model")


def test_router_returns_shared_decision_and_writes_cache(tmp_path):
    class FakeClient:
        def __init__(self):
            self.calls = 0

        def decide(self, prompt, model):
            self.calls += 1
            return {"apply": True, "reason": "public-context-match"}

    client = FakeClient()
    router = SemanticRouter(client=client, model="test-model", cache_path=tmp_path / "cache.json")
    first = router.decide({"workload": {"graph_size": 128}}, "skill")
    second = router.decide({"workload": {"graph_size": 128}}, "skill")
    assert isinstance(first, Decision)
    assert first.action == "reuse"
    assert second == first
    assert client.calls == 1
    assert json.loads((tmp_path / "cache.json").read_text())["entries"]


def test_router_requires_provider_when_no_client_or_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    router = SemanticRouter(model="test-model", cache_path=tmp_path / "cache.json")
    with pytest.raises(RuntimeError, match="provider credentials"):
        router.decide({"workload": {"graph_size": 128}}, "skill")
