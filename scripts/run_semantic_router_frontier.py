#!/usr/bin/env python3
"""Provider-backed semantic router with auditable deterministic caching."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from benchmark.boundary.policy_eval import Decision


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def router_cache_key(
    context: Mapping[str, Any],
    skill_text: str,
    model: str,
    *,
    provider: str = "openrouter",
    router_version: str = "context-indexed-router-v1",
) -> str:
    payload = {
        "context": dict(context),
        "skill_text": str(skill_text),
        "model": str(model),
        "provider": str(provider),
        "router_version": str(router_version),
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def build_router_prompt(context: Mapping[str, Any], skill_text: str, router_version: str) -> str:
    return (
        "Decide whether to apply the fixed intervention to this public context.\n"
        f"Router version: {router_version}\n"
        f"Public context JSON: {_canonical(dict(context))}\n"
        f"Intervention description: {skill_text}\n"
        "Return JSON only: {\"apply\": true|false, \"reason\": \"short explanation\"}."
    )


class _OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str, *, disable_thinking: bool = False, timeout: float = 180.0):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("provider credentials require the openai package") from exc
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(timeout),
        )
        self._disable_thinking = disable_thinking

    def decide(self, prompt: str, model: str) -> dict[str, Any]:
        request = dict(
            model=model,
            messages=[
                {"role": "system", "content": "You are a deterministic semantic applicability router."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        if self._disable_thinking:
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        response = self._client.chat.completions.create(**request)
        raw = response.choices[0].message.content or "{}"
        result = json.loads(raw)
        usage = getattr(response, "usage", None)
        if usage is not None:
            result["_usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return result


class SemanticRouter:
    def __init__(
        self,
        *,
        client: Any | None = None,
        provider: str = "openrouter",
        model: str | None = None,
        cache_path: Path | None = None,
        router_version: str = "context-indexed-router-v1",
    ):
        self.client = client
        self.provider = provider
        if provider not in {"openrouter", "deepseek"}:
            raise ValueError(f"unsupported provider: {provider}")
        self.model = model or (
            os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
            if provider == "deepseek"
            else "openai/gpt-5.4-mini"
        )
        self.cache_path = Path(cache_path) if cache_path else None
        self.router_version = router_version
        self._entries: dict[str, dict[str, Any]] = {}
        self.provider_calls = 0
        self.usage_receipts: list[dict[str, Any]] = []
        if self.cache_path and self.cache_path.exists():
            loaded = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self._entries = dict(loaded.get("entries", {}))

    def _provider(self) -> Any:
        if self.client is not None:
            return self.client
        if self.provider == "deepseek":
            api_key = os.environ.get("MP_API_KEY", "").strip()
            base_url = os.environ.get("DEEPSEEK_BASE_URL", "").strip()
            if not api_key or not base_url:
                raise RuntimeError("DeepSeek credentials require MP_API_KEY and DEEPSEEK_BASE_URL")
        else:
            api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            base_url = "https://openrouter.ai/api/v1"
        if not api_key:
            raise RuntimeError("provider credentials required for semantic router")
        self.client = _OpenAICompatibleClient(api_key, base_url, disable_thinking=self.provider == "deepseek")
        return self.client

    def _save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps({"schema_version": 2, "entries": self._entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def decide(self, context: Mapping[str, Any], skill_text: str) -> Decision:
        key = router_cache_key(
            context,
            skill_text,
            self.model,
            provider=self.provider,
            router_version=self.router_version,
        )
        if key in self._entries:
            result = self._entries[key]["result"]
        else:
            prompt = build_router_prompt(context, skill_text, self.router_version)
            result = self._provider().decide(prompt, self.model)
            self.provider_calls += 1
            usage = result.pop("_usage", None) if isinstance(result, dict) else None
            if usage is not None:
                self.usage_receipts.append(dict(usage))
            apply_value = result.get("apply", False)
            if isinstance(apply_value, str):
                apply_value = apply_value.strip().lower() in {"true", "yes", "1"}
            result = {"apply": bool(apply_value), "reason": str(result.get("reason", "")).strip() or "no_reason"}
            self._entries[key] = {"provider": self.provider, "model": self.model, "result": result}
            self._save()
        admitted = bool(result["apply"])
        return Decision("reuse" if admitted else "abstain", 1.0 if admitted else 0.0, 0.0, f"semantic_router:{result['reason']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON list of {id, context, skill_text}")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--provider", choices=("openrouter", "deepseek"), default="openrouter")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    try:
        router = SemanticRouter(provider=args.provider, model=args.model, cache_path=args.cache)
        decisions = []
        for row in rows:
            decision = router.decide(row["context"], row["skill_text"])
            decisions.append({"id": row["id"], "action": decision.action, "score": decision.score, "reason": decision.reason})
        report = {
            "schema_version": 2,
            "status": "complete",
            "provider": args.provider,
            "provider_calls": router.provider_calls,
            "usage_receipts": router.usage_receipts,
            "model": router.model,
            "decisions": decisions,
            "cache": str(args.cache),
            "claim_boundary": "semantic routing decisions only; no held-out outcomes consumed",
        }
    except Exception as exc:
        report = {
            "schema_version": 2,
            "status": "blocked",
            "provider": args.provider,
            "provider_calls": 0,
            "model": args.model,
            "error_type": type(exc).__name__,
            "reason": str(exc),
            "claim_boundary": "semantic router unavailable",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "provider_calls": report["provider_calls"], "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
