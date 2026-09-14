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


def router_cache_key(context: Mapping[str, Any], skill_text: str, model: str) -> str:
    payload = {"context": dict(context), "skill_text": str(skill_text), "model": str(model)}
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def build_router_prompt(context: Mapping[str, Any], skill_text: str, router_version: str) -> str:
    return (
        "Decide whether to apply the fixed intervention to this public context.\n"
        f"Router version: {router_version}\n"
        f"Public context JSON: {_canonical(dict(context))}\n"
        f"Intervention description: {skill_text}\n"
        "Return JSON only: {\"apply\": true|false, \"reason\": \"short explanation\"}."
    )


class _OpenRouterClient:
    def __init__(self, api_key: str, timeout: float = 180.0):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("provider credentials require the openai package") from exc
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=float(timeout),
        )

    def decide(self, prompt: str, model: str) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a deterministic semantic applicability router."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)


class SemanticRouter:
    def __init__(self, *, client: Any | None = None, model: str = "openai/gpt-5.4-mini", cache_path: Path | None = None, router_version: str = "context-indexed-router-v1"):
        self.client = client
        self.model = model
        self.cache_path = Path(cache_path) if cache_path else None
        self.router_version = router_version
        self._entries: dict[str, dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            loaded = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self._entries = dict(loaded.get("entries", {}))

    def _provider(self) -> Any:
        if self.client is not None:
            return self.client
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("provider credentials required for semantic router")
        self.client = _OpenRouterClient(api_key)
        return self.client

    def _save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps({"schema_version": 1, "entries": self._entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def decide(self, context: Mapping[str, Any], skill_text: str) -> Decision:
        key = router_cache_key(context, skill_text, self.model)
        if key in self._entries:
            result = self._entries[key]["result"]
        else:
            prompt = build_router_prompt(context, skill_text, self.router_version)
            result = self._provider().decide(prompt, self.model)
            apply_value = result.get("apply", False)
            if isinstance(apply_value, str):
                apply_value = apply_value.strip().lower() in {"true", "yes", "1"}
            result = {"apply": bool(apply_value), "reason": str(result.get("reason", "")).strip() or "no_reason"}
            self._entries[key] = {"model": self.model, "result": result}
            self._save()
        admitted = bool(result["apply"])
        return Decision("reuse" if admitted else "abstain", 1.0 if admitted else 0.0, 0.0, f"semantic_router:{result['reason']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON list of {id, context, skill_text}")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-5.4-mini")
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    try:
        router = SemanticRouter(model=args.model, cache_path=args.cache)
        decisions = []
        for row in rows:
            decision = router.decide(row["context"], row["skill_text"])
            decisions.append({"id": row["id"], "action": decision.action, "score": decision.score, "reason": decision.reason})
        report = {"schema_version": 1, "status": "complete", "provider_calls": len(decisions), "model": args.model, "decisions": decisions, "cache": str(args.cache), "claim_boundary": "semantic routing decisions only; no held-out outcomes consumed"}
    except RuntimeError as exc:
        report = {"schema_version": 1, "status": "blocked", "provider_calls": 0, "model": args.model, "reason": str(exc), "claim_boundary": "semantic router unavailable"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "provider_calls": report["provider_calls"], "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
