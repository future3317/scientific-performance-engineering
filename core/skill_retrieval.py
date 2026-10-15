"""Lexical candidate retrieval followed by the existing governed routing path."""

from collections import Counter
import math
import re
from typing import Any, Mapping, Sequence

from core.models import RuleSpec


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def bm25_candidates(specs: Sequence[RuleSpec], query: str, *, top_k: int = 5,
                    include_body: bool = False) -> list[str]:
    """Deterministic BM25 (k1=1.2, b=.75); retrieval conveys no permission."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    documents = [_tokens(f"{s.rule_id} {s.expected_mechanism}" + (f" {s.text}" if include_body else "")) for s in specs]
    if not documents:
        return []
    terms = set(_tokens(query))
    average_length = sum(map(len, documents)) / len(documents) or 1.0
    frequencies = Counter(term for doc in documents for term in set(doc))
    scored = []
    for spec, doc in zip(specs, documents):
        counts = Counter(doc)
        score = 0.0
        for term in terms:
            frequency = counts[term]
            idf = math.log(1 + (len(documents) - frequencies[term] + .5) / (frequencies[term] + .5))
            score += idf * frequency * 2.2 / (frequency + 1.2 * (.25 + .75 * len(doc) / average_length))
        scored.append((-score, spec.rule_id))
    return [rule_id for _, rule_id in sorted(scored)[:top_k]]


def retrieve_and_route(engine: Any, query: str, context: Mapping[str, Any], *,
                       top_k: int = 5, support_regimes: Mapping[str, Any] | None = None,
                       runtime_versions: Mapping[str, str] | None = None) -> dict[str, Any]:
    metadata_ids = bm25_candidates(engine.rule_specs, query, top_k=top_k)
    candidates = [s for s in engine.rule_specs if s.rule_id in metadata_ids]
    body_ids = bm25_candidates(candidates, query, top_k=top_k, include_body=True)
    decision = engine.route(context, candidate_ids=body_ids, support_regimes=support_regimes,
                            runtime_versions=runtime_versions)
    return {"retrieved_ids": metadata_ids, "body_ranked_ids": body_ids,
            "selected_ids": list(decision.selected_rule_ids), "decision": decision,
            "activation": "not_observed"}
