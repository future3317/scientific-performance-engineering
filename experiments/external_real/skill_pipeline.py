from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Sequence


def _canonical(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


@dataclass(frozen=True)
class FrozenSkill:
    benchmark: str
    generation_seed: int
    body: str
    scripts: tuple[str, ...]
    resources: tuple[str, ...]
    semantic_abstract: str
    skill_hash: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def freeze_skill(
    body: str,
    scripts: Sequence[str],
    resources: Sequence[str],
    semantic_abstract: str,
    benchmark: str,
    generation_seed: int,
) -> FrozenSkill:
    payload = {
        "benchmark": benchmark,
        "body": body,
        "generation_seed": int(generation_seed),
        "resources": list(resources),
        "scripts": list(scripts),
        "semantic_abstract": semantic_abstract,
    }
    return FrozenSkill(
        benchmark=benchmark,
        generation_seed=int(generation_seed),
        body=body,
        scripts=tuple(scripts),
        resources=tuple(resources),
        semantic_abstract=semantic_abstract,
        skill_hash=hashlib.sha256(_canonical(payload)).hexdigest(),
    )
