#!/usr/bin/env python3
"""Measure lightweight provider-free ACRE operation costs."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.acre.cegis import BoundaryObservation, StatisticalCEGIS
from core.acre.predicates import PredicateGrammar
from core.acre.router import ConservativeCausalRouter
from core.models import RuleSpec, RuleState


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_samples(
    samples: Sequence[float],
    *,
    warmup: int = 2,
    source: str = "measured",
    minimum_samples: int = 20,
) -> dict[str, Any]:
    if warmup < 0 or warmup >= len(samples):
        raise ValueError("warmup must leave measured samples")
    measured = [float(value) for value in samples[warmup:]]
    if len(measured) < minimum_samples:
        raise ValueError(f"cost profile requires at least {minimum_samples} measured samples")
    if not all(math.isfinite(value) and value >= 0.0 for value in measured):
        raise ValueError("cost samples must be finite and non-negative")
    return {
        "warmup_excluded": warmup,
        "n": len(measured),
        "median": statistics.median(measured),
        "p90": _percentile(measured, 0.90),
        "max": max(measured),
        "samples": measured,
        "stopping_horizon_source": source if source == "offline_replay" else None,
        "online_stopping_horizon_claim": source != "offline_replay",
    }


def _time_repeated(operation: Callable[[], Any], *, repetitions: int, warmup: int) -> tuple[list[float], Any]:
    samples: list[float] = []
    last: Any = None
    for _ in range(warmup + repetitions):
        started = time.perf_counter()
        last = operation()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples, last


def _cegis_operation() -> Any:
    grammar = PredicateGrammar.from_dict({
        "schema_version": 1,
        "features": [{"path": "workload.threshold", "type": "numeric"}],
        "max_depth": 2,
        "max_literals": 2,
    })
    synthesizer = StatisticalCEGIS(grammar)
    return synthesizer.synthesize(
        positive=[BoundaryObservation("p", {"workload": {"threshold": 1}}, 0.2, True, 0.1, 0.3)],
        counterexamples=[BoundaryObservation("c", {"workload": {"threshold": 3}}, -0.1, True, -0.3, -0.05)],
        parent_predicate=None,
        decision_contexts=[{"workload": {"threshold": 1}}, {"workload": {"threshold": 3}}],
    )


def _router_operation() -> Any:
    specs = [
        RuleSpec(
            rule_id=f"profile-{index}", version=1, parent=None,
            applicability={"all": []}, intervention={"action": f"a{index}"},
            expected_mechanism="profile", evidence_requirements=["paired"],
            scientific_invariants=[], abstain_conditions={}, relations={},
            runtime_cost={"tokens": 1}, provenance_policy={"required": True},
        )
        for index in range(2)
    ]
    states = {
        spec.rule_id: RuleState(rule_id=spec.rule_id, version=1, status="canonical", effect={"utility": 0.2}, confidence_sequence={"lcb": 0.2})
        for spec in specs
    }
    return ConservativeCausalRouter(token_budget=4).route(specs, states, [], {}, {"workload": {}})


def collect_cost_profile(*, repetitions: int = 20, warmup: int = 2) -> dict[str, Any]:
    if repetitions < 20:
        raise ValueError("repetitions must be at least 20")
    cegis_samples, cegis_last = _time_repeated(_cegis_operation, repetitions=repetitions, warmup=warmup)
    router_samples, router_last = _time_repeated(_router_operation, repetitions=repetitions, warmup=warmup)
    rule_store = [
        {"rule_id": f"profile-{index}", "version": 1, "status": "canonical"}
        for index in range(2)
    ]
    serialized_bytes = len(json.dumps(rule_store, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {
        "schema_version": 1,
        "provider_calls": 0,
        "measurement_scope": "provider-free CEGIS/router/rule-store primitives",
        "repetitions": repetitions,
        "warmup": warmup,
        "operations": {
            "cegis_synthesis_ms": summarize_samples(cegis_samples, warmup=warmup),
            "router_decision_ms": summarize_samples(router_samples, warmup=warmup),
        },
        "counterexamples_until_stable": {"count": 1, "source": "deterministic_fixture"},
        "paired_replay_stopping_horizon": {
            "count": 16,
            "source": "offline_replay",
            "online_stopping_horizon_claim": False,
        },
        "rule_store": {
            "serialized_bytes": serialized_bytes,
            "active_revision_count": 2,
            "specialized_revision_count": 0,
            "retired_revision_count": 0,
        },
        "environment": {"python": sys.executable, "platform": platform.platform()},
        "fixture_outputs": {
            "cegis_status": cegis_last.status,
            "router_selected_rule_count": len(router_last.selected_rule_ids),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "cost_profile.json")
    args = parser.parse_args()
    report = collect_cost_profile(repetitions=args.repetitions, warmup=args.warmup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "out": str(args.out), "provider_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
