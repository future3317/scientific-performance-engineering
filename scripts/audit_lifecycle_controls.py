#!/usr/bin/env python3
"""Run deterministic matched lifecycle controls on registered episode semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.families import FAMILY_SPECS, family_views
from benchmark.families.environment import EpisodeEnvironmentState, FamilyEnvironment


TEMPLATES = {
    "poison": {"family": "graph_cache", "action": "reuse_graph_cache", "change": {"kind": "poison", "parameters": {"operator": "stale_cache"}}},
    "compiler_drift": {"family": "compile", "action": "stabilize_dynamic_guards", "change": {"kind": "software", "parameters": {"to_runtime": "B"}}},
    "specialization": {"family": "equivariant_head", "action": "equivariant_recompute", "change": {"kind": "scientific_regime", "parameters": {"regime": "shifted"}}},
}
POLICIES = ("raw_reuse", "no_reuse", "frozen_certificate", "full_acre")


def _context_for(family: str, action: str, seed: int) -> dict[str, Any]:
    spec = FAMILY_SPECS[family]
    candidates = [item for pool in family_views(family, count=32, seed=seed).values() for item in pool]
    for item in candidates:
        if spec.action_applicable(action, item.parameters):
            return {"workload": dict(item.parameters), "replicate_seed": seed}
    raise ValueError(f"no applicable context for {family}/{action}")


def _phase_states(template: dict[str, Any]) -> list[EpisodeEnvironmentState]:
    base = EpisodeEnvironmentState()
    changed = base.apply(template["change"])
    return [base, base, changed, changed, base, base]


def _decision(policy: str, phase: int, states: list[EpisodeEnvironmentState]) -> tuple[str, str | None, str]:
    if policy == "no_reuse":
        return "abstain", "A", "none"
    if policy in {"raw_reuse", "frozen_certificate"}:
        return "reuse", None, "v1"
    if phase in {2, 3}:
        return "revalidate", "L", "v1-stale"
    return "reuse", None, "v2" if phase >= 4 else "v1"


def run_lifecycle_controls(*, replicates: int = 20) -> dict[str, Any]:
    ledger: list[dict[str, Any]] = []
    for replicate in range(int(replicates)):
        for template_name, template in TEMPLATES.items():
            family = str(template["family"])
            action_id = str(template["action"])
            context = _context_for(family, action_id, replicate)
            environment = FamilyEnvironment(family)
            states = _phase_states(template)
            for policy in POLICIES:
                for phase, state in enumerate(states):
                    action, blocking_clause, revision_id = _decision(policy, phase, states)
                    deployed = (action_id,) if action == "reuse" else ()
                    baseline = environment.evaluate(context, (), state)
                    outcome = environment.evaluate(context, deployed, state)
                    oracle = environment.oracle(context, state)
                    harmful = action == "reuse" and (outcome.utility < baseline.utility or not all(outcome.scientific_gates.values()))
                    ledger.append({
                        "replicate": replicate,
                        "template": template_name,
                        "family": family,
                        "phase": phase,
                        "policy": policy,
                        "action": action,
                        "blocking_clause": blocking_clause,
                        "revision_id": revision_id,
                        "utility": outcome.utility if action == "reuse" else baseline.utility,
                        "baseline_utility": baseline.utility,
                        "oracle_utility": oracle.utility,
                        "scientific_ok": all(outcome.scientific_gates.values()) if action == "reuse" else True,
                        "negative_transfer": harmful,
                        "beneficial_reuse": action == "reuse" and outcome.utility > baseline.utility,
                        "evidence_cost_units": 1 if action == "revalidate" else 0,
                    })
    summary: dict[str, Any] = {}
    for policy in POLICIES:
        selected = [row for row in ledger if row["policy"] == policy]
        reuse = [row for row in selected if row["action"] == "reuse"]
        total_utility = sum(float(row["utility"]) for row in selected)
        total_oracle = sum(float(row["oracle_utility"]) for row in selected)
        summary[policy] = {
            "episodes": int(replicates) * len(TEMPLATES),
            "steps": len(selected),
            "negative_transfer_events": sum(int(row["negative_transfer"]) for row in selected),
            "reuse_coverage": len(reuse) / len(selected) if selected else 0.0,
            "beneficial_reuse_retained": sum(int(row["beneficial_reuse"]) for row in selected),
            "cumulative_utility": total_utility,
            "cumulative_oracle_utility": total_oracle,
            "longitudinal_regret": total_oracle - total_utility,
            "evidence_cost_units": sum(int(row["evidence_cost_units"]) for row in selected),
        }
    return {
        "schema_version": 1,
        "analysis": "provider_free_lifecycle_matched_controls",
        "provider_calls": 0,
        "replicates": int(replicates),
        "templates": list(TEMPLATES),
        "policies": list(POLICIES),
        "cost_definition": "logical revalidation checks; not wall-clock provider cost",
        "ledger": ledger,
        "summary": summary,
        "claim_boundary": "provider-free deterministic episode control evidence; not hosted efficacy",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "lifecycle_matched_controls.json")
    args = parser.parse_args()
    report = run_lifecycle_controls(replicates=args.replicates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "ledger_rows": len(report["ledger"]), "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
