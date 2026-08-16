#!/usr/bin/env python3
"""Validate the ACRE-v0 grammar and deterministic BoundaryBench pilot."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.boundary.families import family_cases, run_boundary_family
from benchmark.interaction.acquisition_bench import run_acquisition_benchmark
from benchmark.interaction.factorial_bench import run_factorial_benchmark, run_higher_order_benchmark, run_interaction_power_curve
from benchmark.interaction.router_bench import run_router_benchmark
from benchmark.families import family_predicate_grammar
from core.acre.predicates import PredicateGrammar, SYNTHESIZER_VERSION


def validate_method_ownership(root: Path) -> list[str]:
    """Keep method semantics in core; benchmark modules only build/evaluate fixtures."""
    errors: list[str] = []
    boundary_root = root / "benchmark" / "boundary"
    interaction_root = root / "benchmark" / "interaction"
    if (boundary_root / "cegis.py").exists():
        errors.append("benchmark/boundary/cegis.py must not exist; CEGIS is core-owned")
    forbidden_classes = {"StatisticalCEGIS", "ConservativeCausalRouter", "FactorialEngine", "AcreEngine"}
    forbidden_names = {"RuleCandidate", "InteractionEvidence", "EvolutionDecision"}
    for directory in (boundary_root, interaction_root, root / "scripts"):
        for path in directory.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError) as exc:
                errors.append(f"ownership parse failure {path}: {exc}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name in forbidden_classes | forbidden_names:
                    errors.append(f"benchmark method implementation {node.name} in {path}")
    return errors


def validate(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_method_ownership(root))
    for family in ("compile", "graph_cache", "h2d_pipeline", "checkpoint", "scalar_sync"):
        try:
            grammar = PredicateGrammar.from_dict(family_predicate_grammar(family))
        except (KeyError, ValueError) as exc:
            errors.append(f"{family}: invalid FamilySpec predicate grammar: {exc}")
            continue
        if grammar.max_depth > 3 or grammar.max_literals > 4:
            errors.append(f"{family}: grammar bounds exceed ACRE-v0 limits")
    # Canonical families are the production BoundaryBench views.  The two
    # historical names remain covered by benchmark/boundary/test_cegis.py as
    # compatibility aliases, not as a second source of workload semantics.
    for family in ("compile", "graph_cache", "h2d_pipeline", "checkpoint", "scalar_sync"):
        pools = family_cases(family)
        seen: set[str] = set()
        for pool_name, pool in pools.items():
            for item in pool:
                if item.observation_id in seen:
                    errors.append(f"{family}: duplicate observation id {item.observation_id}")
                seen.add(item.observation_id)
            if not pool:
                errors.append(f"{family}: empty {pool_name}")
        first = run_boundary_family(family)
        second = run_boundary_family(family)
        if first != second:
            errors.append(f"{family}: synthesis is not deterministic")
        if first["status"] not in {"identified", "underidentified"} or first["sealed_errors"]:
            errors.append(f"{family}: boundary synthesis failed: {first}")
        if first["result"]["synthesizer_version"] != SYNTHESIZER_VERSION:
            errors.append(f"{family}: missing synthesizer provenance")
        generated = family_cases(family, surface_count=100)
        if sum(len(pool) for pool in generated.values()) != 100 or any(not generated[name] for name in generated):
            errors.append(f"{family}: parameterized surface generation is incomplete")
    factorial = run_factorial_benchmark()
    expected_kinds = {"confirmed_synergy", "confirmed_antagonism", "confirmed_independence", "prerequisite_a_to_b"}
    if set(factorial["classifications"].values()) != expected_kinds:
        errors.append("factorial: interaction classes were not recovered")
    if float(factorial["coverage"]) < 0.93:
        errors.append("factorial: confidence interval coverage below pilot threshold")
    power = run_interaction_power_curve(blocks=(8, 16, 32, 64), repetitions=2)
    if any(abs(float(row["realized_gamma"]) - float(row["target_gamma"])) > 1e-12 for row in power["results"]):
        errors.append("interaction: power-curve target contrast was altered during construction")
    if len({float(row["target_gamma"]) for row in power["results"]}) < 4:
        errors.append("interaction: power curve has no effect-strength ladder")
    higher = run_higher_order_benchmark(count=5, blocks=(8, 16, 32, 64))
    if not all("raw_residual" in row and "normalized_residual" in row for row in higher["results"]):
        errors.append("higher-order: raw and normalized residuals are not recorded")
    acquisition = run_acquisition_benchmark()
    if set(acquisition["cost_to_target"]) != {"random", "uncertainty-only", "decision-aware"}:
        errors.append("acquisition: missing policy comparison")
    if any(cost is None for cost in acquisition["cost_to_target"].values()):
        errors.append("acquisition: a policy did not reach the declared target error")
    router = run_router_benchmark()
    expected_variants = {"current_governed_D", "D_plus_CEGIS", "D_plus_causal_interaction", "full_ACRE"}
    if set(router) != expected_variants:
        errors.append("router: missing ACRE comparison variant")
    if router["full_ACRE"]["objective"] <= router["current_governed_D"]["objective"]:
        errors.append("router: full ACRE did not improve the conservative objective")
    return errors


def main() -> int:
    errors = validate()
    print(json.dumps({"acre_version": "v0", "errors": errors}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
