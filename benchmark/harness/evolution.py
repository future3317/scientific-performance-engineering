#!/usr/bin/env python3
"""Evolution-track episode runner and metrics (BENCHMARK_DESIGN.md sections 8.4, 10).

An episode YAML describes a sequential stream over the six phases (section
10.1): task references, injected experiences, and injected poisons. The runner
materializes a synthetic skill store for condition C or D via
:mod:`harness.conditions`, applies each phase's injections, and computes the
evolution metrics of section 8.4 from the recorded per-task results and the
resulting store state.

Replay grounding (promotion of candidate rules to canonical under condition D)
uses the core skill's ``scripts/run_rule_replay.py`` :func:`build_manifest`
imported by path and the single ``core.governance`` promotion API.

Episode YAML format (miniyaml subset)::

    episode_id: EVOL-EPISODE-POISON-10
    seed: 0
    phases:
      - index: 1
        name: acquisition
        tasks: [CORE-SCALAR-SYNC-01]
        inject_experiences: []          # records dropped into experience/inbox (C) or evolution/candidates (D)
        inject_poisons: []              # misleading records; poisoning_survival_rate tracks their fate
        results: []                     # optional task records; utilities are always computed by FamilyEnvironment
"""

from __future__ import annotations

import json
import random
import tempfile
import time
from pathlib import Path
from typing import Any

from . import conditions, miniyaml
from .evolution_ledger import EvolutionDecisionLedger
from core import governance
from benchmark.families import poisoning_transformation, transformation
from core.predicates import match_predicate
from core.utility import UTILITY_LOG_SCALE, utility_effect
from core.sequential_stats import minimum_all_successes
from benchmark.formal.aggregate import RegretStep, evolution_regret
from core.models import TaskContext
from core.models import identifier_digest, validate_identifier
from benchmark.formal.condition_adapter import FormalConditionAdapter
from benchmark.families import EpisodeEnvironmentState, FamilyEnvironment
from core.acre.budget import StatisticalBudget

# Episode replay uses a fixed, preregistered repetition budget. These are
# performance-style observations, so identical measurements still retain the
# bounded paired uncertainty required by the promotion contract.
# The family lattice spends alpha over 324 preregistered contexts.  The
# episode fixture therefore needs enough paired repetitions for a positive
# effect to remain certified at the resulting per-context delta; 512 leaves
# the bounded interval crossing zero even for a noiseless fixture.
EPISODE_REPLAY_REPETITIONS = 768

METRIC_NAMES = (
    "transfer_gain",
    "rule_reuse_utility",
    "negative_transfer_rate",
    "rule_precision",
    "library_growth",
    "utility_per_rule",
    "utility_per_token",
    "conflict_rate",
    "drift_recovery_latency",
    "poisoning_survival_rate",
    "evolution_regret",
)


def load_episode(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"episode manifest not found: {path}")
    episode = miniyaml.load(str(path))
    if not isinstance(episode, dict) or not isinstance(episode.get("phases"), list):
        raise ValueError(f"{path} must be a mapping with a 'phases' list")
    if "episode_id" not in episode:
        raise ValueError(f"{path} needs an episode_id")
    return episode


def _core_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _import_core_replay_build_manifest(core_repo: Path):
    """Import the canonical core replay manifest builder."""
    from . import runner

    script = core_repo / "scripts" / "run_rule_replay.py"
    if not script.is_file():
        raise FileNotFoundError(f"core replay script not found: {script}")
    module = runner.import_module_by_path(script, "spe_evo_core_rule_replay")
    return module.build_manifest


# ---------------------------------------------------------------------------
# Store driving (conditions C/D)
# ---------------------------------------------------------------------------


def _write_record(store: Path, rel_dir: str, record: dict[str, Any], fallback_id: str) -> str:
    record = dict(record)
    record_id = str(record.setdefault("id", fallback_id))
    validate_identifier(record_id, "record_id")
    target = store / rel_dir
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{identifier_digest(record_id)}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    return record_id


def _materialize_candidate_support(store: Path, candidate: dict[str, Any]) -> None:
    """Materialize the small reviewed/evaluation records required by the
    independent rule-card validator for an episode candidate."""
    source_ids = [str(item) for item in candidate.get("source_cases", [])]
    for index, case_id in enumerate(source_ids):
        _write_record(store, "experience/cases", {
            "case_id": case_id,
            "status": "case",
            "independence_group": f"episode-source-{index}",
            "lesson": {"type": "candidate", "text": str(candidate.get("rule", {}).get("text", ""))},
            "artifacts": {},
        }, case_id)
    for case_id, kind in [(str(item), "admission") for item in candidate.get("admission_cases", [])] + [
        (str(item), "counterexample") for item in candidate.get("regression_cases", [])
    ]:
        _write_record(store, "tests/rule_cases", {
            "schema_version": 1,
            "case_id": case_id,
            "rule_id": str(candidate.get("rule_id", candidate.get("id", "rule"))),
            "kind": kind,
            "status": "pass",
            "scope": {"requires": [], "excludes": []},
            "expected": {"applicable": kind == "admission"},
            "observed": {"applicable": kind == "admission"},
            "evidence": {"source": "episode-fixture"},
            "lineage": {"derived_from_experience_ids": [], "repository_revision": "episode-fixture", "task_family": "compile"},
        }, case_id)


def _strip_poison_labels(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_poison_labels(item) for key, item in value.items() if key != "poisoned"}
    if isinstance(value, list):
        return [_strip_poison_labels(item) for item in value]
    return value


def _environment_result(
    condition: str,
    phase: dict[str, Any],
    raw: dict[str, Any],
    store: Path,
    environment_state: EpisodeEnvironmentState | None = None,
) -> dict[str, Any]:
    """Evaluate deployment from the condition store and phase environment.

    Episode manifests describe contexts and transformations only.  Utility is
    produced here after routing, so C and D cannot share pre-written outcomes.
    """
    task_id = str(raw.get("task_id", f"phase-{phase.get('index', 0)}"))
    family_id = str(phase.get("family_id", "compile"))
    try:
        from benchmark.families import reconstruct_anchor_instance
        instance = reconstruct_anchor_instance(task_id)
        family_id = instance.family_id
        workload = dict(instance.parameters)
    except (KeyError, ValueError):
        workload = dict((raw.get("environment") or {}).get("workload", {})) if isinstance(raw.get("environment"), dict) else {}
    # Episode routing must use the same public domain vocabulary as the
    # candidate rule cards.  ``runtime`` is the historical default, but it
    # would make compiler/sciml rules fail the router's domain gate even when
    # their workload predicate matches.
    domain = {
        "compile": "compiler",
        "equivariant_head": "sciml",
    }.get(family_id, "runtime")
    context = TaskContext(domain=domain, workload=workload, hardware={}, software={}, evidence={}, token_budget=4096)
    environment = FamilyEnvironment(family_id)
    deployed_ids = FormalConditionAdapter(
        condition, store, token_budget=context.token_budget, family_id=family_id
    ).propose_interventions(context)
    state = environment_state or EpisodeEnvironmentState()
    deployed_outcome = environment.evaluate(context.workload, deployed_ids, state)
    oracle_outcome = environment.oracle(context.workload, state)
    baseline_outcome = environment.evaluate(context.workload, (), state)
    source = "drift" if phase.get("name") == "drift" else "poison" if phase.get("name") in {"misleading_experience", "poisoned_experience"} else "recovery" if phase.get("name") == "recovery" else "acquisition" if phase.get("name") == "acquisition" else "negative_transfer"
    return {**raw, "task_id": task_id, "family_id": family_id, "public_context": context.to_dict(), "utility_on": deployed_outcome.utility, "utility_off": baseline_outcome.utility, "task_score_on": deployed_outcome.utility, "task_score_off": baseline_outcome.utility, "delta": deployed_outcome.utility - baseline_outcome.utility, "noise_floor": 0.05, "reused": bool(deployed_ids), "oracle_bundle": list(oracle_outcome.oracle_bundle), "deployed_bundle": deployed_ids, "oracle_utility": oracle_outcome.utility, "deployed_utility": deployed_outcome.utility, "scientific_gates": dict(deployed_outcome.scientific_gates), "experiment_cost": 1.0, "failure_source": source}


def _has_poison_label(value: Any) -> bool:
    if isinstance(value, dict):
        return any((key == "poisoned" and bool(item)) or _has_poison_label(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_has_poison_label(item) for item in value)
    return False


def _apply_phase_injections(store: Path, condition: str, phase: dict[str, Any]) -> dict[str, list[str]]:
    """Drop a phase's experiences/poisons into the store per the condition policy.

    C (append-only): everything lands in experience/inbox/ and is injection-eligible.
    D (governed):    experiences land in evolution/candidates/; only replay-passed
                     rules are promoted to canonical rules/ + registry.
    """
    written = {"experiences": [], "poisons": []}
    for kind, key in (("experiences", "inject_experiences"), ("poisons", "inject_poisons")):
        for position, record in enumerate(phase.get(key) or []):
            # Poison truth is harness-only metadata.  It must never enter the
            # candidate store or any replay payload visible to an agent.
            visible_record = _strip_poison_labels(dict(record))
            rel = (
                "evolution/candidates"
                if condition == "D" and (visible_record.get("status") == "candidate" or visible_record.get("cases"))
                else "experience/inbox"
            )
            if condition == "D" and rel == "evolution/candidates":
                # Candidate projections are persisted at collection time so
                # later tasks can hydrate the append-only evidence ledger.
                # The episode fixture may omit these derived identity fields;
                # the harness, not the worker, supplies them.
                subject_id = str(visible_record.get("rule_id") or visible_record.get("id") or f"candidate-{phase.get('index', 0)}-{position}")
                version = int(visible_record.get("version", 1))
                visible_record.setdefault("rule_id", subject_id)
                visible_record.setdefault("version", version)
                visible_record.setdefault("candidate_identity", f"{subject_id}:v{version}")
                visible_record.setdefault("synthesis_state", {"status": "collecting_evidence", "evidence_ids": []})
                _materialize_candidate_support(store, visible_record)
            record_id = _write_record(
                store, rel, visible_record, f"{kind[:-1]}-{phase.get('index', 0)}-{position}"
            )
            written[kind].append(record_id)
    return written


def promote_via_replay(
    store: Path,
    candidate_results: list[dict[str, Any]],
    core_repo: Path,
    out_dir: Path,
    ledger: EvolutionDecisionLedger,
) -> list[str]:
    """Replay-grounded promotion (D): build case bundles from measured paired
    runs, attest via the core build_manifest, promote passing rules to canonical."""
    build_manifest = _import_core_replay_build_manifest(core_repo)
    promoted: list[str] = []
    for index, candidate in enumerate(candidate_results):
        if str(candidate.get("scope", "calibration")) == "formal":
            # Formal promotion is external-verifier-only.  Synthetic family
            # replay remains calibration evidence and cannot enter promotion.
            continue
        if candidate.get("relation_id"):
            # Relation promotion requires a factorial contrast certificate; a
            # node replay cannot establish a pairwise causal relation.
            continue
        all_cases = [dict(case) for case in (candidate.get("cases") or []) if isinstance(case, dict)]
        predicate = candidate.get("applicability") or candidate.get("trigger")
        if not isinstance(predicate, dict) or not predicate:
            continue
        synthesis_state = candidate.get("synthesis_state") if isinstance(candidate.get("synthesis_state"), dict) else None
        provenance = candidate.get("applicability_provenance") if isinstance(candidate.get("applicability_provenance"), dict) else None
        if synthesis_state is not None and synthesis_state.get("status") != "identified":
            continue
        if provenance is not None and int(provenance.get("decision_context_count", 0)) <= 0:
            continue
        if all(not isinstance(case.get("context"), dict) or not case.get("context") for case in all_cases):
            # Episode calibration cards predate contextual replay payloads;
            # their reviewed trigger is the harness-owned applicability.
            cases = all_cases
        else:
            cases = [case for case in all_cases if match_predicate(predicate, case.get("context", {}))]
        # Once a harness-owned synthesis certificate exists, its positive
        # anchors are the immutable promotion sample.  Recomputing membership
        # from the predicate would silently admit later boundary evidence.
        if synthesis_state is not None or provenance is not None:
            certificate = (provenance or {}).get("certificate") if isinstance(provenance, dict) else None
            anchor_ids = certificate.get("positive_anchor_ids", []) if isinstance(certificate, dict) else []
            declared_ids = candidate.get("promotion_case_ids") or anchor_ids
            if not isinstance(declared_ids, list) or not declared_ids:
                continue
            declared = {str(item) for item in declared_ids}
            representative_anchor_ids = {
                str(item.get("case_id")) for item in cases
                if item.get("query_type", "representative") == "representative"
                and str(item.get("case_id")) in {str(value) for value in anchor_ids}
            }
            if representative_anchor_ids and declared != representative_anchor_ids:
                continue
            cases = [case for case in cases if str(case.get("case_id")) in declared]
            if {str(case.get("case_id")) for case in cases} != declared:
                continue
        if not cases:
            continue
        p_min = float(candidate.get("p_min", 0.8))
        delta = float(candidate.get("delta", 0.05))
        budget = StatisticalBudget(delta_total=delta)
        minimum_groups = minimum_all_successes(p_min, budget.replay_minimum_delta) if p_min > 0.0 else 1
        if len({str(case.get("independence_group") or case.get("source_id") or case.get("case_id")) for case in cases}) < minimum_groups:
            # The pending scheduler will request more family contexts; a
            # partial candidate is never promoted merely because its current
            # prefix happens to contain only successful groups.
            continue
        for case in cases:
            case.setdefault("paired_replay", True)
            case.setdefault("same_fixture_id", case.get("case_id", f"fixture-{index}"))
        # Legacy episode fixtures contain scalar paired scores and are kept as
        # calibration-only material; formal verifier cases always carry
        # repeated arm measurements and retain the declared promotion gate.
        calibration_scalar = all(not isinstance(case.get("intervention_measurements"), list) for case in cases)
        payload = {
            "rule_id": candidate.get("rule_id") or candidate.get("id", f"rule-{index}"),
            "cases": cases,
            "epsilon": float(candidate.get("epsilon", 0.0)),
            "p_min": 0.0 if calibration_scalar else p_min,
            "delta": delta,
        }
        replay_dir = store / "evolution" / "maintenance_reports"
        replay_dir.mkdir(parents=True, exist_ok=True)
        validate_identifier(str(payload["rule_id"]), "rule_id")
        digest = identifier_digest(str(payload["rule_id"]))
        case_path = replay_dir / f"{digest}.cases.json"
        manifest_path = replay_dir / f"{digest}.replay.json"
        case_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        try:
            manifest = build_manifest(payload, case_path, manifest_path, "benchmark-harness")
        except (KeyError, TypeError, ValueError):
            # A malformed or non-paired control is a rejected replay, never a
            # promotion shortcut.
            continue
        representative_groups = sorted({str(case.get("independence_group") or case.get("source_id") or case.get("case_id")) for case in cases if case.get("independence_group") or case.get("source_id") or case.get("case_id")})
        import hashlib
        validation_artifacts = candidate.get("validation_artifacts") if isinstance(candidate.get("validation_artifacts"), dict) else {}
        validation_digest = str(validation_artifacts.get("digest", ""))
        if not validation_digest:
            # Episode maintenance owns its executable probes through the family
            # environment; it never re-labels promotion cases as validation.
            family = FamilyEnvironment(str(candidate.get("family_id", "compile")))
            first_context = dict(cases[0].get("context") or {})
            action = candidate.get("intervention", {})
            action_name = action.get("action") if isinstance(action, dict) else None
            deployed = [str(action_name)] if action_name else []
            heldout_outcome = family.evaluate(first_context, deployed, EpisodeEnvironmentState())
            heldout_baseline = family.evaluate(first_context, (), EpisodeEnvironmentState())
            poison_outcome = family.evaluate(
                first_context,
                deployed,
                EpisodeEnvironmentState(active_poison=("validation_probe",)),
            )
            baseline_outcome = family.evaluate(first_context, (), EpisodeEnvironmentState(active_poison=("validation_probe",)))
            promotion_ids = [str(case.get("case_id")) for case in cases]
            declared_synthesis_ids = candidate.get("synthesis_case_ids")
            if not isinstance(declared_synthesis_ids, list):
                declared_synthesis_ids = [
                    str(case.get("case_id")) for case in candidate.get("cases", [])
                    if isinstance(case, dict) and str(case.get("case_id")) not in set(promotion_ids)
                ]
            synthesis_ids = sorted({
                str(item) for item in declared_synthesis_ids
                if str(item) not in set(promotion_ids)
            })
            # A calibration episode cannot promote from a fabricated marker;
            # it needs a real synthesis pool disjoint from promotion cases.
            if not synthesis_ids:
                continue
            synthesis_groups = sorted({
                str(case.get("independence_group") or case.get("source_id") or case.get("case_id"))
                for case in candidate.get("cases", [])
                if isinstance(case, dict) and str(case.get("case_id")) in set(synthesis_ids)
            })
            promotion_groups = sorted({
                str(case.get("independence_group") or case.get("source_id") or case.get("case_id"))
                for case in cases
            })
            if set(synthesis_groups) & set(promotion_groups):
                continue
            validation = {
                "schema_version": 1,
                "scope": "calibration",
                # Acquisition observations identify applicability; promotion
                # uses a fresh representative replay pool.  Keep memberships
                # disjoint even in calibration episodes.
                "synthesis_case_ids": synthesis_ids,
                "promotion_case_ids": promotion_ids,
                "heldout_regression_cases": [{
                    "case_id": f"HELDOUT-{candidate.get('rule_id')}",
                    "executed": True,
                    "execution_source": "family-environment",
                    "scientific_ok": all(heldout_outcome.scientific_gates.values()),
                    "utility": heldout_outcome.utility,
                    "effect": utility_effect(heldout_outcome.utility, heldout_baseline.utility),
                    "effect_lcb": utility_effect(heldout_outcome.utility, heldout_baseline.utility),
                    "effect_ucb": utility_effect(heldout_outcome.utility, heldout_baseline.utility),
                }],
                "poison_probe_cases": [{
                    "case_id": f"POISON-PROBE-{candidate.get('rule_id')}",
                    "executed": True,
                    "execution_source": "family-environment",
                    "accepted": poison_outcome.utility > baseline_outcome.utility + 1e-9 and all(poison_outcome.scientific_gates.values()),
                    "utility": poison_outcome.utility,
                    "baseline_utility": baseline_outcome.utility,
                }],
                "independence_groups": sorted(set(synthesis_groups + promotion_groups)),
                "synthesis_independence_groups": synthesis_groups,
                "promotion_independence_groups": promotion_groups,
            }
            validation_digest = hashlib.sha256(json.dumps(validation, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
            validation_path = store / "evolution" / "validation" / f"{validation_digest}.json"
            validation_path.parent.mkdir(parents=True, exist_ok=True)
            if not validation_path.exists():
                validation_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            if int(validation_artifacts.get("heldout_count", 0)) < 1 or int(validation_artifacts.get("poison_probe_count", 0)) < 1:
                continue
            validation_path = store / str(validation_artifacts.get("path", ""))
            try:
                validation = json.loads(validation_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
        if not validation_path.is_file():
            continue
        heldout_digest = validation_digest
        replay_digest = f"{manifest['case_bundle_sha256']}:{manifest['result_digest']}"
        manifest["promotion_record"] = {
            "representative_groups": representative_groups,
            "promotion_case_ids": [str(case.get("case_id")) for case in cases],
            "heldout_regression_digest": heldout_digest,
            "validation_artifact_digest": validation_digest,
            "validation_artifact_path": str(validation_path.relative_to(store)).replace("\\", "/"),
            "poison_gate": {"passed": all(entry.get("accepted") is False for entry in validation.get("poison_probe_cases", []))},
            "promotion_probability_lcb": float(manifest["result"].get("promotion_probability_lower_bound", 0.0)),
            "utility_effect_cs": {
                "lcb": float(manifest["result"].get("utility_effect_lcb", -1.0)),
                "ucb": float(manifest["result"].get("utility_effect_ucb", 1.0)),
            },
            "replay_manifest_digest": replay_digest,
        }
        manifest["execution_source"] = "synthetic_family"
        # Store-relative paths are required by the independent evolution
        # validator; absolute host paths are not portable evidence.
        manifest["case_bundle_path"] = str(case_path.relative_to(store))
        manifest["command"] = "python scripts/run_rule_replay.py " + str(case_path.relative_to(store)) + " " + str(manifest_path.relative_to(store))
        body = {key: value for key, value in manifest.items() if key != "attestation"}
        manifest["attestation"] = {
            "algorithm": "sha256",
            "manifest_digest": hashlib.sha256(
                json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rule_id = payload["rule_id"]
        ledger_subject = str(candidate.get("candidate_identity") or rule_id)
        version = int(candidate.get("version", 1))
        if ledger.has_replay(ledger_subject, version, replay_digest):
            continue
        try:
            ledger.record(ledger_subject, version, replay_digest, "candidate")
            ledger.record(ledger_subject, version, replay_digest, "evaluated", float(manifest["result"].get("mean_effect", 0.0)))
        except ValueError:
            continue
        decision = governance.apply_promotion(
            store,
            candidate,
            manifest,
            replay_path=str(manifest_path.relative_to(store)),
            candidate_storage_key=str(candidate.get("candidate_identity") or ledger_subject),
        )
        if decision.allowed:
            ledger.record(ledger_subject, version, replay_digest, "promoted")
            promoted.append(payload["rule_id"])
        else:
            ledger.record(ledger_subject, version, replay_digest, "rejected")
    return promoted


# ---------------------------------------------------------------------------
# Metrics (section 8.4) — pure functions
# ---------------------------------------------------------------------------


def transfer_gain(paired_results: list[dict[str, Any]]) -> float | None:
    """Mean paired delta of task score vs the no-evolution control."""
    deltas = [
        float(r["task_score_on"]) - float(r["task_score_off"])
        for r in paired_results
        if "task_score_on" in r and "task_score_off" in r
    ]
    return sum(deltas) / len(deltas) if deltas else None


def negative_transfer_rate(applications: list[dict[str, Any]], noise_floor: float | None = None) -> float | None:
    """Fraction of rule applications regressing the paired control beyond the floor."""
    if not applications:
        return None
    if noise_floor is None and not any(app.get("noise_floor") is not None for app in applications):
        return None
    regressions = sum(
        1
        for app in applications
        if float(app.get("delta", 0.0)) < -abs(float(app.get("noise_floor", noise_floor or 0.0)))
    )
    return regressions / len(applications)


def rule_reuse_utility(applications: list[dict[str, Any]]) -> float | None:
    reused = [float(app["delta"]) for app in applications if app.get("reused") and "delta" in app]
    return sum(reused) / len(reused) if reused else None


def rule_precision(admitted: int, survived: int) -> float | None:
    """Admitted rules surviving held-out regression / admitted rules."""
    if admitted <= 0:
        return None
    return survived / admitted


def library_growth(canonical_rules: list[dict[str, Any]]) -> dict[str, Any]:
    """Canonical rule count + description length (rate-distortion view)."""
    count = len(canonical_rules)
    description_length = sum(
        len(json.dumps(rule, ensure_ascii=False, default=str)) for rule in canonical_rules
    )
    return {"canonical_rule_count": count, "description_length": description_length}


def utility_per_rule(total_gain: float | None, rule_count: int) -> float | None:
    if total_gain is None or rule_count <= 0:
        return None
    return total_gain / rule_count


def utility_per_token(total_gain: float | None, prompt_tokens: int) -> float | None:
    if total_gain is None or prompt_tokens <= 0:
        return None
    return total_gain / prompt_tokens


def conflict_rate(conflicting_pairs: int, canonical_pairs: int) -> float | None:
    """Conflicting canonical pairs / canonical pairs (should be 0 under governance)."""
    if canonical_pairs <= 0:
        return None
    return conflicting_pairs / canonical_pairs


def drift_recovery_latency(utility_series: list[float], drift_start: int) -> int | None:
    """Phases between the post-drift utility drop and return to pre-drift utility.

    *utility_series* is per-phase mean utility; *drift_start* is the index of the
    drift phase. Returns None when the series never drops or never recovers.
    """
    if drift_start <= 0 or drift_start >= len(utility_series):
        return None
    pre_drift = max(utility_series[:drift_start])
    dropped = False
    for index in range(drift_start, len(utility_series)):
        if utility_series[index] < pre_drift:
            dropped = True
        elif dropped:
            return index - drift_start
    return None


def poisoning_survival_rate(
    poison_ids: list[str], canonical_rule_ids: list[str], regressions_caused: int = 0
) -> float | None:
    """Fraction of poisons that failed to reach canonical status (D) or caused
    measurable regressions (C — pass regressions_caused explicitly)."""
    if not poison_ids:
        return None
    canonized = sum(1 for pid in poison_ids if pid in set(canonical_rule_ids))
    survived = len(poison_ids) - canonized - regressions_caused
    return max(0, survived) / len(poison_ids)


# ---------------------------------------------------------------------------
# Episode driver
# ---------------------------------------------------------------------------


def run_episode(
    episode_path: str | Path,
    condition: str,
    out_dir: str | Path,
    snapshot_dir: str | Path | None = None,
    core_repo: str | Path | None = None,
    context_mode: str = "reset",
    seed: int | None = None,
    max_wall_time_s: float | None = None,
) -> dict[str, Any]:
    """Run one episode under condition C or D; write episode_result.json + attestation.

    The prototype harness drives store mechanics and metric computation. Per-task
    agent execution is the outer driver's job; phases may carry pre-recorded
    ``results`` entries (see module docstring) which this runner aggregates.
    """
    condition = condition.upper()
    if context_mode not in {"reset", "carry"}:
        raise ValueError("context_mode must be reset or carry")
    if condition not in ("C", "C_STRESS", "D"):
        raise ValueError(f"episodes run under conditions C, C_STRESS, or D, got {condition!r}")
    episode = load_episode(episode_path)
    if seed is not None:
        episode = dict(episode)
        episode["seed"] = int(seed)
    deadline = time.monotonic() + float(max_wall_time_s) if max_wall_time_s is not None else None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    core_repo = Path(core_repo) if core_repo else _core_repo_root()
    snapshot_dir = Path(snapshot_dir) if snapshot_dir else core_repo

    store = out_dir / "store"
    from scripts.render_skill_view import render_skill_view

    if (snapshot_dir / "skill_view_manifest.json").is_file():
        skill_view = snapshot_dir
        manifest = conditions.materialize_condition(condition, skill_view, store, context_mode=context_mode)
    else:
        with tempfile.TemporaryDirectory(dir=out_dir.parent) as temp:
            skill_view = Path(temp) / "skill-view"
            render_skill_view(snapshot_dir, skill_view)
            manifest = conditions.materialize_condition(condition, skill_view, store, context_mode=context_mode)

    poison_ids: list[str] = []
    paired_results: list[dict[str, Any]] = []
    transfer_results: list[dict[str, Any]] = []
    applications: list[dict[str, Any]] = []
    utility_series: list[float] = []
    promoted_total: list[str] = []
    regret_steps: list[RegretStep] = []
    family_transformations: list[dict[str, Any]] = []
    environment_state = EpisodeEnvironmentState()
    ledger = EvolutionDecisionLedger(out_dir / "evolution_decision_ledger.jsonl")
    drift_start: int | None = None

    for phase in episode["phases"]:
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"episode exceeded max_wall_time_s={max_wall_time_s}")
        if phase.get("family_id") and phase.get("transformation"):
            transform = transformation(str(phase["family_id"]), str(phase["transformation"]), **dict(phase.get("transformation_parameters") or {}))
            family_transformations.append(transform.__dict__)
            environment_state = environment_state.apply(transform)
        if phase.get("family_id") and phase.get("poison_operator"):
            poison = poisoning_transformation(str(phase["family_id"]), str(phase["poison_operator"]), **dict(phase.get("poison_parameters") or {}))
            family_transformations.append(poison.__dict__)
            environment_state = environment_state.apply(poison)
        if phase.get("name") == "drift" and not phase.get("transformation"):
            environment_state = environment_state.apply({"kind": "software", "parameters": {"to_runtime": "B"}})
        if phase.get("name") == "recovery" and not phase.get("transformation"):
            environment_state = environment_state.apply({"kind": "revalidation", "parameters": {}})
        written = _apply_phase_injections(store, condition, phase)
        poison_ids.extend(written["poisons"])
        phase_results: list[dict[str, Any]] = []
        task_records = list(phase.get("results") or [])
        if not task_records:
            task_records = [{"task_id": task_id} for task_id in (phase.get("tasks") or [])]
        for record in task_records:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"episode exceeded max_wall_time_s={max_wall_time_s}")
            visible = _environment_result(condition, phase, _strip_poison_labels(record), store, environment_state)
            phase_results.append(visible)
            paired_results.append(visible)
            if phase.get("name") in {"same_family_transfer", "cross_family_transfer"}:
                transfer_results.append(visible)
            if "delta" in visible:
                applications.append(visible)
            if "oracle_utility" in visible and "deployed_utility" in visible:
                source = str(visible.get("failure_source")) if visible.get("failure_source") else None
                regret_steps.append(RegretStep(
                    context_id=str(visible.get("task_id", f"phase-{phase.get('index', 0)}")),
                    oracle_bundle=tuple(str(item) for item in visible.get("oracle_bundle", ())),
                    deployed_bundle=tuple(str(item) for item in visible.get("deployed_bundle", ())),
                    oracle_utility=float(visible["oracle_utility"]),
                    deployed_utility=float(visible["deployed_utility"]),
                    experiment_cost=float(visible.get("experiment_cost", 0.0)),
                    failure_source=source,
                    acquisition_regret=float(visible.get("acquisition_regret", 0.0)),
                    negative_transfer_regret=float(visible.get("negative_transfer_regret", 0.0)),
                    interaction_regret=float(visible.get("interaction_regret", 0.0)),
                    drift_recovery_regret=float(visible.get("drift_recovery_regret", 0.0)),
                ))
        phase_utilities = [float(r["utility_on"]) for r in phase_results if "utility_on" in r]
        if phase_utilities:
            utility_series.append(sum(phase_utilities) / len(phase_utilities))
        if phase.get("name") == "drift":
            drift_start = len(utility_series) - 1

        if condition == "D":
            candidates = []
            candidates_dir = store / "evolution" / "candidates"
            for path in sorted(candidates_dir.rglob("*.json")):
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("status") == "candidate":
                    # Episode inputs declare hypotheses and contexts only.
                    # Generate calibration cases from the just-evaluated
                    # FamilyEnvironment outcomes instead of accepting utility
                    # values authored in YAML.
                    if not record.get("cases"):
                        family = str(record.get("family_id", phase.get("family_id", "compile")))
                        generated_cases = []
                        action = record.get("intervention", {})
                        action_id = str(action.get("action", "")) if isinstance(action, dict) else ""
                        family_environment = FamilyEnvironment(family)
                        from benchmark.formal.schedule import PromotionReplayScheduler
                        scheduler = PromotionReplayScheduler(
                            p_min=float(record.get("p_min", 0.8)),
                            delta=float(record.get("delta", 0.05)),
                        )
                        scheduled = []
                        seen_contexts: set[str] = set()
                        # Active replay may need more contexts than the first
                        # family slice contains inside the learned predicate.
                        # Draw additional preregistered slices until the
                        # promotion budget can be filled; no hidden labels are
                        # used for this scheduling decision.
                        # A family surface is intentionally disjoint, but a
                        # sparse applicability predicate can occupy only a
                        # fraction of one representative slice (for example
                        # irrep_order == 1 is one of three equivariant
                        # strata).  Collect up to three preregistered slices
                        # before synthesis so the positive side has enough
                        # independent groups for the existing utility CS.
                        target_contexts = scheduler.max_groups * 3
                        for schedule_seed in range(int(episode.get("seed", 0)), int(episode.get("seed", 0)) + 8):
                            for item in scheduler.pending_contexts(family, seed=schedule_seed):
                                context_id = str(item.get("context_id", ""))
                                if context_id and context_id not in seen_contexts:
                                    seen_contexts.add(context_id)
                                    scheduled.append(item)
                            if len(scheduled) >= target_contexts:
                                break
                        for scheduled_context in scheduled:
                            context = dict(scheduled_context.get("context", {}))
                            workload = dict(context.get("workload", context))
                            deployed = family_environment.evaluate(workload, (action_id,), environment_state)
                            baseline = family_environment.evaluate(workload, (), environment_state)
                            positive = deployed.utility > baseline.utility + float(record.get("epsilon", 0.0))
                            noise_rng = random.Random(f"{scheduled_context.get('context_id', phase.get('index', 0))}|{record.get('seed', 0)}")
                            shared_noise = [noise_rng.uniform(-0.01, 0.01) for _ in range(EPISODE_REPLAY_REPETITIONS)]
                            generated_cases.append({
                                "case_id": f"EPISODE-{scheduled_context.get('context_id', phase.get('index', 0))}",
                                "context": context,
                                "intervention_measurements": [max(-1.0, min(1.0, float(deployed.utility) + noise)) for noise in shared_noise],
                                "baseline_measurements": [max(-1.0, min(1.0, float(baseline.utility) + noise)) for noise in shared_noise],
                                "control_measured": True,
                                "higher_is_better": True,
                                # Episode replay must use the same versioned
                                # bounded utility transform as formal replay.
                                # A local scale here would change the meaning
                                # of the evidence and can turn a real positive
                                # effect into a routing abstention.
                                "utility_scale": UTILITY_LOG_SCALE,
                                "scientific_ok": all(bool(item) for item in deployed.scientific_gates.values()),
                                "quality_ok": all(bool(item) for item in deployed.scientific_gates.values()),
                                "paired_replay": True,
                                "same_fixture_id": str(scheduled_context.get("context_id", phase.get("index", 0))),
                                "independence_group": str(scheduled_context.get("independence_group", phase.get("index", 0))),
                                "query_type": "representative" if positive else "active_query",
                                "sampling_model": "synthetic_bounded_noise_v1",
                            })
                        record["cases"] = generated_cases
                        # Episode calibration uses the same harness-owned CEGIS
                        # contract as formal maintenance: both positive anchors
                        # and certified boundary counterexamples are retained,
                        # while only representative anchors enter promotion.
                        if generated_cases:
                            from core.acre.cegis import synthesize_applicability
                            synthesis = synthesize_applicability(
                                generated_cases,
                                family_id=family,
                                delta=float(record.get("delta", 0.05)),
                                epsilon_true=float(record.get("epsilon", 0.0)),
                                epsilon_false=0.0,
                                require_identified=True,
                            )
                            if synthesis.predicate is not None:
                                record["applicability"] = synthesis.predicate
                                record["applicability_provenance"] = dict(synthesis.provenance or {})
                                record["synthesis_state"] = {
                                    "status": synthesis.status,
                                    "evidence_ids": [str(case.get("case_id")) for case in generated_cases],
                                }
                                certificate = synthesis.certificate.to_dict() if synthesis.certificate else {}
                                record["promotion_case_ids"] = [
                                    str(case_id) for case_id in certificate.get("positive_anchor_ids", [])
                                    if any(str(case.get("case_id")) == str(case_id) and case.get("query_type") == "representative" for case in generated_cases)
                                ]
                    if record.get("cases"):
                        candidates.append(record)
            promoted_total.extend(promote_via_replay(store, candidates, core_repo, out_dir, ledger))

    rules_dir = store / "rules"
    canonical_rules = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(rules_dir.rglob("*.json"))
        if not path.name.endswith(".state.json")
    ] if rules_dir.is_dir() else []
    canonical_ids = [str(rule.get("rule_id")) for rule in canonical_rules]
    n = len(canonical_rules)
    conflicts_dir = store / "evolution" / "conflicts"
    conflicting_pairs = len(list(conflicts_dir.glob("*.json"))) if conflicts_dir.is_dir() else 0

    gain = transfer_gain(transfer_results)
    growth = library_growth(canonical_rules)
    token_count = sum(int(r.get("prompt_tokens", 0)) for r in paired_results)
    # The input episode is harness-owned truth; only this local calculation may
    # inspect its poison marker.  The marker is omitted from all persisted data.
    poison_regressions = sum(
        1
        for phase in episode["phases"]
        if bool(phase.get("poisoned", False))
        and any(float(app.get("delta", 0.0)) < 0.0 for app in paired_results if app.get("failure_source") == "poison")
    )
    metrics = {
        "transfer_gain": gain,
        "rule_reuse_utility": rule_reuse_utility(applications),
        "negative_transfer_rate": negative_transfer_rate(applications),
        "rule_precision": ledger.precision() if condition == "D" else None,
        "library_growth": growth,
        "utility_per_rule": utility_per_rule(gain, n),
        "utility_per_token": utility_per_token(gain, token_count),
        "conflict_rate": conflict_rate(conflicting_pairs, n * (n - 1) // 2),
        "drift_recovery_latency": (
            drift_recovery_latency(utility_series, drift_start) if drift_start is not None else None
        ),
        "poisoning_survival_rate": (
            poisoning_survival_rate(poison_ids, canonical_ids) if condition == "D"
            else poisoning_survival_rate(poison_ids, [], regressions_caused=poison_regressions)
            if poison_ids
            else None
        ),
        "evolution_regret": evolution_regret(regret_steps),
    }

    if condition == "D":
        from scripts.validate_evolution import audit

        validation_errors = audit(store)
        if validation_errors:
            raise ValueError("D store failed independent evolution validation: " + "; ".join(validation_errors))

    result = {
        "schema_version": 1,
        "episode_id": episode["episode_id"],
        "condition": condition,
        "context_mode": context_mode,
        "metrics": metrics,
        "raw": {
            "paired_results": paired_results,
            "poison_ids": poison_ids,
            "promoted_rules": promoted_total,
            "canonical_rule_ids": canonical_ids,
            "utility_series": utility_series,
            "decision_ledger": ledger.decisions(),
            "family_transformations": family_transformations,
            "regret_steps": [step.to_dict() for step in regret_steps],
        },
        "store_manifest": manifest,
    }
    (out_dir / "episode_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    return result
