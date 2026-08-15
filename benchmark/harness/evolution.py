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
        results: []                     # optional pre-recorded per-task result dicts (task_id, utility_on, utility_off, task_score_on, task_score_off)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from . import conditions, miniyaml
from .evolution_ledger import EvolutionDecisionLedger
from core import governance
from benchmark.families import poisoning_transformation, transformation
from benchmark.formal.aggregate import RegretStep, evolution_regret
from core.models import TaskContext
from benchmark.formal.condition_adapter import FormalConditionAdapter
from benchmark.families import EpisodeEnvironmentState, FamilyEnvironment

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
    """Import build_manifest from core scripts/run_rule_replay.py by path (R2 workaround)."""
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
    target = store / rel_dir
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{record_id}.json").write_text(
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
    mechanism = "compile" if "COMPILE" in str(raw.get("task_id", "")) else "runtime"
    context = TaskContext(domain="runtime", workload={"workload": {"mechanism": mechanism}}, hardware={}, software={}, evidence={}, token_budget=4096)
    environment = FamilyEnvironment(str(phase.get("family_id", "compile")))
    deployed_ids = FormalConditionAdapter(condition, store, token_budget=context.token_budget).propose_interventions(context)
    state = environment_state or EpisodeEnvironmentState()
    deployed_outcome = environment.evaluate(context.workload, deployed_ids, state)
    oracle_outcome = environment.oracle(context.workload, state)
    baseline_outcome = environment.evaluate(context.workload, (), state)
    source = "drift" if phase.get("name") == "drift" else "poison" if phase.get("name") == "misleading_experience" else "recovery" if phase.get("name") == "recovery" else "acquisition" if phase.get("name") == "acquisition" else "negative_transfer"
    return {**raw, "task_id": str(raw.get("task_id", f"phase-{phase.get('index', 0)}")), "utility_on": deployed_outcome.utility, "utility_off": baseline_outcome.utility, "task_score_on": deployed_outcome.utility, "task_score_off": baseline_outcome.utility, "delta": deployed_outcome.utility - baseline_outcome.utility, "noise_floor": 0.05, "reused": bool(deployed_ids), "oracle_bundle": list(deployed_outcome.oracle_bundle), "deployed_bundle": deployed_ids, "oracle_utility": oracle_outcome.utility, "deployed_utility": deployed_outcome.utility, "scientific_gates": dict(deployed_outcome.scientific_gates), "experiment_cost": 1.0, "failure_source": source}


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
        cases = candidate.get("cases") or []
        if not cases:
            continue
        payload = {
            "rule_id": candidate.get("id", f"rule-{index}"),
            "cases": cases,
            "epsilon": float(candidate.get("epsilon", 0.0)),
            "p_min": float(candidate.get("p_min", 0.8)),
            "delta": float(candidate.get("delta", 0.05)),
        }
        replay_dir = store / "evolution" / "maintenance_reports"
        replay_dir.mkdir(parents=True, exist_ok=True)
        case_path = replay_dir / f"{payload['rule_id']}.cases.json"
        manifest_path = replay_dir / f"{payload['rule_id']}.replay.json"
        case_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest = build_manifest(payload, case_path, manifest_path, "benchmark-harness")
        # Store-relative paths are required by the independent evolution
        # validator; absolute host paths are not portable evidence.
        manifest["case_bundle_path"] = str(case_path.relative_to(store))
        manifest["command"] = "python scripts/run_rule_replay.py " + str(case_path.relative_to(store)) + " " + str(manifest_path.relative_to(store))
        import hashlib

        body = {key: value for key, value in manifest.items() if key != "attestation"}
        manifest["attestation"] = {
            "algorithm": "sha256",
            "manifest_digest": hashlib.sha256(
                json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rule_id = payload["rule_id"]
        version = int(candidate.get("version", 1))
        replay_digest = f"{manifest['case_bundle_sha256']}:{manifest['result_digest']}"
        if ledger.has_replay(rule_id, version, replay_digest):
            continue
        try:
            ledger.record(rule_id, version, replay_digest, "candidate")
            ledger.record(rule_id, version, replay_digest, "evaluated", float(manifest["result"].get("mean_effect", 0.0)))
        except ValueError:
            continue
        decision = governance.apply_promotion(
            store,
            candidate,
            manifest,
            replay_path=str(manifest_path.relative_to(store)),
        )
        if decision.allowed:
            ledger.record(rule_id, version, replay_digest, "promoted")
            promoted.append(payload["rule_id"])
        else:
            ledger.record(rule_id, version, replay_digest, "rejected")
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
    ledger = EvolutionDecisionLedger()
    drift_start: int | None = None

    for phase in episode["phases"]:
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
        for record in phase.get("results") or []:
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
            for path in sorted(candidates_dir.glob("*.json")):
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("cases") and record.get("status") == "candidate":
                    candidates.append(record)
            promoted_total.extend(promote_via_replay(store, candidates, core_repo, out_dir, ledger))

    rules_dir = store / "rules"
    canonical_rules = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(rules_dir.glob("*.json"))
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
        for app in phase.get("results") or []
        if _has_poison_label(app) and float(app.get("delta", 0.0)) < 0.0
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
