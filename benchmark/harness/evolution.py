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
imported by path. The core CLI has a NameError bug (INTEGRATION_REQUIREMENTS.md
R2) so the CLI is never invoked; core files are never modified.

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
from pathlib import Path
from typing import Any

from . import conditions, miniyaml

METRIC_NAMES = (
    "transfer_gain",
    "negative_transfer_rate",
    "rule_precision",
    "library_growth",
    "utility_per_rule",
    "conflict_rate",
    "drift_recovery_latency",
    "poisoning_survival_rate",
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


def _apply_phase_injections(store: Path, condition: str, phase: dict[str, Any]) -> dict[str, list[str]]:
    """Drop a phase's experiences/poisons into the store per the condition policy.

    C (append-only): everything lands in experience/inbox/ and is injection-eligible.
    D (governed):    experiences land in evolution/candidates/; only replay-passed
                     rules are promoted to canonical rules/ + registry.
    """
    written = {"experiences": [], "poisons": []}
    rel = "experience/inbox" if condition == "C" else "evolution/candidates"
    for kind, key in (("experiences", "inject_experiences"), ("poisons", "inject_poisons")):
        for position, record in enumerate(phase.get(key) or []):
            record = dict(record, poisoned=(kind == "poisons"))
            record_id = _write_record(store, rel, record, f"{kind[:-1]}-{phase.get('index', 0)}-{position}")
            written[kind].append(record_id)
    return written


def _promote_via_replay(
    store: Path,
    candidate_results: list[dict[str, Any]],
    core_repo: Path,
    out_dir: Path,
) -> list[str]:
    """Replay-grounded promotion (D): build case bundles from measured paired
    runs, attest via the core build_manifest, promote passing rules to canonical."""
    build_manifest = _import_core_replay_build_manifest(core_repo)
    promoted: list[str] = []
    registry_path = store / "registry" / "rules.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {"schema_version": 1, "rules": []}
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
        manifest_path = out_dir / "replay_manifests" / f"{payload['rule_id']}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(payload, manifest_path, manifest_path, "benchmark-harness")
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if manifest["outcome"] == "passed" and not candidate.get("poisoned", False):
            promoted.append(payload["rule_id"])
            _write_record(store, "rules", {"rule_id": payload["rule_id"], "status": "canonical"}, payload["rule_id"])
            registry["rules"].append({"rule_id": payload["rule_id"], "status": "canonical"})
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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


def negative_transfer_rate(applications: list[dict[str, Any]], noise_floor: float = 0.0) -> float | None:
    """Fraction of rule applications regressing the paired control beyond the floor."""
    if not applications:
        return None
    regressions = sum(
        1
        for app in applications
        if float(app.get("delta", 0.0)) < -abs(noise_floor)
    )
    return regressions / len(applications)


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
) -> dict[str, Any]:
    """Run one episode under condition C or D; write episode_result.json + attestation.

    The prototype harness drives store mechanics and metric computation. Per-task
    agent execution is the outer driver's job; phases may carry pre-recorded
    ``results`` entries (see module docstring) which this runner aggregates.
    """
    condition = condition.upper()
    if condition not in ("C", "D"):
        raise ValueError(f"episodes run under conditions C or D, got {condition!r}")
    episode = load_episode(episode_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    core_repo = Path(core_repo) if core_repo else _core_repo_root()
    snapshot_dir = Path(snapshot_dir) if snapshot_dir else core_repo

    store = out_dir / "store"
    manifest = conditions.materialize_condition(condition, snapshot_dir, store)

    poison_ids: list[str] = []
    paired_results: list[dict[str, Any]] = []
    applications: list[dict[str, Any]] = []
    utility_series: list[float] = []
    admitted = 0
    promoted_total: list[str] = []
    drift_start: int | None = None

    for phase in episode["phases"]:
        written = _apply_phase_injections(store, condition, phase)
        poison_ids.extend(written["poisons"])
        for record in phase.get("results") or []:
            paired_results.append(record)
            if "delta" in record:
                applications.append(record)
        phase_utilities = [
            float(r["utility_on"]) for r in (phase.get("results") or []) if "utility_on" in r
        ]
        utility_series.append(sum(phase_utilities) / len(phase_utilities) if phase_utilities else 0.0)
        if phase.get("name") == "drift":
            drift_start = len(utility_series) - 1

        if condition == "D":
            candidates = []
            candidates_dir = store / "evolution" / "candidates"
            for path in sorted(candidates_dir.glob("*.json")):
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("cases"):
                    candidates.append(record)
                    admitted += 1
            promoted_total.extend(_promote_via_replay(store, candidates, core_repo, out_dir))

    rules_dir = store / "rules"
    canonical_rules = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(rules_dir.glob("*.json"))
    ] if rules_dir.is_dir() else []
    canonical_ids = [str(rule.get("rule_id")) for rule in canonical_rules]
    n = len(canonical_rules)
    conflicts_dir = store / "evolution" / "conflicts"
    conflicting_pairs = len(list(conflicts_dir.glob("*.json"))) if conflicts_dir.is_dir() else 0

    gain = transfer_gain(paired_results)
    growth = library_growth(canonical_rules)
    poison_regressions = sum(
        1 for app in applications if app.get("poisoned") and float(app.get("delta", 0.0)) < 0.0
    )
    metrics = {
        "transfer_gain": gain,
        "negative_transfer_rate": negative_transfer_rate(applications),
        "rule_precision": rule_precision(admitted, len(promoted_total)) if condition == "D" else None,
        "library_growth": growth,
        "utility_per_rule": utility_per_rule(gain, n),
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
    }

    result = {
        "schema_version": 1,
        "episode_id": episode["episode_id"],
        "condition": condition,
        "metrics": metrics,
        "raw": {
            "paired_results": paired_results,
            "poison_ids": poison_ids,
            "promoted_rules": promoted_total,
            "canonical_rule_ids": canonical_ids,
            "utility_series": utility_series,
        },
        "store_manifest": manifest,
    }
    (out_dir / "episode_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    return result
