#!/usr/bin/env python3
"""Run or dry-run a complete SPE-EvoBench formal campaign.

Without ``--agent-command`` this writes only a frozen campaign plan and never
claims benchmark results. With a command, the driver gives the agent a fresh
solution workspace for each task and then invokes the immutable verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.harness import conditions, miniyaml, scoring, verifier
from benchmark.harness.evolution_ledger import EvolutionDecisionLedger
from benchmark.harness.fingerprint import capture_fingerprint
from benchmark.formal import aggregate, attest, budget, schedule
from benchmark.formal.condition_adapter import FormalConditionAdapter
from benchmark.harness.evolution import promote_via_replay
from benchmark.harness.evolution_ledger import CandidateEvidenceLedger
from benchmark.families import EpisodeEnvironmentState, FamilyEnvironment
from scripts.render_skill_view import render_skill_view, validate_skill_view_bundle
from core.models import identifier_digest, validate_identifier, ActionSpec, RealizationRecord
from core.sequential_stats import bounded_mean_interval, paired_repetition_interval, minimum_all_successes
from core.utility import UTILITY_LOG_SCALE, practical_effect_threshold, utility_effect
from scripts.run_rule_replay import evaluate_cases


def canonical_public_context(value: dict[str, Any] | None) -> dict[str, Any]:
    """Return the single public routing shape shared by C and D.

    Family generators historically nested parameters below
    ``workload.family_parameters`` while the core predicate DSL addresses
    ``workload.<feature>``.  Flattening that declared public mapping at the
    boundary prevents task-local IDs and hidden mechanism labels from entering
    either retrieval or routing.
    """
    source = value if isinstance(value, dict) else {}
    result = {key: dict(source[key]) if isinstance(source.get(key), dict) else source[key]
              for key in ("domain", "workload", "hardware", "software", "evidence")
              if key in source}
    workload = result.setdefault("workload", {})
    nested = workload.pop("family_parameters", None)
    if isinstance(nested, dict):
        workload = {**nested, **workload}
        result["workload"] = workload
    return result


class InterventionRealizer:
    """Materialize one explicit worker patch as an executable intervention."""

    @staticmethod
    def realize(baseline: Path, destination: Path, proposal: dict[str, Any]) -> Path:
        if not baseline.is_dir():
            raise ValueError("intervention baseline directory is missing")
        if not isinstance(proposal, dict):
            raise ValueError("intervention proposal must be an object")
        intervention = proposal.get("intervention")
        if not isinstance(intervention, dict) or not intervention:
            raise ValueError("intervention proposal needs an explicit patch")
        relative_file = intervention.get("file")
        replacements = intervention.get("replacements")
        if not isinstance(relative_file, str) or not relative_file or not isinstance(replacements, list) or not replacements:
            raise ValueError("intervention patch requires file and replacements")
        relative = Path(relative_file)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("intervention file must be baseline-relative")
        source = (baseline / relative).resolve()
        baseline_root = baseline.resolve()
        if baseline_root not in source.parents or not source.is_file() or source.is_symlink():
            raise ValueError("intervention file must be a regular baseline file")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(baseline, destination)
        target = destination / relative
        content = target.read_text(encoding="utf-8")
        for replacement in replacements:
            if not isinstance(replacement, dict) or not isinstance(replacement.get("old"), str) or not isinstance(replacement.get("new"), str):
                raise ValueError("intervention replacements must contain old and new text")
            old, new = replacement["old"], replacement["new"]
            if content.count(old) != 1:
                raise ValueError("intervention replacement must match exactly once")
            content = content.replace(old, new, 1)
        target.write_text(content, encoding="utf-8")
        return destination

    @staticmethod
    def realize_action(
        baseline: Path,
        destination: Path,
        proposal: dict[str, Any],
        *,
        family_id: str,
        task_id: str,
        context_id: str,
        verifier_digest: str = "unverified",
    ) -> RealizationRecord:
        """Materialize a reusable semantic action and record its realization.

        The source patch is a realization detail.  The candidate identity and
        governance path use the ActionSpec digest; this record links that
        semantic action to the task-local artifact without making source text
        part of the canonical rule meaning.
        """
        from core.acre.actions import action_from_proposal
        action = action_from_proposal(family_id, proposal)
        output = InterventionRealizer.realize(baseline, destination, proposal)
        def digest_tree(root: Path) -> str:
            digest = hashlib.sha256()
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
                digest.update(path.read_bytes())
            return digest.hexdigest()
        record = RealizationRecord(
            action_id=action.action_id,
            task_id=task_id,
            context_id=context_id,
            baseline_digest=digest_tree(baseline),
            patch=dict(proposal.get("intervention") or {}),
            realized_digest=digest_tree(output),
            verifier_digest=verifier_digest,
        )
        (output / "realization_record.json").write_text(
            json.dumps(record.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return record


def sanitize_submission(
    source: Path,
    destination: Path,
    allowed_files: set[str],
    *,
    baseline: Path | None = None,
    max_file_bytes: int = 8 * 1024 * 1024,
    max_total_bytes: int = 32 * 1024 * 1024,
) -> Path:
    """Copy only allowlisted regular files out of a worker namespace."""
    source = Path(source)
    destination = Path(destination)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("worker solution root must be a regular directory")
    normalized_allowlist = {str(Path(item)) for item in allowed_files}
    if not normalized_allowlist:
        raise ValueError("submission allowlist must not be empty")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        relative_text = str(relative)
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"submission contains symlink: {relative_text}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"submission contains non-regular file: {relative_text}")
        if relative_text not in normalized_allowlist:
            if baseline is None:
                raise ValueError(f"submission file is not allowlisted: {relative_text}")
            baseline_path = baseline / relative
            if not baseline_path.is_file() or baseline_path.read_bytes() != path.read_bytes():
                raise ValueError(f"submission changes outside public change surface: {relative_text}")
        if info.st_size > max_file_bytes:
            raise ValueError(f"submission file exceeds size limit: {relative_text}")
        total_bytes += info.st_size
        if total_bytes > max_total_bytes:
            raise ValueError("submission exceeds total size limit")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    source_files = {str(path.relative_to(source)) for path in source.rglob("*") if path.is_file() and not path.is_symlink()}
    required_files = normalized_allowlist
    if baseline is not None:
        required_files = {
            str(path.relative_to(baseline))
            for path in baseline.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
    missing = sorted(required_files - source_files)
    if missing:
        raise ValueError(f"submission is missing allowlisted files: {missing}")
    return destination


def _case_effect_interval(case: dict[str, Any], *, delta: float = 0.05) -> tuple[float, float, float] | None:
    intervention = case.get("intervention_measurements")
    baseline = case.get("baseline_measurements")
    higher_is_better = bool(case.get("higher_is_better", True))
    log_scale = float(case.get("utility_scale", UTILITY_LOG_SCALE))
    if isinstance(intervention, list) and isinstance(baseline, list):
        if not intervention or len(intervention) != len(baseline):
            return None
        effects = [utility_effect(float(on), float(off), higher_is_better=higher_is_better, log_scale=log_scale) for on, off in zip(intervention, baseline)]
        lower, upper = paired_repetition_interval(effects, delta)
        return sum(effects) / len(effects), lower, upper
    try:
        effect = utility_effect(
            float(case["utility_on"]), float(case["utility_off"]),
            higher_is_better=higher_is_better, log_scale=log_scale,
        )
    except (KeyError, TypeError, ValueError):
        return None
    # A scalar score has no sampling uncertainty and cannot certify a boundary
    # anchor or counterexample.  Formal CEGIS consumes paired repetitions only.
    return effect, -1.0, 1.0


def _family_decision_lattice(family_id: str | None) -> list[dict[str, Any]]:
    if not family_id:
        return []
    try:
        from benchmark.families import family_instances
        instances = family_instances(family_id, count=24, seed=0)
    except (KeyError, ValueError):
        return []
    return [{"workload": dict(instance.parameters)} for instance in instances]


def synthesize_applicability(
    cases: list[dict[str, Any]],
    *,
    family_id: str | None = None,
    decision_contexts: list[dict[str, Any]] | None = None,
    delta: float = 0.05,
    epsilon_true: float = 0.0,
    epsilon_false: float = 0.0,
    require_identified: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Derive a worker rule boundary from harness-owned public observations."""
    from core.acre.cegis import BoundaryObservation, StatisticalCEGIS
    from core.acre.predicates import PredicateGrammar

    lattice = list(decision_contexts or _family_decision_lattice(family_id))
    if require_identified and not lattice:
        return None
    context_delta = float(delta) / max(1, len(lattice))
    observations: list[BoundaryObservation] = []
    for index, case in enumerate(cases):
        context = case.get("context") if isinstance(case.get("context"), dict) else {}
        if not context:
            continue
        interval = _case_effect_interval(case, delta=context_delta)
        if interval is None:
            continue
        effect, effect_lower, effect_upper = interval
        observations.append(BoundaryObservation(
            observation_id=str(case.get("case_id", f"case-{index}")),
            context=context,
            effect=effect,
            gate_passed=bool(case.get("scientific_ok", False)) and bool(case.get("quality_ok", True)),
            effect_lower=effect_lower,
            effect_upper=effect_upper,
        ))
    if not observations:
        return None
    from benchmark.families import family_predicate_grammar
    grammar_payload = family_predicate_grammar(family_id) if family_id else {}
    if not grammar_payload:
        grammar_path = Path(__file__).resolve().parents[2] / "assets" / "predicate_grammar.json"
        grammar_payload = json.loads(grammar_path.read_text(encoding="utf-8"))
    observed_paths = set()
    for observation in observations:
        for feature in grammar_payload["features"]:
            value: Any = observation.context
            for part in feature["path"].split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if value is not None:
                observed_paths.add(feature["path"])
    grammar_payload["features"] = [feature for feature in grammar_payload["features"] if feature["path"] in observed_paths]
    grammar_payload["threshold_universe"] = {
        path: values for path, values in grammar_payload.get("threshold_universe", {}).items() if path in observed_paths
    }
    if not grammar_payload["features"]:
        return None
    grammar = PredicateGrammar.from_dict(grammar_payload)
    cegis = StatisticalCEGIS(grammar, epsilon_true=epsilon_true, epsilon_false=epsilon_false)
    positives = [item for item in observations if item.positive_anchor(epsilon_true)]
    negatives = [item for item in observations if item.certified_counterexample(epsilon_false)]
    result = cegis.synthesize(
        positive=positives,
        counterexamples=negatives,
        parent_predicate=None,
        decision_contexts=lattice,
    )
    if result.predicate is None or (require_identified and result.status != "identified"):
        return None
    return result.predicate, {"source": "harness-cegis", **(result.provenance or {}), "status": result.status}


def representative_case_ids(predicate: dict[str, Any], cases: list[dict[str, Any]]) -> list[str]:
    """Return only certified representative cases covered by a learned predicate."""
    from core.predicates import match_predicate

    return sorted({
        str(case["case_id"])
        for case in cases
        if isinstance(case, dict)
        and isinstance(case.get("case_id"), str)
        and isinstance(case.get("context"), dict)
        and match_predicate(predicate, case["context"])
        and (_case_effect_interval(case) is not None)
        and _case_effect_interval(case)[1] > float(case.get("epsilon", 0.0))
        and bool(case.get("scientific_ok", False))
    })


def rewrite_validation_membership(
    store: Path,
    candidate: dict[str, Any],
    *,
    synthesis_case_ids: list[str],
    promotion_case_ids: list[str],
) -> None:
    """Bind validation membership after CEGIS has identified applicability."""
    reference = candidate.get("validation_artifacts")
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise ValueError("candidate validation artifact is missing")
    path = store / str(reference["path"])
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("candidate validation artifact is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("candidate validation artifact must be an object")
    value["synthesis_case_ids"] = sorted(set(str(item) for item in synthesis_case_ids))
    value["promotion_case_ids"] = sorted(set(str(item) for item in promotion_case_ids))
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    target = store / "evolution" / "validation" / f"{digest}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    candidate["validation_artifacts"] = {
        "path": str(target.relative_to(store)).replace("\\", "/"),
        "digest": digest,
        "heldout_count": len(value.get("heldout_regression_cases", [])),
        "poison_probe_count": len(value.get("poison_probe_cases", [])),
    }


def hydrate_candidate_cases(
    store: Path,
    candidate: dict[str, Any],
    ledger: CandidateEvidenceLedger,
) -> list[dict[str, Any]]:
    """Rebuild all immutable case payloads recorded for a candidate revision."""
    subject_id = str(candidate.get("candidate_identity") or candidate.get("rule_id") or candidate.get("relation_id") or candidate.get("id") or "")
    version = int(candidate.get("version", 1))
    action_digest = str(candidate.get("action_semantic_digest", "")) or None
    memberships = ledger.members(subject_id, version, action_digest=action_digest)
    paths: dict[str, Path] = {}
    for membership in memberships:
        case_id = membership.get("case_id")
        case_path = membership.get("case_path")
        if not isinstance(case_id, str) or not isinstance(case_path, str):
            continue
        relative = Path(case_path)
        if relative.is_absolute() or ".." in relative.parts:
            continue
        paths[case_id] = store / relative
    values: list[dict[str, Any]] = []
    for case_id in sorted(paths):
        try:
            value = json.loads(paths[case_id].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            case_for_hash = {key: item for key, item in value.items() if key != "case_path"}
            expected = next((item.get("case_sha256") for item in memberships if item.get("case_id") == case_id), None)
            actual = hashlib.sha256(json.dumps(case_for_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
            if expected and actual != expected:
                raise ValueError(f"candidate evidence digest mismatch: {case_id}")
            values.append(value)
    return values


def candidate_intervention_digest(intervention: dict[str, Any]) -> str:
    """Digest the semantic proposal identity used to partition evidence."""
    return hashlib.sha256(json.dumps(intervention, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def candidate_identity(rule_id: str, version: int, intervention: dict[str, Any]) -> str:
    return f"{rule_id}:v{int(version)}:{candidate_intervention_digest(intervention)[:24]}"


def semantic_action_spec(family_id: str | None, proposal: dict[str, Any]) -> dict[str, Any]:
    """Project a task patch onto the reusable family action vocabulary."""
    from core.acre.actions import action_from_proposal
    resolved_family = str(family_id) if family_id is not None else ""
    try:
        from benchmark.families import resolve_family_id
        resolved_family = resolve_family_id(resolved_family)
    except (KeyError, ValueError):
        pass
    action = action_from_proposal(resolved_family or None, proposal)
    return {"action": action.action_id, "action_id": action.action_id, "family": action.family, "parameters": dict(action.parameters), "preconditions": dict(action.preconditions), "preserves": list(action.preserves), "risk_class": action.risk_class}


def persist_collecting_proposals(
    store: Path,
    proposals: list[dict[str, Any]],
    case_ids: list[str],
    *,
    family_id: str | None = None,
) -> None:
    """Persist worker hypotheses before a boundary predicate is identifiable."""
    candidates_dir = Path(store) / "evolution" / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    for proposal in proposals:
        if not isinstance(proposal, dict) or proposal.get("relation_id"):
            continue
        identifier = proposal.get("rule_id") or proposal.get("id")
        if not isinstance(identifier, str):
            continue
        try:
            validate_identifier(identifier, "candidate_id")
        except ValueError:
            continue
        version = int(proposal.get("version", 1))
        realization = proposal.get("intervention") if isinstance(proposal.get("intervention"), dict) else {"action": "measure"}
        intervention = semantic_action_spec(family_id, proposal)
        intervention_digest = candidate_intervention_digest(intervention)
        identity = candidate_identity(identifier, version, intervention)
        path = candidates_dir / f"{identifier_digest(identity)}.json"
        existing: dict[str, Any] = {}
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
        evidence_ids = sorted(set(str(item) for item in existing.get("synthesis_state", {}).get("evidence_ids", []) if isinstance(item, str)) | set(case_ids))
        candidate = {
            "rule_id": identifier,
            "family_id": family_id,
            "version": int(proposal.get("version", existing.get("version", 1))),
            "candidate_identity": identity,
            "intervention_digest": intervention_digest,
            "parent": None,
            "hypothesis": {
                "intervention": intervention,
                "realization": realization,
                "expected_mechanism": str(proposal.get("expected_mechanism") or proposal.get("hypothesis") or "task-local performance mechanism"),
            },
            "synthesis_state": existing.get("synthesis_state", {
                "status": "collecting_evidence",
                "predicate": None,
                "version_space_digest": None,
                "evidence_ids": evidence_ids,
            }),
            "intervention": intervention,
            "realization": realization,
            "expected_mechanism": str(proposal.get("expected_mechanism") or proposal.get("hypothesis") or "task-local performance mechanism"),
            "status": "collecting_evidence",
            "p_min": 0.8,
            "delta": 0.05,
            "cases": sorted(set(str(item) for item in existing.get("cases", []) if isinstance(item, str)) | set(case_ids)),
        }
        candidate["synthesis_state"]["evidence_ids"] = evidence_ids
        payload = {**existing, **candidate}
        if payload.get("status") == "collecting_evidence":
            payload.pop("applicability", None)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def execute_poison_probe(
    task_spec: dict[str, Any],
    public_context: dict[str, Any],
    proposal: dict[str, Any],
    realized_solution: Path,
    baseline_solution: Path,
    task_dir: Path | None = None,
    verifier_out: Path | None = None,
) -> dict[str, Any]:
    """Execute a poison probe against a materialized, non-empty intervention."""
    changed = False
    baseline_files = {path.relative_to(baseline_solution) for path in baseline_solution.rglob("*") if path.is_file()}
    realized_files = {path.relative_to(realized_solution) for path in realized_solution.rglob("*") if path.is_file()}
    for relative in baseline_files | realized_files:
        baseline_path, realized_path = baseline_solution / relative, realized_solution / relative
        if not baseline_path.is_file() or not realized_path.is_file() or baseline_path.read_bytes() != realized_path.read_bytes():
            changed = True
            break
    if not changed:
        raise ValueError("poison probe requires a realized intervention different from baseline")
    family_id = str(task_spec.get("family_id", task_spec.get("family", "compile")))
    from core.acre.actions import action_from_proposal
    intervention_id = action_from_proposal(family_id, proposal).action_id
    deployed = [intervention_id]
    verifier_executed = False
    verifier_scientific_ok = True
    verifier_result_digest: str | None = None
    if task_dir is not None:
        result = verifier.verify_task(
            task_dir,
            realized_solution,
            out_path=verifier_out or realized_solution.parent / "poison-result.json",
            condition="D",
            context_mode="reset",
            seed=0,
        )
        verifier_executed = True
        verifier_scientific_ok = bool(scoring.score_task(result).get("gates_passed", False))
        verifier_result_digest = hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    environment = FamilyEnvironment(family_id)
    state = EpisodeEnvironmentState(active_poison=("formal_validation_probe",))
    outcome = environment.evaluate(public_context.get("workload", {}), deployed, state)
    baseline = environment.evaluate(public_context.get("workload", {}), (), state)
    return {
        "case_id": f"POISON-PROBE-{task_spec.get('task_id', 'task')}",
        "executed": True,
        "execution_source": "verifier+family-environment" if verifier_executed else "family-environment",
        "validation_class": "executable_adversarial" if verifier_executed else "synthetic_validation_only",
        "accepted": bool(
            outcome.utility > baseline.utility + 1e-9
            and all(outcome.scientific_gates.values())
            and verifier_scientific_ok
        ),
        "utility": outcome.utility,
        "baseline_utility": baseline.utility,
        "intervention_id": intervention_id,
        "realized_changed": True,
        "verifier_scientific_ok": verifier_scientific_ok,
        "verifier_result_digest": verifier_result_digest,
    }


def _copy_workspace(task_dir: Path, destination: Path) -> None:
    source = task_dir / "workspace"
    if not source.is_dir():
        raise FileNotFoundError(f"workspace missing: {source}")
    shutil.copytree(source, destination, dirs_exist_ok=True)


def materialize_agent_task(task_dir: Path, destination: Path) -> None:
    """Create the public task view used by the external worker.

    Oracle and verifier code are harness-only.  The worker receives the task
    contract, workspace, and public tests, never the task package root.
    """
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    task = miniyaml.load(str(task_dir / "task.yaml"))
    measurement = task.get("measurement", {}) if isinstance(task, dict) else {}
    correctness = task.get("correctness", {}) if isinstance(task, dict) else {}
    declared_public_context = task.get("public_context") if isinstance(task.get("public_context"), dict) else None
    if declared_public_context is None:
        # Family parameters are observable workload facts, unlike the hidden
        # mechanism/oracle labels.  Existing anchors are projected through
        # this same public contract until their YAML declarations are rebuilt.
        family_parameters = task.get("family_parameters") if isinstance(task.get("family_parameters"), dict) else {}
        declared_public_context = {"workload": dict(family_parameters)}
    # Optional observable platform facts are part of the common public
    # context, never condition-specific metadata.
    if isinstance(task.get("public_hardware"), dict):
        declared_public_context.setdefault("hardware", dict(task["public_hardware"]))
    if isinstance(task.get("public_software"), dict):
        declared_public_context.setdefault("software", dict(task["public_software"]))
    if isinstance(task.get("pre_task_telemetry"), dict):
        declared_public_context.setdefault("evidence", dict(task["pre_task_telemetry"]))
    declared_public_context = canonical_public_context(declared_public_context)
    public_task = {
        "schema_version": 1,
        "task_id": task.get("task_id", task_dir.name),
        "title": task.get("title", ""),
        "requires_cuda": bool(task.get("requires_cuda", False)),
        "time_budget_s": int(task.get("time_budget_s", 0)),
        "workspace": dict(task.get("workspace", {})),
        "allowed_files": [str(task.get("workspace", {}).get("entrypoint", "solution.py"))],
        "scientific_gates": list(task.get("scientific_gates", [])),
        "scientific_contract": "scientific_contract.py",
        "measurement_contract": {
            "primary_metric": measurement.get("primary_metric"),
            "higher_is_better": measurement.get("higher_is_better"),
            "warmup_iterations": measurement.get("warmup_iterations"),
            "measured_iterations": measurement.get("measured_iterations"),
            "repetitions": measurement.get("repetitions"),
        },
        "correctness_contract": correctness,
        # Only fields explicitly declared public may drive retrieval/routing.
        "routing_context": declared_public_context,
    }
    (destination / "public_task.json").write_text(
        json.dumps(public_task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    source = task_dir / "scientific_contract.py"
    if source.is_file():
        shutil.copy2(source, destination / "scientific_contract.py")
    for name in ("workspace", "public_tests"):
        source = task_dir / name
        if source.is_dir():
            shutil.copytree(source, destination / name, dirs_exist_ok=True)


def _run_agent(command_template: str, env: dict[str, str], cwd: Path, timeout: float) -> dict[str, Any]:
    command = command_template.format(**env)
    try:
        inherited = {key: value for key, value in os.environ.items() if not key.startswith("SPE_") and key not in {"PYTHONPATH", "PYTHONHOME"}}
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            env={**inherited, **env},
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + "timeout",
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _run_isolated_agent(
    command_template: str,
    executor_command: str,
    env: dict[str, str],
    worker_root: Path,
    timeout: float,
) -> dict[str, Any]:
    """Run through an externally supplied namespace/container executor.

    The executor owns the filesystem boundary.  The campaign never exposes the
    condition store, verifier, or benchmark root as worker paths.
    """
    executor = executor_command.format(
        agent_command=shlex.quote(command_template),
        worker_root=shlex.quote(str(worker_root)),
        task_dir=shlex.quote(env["SPE_TASK_DIR"]),
        solution_dir=shlex.quote(env["SPE_SOLUTION_DIR"]),
        retrieved_context=shlex.quote(env["SPE_RETRIEVED_CONTEXT"]),
        skill_view=shlex.quote(env.get("SPE_SKILL_VIEW_DIR", "")),
        executor_receipt=shlex.quote(env["SPE_EXECUTOR_RECEIPT_PATH"]),
    )
    return _run_agent(executor, env, worker_root, timeout)


def _prepare_worker_root(
    trial_dir: Path,
    agent_task_dir: Path,
    solution_dir: Path,
    skill_view: Path | None,
) -> Path:
    """Create the only host directory visible to an external executor."""
    worker_root = trial_dir / "worker"
    if worker_root.exists():
        shutil.rmtree(worker_root)
    worker_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(agent_task_dir, worker_root / "task")
    shutil.copytree(solution_dir, worker_root / "solution")
    if skill_view is not None:
        shutil.copytree(skill_view, worker_root / "skill_view")
    return worker_root


def _verify_baseline(
    task_dir: Path,
    solution_dir: Path,
    out_path: Path,
    *,
    condition: str,
    context_mode: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score the untouched arm on the same fixture before candidate scoring."""
    baseline = verifier.verify_task(
        task_dir,
        solution_dir,
        out_path=out_path,
        condition=condition,
        context_mode=context_mode,
        seed=seed,
    )
    return baseline, scoring.score_task(baseline)


def _read_executor_receipt(path: Path, skill_digest: str | None, context_mode: str = "reset") -> tuple[dict[str, Any], list[str]]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, ["external executor receipt missing or invalid"]
    if not isinstance(receipt, dict):
        return {}, ["external executor receipt must be an object"]
    errors: list[str] = []
    required = (
        "mode", "network_mode", "mount_allowlist", "executor_digest", "worker_uid", "usage",
        "network_namespace_attested", "mount_receipt", "isolation_canary", "usage_meter_source",
    )
    errors.extend(f"executor receipt missing {key}" for key in required if key not in receipt)
    if receipt.get("mode") != "external_namespace_executor":
        errors.append("executor receipt mode mismatch")
    if receipt.get("network_mode") != "none":
        errors.append("external executor must declare network_mode=none")
    if receipt.get("network_namespace_attested") is not True:
        errors.append("external executor network namespace is not attested")
    if not isinstance(receipt.get("mount_receipt"), dict) or receipt.get("mount_receipt", {}).get("verified") is not True:
        errors.append("external executor mount receipt is not verified")
    if receipt.get("isolation_canary") is not True:
        errors.append("external executor isolation canary did not pass")
    if not isinstance(receipt.get("usage_meter_source"), str) or not receipt.get("usage_meter_source"):
        errors.append("external executor usage meter source is required")
    mounts = receipt.get("mount_allowlist")
    if not isinstance(mounts, list) or not mounts:
        errors.append("executor receipt mount_allowlist must be non-empty")
    else:
        allowed_mounts = {"task", "solution", "skill_view", "retrieved_context", "context_state", "result", "executor_receipt"}
        unexpected = sorted(set(str(item) for item in mounts) - allowed_mounts)
        if unexpected:
            errors.append(f"executor receipt contains disallowed mounts: {unexpected}")
        required_mounts = {"task", "solution", "retrieved_context", "result", "executor_receipt"}
        if skill_digest is not None:
            required_mounts.add("skill_view")
        elif "skill_view" in set(str(item) for item in mounts):
            errors.append("condition A must not mount skill_view")
        if context_mode == "carry":
            required_mounts.add("context_state")
        missing_mounts = sorted(required_mounts - set(str(item) for item in mounts))
        if missing_mounts:
            errors.append(f"executor receipt missing required mounts: {missing_mounts}")
    for key in ("executor_digest", "worker_uid"):
        if not isinstance(receipt.get(key), str) or not receipt.get(key):
            errors.append(f"executor receipt {key} must be non-empty")
    if skill_digest is not None and receipt.get("skill_view_digest") != skill_digest:
        errors.append("executor receipt skill_view_digest mismatch")
    if skill_digest is None and receipt.get("skill_view_digest") not in {None, ""}:
        errors.append("condition A must not attest a skill_view_digest")
    usage = receipt.get("usage")
    if not isinstance(usage, dict):
        errors.append("executor receipt usage must be an object")
    return receipt, errors


def _read_agent_extensions(path: Path) -> dict[str, Any]:
    """Read only worker-supplied lesson/proposal extensions before verification.

    The verifier remains authoritative for scores, gates, and verdicts.  A
    worker may contribute a typed proposal, but cannot author evidence,
    replay cases, confidence, or any scored field.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    extensions: dict[str, Any] = {}
    for key in ("lesson", "acre_proposals", "predicted_mechanisms", "abstain", "abstain_reason"):
        if key in payload:
            extensions[key] = payload[key]
    if not isinstance(extensions.get("lesson", {}), dict):
        extensions["lesson"] = {}
    mechanisms = extensions.get("predicted_mechanisms", [])
    if not isinstance(mechanisms, list) or any(not isinstance(item, str) or not item for item in mechanisms):
        extensions["predicted_mechanisms"] = []
    if not isinstance(extensions.get("abstain", False), bool):
        extensions["abstain"] = False
    if not isinstance(extensions.get("abstain_reason", ""), str):
        extensions["abstain_reason"] = ""
    proposals = extensions.get("acre_proposals", [])
    if not isinstance(proposals, list):
        extensions["acre_proposals"] = []
    else:
        clean: list[dict[str, Any]] = []
        proposal_fields = {"rule_id", "relation_id", "id", "expected_mechanism", "intervention", "text", "hypothesis", "query"}
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            if any(key in proposal for key in ("cases", "confidence", "promotion", "p_min", "delta", "epsilon", "effect", "scientific_gates")):
                continue
            identifier = proposal.get("rule_id") or proposal.get("relation_id") or proposal.get("id")
            if not isinstance(identifier, str):
                continue
            try:
                validate_identifier(identifier, "proposal_id")
            except ValueError:
                continue
            clean.append({key: value for key, value in proposal.items() if key in proposal_fields})
        extensions["acre_proposals"] = clean
    return extensions


def post_task_update(
    *,
    condition: str,
    store: Path,
    task_id: str,
    result: dict[str, Any],
    scored: dict[str, Any],
    core_repo: Path,
    out_dir: Path,
    context_mode: str = "reset",
    family_id: str | None = None,
    ledger: EvolutionDecisionLedger | None = None,
    allow_maintenance: bool = True,
    control_result: dict[str, Any] | None = None,
    control_scored: dict[str, Any] | None = None,
    public_context: dict[str, Any] | None = None,
    causal_result: dict[str, Any] | None = None,
    causal_scored: dict[str, Any] | None = None,
    validation_evidence: dict[str, Any] | None = None,
    practical_epsilon: float = 0.0,
) -> dict[str, Any]:
    """Run the explicit execute -> evidence -> maintenance -> attest transition."""
    condition = condition.upper()
    pre_digest = conditions.store_digest(store)
    experience_id = f"EXP-{task_id}"
    added_experience_ids: list[str] = []
    added_replay_case_ids: list[str] = []
    maintenance_decisions: list[dict[str, Any]] = []
    promoted_rule_ids: list[str] = []
    if condition in {"C", "C_STRESS", "D"}:
        experience = {
            "schema_version": 1,
            "record_type": "task_experience",
            "id": experience_id,
            "experience_id": experience_id,
            "task_id": task_id,
            "condition": condition,
            "context_mode": context_mode,
            "public_context": canonical_public_context(public_context),
            "retrieval_query": json.dumps(canonical_public_context(public_context).get("workload", {}), sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            "observation": {
                "verdict": result.get("verdict"),
                "task_score": scored.get("task_score"),
                "scientific_gates": result.get("scientific_gates", {}),
                "measurement": result.get("measurement", {}),
            },
            "lesson": result.get("lesson", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        inbox = store / "experience" / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        validate_identifier(experience_id, "experience_id")
        experience_path = inbox / f"{identifier_digest(experience_id)}.json"
        if not experience_path.exists():
            experience_path.write_text(json.dumps(experience, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            added_experience_ids.append(experience_id)

        if condition == "D" and causal_result is not None and causal_scored is not None and control_result is not None and control_scored is not None:
            # The worker cannot author evidence.  The harness derives paired
            # case records from the verifier-owned outcome below.
            evidence_dir = store / "experience" / "cases"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            case_id = f"CASE-{task_id}"
            validate_identifier(case_id, "case_id")
            measurement = causal_result.get("measurement", {}) if isinstance(causal_result.get("measurement"), dict) else {}
            control_measurement = control_result.get("measurement", {}) if isinstance(control_result.get("measurement"), dict) else {}
            intervention_runs = list(measurement.get("candidate_runs", []))
            baseline_runs = list(measurement.get("baseline_runs", []))
            fixture_hashes = measurement.get("fixture_hashes", {})
            control_fixture_hashes = control_measurement.get("fixture_hashes", {})
            if not intervention_runs or len(intervention_runs) != len(baseline_runs) or not fixture_hashes or fixture_hashes != control_fixture_hashes:
                # The verifier's paired record is the only accepted source of
                # causal measurements; no task-score fallback is fabricated.
                persist_collecting_proposals(store, [item for item in result.get("acre_proposals", []) if isinstance(item, dict)], added_replay_case_ids, family_id=family_id)
                return {"transition": "no_replay_measurement", "added_experience_ids": added_experience_ids}
            case = {
                "schema_version": 1,
                "record_type": "paired_replay_case",
                "case_id": case_id,
                "utility_on": float(causal_scored.get("task_score", 0.0)),
                "utility_off": float(control_scored.get("task_score", 0.0)),
                "control_measured": bool(control_measurement.get("candidate_runs")),
                "baseline_measurements": baseline_runs,
                "intervention_measurements": intervention_runs,
                "same_fixture_digest": measurement.get("fixture_hashes", {}),
                "same_seed": causal_result.get("seed"),
                "same_work_units": measurement.get("work_units", {}),
                "higher_is_better": bool(measurement.get("higher_is_better", False)),
                "utility_scale": UTILITY_LOG_SCALE,
                "scientific_ok": bool(causal_scored.get("gates_passed", False)),
                "quality_ok": bool(causal_scored.get("gates_passed", False)) and bool(control_scored.get("gates_passed", False)),
                "paired_replay": True,
                "same_fixture_id": task_id,
                "candidate_result_digest": hashlib.sha256(json.dumps(causal_result, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
                "control_result_digest": hashlib.sha256(json.dumps(control_result, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
                "scientific_gates_on": dict(causal_result.get("scientific_gates", {})),
                "scientific_gates_off": dict(control_result.get("scientific_gates", {})),
                "source_id": f"verifier-{task_id}",
                "independence_group": f"task-{task_id}",
                "context": canonical_public_context(public_context),
                "context_mode": context_mode,
            }
            evidence_path = evidence_dir / f"{identifier_digest(case_id)}.json"
            if not evidence_path.exists():
                evidence_path.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            added_replay_case_ids.append(case_id)
            validation_dir = store / "evolution" / "validation"
            validation_dir.mkdir(parents=True, exist_ok=True)
            if not isinstance(validation_evidence, dict):
                persist_collecting_proposals(store, [item for item in result.get("acre_proposals", []) if isinstance(item, dict)], added_replay_case_ids, family_id=family_id)
                return {"transition": "no_independent_validation", "added_experience_ids": added_experience_ids}
            validation = {
                "schema_version": 1,
                "scope": "formal",
                "subject_context": canonical_public_context(public_context),
                "synthesis_case_ids": [case_id],
                "promotion_case_ids": [],
                "heldout_regression_cases": list(validation_evidence.get("heldout_regression_cases", [])),
                "poison_probe_cases": list(validation_evidence.get("poison_probe_cases", [])),
                "independence_groups": [case["independence_group"]],
            }
            validation_digest = hashlib.sha256(json.dumps(validation, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
            validation_path = validation_dir / f"{validation_digest}.json"
            if not validation_path.exists():
                validation_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if condition == "D" and allow_maintenance:
        valid, policy_errors = conditions.verify_condition_policy(store)
        if not valid:
            raise ValueError("governed transition failed store policy: " + "; ".join(policy_errors))
        from dataclasses import asdict
        from core.acre.engine import AcreEngine
        engine = AcreEngine.from_store(store)
        # Core owns the evidence-to-lifecycle reducer.  The formal driver
        # only supplies this task's verifier-produced events and persistence
        # callbacks; it does not make a second lifecycle decision.
        maintenance_step = engine.maintainer.step(
            events=[item for item in result.get("evidence_events", []) if isinstance(item, dict)],
        )
        maintenance_decisions.append({"operation": "OBSERVE", "observed": maintenance_step.observed, "assessment": maintenance_step.assessment})
        for subject_id in (*engine.rule_states, *engine.relation_states):
            maintenance_decisions.append(asdict(engine.evolve(subject_id)))
        active_ledger = ledger or EvolutionDecisionLedger()
        candidates_dir = store / "evolution" / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        from benchmark.harness.evolution_ledger import CandidateEvidenceLedger
        candidate_evidence = CandidateEvidenceLedger(store / "evolution" / "candidate_evidence.jsonl")
        candidates_by_id: dict[str, dict[str, Any]] = {}
        proposals = [item for item in result.get("acre_proposals", []) if isinstance(item, dict)]
        if len(proposals) != 1:
            proposals = []
        for candidate in proposals:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("relation_id"):
                # Relations use an explicit factorial experiment and never
                # enter the node utility replay path.
                from benchmark.formal.schedule import RelationExperimentScheduler
                relation_id = str(candidate.get("relation_id") or candidate.get("id") or "")
                validate_identifier(relation_id, "relation_id")
                relation_schedule = RelationExperimentScheduler().schedule(candidate, str(family_id or "compile"))
                relation_dir = store / "evolution" / "relation_experiments"
                relation_dir.mkdir(parents=True, exist_ok=True)
                (relation_dir / f"{identifier_digest(relation_id)}.json").write_text(
                    json.dumps(relation_schedule, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                continue
            identifier = str(candidate.get("relation_id") or candidate.get("rule_id") or candidate.get("id") or "")
            if not identifier:
                continue
            candidate = dict(candidate)
            validate_identifier(identifier, "candidate_id")
            task_patch = candidate.get("intervention") if isinstance(candidate.get("intervention"), dict) else {"action": "measure"}
            version = int(candidate.get("version", 1))
            action_spec = semantic_action_spec(family_id, candidate)
            intervention_digest = candidate_intervention_digest(action_spec)
            identity = candidate_identity(identifier, version, action_spec)
            try:
                from benchmark.families import FAMILY_SPECS, resolve_family_id
                family_spec = FAMILY_SPECS.get(resolve_family_id(str(family_id)))
            except (ImportError, AttributeError):
                family_spec = None
            scientific_invariants = list(getattr(family_spec, "scientific_invariants", ()) or ()) or ["task_scientific_gates"]
            severity = str(getattr(family_spec, "default_severity", "P2"))
            runtime_tokens = max(1.0, len(json.dumps(action_spec, sort_keys=True, separators=(",", ":"))) / 4.0)
            # Proposal fields are hypotheses only.  Applicability is generated
            # below by the harness-owned CEGIS pass.
            candidate = {
                "rule_id": identifier,
                "family_id": family_id,
                "version": version,
                "candidate_identity": identity,
                "intervention_digest": intervention_digest,
                "parent": None,
                "applicability": {"all": []},
                "hypothesis": {
                    "intervention": action_spec,
                    "expected_mechanism": str(candidate.get("expected_mechanism") or candidate.get("hypothesis") or "task-local performance mechanism"),
                },
                "synthesis_state": {
                    "status": "collecting_evidence",
                    "predicate": None,
                    "version_space_digest": None,
                    "evidence_ids": [],
                },
                "intervention": action_spec,
                "realization": task_patch,
                "expected_mechanism": str(candidate.get("expected_mechanism") or candidate.get("hypothesis") or "task-local performance mechanism"),
                "evidence_requirements": ["paired_replay"],
                "scientific_invariants": scientific_invariants,
                "abstain_conditions": {},
                "relations": {},
                "runtime_cost": {"tokens": runtime_tokens},
                "provenance_policy": {"required": True},
                "severity": severity,
                "domain": "runtime",
                "text": str(candidate.get("text", "")),
                "status": "candidate",
                "epsilon": float(practical_epsilon),
                "p_min": 0.8,
                "delta": 0.05,
            }
            candidate["cases"] = list(added_replay_case_ids)
            candidate["validation_artifacts"] = {
                "path": str(validation_path.relative_to(store)).replace("\\", "/"),
                "digest": validation_digest,
                "heldout_count": 1,
                "poison_probe_count": 1,
            }
            path = candidates_dir / f"{identifier_digest(identity)}.json"
            existing: dict[str, Any] = {}
            if path.is_file():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
            if existing:
                immutable = ("rule_id", "relation_id", "version", "intervention", "expected_mechanism")
                for field in immutable:
                    if field in existing and field in candidate and existing[field] != candidate[field]:
                        raise ValueError(f"candidate immutable field changed for {identifier}: {field}")
                merged = dict(existing)
                merged.update({key: value for key, value in candidate.items() if key not in {"cases", "applicability", "synthesis_state"}})
                candidate = merged
                candidate["applicability"] = existing.get("applicability", {"all": []})
                candidate["synthesis_state"] = existing.get("synthesis_state", {
                    "status": "collecting_evidence",
                    "predicate": None,
                    "version_space_digest": None,
                    "evidence_ids": [],
                })
            # Membership is authoritative in CandidateEvidenceLedger.  The
            # candidate projection may retain case ids for display only; it
            # must never union mutable worker data into the evidence set.
            candidate["cases"] = sorted(set(candidate.get("cases", [])))
            # A candidate may accumulate independent replay cases over several
            # tasks.  Keep the validation artifact aligned with the complete
            # promotion bundle instead of leaving only the latest case id.
            validation_sources: list[dict[str, Any]] = []
            for artifact_owner in (existing, candidate):
                artifact_ref = artifact_owner.get("validation_artifacts") if isinstance(artifact_owner, dict) else None
                if not isinstance(artifact_ref, dict):
                    continue
                artifact_path = artifact_ref.get("path")
                if not isinstance(artifact_path, str):
                    continue
                try:
                    artifact_value = json.loads((store / artifact_path).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(artifact_value, dict):
                    validation_sources.append(artifact_value)
            if validation_sources:
                heldout_cases: dict[str, dict[str, Any]] = {}
                poison_cases: dict[str, dict[str, Any]] = {}
                independence_groups: set[str] = set()
                for artifact_value in validation_sources:
                    groups = artifact_value.get("independence_groups", [])
                    if isinstance(groups, list):
                        independence_groups.update(str(group) for group in groups if isinstance(group, str))
                    for key, target in (("heldout_regression_cases", heldout_cases), ("poison_probe_cases", poison_cases)):
                        entries = artifact_value.get(key, [])
                        if not isinstance(entries, list):
                            continue
                        for entry in entries:
                            if isinstance(entry, dict) and isinstance(entry.get("case_id"), str):
                                target.setdefault(entry["case_id"], entry)
                merged_validation = dict(validation_sources[-1])
                merged_validation["synthesis_case_ids"] = sorted({str(item) for item in candidate["cases"]})
                merged_validation["promotion_case_ids"] = sorted({str(item) for item in merged_validation.get("promotion_case_ids", [])})
                merged_validation["heldout_regression_cases"] = list(heldout_cases.values())
                merged_validation["poison_probe_cases"] = list(poison_cases.values())
                merged_validation["independence_groups"] = sorted(independence_groups)
                validation_digest = hashlib.sha256(json.dumps(
                    merged_validation, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")).hexdigest()
                validation_path = store / "evolution" / "validation" / f"{validation_digest}.json"
                validation_path.parent.mkdir(parents=True, exist_ok=True)
                if not validation_path.exists():
                    validation_path.write_text(json.dumps(merged_validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                candidate["validation_artifacts"] = {
                    "path": str(validation_path.relative_to(store)).replace("\\", "/"),
                    "digest": validation_digest,
                    "heldout_count": len(merged_validation["heldout_regression_cases"]),
                    "poison_probe_count": len(merged_validation["poison_probe_cases"]),
                }
            case_values_for_cegis: list[dict[str, Any]] = []
            for case_id in candidate.get("cases", []):
                try:
                    case_value = json.loads((store / "experience" / "cases" / f"{identifier_digest(str(case_id))}.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
                if isinstance(case_value, dict):
                    case_values_for_cegis.append(case_value)
                    case_value = dict(case_value)
                    case_value["case_path"] = str((store / "experience" / "cases" / f"{identifier_digest(str(case_id))}.json").relative_to(store)).replace("\\", "/")
                    candidate_evidence.append(
                        str(candidate.get("candidate_identity") or identifier),
                        int(candidate.get("version", 1)),
                        case_value,
                        action_digest=str(candidate.get("intervention_digest", "")),
                    )
            case_values_for_cegis = hydrate_candidate_cases(store, candidate, candidate_evidence)
            candidate["cases"] = sorted({str(case.get("case_id")) for case in case_values_for_cegis if case.get("case_id")})
            from benchmark.formal.schedule import PromotionReplayScheduler
            scheduler = PromotionReplayScheduler()
            seen_groups = {str(case.get("independence_group")) for case in case_values_for_cegis if case.get("independence_group")}
            candidate["replay_schedule"] = {
                "minimum_groups": scheduler.minimum_groups,
                "pending_contexts": scheduler.pending_contexts(
                    str(family_id or "compile"), seen_group_ids=seen_groups,
                ),
            }
            candidate["replay_schedule"]["experiment_cost"] = float(
                len(candidate["replay_schedule"]["pending_contexts"])
            )
            candidate["status"] = "collecting_evidence"
            candidate["synthesis_state"] = {
                **dict(candidate.get("synthesis_state") or {}),
                "status": "collecting_evidence",
                "evidence_ids": list(candidate["cases"]),
            }
            candidate.pop("applicability", None)
            candidate.pop("applicability_provenance", None)
            path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            synthesized = synthesize_applicability(
                case_values_for_cegis,
                family_id=family_id,
                delta=0.05,
                epsilon_true=float(practical_epsilon),
                epsilon_false=0.0,
                require_identified=True,
            )
            if synthesized is None:
                continue
            intervention = candidate.get("realization") or candidate.get("intervention")
            if not isinstance(intervention, dict) or not isinstance(intervention.get("file"), str) or not isinstance(intervention.get("replacements"), list):
                continue
            predicate, provenance = synthesized
            promotion_ids = representative_case_ids(predicate, case_values_for_cegis)
            if not promotion_ids or provenance.get("status") != "identified":
                continue
            candidate["applicability"], candidate["applicability_provenance"] = predicate, provenance
            candidate["synthesis_state"] = {
                "status": "identified",
                "predicate": predicate,
                "version_space_digest": provenance.get("version_space_digest"),
                "evidence_ids": list(candidate["cases"]),
            }
            rewrite_validation_membership(
                store,
                candidate,
                synthesis_case_ids=list(candidate["cases"]),
                promotion_case_ids=promotion_ids,
            )
            candidate["status"] = "candidate"
            path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            candidates_by_id[str(candidate.get("candidate_identity") or identifier)] = candidate
        for path in sorted(candidates_dir.glob("*.json")):
            candidate = json.loads(path.read_text(encoding="utf-8"))
            synthesis_state = candidate.get("synthesis_state") if isinstance(candidate.get("synthesis_state"), dict) else {}
            provenance = candidate.get("applicability_provenance") if isinstance(candidate.get("applicability_provenance"), dict) else {}
            if (
                candidate.get("cases")
                and candidate.get("status") == "candidate"
                and synthesis_state.get("status") == "identified"
                and int(provenance.get("decision_context_count", 0)) > 0
            ):
                identifier = str(candidate.get("relation_id") or candidate.get("rule_id") or candidate.get("id") or path.stem)
                candidates_by_id[str(candidate.get("candidate_identity") or identifier)] = candidate
        candidates = list(candidates_by_id.values())
        for candidate in candidates:
            hydrated_cases: list[dict[str, Any]] = []
            subject_id = str(candidate.get("candidate_identity") or candidate.get("rule_id") or candidate.get("id") or "")
            memberships = candidate_evidence.members(subject_id, int(candidate.get("version", 1)), action_digest=str(candidate.get("intervention_digest", "")) or None)
            membership_by_id = {str(item.get("case_id")): item for item in memberships}
            case_ids = list(membership_by_id)
            for case_id in case_ids:
                try:
                    case_path = store / "experience" / "cases" / f"{identifier_digest(str(case_id))}.json"
                    case_value = json.loads(case_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
                if isinstance(case_value, dict):
                    membership = membership_by_id.get(str(case_id))
                    if membership and membership.get("case_sha256"):
                        actual_case_digest = hashlib.sha256(json.dumps(case_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
                        if actual_case_digest != membership["case_sha256"]:
                            continue
                    hydrated_cases.append(case_value)
            candidate["cases"] = hydrated_cases
        if candidates:
            promoted_rule_ids = promote_via_replay(store, candidates, core_repo, out_dir, active_ledger)
        from scripts.validate_evolution import audit

        errors = audit(store, schema_root=core_repo)
        if errors:
            raise ValueError("D maintenance transition failed validation: " + "; ".join(errors))
        transition = "governed_maintenance"
    elif condition == "D":
        valid, policy_errors = conditions.verify_condition_policy(store)
        if not valid:
            raise ValueError("governed transition failed store policy: " + "; ".join(policy_errors))
        from scripts.validate_evolution import audit

        errors = audit(store, schema_root=core_repo)
        if errors:
            raise ValueError("D maintenance transition failed validation: " + "; ".join(errors))
        transition = "maintenance_blocked"
    elif condition in {"C", "C_STRESS"}:
        valid, errors = conditions.verify_condition_policy(store)
        if not valid:
            raise ValueError("raw-experience transition failed policy validation: " + "; ".join(errors))
        transition = "raw_experience_capture"
    else:
        transition = "no_op"

    if condition in {"C", "C_STRESS", "D"}:
        conditions.refresh_attestation(store)
    post_digest = conditions.store_digest(store)
    return {
        "status": transition,
        "pre_store_digest": pre_digest,
        "post_store_digest": post_digest,
        "added_experience_ids": added_experience_ids,
        "added_replay_case_ids": added_replay_case_ids,
        "maintenance_decisions": maintenance_decisions,
        "promoted_rule_ids": promoted_rule_ids,
    }


def _experiment_manifest(
    *,
    repo_root: Path,
    tasks_root: Path,
    skill_digest: str,
    task_digest: str,
    task_ids: list[str],
    item: dict[str, Any],
    model_id: str,
    agent_config: dict[str, Any],
    budgets: budget.Budget,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    task_spec = miniyaml.load(str(tasks_root / item["task_id"] / "task.yaml"))
    lineage = task_spec.get("lineage", {}) if isinstance(task_spec, dict) else {}
    return {
        "schema_version": 1,
        "experiment_id": f"SPE-EvoBench-v1.0-20-{item['stream_id']}-{item['task_id']}",
        "benchmark_revision": attest.benchmark_revision(repo_root),
        "skill_view_digest": skill_digest,
        "task_manifest_digest": task_digest,
        "agent_model_id": model_id,
        "agent_config": agent_config,
        "condition": item["condition"],
        "context_mode": item["context_mode"],
        "worker_isolation": {
            "mode": "external_namespace_executor",
            "network_mode": "none",
            "mount_allowlist": ["task", "solution", "skill_view", "retrieved_context", "result", "executor_receipt"],
            "executor_receipt": None,
        },
        "task_order": task_ids,
        "family_id": task_spec.get("family_id", task_spec.get("family", "unknown")),
        "anchor_instance_id": task_spec.get("anchor_instance_id", item["task_id"]),
        "family_instance_digest": task_spec.get("family_instance_digest"),
        "lineage_id": lineage.get("mutation_template_id", item["task_id"]),
        "outer_trial_id": item["outer_trial_id"],
        "budgets": budgets.as_dict(),
        "hardware_fingerprint": fingerprint,
        "software_fingerprint": {
            "python_version": fingerprint.get("python_version"),
            "platform": fingerprint.get("platform"),
            "torch_geometric_version": fingerprint.get("torch_geometric_version"),
        },
        "torch_version": fingerprint.get("torch_version"),
        "cuda_version": fingerprint.get("cuda_version"),
    }


def _formal_claim_gate(campaign: dict[str, Any], records: list[dict[str, Any]], report_path: Path) -> bool:
    """Allow a claim only after an attested calibration approval."""
    if campaign.get("status") != "complete" or not records:
        return False
    if any(record.get("validity") != "valid" for record in records):
        return False
    expected = int(campaign.get("schedule_size", 0))
    if len(records) != expected:
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    calibration = report.get("empirical_calibration", {})
    if calibration.get("calibration_gate") != "passed":
        return False
    approval_path = report_path.with_name("calibration_approval.json")
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(approval, dict) or approval.get("schema_version") != 1 or approval.get("approved") is not True:
        return False
    body = {key: value for key, value in approval.items() if key != "approval_digest"}
    expected = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if approval.get("approval_digest") != expected:
        return False
    report_digest = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if approval.get("population_report_digest") != report_digest:
        return False
    source = calibration.get("source")
    if not source:
        return False
    try:
        empirical = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    empirical_digest = hashlib.sha256(
        json.dumps(empirical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return approval.get("empirical_report_digest") == empirical_digest


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    tasks_root = Path(args.tasks_root).resolve()
    split_path = Path(args.split).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    task_items = schedule.task_order(split_path)
    task_ids = [task_id for _, task_id in task_items]
    if args.skill_view:
        skill_view = Path(args.skill_view).resolve()
        if not (skill_view / "skill_view_manifest.json").is_file():
            raise ValueError("--skill-view must point to a rendered skill bundle")
        if validate_skill_view_bundle(skill_view):
            raise ValueError("--skill-view failed bundle validation")
    else:
        skill_view = out_dir / "skill-view"
        render_skill_view(args.skill_source, skill_view)
    skill_digest = attest.skill_view_digest(skill_view)
    task_digest = attest.task_manifest_digest(tasks_root, task_ids)
    conditions_list = tuple(item.strip().upper() for item in args.conditions.split(",") if item.strip())
    context_modes = tuple(item.strip() for item in args.context_modes.split(",") if item.strip())
    plan = schedule.build_schedule(
        split_path,
        conditions=conditions_list,
        context_modes=context_modes,
        outer_trials=args.outer_trials,
    )
    budgets = budget.parse_budget(json.loads(args.budgets) if args.budgets else None)
    fingerprint = capture_fingerprint()
    campaign = {
        "schema_version": 1,
        "population_id": "SPE-EvoBench-v1.0-20-pilot",
        "status": "planned" if not args.agent_command else "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_revision": attest.benchmark_revision(repo_root),
        "skill_view_digest": skill_digest,
        "task_manifest_digest": task_digest,
        "conditions": list(conditions_list),
        "context_modes": list(context_modes),
        "outer_trials": args.outer_trials,
        "task_order": task_ids,
        "budgets": budgets.as_dict(),
        "agent_model_id": args.model_id,
        "agent_config": json.loads(args.agent_config),
        "schedule_size": len(plan),
        "results_claimed": False,
    }
    (out_dir / "campaign.json").write_text(json.dumps(campaign, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.agent_command:
        (out_dir / "schedule.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return campaign

    records: list[dict[str, Any]] = []
    stores: dict[str, Path] = {}
    context_paths: dict[str, Path] = {}
    ledgers: dict[str, EvolutionDecisionLedger] = {}
    for item in plan:
        stream_id = str(item["stream_id"])
        task_id = str(item["task_id"])
        task_spec = miniyaml.load(str(tasks_root / task_id / "task.yaml"))
        trial_dir = out_dir / "trials" / stream_id / task_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        manifest = _experiment_manifest(
            repo_root=repo_root,
            tasks_root=tasks_root,
            skill_digest=skill_digest,
            task_digest=task_digest,
            task_ids=task_ids,
            item=item,
            model_id=args.model_id,
            agent_config=json.loads(args.agent_config),
            budgets=budgets,
            fingerprint=fingerprint,
        )
        attest.write_experiment(trial_dir / "experiment.json", manifest)
        state_path = context_paths.setdefault(stream_id, trial_dir.parent / "context.json")
        if item["context_mode"] == "reset" and state_path.exists():
            state_path.unlink()
        store = stores.get(stream_id)
        if store is None:
            store = trial_dir.parent / "condition-store"
            conditions.materialize_condition(item["condition"], skill_view if item["condition"] != "A" else None, store, item["context_mode"])
            stores[stream_id] = store
        solution_dir = trial_dir / "solution"
        _copy_workspace(tasks_root / task_id, solution_dir)
        agent_task_dir = trial_dir / "agent-task"
        materialize_agent_task(tasks_root / task_id, agent_task_dir)
        public_task = json.loads((agent_task_dir / "public_task.json").read_text(encoding="utf-8"))
        public_routing = public_task.get("routing_context", {}) if isinstance(public_task, dict) else {}
        if not isinstance(public_routing, dict):
            public_routing = {}
        retrieved_context_path = agent_task_dir / "retrieved_context.json"
        if str(item["condition"]) in {"C", "C_STRESS", "D"}:
            adapter = FormalConditionAdapter(str(item["condition"]), store, token_budget=budgets.tokens)
            retrieval_input = {
                "domain": public_routing.get("domain", "scientific-performance"),
                "workload": dict(public_routing.get("workload", {})),
                "hardware": dict(public_routing.get("hardware", {})),
                "software": dict(public_routing.get("software", {})),
                "evidence": dict(public_routing.get("evidence", {})),
                "token_budget": budgets.tokens,
            }
            retrieved_context = adapter.retrieved_context(retrieval_input)
            exposed_context = retrieved_context.get("context", {})
            for key in ("domain", "workload", "hardware", "software", "evidence"):
                if exposed_context.get(key) != retrieval_input.get(key):
                    raise ValueError(f"retrieval context escaped public task context at {key}")
        else:
            retrieved_context = {
                "schema_version": 1,
                "condition": str(item["condition"]),
                "context": {"task_id": task_id, "context_mode": str(item["context_mode"])},
                "proposed_interventions": [],
            }
        retrieved_context_path.write_text(json.dumps(retrieved_context, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        worker_root = _prepare_worker_root(trial_dir, agent_task_dir, solution_dir, None if item["condition"] == "A" else skill_view)
        worker_context_state = worker_root / "context_state.json"
        if item["context_mode"] == "carry" and state_path.is_file():
            shutil.copy2(state_path, worker_context_state)
        else:
            worker_context_state.write_text(json.dumps({"context_mode": item["context_mode"], "trajectory": []}) + "\n", encoding="utf-8")
        worker_task_dir = worker_root / "task"
        worker_solution_dir = worker_root / "solution"
        worker_retrieved_context_path = worker_task_dir / "retrieved_context.json"
        worker_retrieved_context_path.write_text(retrieved_context_path.read_text(encoding="utf-8"), encoding="utf-8")
        worker_result_path = worker_root / "worker_result.json"
        # The receipt is written by the executor outside the worker namespace;
        # placing it under worker/ would let the worker author its own trust
        # metadata.
        receipt_path = trial_dir / "executor_receipt.json"
        env = {
            "SPE_TASK_ID": task_id,
            "SPE_TASK_DIR": str(worker_task_dir),
            "SPE_SOLUTION_DIR": str(worker_solution_dir),
            "SPE_CONDITION": str(item["condition"]),
            "SPE_CONTEXT_MODE": str(item["context_mode"]),
            "SPE_RETRIEVED_CONTEXT": str(worker_retrieved_context_path),
            "SPE_RESULT_PATH": str(worker_result_path),
            "SPE_AGENT_USAGE_PATH": str(worker_root / "agent_usage.json"),
            "SPE_EXECUTOR_RECEIPT_PATH": str(receipt_path),
            "SPE_SKILL_VIEW_DIR": str(worker_root / "skill_view") if item["condition"] != "A" else "",
            "SPE_BUDGET_JSON": json.dumps(budgets.as_dict(), sort_keys=True),
            "SPE_OUTER_TRIAL_ID": str(item["outer_trial_id"]),
            "SPE_CONTEXT_STATE_PATH": str(worker_context_state),
        }
        if not getattr(args, "executor_command", None):
            raise ValueError("formal agent runs require --executor-command with a namespace/container executor")
        agent = _run_isolated_agent(args.agent_command, args.executor_command, env, worker_root, budgets.wall_time_s)
        if item["context_mode"] == "carry" and worker_context_state.is_file():
            shutil.copy2(worker_context_state, state_path)
        agent_extensions = _read_agent_extensions(worker_result_path)
        receipt, receipt_errors = _read_executor_receipt(receipt_path, None if item["condition"] == "A" else skill_digest, str(item["context_mode"]))
        usage = receipt.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        budget_errors = budgets.validate_usage(usage)
        budget_errors.extend(receipt_errors)
        manifest["worker_isolation"]["executor_receipt"] = {
            key: receipt.get(key)
            for key in ("mode", "network_mode", "mount_allowlist", "executor_digest", "worker_uid", "skill_view_digest")
            if key in receipt
        }
        attest.write_experiment(trial_dir / "experiment.json", manifest)
        if agent["returncode"] != 0:
            budget_errors.append(f"agent command failed with return code {agent['returncode']}")
        submitted_solution_dir = trial_dir / "submitted_solution"
        try:
            sanitize_submission(
                worker_solution_dir,
                submitted_solution_dir,
                {str(item) for item in public_task.get("allowed_files", []) if isinstance(item, str)},
                baseline=solution_dir,
            )
        except ValueError as exc:
            budget_errors.append(f"submission sanitization failed: {exc}")
            submitted_solution_dir.mkdir(parents=True, exist_ok=True)
        control_result: dict[str, Any] | None = None
        control_scored: dict[str, Any] | None = None
        causal_result: dict[str, Any] | None = None
        causal_scored: dict[str, Any] | None = None
        validation_evidence: dict[str, Any] | None = None
        if str(item["condition"]) == "D":
            control_result, control_scored = _verify_baseline(
                tasks_root / task_id,
                solution_dir,
                trial_dir / "control-result.json",
                condition=str(item["condition"]),
                context_mode=str(item["context_mode"]),
                seed=int(item["outer_trial_index"]),
            )
        result = verifier.verify_task(
            tasks_root / task_id,
            submitted_solution_dir,
            out_path=trial_dir / "result.json",
            condition=str(item["condition"]),
            context_mode=str(item["context_mode"]),
            seed=int(item["outer_trial_index"]),
        )
        result["seed"] = int(item["outer_trial_index"])
        result.update(agent_extensions)
        expected_mechanism = str((result.get("task") or {}).get("expected_mechanism", ""))
        predicted = [str(value) for value in result.get("predicted_mechanisms", []) if isinstance(value, str)]
        result["diagnosis"] = {
            "predicted_mechanisms": predicted,
            "diagnosis_correct": bool(expected_mechanism and expected_mechanism in predicted),
        }
        result["abstained"] = bool(result.get("abstain", False))
        result["condition_adapter"] = retrieved_context
        result.setdefault("cost", {}).update({
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "tool_calls": usage.get("tool_calls"),
        })
        scored = scoring.score_task(result)
        if str(item["condition"]) == "D":
            proposals = [item for item in agent_extensions.get("acre_proposals", []) if isinstance(item, dict)]
            if len(proposals) == 1:
                try:
                    realized_solution = trial_dir / "realized_solution"
                    InterventionRealizer.realize(solution_dir, realized_solution, proposals[0])
                    causal_result = verifier.verify_task(
                        tasks_root / task_id,
                        realized_solution,
                        out_path=trial_dir / "causal-result.json",
                        condition=str(item["condition"]),
                        context_mode=str(item["context_mode"]),
                        seed=int(item["outer_trial_index"]),
                    )
                    causal_result["seed"] = int(item["outer_trial_index"])
                    causal_scored = scoring.score_task(causal_result)
                    heldout_result = verifier.verify_task(
                        tasks_root / task_id,
                        realized_solution,
                        out_path=trial_dir / "heldout-result.json",
                        condition=str(item["condition"]),
                        context_mode=str(item["context_mode"]),
                        seed=int(item["outer_trial_index"]) + 1000003,
                    )
                    heldout_scored = scoring.score_task(heldout_result)
                    heldout_control, heldout_control_scored = _verify_baseline(
                        tasks_root / task_id,
                        solution_dir,
                        trial_dir / "heldout-control-result.json",
                        condition=str(item["condition"]),
                        context_mode=str(item["context_mode"]),
                        seed=int(item["outer_trial_index"]) + 1000003,
                    )
                    heldout_measurement = heldout_result.get("measurement", {}) if isinstance(heldout_result.get("measurement"), dict) else {}
                    heldout_candidate_runs = list(heldout_measurement.get("candidate_runs", []))
                    heldout_baseline_runs = list(heldout_measurement.get("baseline_runs", []))
                    if not heldout_candidate_runs or len(heldout_candidate_runs) != len(heldout_baseline_runs):
                        raise ValueError("held-out verifier did not produce paired measurements")
                    heldout_case = {
                        "case_id": f"HELDOUT-{task_id}",
                        "paired_replay": True,
                        "same_fixture_id": f"HELDOUT-FIXTURE-{task_id}",
                        "independence_group": f"heldout-{task_id}",
                        "intervention_measurements": heldout_candidate_runs,
                        "baseline_measurements": heldout_baseline_runs,
                        "higher_is_better": bool(heldout_measurement.get("higher_is_better", False)),
                        "utility_scale": UTILITY_LOG_SCALE,
                        "scientific_ok": bool(heldout_scored.get("gates_passed", False)),
                        "quality_ok": bool(heldout_scored.get("gates_passed", False)) and bool(heldout_control_scored.get("gates_passed", False)),
                    }
                    heldout_interval = _case_effect_interval(heldout_case, delta=0.05)
                    if heldout_interval is None:
                        raise ValueError("held-out paired effect interval is unavailable")
                    heldout_effect, heldout_lcb, heldout_ucb = heldout_interval
                    regression_tolerance = float(
                        (task_spec.get("measurement") or {}).get("regression_tolerance", 0.0)
                    )
                    validation_evidence = {
                        "regression_tolerance": regression_tolerance,
                        "heldout_regression_cases": [{
                            "case_id": f"HELDOUT-{task_id}",
                            "executed": True,
                            "execution_source": "verifier",
                            "scientific_ok": bool(heldout_scored.get("gates_passed", False)),
                            "utility": heldout_effect,
                            "effect": heldout_effect,
                            "effect_lcb": heldout_lcb,
                            "effect_ucb": heldout_ucb,
                            "utility_policy_id": "bounded_log_speedup_v1",
                        }],
                        "poison_probe_cases": [execute_poison_probe(
                            task_spec, public_routing, proposals[0], realized_solution, solution_dir,
                            task_dir=tasks_root / task_id,
                            verifier_out=trial_dir / "poison-result.json",
                        )],
                    }
                except (OSError, ValueError, TypeError) as exc:
                    budget_errors.append(f"causal intervention realization failed: {exc}")
        transition = post_task_update(
            condition=str(item["condition"]),
            store=store,
            task_id=task_id,
            result=result,
            scored=scored,
            core_repo=repo_root,
            out_dir=trial_dir,
            context_mode=str(item["context_mode"]),
            # Decisions are append-only and survive a campaign resume.  The
            # stream-local ledger is part of the condition store rather than
            # an in-memory side table.
            ledger=ledgers.setdefault(
                stream_id,
                EvolutionDecisionLedger(store / "evolution" / "decisions.jsonl"),
            ),
            allow_maintenance=not budget_errors,
            control_result=control_result,
            control_scored=control_scored,
            public_context=public_routing,
            family_id=str(task_spec.get("family_id", task_spec.get("family", ""))) or None,
            causal_result=causal_result,
            causal_scored=causal_scored,
            validation_evidence=validation_evidence,
            practical_epsilon=practical_effect_threshold(
                float((task_spec.get("measurement") or {}).get("min_improvement_percent", 0.0))
            ),
        )
        attestation_ok, attestation_errors = conditions.verify_attestation(store)
        if not attestation_ok:
            budget_errors.extend(f"condition attestation failed: {error}" for error in attestation_errors)
        record = {
            "experiment": manifest,
            "task_id": task_id,
            "family": task_spec.get("family"),
            "family_id": task_spec.get("family_id", task_spec.get("family")),
            "anchor_instance_id": task_spec.get("anchor_instance_id", task_id),
            "family_instance_digest": task_spec.get("family_instance_digest"),
            "lineage_id": task_spec.get("lineage", {}).get("mutation_template_id", task_id),
            "kind": task_spec.get("kind"),
            "condition": item["condition"],
            "context_mode": item["context_mode"],
            "outer_trial_id": item["outer_trial_id"],
            "phase": item["phase"],
            "agent": agent,
            "agent_usage": usage,
            "budget_errors": budget_errors,
            "attestation_ok": attestation_ok,
            "validity": "invalid" if budget_errors else "valid",
            "transition": transition,
            "score": scored,
        }
        (trial_dir / "trial.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        records.append(record)
    campaign["status"] = "complete"
    campaign["results_claimed"] = bool(
        getattr(args, "claim_results", False)
        and _formal_claim_gate(campaign, records, repo_root / "benchmark" / "population_report.json")
    )
    campaign["aggregate"] = aggregate.aggregate_trials(records)
    (out_dir / "campaign.json").write_text(json.dumps(campaign, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--tasks-root", type=Path, default=root / "benchmark" / "tasks")
    parser.add_argument("--split", type=Path, default=root / "benchmark" / "split" / "sequential.yaml")
    parser.add_argument("--skill-source", type=Path, default=root)
    parser.add_argument("--skill-view", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conditions", default="A,B,C,D")
    parser.add_argument("--context-modes", default="reset")
    parser.add_argument("--outer-trials", type=int, default=3)
    parser.add_argument("--model-id", default="unidentified-agent")
    parser.add_argument("--agent-config", default="{}")
    parser.add_argument("--budgets", default=None, help='JSON, e.g. {"tokens":12000,"tool_calls":80,"wall_time_s":900}')
    parser.add_argument("--agent-command", default=None, help="shell template; receives SPE_* environment variables")
    parser.add_argument(
        "--executor-command",
        default=None,
        help="namespace/container executor template; receives {agent_command}, {worker_root}, {task_dir}, {solution_dir}, {retrieved_context}, {skill_view}, {executor_receipt}",
    )
    parser.add_argument("--claim-results", action="store_true", help="claim only if the formal calibration gate is passed")
    args = parser.parse_args()
    result = run_campaign(args)
    print(json.dumps({"status": result["status"], "schedule_size": result["schedule_size"], "results_claimed": result["results_claimed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
