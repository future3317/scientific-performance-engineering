"""Sequential A/C/D runner for immutable evolving-skill treatment evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from .contracts import DecisionRecord
from .decision_ledger import DecisionLedger
from .policies import PolicyDecision
from .skill_versions import SkillVersion, SkillVersionStore


def decision_for_condition(
    condition: str,
    context: dict[str, Any],
    raw_experiences: list[str],
    skill_version: SkillVersion | None,
) -> PolicyDecision:
    condition = str(condition).upper()
    if condition == "A":
        return PolicyDecision(False, 0.0, 1.0, "no_persistent_skill")
    if condition == "C":
        return PolicyDecision(bool(raw_experiences), 0.5 if raw_experiences else 1.0, 0.5, "raw_experience_retrieval" if raw_experiences else "no_prior_experience")
    if condition == "D":
        if skill_version is None:
            return PolicyDecision(False, 0.0, 1.0, "no_promoted_skill")
        return PolicyDecision(True, 1.0, 0.0, "acre_certificate", certificate_id=f"cert:{skill_version.version_id}", skill_version=skill_version.version_id)
    raise ValueError(f"unknown evolving condition: {condition}")


def promote_candidate(
    store: SkillVersionStore,
    *,
    task_index: int,
    content: str,
    evidence_ids: Iterable[str],
    prior_effects: list[float],
    current_effect: float,
    minimum_effect: float,
) -> SkillVersion | None:
    if float(current_effect) < float(minimum_effect):
        return None
    if any(float(effect) < 0.0 for effect in prior_effects):
        return None
    parent_id = store.versions()[-1].version_id if store.versions() else None
    return store.create(
        parent_id=parent_id,
        task_index=task_index,
        content=str(content),
        evidence_ids=tuple(str(item) for item in evidence_ids),
        certificate={"status": "promoted", "gate": "paired_nonnegative_effects", "minimum_effect": float(minimum_effect)},
        provenance={"policy": "ACRE", "task_index": int(task_index)},
    )


def build_condition_prompt(task_input: str, condition: str, raw_experiences: list[str], skill_version: SkillVersion | None) -> str:
    condition = str(condition).upper()
    if condition == "A":
        return task_input
    if condition == "C":
        evidence = "\n\n".join(raw_experiences[-3:])
        return task_input + "\n\n## Prior raw experience (reuse only)\n" + evidence
    if condition == "D" and skill_version is not None:
        return task_input + "\n\n## Certified evolving skill\n" + skill_version.content
    return task_input


def run_sequential_campaign(
    tasks: Iterable[dict[str, Any]],
    *,
    condition: str,
    run_id: str,
    out_dir: str | Path,
    agent_call: Callable[[str, dict[str, Any], str], tuple[str, dict[str, Any]]],
    score_call: Callable[[str, dict[str, Any]], float],
    minimum_effect: float = 0.05,
    initial_skill_content: str | None = None,
) -> dict[str, Any]:
    condition = str(condition).upper()
    if condition not in {"A", "C", "D"}:
        raise ValueError(f"unsupported evolving condition: {condition}")
    tasks = list(tasks)
    out_dir = Path(out_dir)
    ledger = DecisionLedger(out_dir / "decision_ledger.jsonl")
    versions = SkillVersionStore(out_dir / "skill_versions.jsonl")
    raw_experiences: list[str] = [str(initial_skill_content)] if initial_skill_content else []
    effects: list[float] = []
    rows: list[dict[str, Any]] = []
    usage: list[dict[str, Any]] = []
    current_skill: SkillVersion | None = None
    if condition == "D" and initial_skill_content:
        current_skill = versions.create(
            parent_id=None,
            task_index=-1,
            content=str(initial_skill_content),
            evidence_ids=("initial-verified-skill",),
            certificate={"status": "verified_snapshot", "source": "protocol_input"},
            provenance={"policy": "shared_v0", "revision": "public-acre-treatment-20260915-v1"},
            version_id="skill-v0",
        )
    for index, task in enumerate(tasks):
        context = dict(task.get("metadata") or {})
        decision = decision_for_condition(condition, context, raw_experiences, current_skill)
        record = DecisionRecord(
            run_id=run_id,
            condition=condition,
            task_id=str(task["instance_id"]),
            task_index=index,
            stage="online_treatment",
            policy=condition,
            decision_id=f"{run_id}:{index}",
            reuse=decision.reuse,
            score=decision.score,
            uncertainty=decision.uncertainty,
            reason=decision.reason,
            certificate_id=decision.certificate_id,
            skill_version=decision.skill_version,
            public_context_digest=str(task.get("public_context_digest", task["instance_id"])),
            available_evidence_ids=tuple(f"experience:{i}" for i in range(len(raw_experiences))),
        )
        ledger.append(record)
        baseline_output, baseline_usage = agent_call(str(task["input"]), task, "A")
        condition_output = baseline_output
        condition_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if decision.reuse:
            condition_output, condition_usage = agent_call(build_condition_prompt(str(task["input"]), condition, raw_experiences, current_skill), task, condition)
        baseline_score = float(score_call(baseline_output, task))
        condition_score = float(score_call(condition_output, task))
        delta = condition_score - baseline_score
        usage.extend([dict(baseline_usage, stage="task_baseline"), dict(condition_usage, stage="task_condition")])
        row = {
            "instance_id": task["instance_id"],
            "condition": condition,
            "task_index": index,
            "baseline_score": baseline_score,
            "skill_score": condition_score,
            "measurement_reached": True,
            "policy_decisions": {condition: decision.to_dict()},
            "delta": delta,
        }
        rows.append(row)
        ledger.append_outcome(record.decision_id, {"task_index": index, "official_score": condition_score, "delta": delta})
        raw_experiences.append(condition_output)
        if condition == "D":
            # A first successful experience can be proposed only after the
            # task decision has been serialized.  The candidate replay is an
            # evolution-cost measurement, not a deployment on this task.
            candidate_delta = delta
            if current_skill is None:
                candidate_text = baseline_output[:4000]
                candidate_output, candidate_usage = agent_call(
                    build_condition_prompt(str(task["input"]), "D", raw_experiences, SkillVersion(
                        version_id="candidate", parent_id=None, task_index=index,
                        content=candidate_text, content_sha256="", evidence_ids=(), certificate={}, provenance={},
                    )),
                    task,
                    "D_evolution_acquisition",
                )
                candidate_delta = float(score_call(candidate_output, task)) - baseline_score
                usage.append(dict(candidate_usage, stage="evolution_acquisition"))
            candidate_content = (
                current_skill.content + "\n\n## Newly observed successful pattern\n" + condition_output[:1200]
                if current_skill is not None
                else baseline_output[:4000]
            )
            candidate = promote_candidate(
                versions,
                task_index=index,
                content=candidate_content,
                evidence_ids=(record.decision_id,),
                prior_effects=effects,
                current_effect=candidate_delta,
                minimum_effect=minimum_effect,
            )
            if candidate is not None:
                current_skill = candidate
        effects.append(delta)
    return {
        "schema_version": 1,
        "condition": condition,
        "run_id": run_id,
        "rows": rows,
        "usage": usage,
        "skill_versions": [version.to_dict() for version in versions.versions()],
        "decision_ledger": ledger.rows(),
        "observation_completion": sum(bool(row["measurement_reached"]) for row in rows) / len(rows) if rows else 0.0,
    }
