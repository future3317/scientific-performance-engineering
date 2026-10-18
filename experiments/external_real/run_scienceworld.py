from __future__ import annotations

import json
import os
import re
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adapters.scienceworld_native import ScienceWorldAdapter
from .freeze import freeze_decisions
from .models import FALLBACK_TO_BASE_AGENT, PolicyDecision
from .paired_runner import run_paired
from .protocol import ProtocolManifest
from .scorer import score_policies
from .skill_pipeline import freeze_skill
from .context.scienceworld import PUBLIC_FIELDS


def parse_action_plan(raw: str) -> list[str]:
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, Mapping):
        actions = parsed.get("actions")
        if isinstance(actions, list):
            return [str(action).strip() for action in actions if str(action).strip()]
    if isinstance(parsed, list):
        return [str(action).strip() for action in parsed if str(action).strip()]
    actions: list[str] = []
    for line in text.splitlines():
        item = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if item:
            actions.append(item)
    return actions


SMOKE_SKILL_TEXT = (
    "Plan from the stated goal before emitting actions. Identify the target object "
    "and required state change. Satisfy navigation and possession prerequisites "
    "before manipulation. Finish with the requested action followed by wait1."
)


def build_scienceworld_smoke_task_manifest(
    scienceworld_root: Path,
    *,
    seed: int = 0,
) -> dict[str, list[dict[str, object]]]:
    """Build a deterministic train/dev-derived manifest without gold fields."""
    import sys

    root = scienceworld_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scienceworld import ScienceWorldEnv

    task_rows = json.loads((root / "scienceworld" / "tasks.json").read_text(encoding="utf-8"))
    topics = {str(row["task_name"]): str(row["topic"]) for row in task_rows}
    env = ScienceWorldEnv("", str(root / "scienceworld" / "scienceworld.jar"), envStepLimit=50)
    train_pairs: list[tuple[str, int]] = []
    dev_pairs: list[tuple[str, int]] = []
    try:
        for task_name in sorted(env.get_task_names()):
            env.load(task_name, 0, "")
            train_pairs.extend((task_name, int(index)) for index in env.get_variations_train())
            dev_pairs.extend((task_name, int(index)) for index in env.get_variations_dev())
        random.Random(seed).shuffle(train_pairs)
        random.Random(seed + 1).shuffle(dev_pairs)
        selected = {
            "construction": train_pairs[:20],
            "verification": train_pairs[20:30],
            "support": train_pairs[30:50],
            "test": dev_pairs[:20],
        }
        if any(len(selected[name]) != expected for name, expected in {
            "construction": 20, "verification": 10, "support": 20, "test": 20
        }.items()):
            raise ValueError("ScienceWorld train/dev pools are too small for the registered smoke manifest")

        manifest: dict[str, list[dict[str, object]]] = {name: [] for name in selected}
        for split, pairs in selected.items():
            for task_name, variation_idx in pairs:
                env.load(task_name, variation_idx, "")
                initial_observation, _ = env.reset()
                task_id = f"scienceworld:{task_name}:{variation_idx}"
                manifest[split].append({
                    "task_id": task_id,
                    "task_name": task_name,
                    "variation_idx": variation_idx,
                    "task_family": topics.get(task_name, "unknown"),
                    "goal_operation": task_name,
                    "target_state_change": env.get_task_description(),
                    "variation_bucket": split,
                    "task_text": env.get_task_description(),
                    "initial_observation": initial_observation,
                })
        return manifest
    finally:
        env.close()


class ScienceWorldProviderBackend:
    def __init__(self, *, scienceworld_root: Path, model_id: str, max_actions: int = 20) -> None:
        self.scienceworld_root = scienceworld_root.resolve()
        self.model_id = model_id
        self.max_actions = max_actions
        self.provider_calls = 0

    def _prompt(self, payload: Mapping[str, object], skill: object | None, observation: str, actions: Sequence[str]) -> str:
        skill_text = getattr(skill, "body", "") if skill is not None else ""
        return (
            "You are operating the public ScienceWorld simulator. Return JSON only with "
            "an actions array. Use only the listed public action examples. Do not explain.\n"
            f"Task: {payload.get('task_description', '')}\n"
            f"Current observation: {observation}\n"
            f"Public action examples: {json.dumps(list(actions)[:120], ensure_ascii=False)}\n"
            f"Optional fixed skill: {skill_text}\n"
            'Schema: {"actions": ["action 1", "action 2"]}'
        )

    def execute(self, native_payload: dict[str, object], skill: object | None, seed: int) -> Mapping[str, object]:
        import sys

        if str(self.scienceworld_root) not in sys.path:
            sys.path.insert(0, str(self.scienceworld_root))
        from scienceworld import ScienceWorldEnv
        from benchmark.formal.host_agent import _api_call_with_retry

        random.seed(seed)
        env = ScienceWorldEnv("", str(self.scienceworld_root / "scienceworld" / "scienceworld.jar"), envStepLimit=50)
        try:
            env.load(str(native_payload["task_name"]), int(native_payload["variation_idx"]), "")
            observation, _ = env.reset()
            public_actions = [str(item) for item in env.get_possible_actions()]
            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": "You produce deterministic ScienceWorld action plans."},
                    {"role": "user", "content": self._prompt(native_payload, skill, str(observation), public_actions)},
                ],
                "temperature": 0.0,
                "top_p": 1,
                "max_tokens": 768,
                "thinking": {"type": "disabled"},
                "reasoning_effort": "none",
            }
            response = _api_call_with_retry(
                os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("MP_API_KEY", ""),
                payload,
                timeout_s=120,
            )
            self.provider_calls += 1
            raw = response["choices"][0]["message"].get("content", "")
            actions = parse_action_plan(str(raw))
            score = 0.0
            completed = False
            executed = 0
            for action in actions[: self.max_actions]:
                _observation, _reward, completed, info = env.step(action)
                executed += 1
                score = float(info.get("score", score))
                if completed:
                    break
            return {
                "official_score": score,
                "success": bool(completed and score >= 100.0),
                "execution_valid": True,
                "native_metrics": {"steps": executed, "plan_actions": len(actions), "completed": completed},
            }
        except Exception as exc:
            return {
                "official_score": 0.0,
                "success": False,
                "execution_valid": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "native_metrics": {},
            }
        finally:
            env.close()


def run_scienceworld_smoke(
    manifest: ProtocolManifest,
    out_dir: Path,
    *,
    task_manifest: Mapping[str, Sequence[Mapping[str, object]]],
    backend: Any,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = ScienceWorldAdapter(
        task_manifest,
        backend=backend,
        benchmark_revision=manifest.benchmark_revision,
        evaluator_revision="ScienceWorld-official-score-1.2.3",
    )
    tasks = adapter.load_tasks("test")
    assert all(set(task.public_context).issubset(set(PUBLIC_FIELDS)) for task in tasks)
    decision_timestamp = datetime.now(timezone.utc).isoformat()
    decisions = {
        "acre": [
            PolicyDecision.create(
                task_id=task.task_id,
                policy_name="acre",
                deploy=False,
                reason=FALLBACK_TO_BASE_AGENT,
                decision_timestamp=decision_timestamp,
            )
            for task in tasks
        ]
    }
    decision_path = out_dir / "test_policy_decisions.jsonl"
    decision_digest = freeze_decisions(decision_path, decisions["acre"])
    outcomes = run_paired(adapter, tasks, freeze_skill(
        SMOKE_SKILL_TEXT, (), (), "ScienceWorld smoke fixed skill", "scienceworld", 0
    ), repeat_seeds=(0,))
    scored = score_policies(decisions, outcomes, lambda outcome, deploy: {
        "score": outcome.skill_score if deploy else outcome.baseline_score,
    })
    valid = sum(outcome.execution_valid for outcome in outcomes)
    report: dict[str, object] = {
        "schema_version": 1,
        "stage": "scienceworld_smoke",
        "provider_calls": int(getattr(backend, "provider_calls", 0)),
        "decision_digest": decision_digest,
        "outcome_before_decision": False,
        "information_flow_valid": True,
        "evaluator_valid": all(outcome.execution_valid for outcome in outcomes),
        "environment_validity": valid / len(outcomes) if outcomes else 0.0,
        "acre_coverage": scored["acre"]["coverage"],
        "metrics": scored,
        "outcomes": [outcome.__dict__ for outcome in outcomes],
        "claim_boundary": "ScienceWorld smoke engineering validation; not treatment efficacy",
    }
    (out_dir / "smoke_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "REPORT.md").write_text(
        "# ScienceWorld smoke\n\n"
        f"- provider calls: {report['provider_calls']}\n"
        f"- environment validity: {report['environment_validity']}\n"
        f"- ACRE coverage diagnostic: {report['acre_coverage']}\n"
        "- claim boundary: engineering validation only; not treatment efficacy\n",
        encoding="utf-8",
    )
    return report


def build_smoke_manifest(
    benchmark_revision: str,
    model_id: str,
    seeds: Sequence[int],
) -> ProtocolManifest:
    if not seeds:
        raise ValueError("smoke requires at least one seed")
    return ProtocolManifest.from_dict({
        "benchmark": "scienceworld",
        "benchmark_revision": benchmark_revision,
        "protocol_revision": "scienceworld-smoke-20260915",
        "split_seed": "scienceworld-smoke-v1",
        "model_id": model_id,
        "environment": {"official_simulator": True, "provider_calls_authorized": True},
        "feature_fields": [
            "task_family",
            "goal_operation",
            "object_category",
            "material_category",
            "target_state_change",
            "requires_measurement",
            "requires_container",
            "variation_bucket",
        ],
        "metadata": {
            "generation_seeds": [int(seed) for seed in seeds],
            "construction_n": 20,
            "verification_n": 10,
            "support_n": 20,
            "test_n": 20,
            "provider_calls_authorized": True,
            "claim_boundary": "ScienceWorld smoke engineering validation; not treatment efficacy",
        },
    })


def evaluate_smoke_gate(report: Mapping[str, object]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    environment_validity = report.get("environment_validity")
    if not isinstance(environment_validity, (int, float)) or float(environment_validity) < 0.95:
        reasons.append("environment validity below 0.95")
    if not report.get("decision_digest"):
        reasons.append("decision digest missing")
    if bool(report.get("outcome_before_decision", False)):
        reasons.append("outcome before decision")
    if report.get("information_flow_valid") is not True:
        reasons.append("information flow invalid")
    if report.get("evaluator_valid") is not True:
        reasons.append("official evaluator invalid")
    return not reasons, reasons
