"""One-shot fixed skill synthesis from ALFWorld train demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

from experiments.external_real.alfworld_runner import DEFAULT_MODEL, DeepSeekClient


FAMILY_TASK_TYPES = {
    "clean_place": "pick_clean_then_place_in_recep",
    "heat_place": "pick_heat_then_place_in_recep",
    "cool_place": "pick_cool_then_place_in_recep",
}

SKILL_PROMPT = (
    "Extract a reusable procedure for this ALFWorld task family.\n"
    "Describe search, required state transformation, and placement steps.\n"
    "Do not mention specific object names or instance-specific answers.\n"
    "Return one concise procedural skill, without discussion of the demonstrations."
)


def build_demo_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    annotations = data.get("turk_annotations", {}).get("anns", [])
    first_annotation = annotations[0] if annotations else {}
    high_pddl = data.get("plan", {}).get("high_pddl", [])
    return {
        "task_type": str(data.get("task_type", "")),
        "task_desc": str(first_annotation.get("task_desc", "")),
        "high_descs": [str(item) for item in first_annotation.get("high_descs", [])],
        "discrete_actions": [dict(item.get("discrete_action", {})) for item in high_pddl],
    }


def select_demo_paths(train_root: Path, task_type: str, count: int, seed: int) -> list[Path]:
    candidates: list[Path] = []
    for path in train_root.rglob("traj_data.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("task_type") == task_type:
            candidates.append(path)
    if len(candidates) < count:
        raise ValueError(f"need {count} demonstrations for {task_type}, found {len(candidates)}")
    return sorted(random.Random(seed).sample(sorted(candidates), count))


def _skill_request(client: DeepSeekClient, prompt: str) -> dict[str, Any]:
    response = client.client.chat.completions.create(
        model=client.model,
        temperature=client.temperature,
        max_tokens=512,
        extra_body={"thinking": {"type": "disabled"}},
        messages=[
            {"role": "system", "content": "Follow the requested procedure format exactly."},
            {"role": "user", "content": prompt},
        ],
    )
    message = response.choices[0].message
    content = message.content or ""
    reasoning = getattr(message, "reasoning_content", "") or ""
    text = content or reasoning
    return {
        "provider_raw": {"content": content, "reasoning_content": reasoning},
        "response_field_used": "message.content" if content else "message.reasoning_content",
        "skill_text": text.strip(),
    }


def synthesize_skills(
    data_root: Path,
    output_dir: Path,
    client: DeepSeekClient,
    count: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    train_root = data_root / "json_2.1.1" / "train"
    skills: dict[str, Any] = {}
    for family, task_type in FAMILY_TASK_TYPES.items():
        paths = select_demo_paths(train_root, task_type, count, seed)
        demos = [build_demo_payload(json.loads(path.read_text(encoding="utf-8"))) for path in paths]
        prompt = SKILL_PROMPT + "\n\nDemonstrations:\n" + json.dumps(demos, indent=2, sort_keys=True)
        response = _skill_request(client, prompt)
        if not response["skill_text"]:
            raise RuntimeError(f"empty skill response for {family}")
        skills[family] = {
            "task_type": task_type,
            "demonstration_count": len(demos),
            "demonstration_paths": [str(path) for path in paths],
            "provider_raw": response["provider_raw"],
            "response_field_used": response["response_field_used"],
            "skill_text": response["skill_text"],
            "sha256": hashlib.sha256(response["skill_text"].encode("utf-8")).hexdigest(),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps({
        "benchmark": "ALFWorld",
        "split": "train",
        "families": list(FAMILY_TASK_TYPES),
        "demonstrations_per_family": count,
        "selection_seed": seed,
        "model": client.model,
        "temperature": client.temperature,
        "thinking": "disabled",
        "refinement_loops": 0,
        "provider_calls": 3,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {"schema_version": 1, "skills": skills, "provider_calls": 3}
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "REPORT.md").write_text(
        "# ALFWorld fixed skill synthesis\n\n"
        "Three family-specific procedural skills were generated with one DeepSeek call per family.\n\n"
        + "\n".join(
            f"- `{family}`: {row['demonstration_count']} demonstrations, SHA256 `{row['sha256']}`"
            for family, row in skills.items()
        )
        + "\n\nNo refinement loop, extra seed, or outcome-driven selection was used.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    args = parser.parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required at runtime")
    client = DeepSeekClient(api_key, args.base_url, args.model)
    synthesize_skills(args.data_root, args.output_dir, client, args.count, args.seed)


if __name__ == "__main__":
    main()
