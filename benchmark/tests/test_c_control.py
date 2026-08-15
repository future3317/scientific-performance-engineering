#!/usr/bin/env python3
"""C is raw retrieval; C_STRESS is the append-only ablation."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.harness import conditions, experience_retrieval
from scripts.render_skill_view import render_skill_view


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "skill"
        root.mkdir()
        (root / "SKILL.md").write_text("# skill\n", encoding="utf-8")
        bundle = Path(tmp) / "bundle"
        render_skill_view(root, bundle)
        c = Path(tmp) / "c"
        manifest = conditions.materialize_condition("C", bundle, c)
        assert manifest["injection_policy"]["mode"] == "raw_experience_retrieval"
        inbox = c / "experience" / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / "exp.json").write_text(json.dumps({"case_id": "EXP-1", "lesson": "cache"}), encoding="utf-8")
        records = experience_retrieval.retrieve_raw_experiences(c, query="cache", token_budget=4096)
        assert records == [{"case_id": "EXP-1", "lesson": "cache"}]
        assert not any("RuleSpec" in str(record) for record in records)

    assert conditions.INJECTION_POLICIES["C_STRESS"]["mode"] == "inbox_any"
    assert conditions.INJECTION_POLICIES["C"]["retrieval_budget_tokens"] == conditions.INJECTION_POLICIES["D"]["retrieval_budget_tokens"]
    print("test_c_control: OK")


if __name__ == "__main__":
    main()
