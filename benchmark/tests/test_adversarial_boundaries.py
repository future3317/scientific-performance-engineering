#!/usr/bin/env python3
"""Negative tests for the P1 condition, attestation, and evolution boundaries."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.harness import conditions, evolution
from benchmark.harness.evolution_ledger import EvolutionDecisionLedger
from core.models import identifier_digest
from scripts.render_skill_view import render_skill_view


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "skill"
        source.mkdir()
        (source / "SKILL.md").write_text("# skill\n", encoding="utf-8")
        bundle = root / "bundle"
        render_skill_view(source, bundle)

        # A forged or incomplete manifest is rejected before materialization.
        manifest_path = bundle / "skill_view_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"] = []
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        try:
            conditions.materialize_condition("C", bundle, root / "forged")
        except ValueError as exc:
            assert "invalid skill-view bundle" in str(exc)
        else:
            raise AssertionError("forged skill-view manifest was accepted")

        # Re-render a valid bundle for the remaining boundary checks.
        render_skill_view(source, bundle)
        c_store = root / "c"
        conditions.materialize_condition("C", bundle, c_store)
        (c_store / "experience" / "inbox" / "raw.json").write_text("{}\n", encoding="utf-8")
        ok, errors = conditions.verify_condition_policy(c_store)
        assert ok and errors == [], errors
        (c_store / "rules").mkdir()
        (c_store / "rules" / "canonical.json").write_text("{}\n", encoding="utf-8")
        ok, errors = conditions.verify_condition_policy(c_store)
        assert not ok and any("outside experience/inbox" in item for item in errors), errors

        # A field-only manifest edit is detected even when file hashes are unchanged.
        b_store = root / "b"
        conditions.materialize_condition("B", bundle, b_store)
        b_manifest = b_store / "condition_manifest.json"
        payload = json.loads(b_manifest.read_text(encoding="utf-8"))
        payload["context_mode"] = "carry"
        b_manifest.chmod(0o644)
        b_manifest.write_text(json.dumps(payload), encoding="utf-8")
        ok, errors = conditions.verify_attestation(b_store)
        assert not ok and any("digest mismatch" in item for item in errors), errors
        ok, errors = conditions.verify_condition_policy(b_store)
        assert not ok and any("digest mismatch" in item for item in errors), errors

        # Poison labels are removed recursively before records become visible.
        poison_store = root / "poison"
        conditions.materialize_condition("C", bundle, poison_store)
        evolution._apply_phase_injections(
            poison_store,
            "C",
            {"index": 1, "inject_experiences": [{"id": "nested", "metadata": {"poisoned": True}}]},
        )
        stored = json.loads((poison_store / "experience" / "inbox" / f"{identifier_digest('nested')}.json").read_text(encoding="utf-8"))
        assert "poisoned" not in json.dumps(stored), stored

        ledger = EvolutionDecisionLedger()
        ledger.record("r1", 1, "digest", "candidate")
        assert ledger.has("r1", 1)
        assert ledger.has_replay("r1", 1, "digest")
        assert not ledger.has_replay("r1", 1, "other-digest")
        assert not ledger.has("r2", 1)

    print("test_adversarial_boundaries: OK")


if __name__ == "__main__":
    main()
