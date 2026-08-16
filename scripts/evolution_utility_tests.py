#!/usr/bin/env python3
"""RED/GREEN fixtures for causal replay, Bayesian admission, and MDL scoring."""

from __future__ import annotations

import importlib.util
import json
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules while a
    # module is executed; mirror normal import semantics for dynamic fixtures.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    replay = load("run_rule_replay.py")
    assert replay.beta_tail_probability(20, 1, 0.8) > 0.8
    assert replay.beta_tail_probability(2, 10, 0.8) < 0.01
    result = replay.evaluate_cases(
        [{"case_id": f"REG-{index}", "paired_replay": True, "same_fixture_id": f"F-{index}", "intervention_measurements": [1.2] * 512, "baseline_measurements": [1.0] * 512, "control_measured": True, "scientific_ok": True} for index in range(200)],
        epsilon=0.05,
        p_min=0.8,
        delta=0.05,
    )
    assert result["outcome"] == "passed"
    assert result["mean_effect"] > 0.05
    assert result["successes"] == 200 and result["failures"] == 0
    payload = {"rule_id": "PERF-SYNC-004", "epsilon": 0.05, "p_min": 0.8, "delta": 0.05, "cases": [{"case_id": f"REG-{index}", "paired_replay": True, "same_fixture_id": f"F-{index}", "intervention_measurements": [1.2] * 512, "baseline_measurements": [1.0] * 512, "control_measured": True, "scientific_ok": True} for index in range(200)]}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "input.json").write_text(json.dumps(payload), encoding="utf-8")
        manifest = replay.build_manifest(payload, Path("input.json"), Path("replay.json"), "a" * 40)
        (root / "replay.json").write_text(json.dumps(manifest), encoding="utf-8")
        validator = load("validate_evolution.py")
        card = {"rule_id": "PERF-SYNC-004", "status": "canonical", "confidence": {
            key: manifest["result"][key] for key in ("prior_alpha", "prior_beta", "successes", "failures", "p_min", "delta", "posterior_probability")
        }, "promotion": {"replay_manifest": {
            "path": "replay.json", "command": manifest["command"], "case_bundle_path": "input.json", "case_bundle_sha256": manifest["case_bundle_sha256"],
            "harness_revision": manifest["harness_revision"], "result_digest": manifest["result_digest"], "outcome": "passed",
        }}}
        assert validator.validate_replay_manifest(card, root) == []
        # Exercise the real CLI, not only imported functions: this catches
        # undefined variables and missing output artifacts in packaging/CI.
        output = root / "cli-replay.json"
        completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_rule_replay.py"), str(root / "input.json"), str(output)], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        assert output.is_file() and json.loads(output.read_text(encoding="utf-8"))["result"]["outcome"] == "passed"
        cli_manifest = json.loads(output.read_text(encoding="utf-8"))
        assert hashlib.sha256(replay.canonical_json(cli_manifest["result"])).hexdigest() == cli_manifest["result_digest"]

    scorer = load("score_rule_library.py")
    metrics = scorer.score_library(
        [{"rule_id": "A", "status": "canonical", "trigger": {"all": ["x"]}, "rule": {"text": "short"}, "conflicts_with": []}],
        {"A": {"reference_utility": 1.0, "library_utility": 0.9}},
    )
    assert metrics["description_length"] > 0
    assert metrics["distortion"] == 0.1
    assert metrics["objective"] > metrics["distortion"]
    print("evolution utility fixtures: ok")


if __name__ == "__main__":
    main()
