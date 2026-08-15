from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from pathlib import Path

from benchmark.formal import run_campaign
from core.acre.factorial import FactorialBlock, FactorialEngine
from core.acre.router import BundleCertificate
from scripts.render_skill_view import render_skill_view


ROOT = Path(__file__).resolve().parents[2]


def test_skill_view_uses_explicit_script_allowlist() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source"
        source.mkdir()
        (source / "SKILL.md").write_text("# skill\n", encoding="utf-8")
        (source / "scripts").mkdir()
        (source / "scripts" / "validate_skill.py").write_text("# allowed\n", encoding="utf-8")
        (source / "scripts" / "run_pilot_surface_experiments.py").write_text("# harness\n", encoding="utf-8")
        manifest = render_skill_view(source, Path(tmp) / "bundle")
        assert "scripts/validate_skill.py" in manifest["files"]
        assert "scripts/run_pilot_surface_experiments.py" not in manifest["files"]
        assert manifest["source_snapshot"] == "redacted"


def test_condition_attestation_redacts_host_snapshot_path() -> None:
    from benchmark.harness.conditions import materialize_condition

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source"
        source.mkdir()
        (source / "SKILL.md").write_text("# skill\n", encoding="utf-8")
        from scripts.render_skill_view import render_skill_view
        bundle = Path(tmp) / "bundle"
        render_skill_view(source, bundle)
        store = Path(tmp) / "store"
        manifest = materialize_condition("C", bundle, store)
        assert manifest["source_snapshot"] == "redacted"
        assert str(bundle) not in (store / "condition_manifest.json").read_text(encoding="utf-8")


def test_bundle_certificate_requires_pairwise_null_residual_for_auto_deployment() -> None:
    assert BundleCertificate(("a", "b", "c"), {}, -0.01, 0.02, "pairwise_certified").bounded_auto_allowed
    assert not BundleCertificate(("a", "b", "c"), {}, 0.12, 0.20, "hyperedge_required").bounded_auto_allowed
    assert not BundleCertificate(("a", "b", "c"), {}, -0.20, -0.12, "hyperedge_required").bounded_auto_allowed


def test_router_marks_significant_three_way_residual_as_hyperedge() -> None:
    from benchmark.tests.test_acre_v04 import _rule, _state
    from core.acre.router import ConservativeCausalRouter
    from core.models import TaskContext

    specs = [_rule("a", "a"), _rule("b", "b"), _rule("c", "c")]
    states = {item.rule_id: _state(item.rule_id) for item in specs}
    decision = ConservativeCausalRouter(token_budget=8).route(
        specs, states, [], {}, TaskContext("x", {}, {}, {}, {}),
        higher_order_evidence={"lcb": 0.12, "ucb": 0.20, "practical_margin": 0.05},
    )
    assert decision.bundle_certificate is not None
    assert decision.bundle_certificate.status == "hyperedge_required"
    assert not decision.bundle_certificate.bounded_auto_allowed


def test_power_curve_realizes_requested_gamma_without_clipping() -> None:
    from benchmark.interaction.factorial_bench import run_interaction_power_curve

    report = run_interaction_power_curve(blocks=(8,), repetitions=1)
    for row in report["results"]:
        assert abs(row["realized_gamma"] - row["target_gamma"]) < 1e-12


def test_higher_order_null_state_is_distinct_from_unresolved() -> None:
    from core.acre.factorial import ThreeWayBlock, estimate_higher_order

    blocks = [ThreeWayBlock(str(index), {arm: 0.0 for arm in ("000", "001", "010", "011", "100", "101", "110", "111")}) for index in range(2048)]
    estimate = estimate_higher_order(blocks, practical_margin=0.05)
    # The formal certificate now uses the coverage-preserving Hoeffding bound;
    # a null point estimate need not be certified inside a narrow margin yet.
    assert estimate.status == "unresolved"


def test_formal_claim_requires_explicit_gate_and_complete_records() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        args = Namespace(
            repo_root=ROOT,
            tasks_root=ROOT / "benchmark" / "tasks",
            split=ROOT / "benchmark" / "split" / "sequential.yaml",
            skill_source=ROOT,
            skill_view=None,
            out=Path(tmp) / "dry",
            conditions="A,B,C,D",
            context_modes="reset",
            outer_trials=1,
            model_id="test",
            agent_config="{}",
            budgets=None,
            agent_command=None,
            claim_results=True,
        )
        result = run_campaign.run_campaign(args)
        assert result["results_claimed"] is False


def test_formal_agent_view_excludes_hidden_task_truth_and_isolated_cwd() -> None:
    from benchmark.formal.run_campaign import materialize_agent_task

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "task"
        (source / "workspace").mkdir(parents=True)
        (source / "public_tests").mkdir()
        (source / "hidden_verifier").mkdir()
        (source / "oracle").mkdir()
        (source / "task.yaml").write_text(
            "schema_version: 1\ntask_id: T\ntitle: public\ntrack: spe_core\nfamily: compiler\nmechanism: hidden\nkind: counterexample\n"
            "requires_cuda: false\ntime_budget_s: 10\nworkspace:\n  entrypoint: solution.py\n  api: train_loop_v1\n"
            "measurement:\n  primary_metric: step_ms_p50\n  higher_is_better: false\n  warmup_iterations: 1\n  measured_iterations: 2\n  repetitions: 1\ncorrectness:\n  num_fresh_inputs: 1\n  reference: fp64_recompute\n  tolerance:\n    rtol: 0.1\n    atol: 0.1\nscientific_gates: [finite_loss]\noracle:\n  expected_speedup_range: [0.9, 1.1]\nlineage:\n  source: synthetic\n  mutation_template_id: MT-HIDDEN\nauthoring_note: secret\n",
            encoding="utf-8",
        )
        (source / "metadata.json").write_text('{"authoring_note":"secret"}\n', encoding="utf-8")
        (source / "workspace" / "solution.py").write_text("x = 1\n", encoding="utf-8")
        (source / "hidden_verifier" / "checks.py").write_text("secret = 1\n", encoding="utf-8")
        (source / "oracle" / "truth.json").write_text("{}\n", encoding="utf-8")
        public = Path(tmp) / "public"
        materialize_agent_task(source, public)
        files = {path.relative_to(public).as_posix() for path in public.rglob("*") if path.is_file()}
        assert "public_task.json" in files and "workspace/solution.py" in files
        assert "task.yaml" not in files and "metadata.json" not in files
        public_task = json.loads((public / "public_task.json").read_text(encoding="utf-8"))
        serialized = json.dumps(public_task)
        assert "counterexample" not in serialized
        assert "oracle" not in serialized
        assert "hidden" not in serialized
        assert not any(name.startswith("oracle/") or name.startswith("hidden_verifier/") for name in files)


def test_factorial_engine_and_relation_identifier_share_decision_policy() -> None:
    from core.acre.relation import RelationIdentifier

    engine = FactorialEngine(delta=0.05, practical_margin=0.05)
    values = {"00": 0.0, "10": 0.1, "01": 0.1, "11": 0.9}
    for index in range(256):
        engine.add_block(FactorialBlock(str(index), values))
    estimate = engine.estimate()
    identified = RelationIdentifier(practical_margin=0.05).identify({"context": estimate})
    assert identified.context_decisions["context"] == estimate.decision
