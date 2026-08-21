from __future__ import annotations

import ast
from pathlib import Path


def test_api_registry_owns_execution_class():
    from benchmark.harness.api import execution_class_for_api

    assert execution_class_for_api("train_loop_v1") == "atomic"
    assert execution_class_for_api("episode_v1") == "episode"


def test_calibration_authority_modules_are_public():
    from benchmark.calibration import approval, bundle, execution, identity, protocol, report, state

    assert callable(approval.issue_calibration_approval)
    assert callable(bundle.classify_result)
    assert callable(execution.CellExecutor)
    assert callable(identity.canonical_cell_identity)
    assert callable(protocol.load_calibration_protocol)
    assert callable(report.rebuild_calibration_views)
    assert state.derive_cell_state({"execution_validity": "valid", "efficacy_eligible": True, "calibration_status": "eligible"}) == "eligible"


def test_calibration_cli_scripts_do_not_import_private_helpers():
    root = Path(__file__).parents[2]
    for name in ("run_active30_calibration.py", "audit_calibration_resume.py", "approve_calibration.py"):
        tree = ast.parse((root / "scripts" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert all(not alias.name.startswith("_") for alias in node.names), (name, ast.dump(node))
                assert not (node.module or "").startswith("scripts."), (name, ast.dump(node))


def test_active30_cli_is_only_argument_parsing_and_orchestration():
    root = Path(__file__).parents[2]
    source = (root / "scripts" / "run_active30_calibration.py").read_text(encoding="utf-8")
    assert "def _bounded_verifier_result" not in source
    assert "def _resource_blocked_result" not in source
    assert "def _calibration_record" not in source
    assert "run_calibration_campaign" in source
    assert len(source.splitlines()) < 40


def test_resource_blocked_verifier_has_one_cleanup_branch():
    source = (Path(__file__).parents[2] / "benchmark" / "calibration" / "campaign.py").read_text(encoding="utf-8")
    assert source.count('if cleanup.get("residual_detected")') == 1
    assert "noise control left a residual process group" not in source


def test_formal_and_episode_verifier_use_shared_cell_executor():
    root = Path(__file__).parents[2]
    formal = (root / "benchmark" / "formal" / "run_campaign.py").read_text(encoding="utf-8")
    verifier = (root / "benchmark" / "harness" / "verifier.py").read_text(encoding="utf-8")
    assert "from benchmark.calibration.execution import CellExecutor" in formal
    assert "from benchmark.calibration.execution import CellExecutor" in verifier
    assert "subprocess.Popen(" not in verifier


def test_population_structural_checks_have_a_single_module():
    root = Path(__file__).parents[2]
    structural = (root / "benchmark" / "population" / "structural.py").read_text(encoding="utf-8")
    validator = (root / "benchmark" / "taskgen" / "validate_population.py").read_text(encoding="utf-8")
    assert "def artifact_findings" in structural
    assert "def metadata_findings" in structural
    assert "def _artifact_findings" not in validator
    assert "def _metadata_findings" not in validator
    assert "def build_report" not in validator
    assert "def build_pilot_calibration" not in validator


def test_executor_digest_excludes_formal_reporting_surface():
    source = (Path(__file__).parents[2] / "benchmark" / "calibration" / "execution.py").read_text(encoding="utf-8")
    assert "approval.py" not in source
    assert "report.py" not in source
    assert "bundle.py" not in source
    assert "calibration_protocol.json" in source


def test_production_benchmark_layers_do_not_import_cli_scripts():
    root = Path(__file__).parents[2]
    for directory in (root / "benchmark" / "formal", root / "benchmark" / "harness"):
        for path in directory.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "from scripts." not in source and "import scripts." not in source, path


def test_calibration_authority_does_not_depend_on_formal_or_population_cli():
    root = Path(__file__).parents[2]
    calibration = root / "benchmark" / "calibration"
    for path in calibration.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "benchmark.formal" not in source, path
        assert "benchmark.taskgen.validate_population" not in source, path


def test_population_cli_is_only_a_wrapper_around_current_authority():
    root = Path(__file__).parents[2]
    source = (root / "benchmark" / "taskgen" / "validate_population.py").read_text(encoding="utf-8")
    assert "from benchmark.calibration.report import main" in source
    assert "def build_report" not in source
    assert "def build_pilot_calibration" not in source
