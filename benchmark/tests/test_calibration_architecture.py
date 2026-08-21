from __future__ import annotations

import ast
from pathlib import Path


def test_api_registry_owns_execution_class():
    from benchmark.harness.api import execution_class_for_api

    assert execution_class_for_api("train_loop_v1") == "atomic"
    assert execution_class_for_api("episode_v1") == "episode"


def test_calibration_authority_modules_are_public():
    from benchmark.calibration import approval, bundle, execution, identity, protocol, report

    assert callable(approval.issue_calibration_approval)
    assert callable(bundle.classify_result)
    assert callable(execution.CellExecutor)
    assert callable(identity.canonical_cell_identity)
    assert callable(protocol.load_calibration_protocol)
    assert callable(report.rebuild_calibration_views)


def test_calibration_cli_scripts_do_not_import_private_helpers():
    root = Path(__file__).parents[2]
    for name in ("run_active30_calibration.py", "audit_calibration_resume.py", "approve_calibration.py"):
        tree = ast.parse((root / "scripts" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert all(not alias.name.startswith("_") for alias in node.names), (name, ast.dump(node))
                assert not (node.module or "").startswith("scripts."), (name, ast.dump(node))
