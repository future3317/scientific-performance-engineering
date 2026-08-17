from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from benchmark.formal.reference_executor import ReferenceExecutor


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is required for namespace smoke")
def test_reference_executor_real_worker_and_negative_probes() -> None:
    executor = ReferenceExecutor()
    with tempfile.TemporaryDirectory(prefix="acre-reference-worker-", dir=str(Path.home())) as raw_root:
        root = Path(raw_root)
        for name in ("task", "skill_view", "retrieved_context", "context_state", "solution", "result", "executor_receipt"):
            (root / name).mkdir()
        (root / "task" / "public_task.json").write_text("{}\n", encoding="utf-8")
        worker = (
            "from pathlib import Path; "
            "assert Path('/worker/task/public_task.json').is_file(); "
            "Path('/worker/solution/worker-output.json').write_text('{\\\"ok\\\":true}'); "
            "Path('/worker/result/result.json').write_text('{\\\"worker\\\":true}')"
        )
        receipt_path = root / "executor_receipt" / "receipt.json"
        completed = executor.execute([sys.executable, "-c", worker], root, receipt_path=receipt_path, worker_uid="wsl-test")
        assert completed.returncode == 0, completed.stderr
        assert json.loads((root / "solution" / "worker-output.json").read_text(encoding="utf-8")) == {"ok": True}
        assert json.loads((root / "result" / "result.json").read_text(encoding="utf-8")) == {"worker": True}

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["mode"] == "external_namespace_executor"
        assert receipt["network_mode"] == "none"
        assert receipt["isolation_canary"] is True
        assert receipt["mount_receipt"]["verified"] is True
        assert receipt["network_namespace_attested"] is True
        assert all(receipt["canary_checks"].values())
