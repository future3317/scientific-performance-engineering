from pathlib import Path

import pytest

from scripts.run_active30_calibration import _copy_oracle


@pytest.mark.parametrize(
    "task_id",
    [
        "SCIML-CRYSTAL-DIFFUSION-07",  # a/solution.py
        "SCIML-EQUIV-RECOMPUTE-06",  # a/workspace/solution.py
        "CORE-KERNEL-FUSION-09",  # solution.py
    ],
)
def test_copy_oracle_applies_mixed_reference_patch_headers(task_id, tmp_path):
    task_dir = Path(__file__).parents[1] / "tasks" / task_id

    _copy_oracle(task_dir, tmp_path)

    assert (tmp_path / "solution.py").read_text(encoding="utf-8") != (
        task_dir / "workspace" / "solution.py"
    ).read_text(encoding="utf-8")
