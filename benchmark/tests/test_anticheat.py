#!/usr/bin/env python3
"""Standalone assert-script tests for harness/anticheat.py (no pytest)."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.harness import anticheat


def _rules(findings):
    return {f["rule"] for f in findings}


def _hard_rules(findings):
    return {f["rule"] for f in findings if f["severity"] == "hard_fail"}


def main() -> None:
    # --- hard-fail patterns ----------------------------------------------------
    cases = {
        "monkey_patch": "import time\ntime.sleep = lambda x: None\n",
        "monkey_patch_via_setattr": "import torch\nsetattr(torch, 'matmul', lambda a, b: a)\n",
        "seed_tampering": "import random\nrandom.seed(1234)\n",
        "sys_modules_mutation": "import sys\nsys.modules['time'] = None\n",
        "network_import": "import socket\n",
        "network_import_from": "from urllib.request import urlopen\n",
        "ctypes_import": "import ctypes\n",
        "process_creation": "import subprocess\nsubprocess.run(['ls'])\n",
        "process_creation_threading": "import threading\nthreading.Thread(target=lambda: None).start()\n",
        "torch_cuda_internal": "import torch\ntorch.cuda._initialized = True\n",
        "dynamic_import": "mod = __import__('requests')\n",
        "outside_workspace_read": "data = open('../../oracle/expected_mechanism.json').read()\n",
        "global_memoization": "_output_cache = {}\n\ndef f(x):\n    global _output_cache\n    if x not in _output_cache:\n        _output_cache[x] = x * 2\n    return _output_cache[x]\n",
    }
    for expected_rule, source in cases.items():
        findings = anticheat.scan_source(source, filename=f"{expected_rule}.py")
        hard = _hard_rules(findings)
        # Case names may carry a variant suffix (e.g. network_import_from).
        assert any(rule == expected_rule or expected_rule.startswith(rule + "_") for rule in hard), (
            expected_rule,
            findings,
        )

    # --- warnings (not hard fails) -----------------------------------------------
    findings = anticheat.scan_source(
        "import torch\ns = torch.cuda.Stream()\nwith torch.cuda.stream(s):\n    pass\n",
        filename="streamy.py",
    )
    assert "cuda_stream_graph" in _rules(findings)
    assert not anticheat.has_hard_fail(findings), findings

    # --- clean code passes -----------------------------------------------------------
    clean = '''
"""Straightforward optimized training step."""
import torch


def train_step(model, batch, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch["x"]).pow(2).mean()
    loss.backward()
    optimizer.step()
    return {"loss": loss, "work_units": {"forward": 1, "backward": 1, "optimizer": 1}}
'''
    findings = anticheat.scan_source(clean, filename="clean.py")
    assert findings == [], findings

    # --- hash_tree / manifests ---------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.py").write_text("print('a')\n", encoding="utf-8")
        (root / "sub").mkdir()
        (root / "sub" / "b.py").write_text("print('b')\n", encoding="utf-8")
        manifest = anticheat.hash_tree(root)
        assert set(manifest) == {"a.py", "sub/b.py"}
        same, diffs = anticheat.manifests_equal(manifest, anticheat.hash_tree(root))
        assert same and diffs == []
        (root / "a.py").write_text("print('changed')\n", encoding="utf-8")
        same, diffs = anticheat.manifests_equal(manifest, anticheat.hash_tree(root))
        assert not same and diffs == ["a.py"]

        # --- assert_no_vcs --------------------------------------------------------------
        anticheat.assert_no_vcs(root)
        (root / ".git").mkdir()
        try:
            anticheat.assert_no_vcs(root)
        except AssertionError:
            pass
        else:
            raise AssertionError("expected AssertionError for .git presence")

    # --- canary ----------------------------------------------------------------------------
    findings = anticheat.canary_check("x = 1  # CANARY-ORACLE-1234", ["CANARY-ORACLE-1234"])
    assert anticheat.has_hard_fail(findings) and findings[0]["rule"] == "canary"
    assert anticheat.canary_check("x = 1", ["CANARY-ORACLE-1234"]) == []
    with tempfile.TemporaryDirectory() as tmp:
        oracle = Path(tmp)
        assert anticheat.load_canaries(oracle) == []
        (oracle / "canaries.txt").write_text("ABC-1\n\nDEF-2\n", encoding="utf-8")
        assert anticheat.load_canaries(oracle) == ["ABC-1", "DEF-2"]

    # --- adversarial probes ------------------------------------------------------------------
    import torch

    def real_fn(x, bias):
        return {"out": x * 2.0 + bias}

    inputs = {"x": torch.ones(4), "bias": torch.tensor(1.0)}
    zeroed = anticheat.zero_input_probe(real_fn, inputs)
    assert torch.allclose(zeroed["out"], torch.zeros(4)), zeroed
    mutated_out = anticheat.mutated_input_probe(real_fn, inputs, seed=0)
    assert not torch.allclose(mutated_out["out"], real_fn(**inputs)["out"])

    # The probe must preserve the input device.  This is especially important
    # for CUDA tasks, where CPU noise would fail before the candidate is tested.
    if torch.cuda.is_available():
        cuda_inputs = {key: value.cuda() for key, value in inputs.items()}
        cuda_out = anticheat.mutated_input_probe(real_fn, cuda_inputs, seed=0)
        assert cuda_out["out"].is_cuda

    # A memoizing cheat returns identical outputs under the mutated probe.
    cached = {"out": real_fn(**inputs)["out"]}
    cheat_out = anticheat.mutated_input_probe(lambda **kw: cached, inputs, seed=0)
    assert torch.allclose(cheat_out["out"], cached["out"])

    # --- tripwire ----------------------------------------------------------------------------
    tripped, message = anticheat.tripwire_check(25.0, 20.0)
    assert tripped and "tripwire" in message
    tripped, _ = anticheat.tripwire_check(3.0, 20.0)
    assert not tripped
    tripped, _ = anticheat.tripwire_check(None, 20.0)
    assert not tripped

    print("test_anticheat: OK")


if __name__ == "__main__":
    main()
