from __future__ import annotations

import argparse
import json
import random
import random
import traceback
from pathlib import Path
from typing import Any


def admit(spec: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Baseline-only load and eager/eager numerical qualification; never runs S1."""
    import torch

    from .numerical_checks import compare_outputs, nested_digest
    from .source_adapters import load_model

    result: dict[str, Any] = {"status": "error", "model": spec.get("model"), "source": spec.get("source"),
                              "arm_executed": "S0_ONLY"}
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        seed = int(spec["seed"])
        _seed_all(torch, seed)
        model, inputs = load_model(spec)
        first_weights = nested_digest(model.state_dict())
        inputs_sha = nested_digest(inputs)
        reference_path = Path(spec["reference_path"])
        reference_path.parent.mkdir(parents=True, exist_ok=True)

        with torch.inference_mode():
            _seed_all(torch, seed)
            first = model(*inputs)
            torch.cuda.synchronize()
            _seed_all(torch, seed)
            second = model(*inputs)
            torch.cuda.synchronize()

        reference = _cpu_tree(first)
        torch.save(reference, reference_path)
        tolerances = spec.get("correctness", {})
        comparison = compare_outputs(reference, _cpu_tree(second),
                                     rtol=float(tolerances.get("rtol", 1e-4)),
                                     atol=float(tolerances.get("atol", 1e-5)))
        result.update({"status": "valid" if comparison.passed else "numeric_mismatch",
                       "admission_status": "ADMITTED_BASELINE_ONLY" if comparison.passed else "BASELINE_AA_NUMERIC_MISMATCH",
                       "correctness": comparison.__dict__, "weights_sha256": first_weights,
                       "inputs_sha256": inputs_sha, "reference_sha256": nested_digest(reference),
                       "reference_path": str(reference_path),
                       "input_kind": "REAL_MODEL_MATERIALIZED_INPUT" if spec["source"] == "torchvision" else "TORCHBENCH_MINIATURE_INPUT"})
        deterministic = _torchbench_determinism(spec)
        if deterministic is not None:
            result["source_determinism_metadata"] = deterministic
            if deterministic is False and comparison.passed:
                result["admission_status"] = "STOCHASTIC_UNQUALIFIED"
    except Exception as exc:
        result.update({"status": "error", "admission_status": "BASELINE_INVALID",
                       "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _torchbench_determinism(spec: dict[str, Any]) -> bool | None:
    if spec.get("source") != "torchbench":
        return None
    import importlib
    import sys

    sys.path.insert(0, str(Path(spec["torchbench_root"]).resolve()))
    module = importlib.import_module(f"torchbenchmark.models.{spec['model']}")
    metadata = getattr(module, "metadata", {})
    value = metadata.get("eval_deterministic") if isinstance(metadata, dict) else None
    return value if isinstance(value, bool) else None


def _seed_all(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def _cpu_tree(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise ValueError("spec must contain a JSON object")
        admit(spec, spec.get("output_path", args.output))
    except Exception as exc:
        result = {"status": "error", "admission_status": "BASELINE_INVALID", "error_type": type(exc).__name__,
                  "error": str(exc), "traceback": traceback.format_exc()}
        Path(args.output).write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
