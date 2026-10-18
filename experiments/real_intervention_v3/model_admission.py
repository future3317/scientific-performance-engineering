from __future__ import annotations

import argparse
import json
import random
import traceback
from pathlib import Path
from typing import Any, Mapping


def admit(spec: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Baseline-only load and eager/eager numerical qualification; never runs S1."""
    import torch

    from .numerical_checks import compare_outputs, configure_determinism, nested_digest
    from .source_adapters import load_model

    spec = dict(spec)
    spec.setdefault("reference_path", str(output_path) + ".reference.pt")
    result: dict[str, Any] = {"status": "error", "model": spec.get("model"), "source": spec.get("source"),
                              "arm_executed": "S0_ONLY"}
    try:
        configure_determinism(torch)
        seed = int(spec["seed"])
        _seed_all(torch, seed)
        model, inputs = load_model(spec)
        first_weights = nested_digest(model.state_dict())
        inputs_sha = nested_digest(inputs)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        actual_input_shapes = _tensor_shapes(inputs)
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
        tolerances = spec.get("correctness", {})
        comparison = compare_outputs(reference, _cpu_tree(second),
                                     rtol=float(tolerances.get("rtol", 1e-4)),
                                     atol=float(tolerances.get("atol", 1e-5)))
        result.update({"status": "valid" if comparison.passed else "numeric_mismatch",
                       "admission_status": "ADMITTED_BASELINE_ONLY" if comparison.passed else "BASELINE_AA_NUMERIC_MISMATCH",
                       "correctness": comparison.__dict__, "weights_sha256": first_weights,
                       "inputs_sha256": inputs_sha, "reference_sha256": nested_digest(reference),
                       "reference_path": str(reference_path),
                       "parameter_count": parameter_count, "actual_input_shapes": actual_input_shapes,
                       "input_kind": "REAL_MODEL_MATERIALIZED_INPUT" if spec["source"] == "torchvision" else "TORCHBENCH_MINIATURE_INPUT"})
        if comparison.passed:
            torch.save(reference, reference_path)
        deterministic = _torchbench_determinism(spec)
        if spec["source"] == "torchbench":
            result["source_determinism_metadata"] = deterministic
            result["source_determinism_status"] = {True: "deterministic", False: "stochastic", None: "unknown"}[deterministic]
            if deterministic is not True and comparison.passed:
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
    import yaml

    metadata_path = Path(spec["torchbench_root"]).resolve() / "torchbenchmark" / "models" / spec["model"] / "metadata.yaml"
    if not metadata_path.is_file():
        return None
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    value = metadata.get("eval_deterministic") if isinstance(metadata, Mapping) else None
    return value if isinstance(value, bool) else None


def _tensor_shapes(value: Any) -> dict[str, list[int]]:
    import torch

    shapes: dict[str, list[int]] = {}

    def visit(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            shapes[path] = list(item.shape)
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "input")
    return shapes


def _seed_all(torch: Any, seed: int) -> None:
    import numpy as np

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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
