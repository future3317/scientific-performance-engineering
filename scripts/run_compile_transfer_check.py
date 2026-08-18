#!/usr/bin/env python3
"""Calibration-only TorchBench compiler-direction sanity check.

This is deliberately outside the formal task population.  It requires an
installed TorchBench checkout; missing TorchBench is reported as blocked rather
than replaced with a synthetic model.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any


def _load_torchbench_model(name: str, device: str) -> tuple[Any, tuple[Any, ...]]:
    from torchbenchmark import ModelLoader

    loaded = ModelLoader(name, device=device).load_model()
    module = loaded.get_module() if hasattr(loaded, "get_module") else loaded.module
    example_inputs = loaded.example_inputs
    if isinstance(example_inputs, tuple):
        inputs = example_inputs
    elif isinstance(example_inputs, list):
        inputs = tuple(example_inputs)
    else:
        inputs = (example_inputs,)
    return module.eval(), inputs


def _first_tensor(inputs: tuple[Any, ...]) -> Any:
    for value in inputs:
        if hasattr(value, "shape") and hasattr(value, "clone"):
            return value
    return None


def _run(module: Any, inputs: tuple[Any, ...], *, targeted_dynamic: bool) -> dict[str, Any]:
    import torch

    reset = getattr(getattr(torch, "compiler", None), "reset", None)
    if callable(reset):
        reset()
    else:
        torch._dynamo.reset()
    compiled = torch.compile(module, dynamic=False)
    call_inputs = list(inputs)
    first = _first_tensor(inputs)
    if targeted_dynamic and first is not None and len(first.shape) > 0:
        dynamic = first.clone()
        torch._dynamo.mark_dynamic(dynamic, 0, min=1, max=int(first.shape[0]))
        for index, value in enumerate(inputs):
            if value is first:
                call_inputs[index] = dynamic
                break
    started = time.perf_counter()
    with torch.no_grad():
        output = compiled(*tuple(call_inputs))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    counters = {
        str(group): {str(key): int(value) for key, value in values.items()}
        for group, values in torch._dynamo.utils.counters.items()
        if values
    }
    flat = output if isinstance(output, (tuple, list)) else (output,)
    finite = all(bool(torch.isfinite(value).all()) for value in flat if hasattr(value, "numel"))
    return {
        "targeted_dynamic": targeted_dynamic,
        "elapsed_ms": elapsed_ms,
        "finite_output": finite,
        "counters": counters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report: dict[str, Any] = {
        "scope": "calibration_only",
        "formal_population": False,
        "promotion_evidence": False,
        "model": args.model,
        "device": args.device,
    }
    try:
        module, inputs = _load_torchbench_model(args.model, args.device)
    except ModuleNotFoundError as exc:
        report.update({"status": "blocked", "reason": "torchbench_unavailable", "error": repr(exc)})
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    except Exception as exc:
        report.update({"status": "inconclusive", "reason": "model_load_failed", "error": repr(exc)})
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    try:
        baseline = _run(module, inputs, targeted_dynamic=False)
        targeted = _run(module, inputs, targeted_dynamic=True)
        report.update(
            {
                "status": "observed",
                "graph_break_direction": "diagnostic_only",
                "baseline": baseline,
                "targeted_dynamic": targeted,
                "direction_sanity": bool(targeted["finite_output"] and baseline["finite_output"]),
            }
        )
    except Exception as exc:
        report.update({"status": "inconclusive", "reason": "compile_probe_failed", "error": repr(exc)})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
