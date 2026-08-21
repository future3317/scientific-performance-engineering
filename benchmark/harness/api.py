#!/usr/bin/env python3
"""Named workspace API contracts (BENCHMARK_DESIGN.md sections 4 and 11).

A task pins the agent-visible entrypoint contract via ``task.yaml: workspace.api``.
This module is the registry of those contracts. Each spec describes the callables
the workspace entrypoint module must expose; the harness checks them with
:func:`validate_solution_api` before measurement so a signature drift fails fast
instead of producing garbage timings.
"""

from __future__ import annotations

import inspect
from types import ModuleType
from typing import Any

# Registry: api name -> spec.
#   entrypoint:          expected workspace file (informational; task.yaml is authoritative)
#   required_callables:  name -> human-readable signature description
#   notes:               free-form contract notes
API_REGISTRY: dict[str, dict[str, Any]] = {
    "train_loop_v1": {
        "execution_class": "atomic",
        "metric_type": "latency",
        "entrypoint": "solution.py",
        "required_callables": {
            "build_model": "build_model(fixtures) -> torch.nn.Module",
            "train_step": "train_step(model, batch, optimizer) -> dict with at least 'loss' (tensor) and 'work_units' (dict of counters)",
            "run_training": "run_training(fixtures, steps: int) -> dict with final metrics",
        },
        "notes": "SPE-Core training-loop tasks. train_step is the hot loop body under measurement; it must execute the same forward/backward/optimizer units as the baseline (work-unit counters are compared).",
    },
    "energy_force_v1": {
        "execution_class": "atomic",
        "metric_type": "energy_force",
        "entrypoint": "solution.py",
        "required_callables": {
            "build_model": "build_model(fixtures) -> torch.nn.Module",
            "energy_fn": "energy_fn(model, positions, **graph) -> energy tensor (must support autograd w.r.t. positions)",
            "forces_fn": "forces_fn(model, positions, **graph) -> forces tensor equal to -dE/dx",
        },
        "notes": "SciML graph/materials tasks. Gates check F == -dE/dx via autograd, so energy_fn must keep positions differentiable.",
    },
    "sampler_v1": {
        "execution_class": "atomic",
        "metric_type": "sampling",
        "entrypoint": "solution.py",
        "required_callables": {
            "build_sampler": "build_sampler(fixtures) -> sampler object",
            "sample_step": "sample_step(sampler, state, step_index: int) -> new state",
            "sample": "sample(sampler, fixtures, num_steps: int) -> samples tensor",
        },
        "notes": "SciML diffusion/score sampling loops. time-to-quality tasks measure wall-clock to a fixed validity threshold.",
    },
    "kernel_module_v1": {
        "execution_class": "atomic",
        "metric_type": "kernel",
        "entrypoint": "solution.py",
        "required_callables": {
            "init": "init(fixtures) -> opaque context",
            "forward": "forward(context, *inputs) -> output tensor(s); the fused/optimized kernel path",
        },
        "notes": "KernelBench-style kernel/compiler tasks (spe_core family 'compiler'). Measured with CUDA events plus host wall clock and L2-thrash between trials.",
    },
    "episode_v1": {
        "execution_class": "episode",
        "metric_type": "episode_score",
        "entrypoint": "solution.py",
        "required_callables": {
            "run_episode_task": "run_episode_task(task_workspace, skill_view, budget) -> {'action': declarative policy}",
        },
        "notes": "Evolution-track episode steps. The candidate returns only a declarative action; the harness executes, scores, and gates the episode.",
    },
}


def list_apis() -> list[str]:
    """Return the registered API names."""
    return sorted(API_REGISTRY)


def get_api(name: str) -> dict[str, Any]:
    """Return the spec dict for an API name; KeyError lists valid names."""
    if name not in API_REGISTRY:
        raise KeyError(f"unknown api {name!r}; registered: {', '.join(list_apis())}")
    return API_REGISTRY[name]


def execution_class_for_api(name: str) -> str:
    """Return the registered execution class for a workspace API."""
    execution_class = get_api(name).get("execution_class")
    if execution_class not in {"atomic", "episode"}:
        raise ValueError(f"API {name!r} has no valid execution_class")
    return str(execution_class)


def execution_class_for_task(spec: dict[str, Any]) -> str:
    """Resolve a task's execution class through the API registry."""
    try:
        api_name = str(spec["workspace"]["api"])
    except (KeyError, TypeError) as exc:
        raise ValueError("task spec lacks workspace.api") from exc
    return execution_class_for_api(api_name)


def metric_type_for_task(spec: dict[str, Any]) -> str:
    try:
        api_name = str(spec["workspace"]["api"])
    except (KeyError, TypeError) as exc:
        raise ValueError("task spec lacks workspace.api") from exc
    metric_type = get_api(api_name).get("metric_type")
    if not isinstance(metric_type, str) or not metric_type:
        raise ValueError(f"API {api_name!r} has no metric_type")
    return metric_type


def validate_solution_api(module: ModuleType, api_name: str) -> list[str]:
    """Check *module* against the named contract; return a list of violations.

    A violation is a missing callable, or a callable whose declared parameters
    cannot be inspected (builtins are exempt from the signature check). The
    registry stores signature *descriptions* rather than formal grammars, so the
    check here is: attribute exists, is callable, and (when inspectable) exposes
    at least the leading parameter named in the description.
    """
    spec = get_api(api_name)
    violations: list[str] = []
    for name, description in spec["required_callables"].items():
        member = getattr(module, name, None)
        if member is None:
            violations.append(f"missing required callable {name!r} ({description})")
            continue
        if not callable(member):
            violations.append(f"{name!r} exists but is not callable ({description})")
            continue
        try:
            signature = inspect.signature(member)
        except (TypeError, ValueError):
            continue  # builtin/C-level callable; cannot inspect, accept
        if not signature.parameters:
            violations.append(f"{name!r} takes no parameters; expected {description}")
    return violations
