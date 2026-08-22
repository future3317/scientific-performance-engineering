from __future__ import annotations
import torch

TASK_VARIANT = "CORE-KERNEL-FUSION-09R2"

def _graph(x, residual, context):
    z = x * context["a1"] + context["b1"]
    for _ in range(max(1, context["graph_size"] // 64)):
        z = z + residual * 0.0
    h = z * torch.sigmoid(z)
    y = torch.clamp(h + residual, context["clamp_min"], context["clamp_max"])
    return y * context["a2"] + context["b2"]

def init(fixtures):
    profile = fixtures["compile_profile"]
    context = {key: float(fixtures[key]) for key in ("a1", "b1", "a2", "b2", "clamp_min", "clamp_max")}
    context.update({"graph_size": int(fixtures["graph_size"]), "logical_steps": int(fixtures["logical_steps"]), "dynamic_shape_rate": float(fixtures["dynamic_shape_rate"]), "compile_profile": dict(profile)})
    context["compiled"] = torch.compile(_graph, backend=profile["baseline_backend"], mode=profile["mode"], dynamic=profile["dynamic"])
    return context

def forward(context, x, residual):
    return context["compiled"](x, residual, context)


