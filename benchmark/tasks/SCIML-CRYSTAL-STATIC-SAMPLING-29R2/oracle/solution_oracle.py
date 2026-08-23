from __future__ import annotations

import torch


TASK_VARIANT = "SCIML-CRYSTAL-STATIC-SAMPLING-29R2"


class Sampler:
    def __init__(self, cutoff, variation):
        self.cutoff = cutoff
        self.variation = variation
        self.rebuild_count = 0


def build_sampler(fixtures):
    config = fixtures["config"]
    return Sampler(config["cutoff"], config["geometry_variation"])


def _fast_edges(pos, cutoff):
    distances = torch.cdist(pos, pos)
    return ((distances < cutoff) & (distances > 0)).nonzero(as_tuple=False).T


def sample_step(sampler, state, step_index):
    pos = state["positions"]
    edges = _fast_edges(pos, sampler.cutoff)
    sampler.rebuild_count += 1
    delta = torch.zeros_like(pos)
    if edges.numel():
        src, dst = edges
        delta.index_add_(0, src, 0.0005 * (1.0 + sampler.variation) * (pos[dst] - pos[src]))
    return {"positions": pos + delta, "edges": edges}


def sample(sampler, fixtures, num_steps):
    state = {"positions": fixtures["initial"].clone(), "edges": None}
    for step_index in range(num_steps):
        state = sample_step(sampler, state, step_index)
    return state["positions"]
