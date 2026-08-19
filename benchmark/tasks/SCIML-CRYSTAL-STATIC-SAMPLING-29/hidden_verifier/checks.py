from __future__ import annotations
import torch

def _reference(fixtures):
    pos = fixtures['initial'].clone()
    cutoff = fixtures['cutoff']
    for _ in range(fixtures['num_steps']):
        d = torch.cdist(pos, pos)
        edges = ((d < cutoff) & (d > 0)).nonzero(as_tuple=False).T
        delta = torch.zeros_like(pos)
        if edges.numel():
            src, dst = edges
            delta.index_add_(0, src, 0.0005 * (pos[dst] - pos[src]))
        pos = pos + delta
    return pos

def check_sample(solution, fixtures):
    sampler = solution.build_sampler(fixtures)
    got = solution.sample(sampler, fixtures, fixtures['num_steps'])
    ref = _reference(fixtures)
    err = float((got-ref).abs().max())
    return {'passed': bool(got.shape == ref.shape and torch.isfinite(got).all() and err < 2e-6), 'output_checksum': float(got.mean()), 'max_trajectory_error': err}
