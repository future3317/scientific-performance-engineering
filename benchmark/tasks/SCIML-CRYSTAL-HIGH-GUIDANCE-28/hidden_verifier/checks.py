from __future__ import annotations
import torch

def check_sample(solution,fixtures):
    s=solution.build_sampler(fixtures); out=solution.sample(s,fixtures,fixtures['num_steps']); return {'passed':bool(out.shape==fixtures['target'].shape and torch.isfinite(out).all()), 'output_checksum':float(out.mean())}
