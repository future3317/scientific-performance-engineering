from __future__ import annotations
import torch

def neighbor_consistency(solution,fixtures):
    s=solution.build_sampler(fixtures); out=solution.sample(s,fixtures,fixtures['num_steps']); d=torch.cdist(out,out); edges=((d<fixtures['cutoff'])&(d>0)).nonzero(as_tuple=False).T; ok=torch.isfinite(out).all().item() and edges.shape[1]>0; return (ok, {'edge_count':int(edges.shape[1])})
