from __future__ import annotations
import torch

def gradient_equivalence(solution,fixtures):
    m=solution.build_model(fixtures); x=fixtures['batch'][0].to(next(m.parameters()).device); j=solution.jacobian_features(m,x)
    return (tuple(j.shape[:2])==(2,8) and torch.isfinite(j).all().item(), {'shape':list(j.shape)})
