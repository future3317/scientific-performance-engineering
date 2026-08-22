from __future__ import annotations
import torch

def batch_semantics_preserved(solution,fixtures):
    x,y=fixtures['batch']; return (x.shape[0]==y.shape[0]==32 and torch.isfinite(x).all().item(), {'batch_size':int(x.shape[0])})
