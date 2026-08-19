from __future__ import annotations
import torch
import torch.nn as nn
TASK_VARIANT="SCIML-GRAPH-CACHE-BOUNDARY-27"
class Model(nn.Module):
    def __init__(self): super().__init__(); self.log_scale=nn.Parameter(torch.tensor(0.0))
def build_model(fixtures): m=Model(); m.load_state_dict(fixtures['init_state']); return m.to(fixtures['device'])
def _edges(pos,cutoff):
    d=torch.cdist(pos,pos); mask=(d<cutoff)&(d>0); return mask.nonzero(as_tuple=False).T
def energy_fn(model,positions,**graph):
    e=_edges(positions,graph['cutoff']); src,dst=e; r2=(positions[src]-positions[dst]).square().sum(-1); return 0.5*model.log_scale.exp() * torch.exp(-r2).sum()
def forces_fn(model,positions,**graph):
    p=positions.detach().clone().requires_grad_(True); e=energy_fn(model,p,**graph); return -torch.autograd.grad(e,p,create_graph=False)[0]
