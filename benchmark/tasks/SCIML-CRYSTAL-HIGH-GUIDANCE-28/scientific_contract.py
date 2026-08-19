from __future__ import annotations
import torch

def structure_validity(solution,fixtures):
    s=solution.build_sampler(fixtures); out=solution.sample(s,fixtures,fixtures['num_steps']); rms=float((out-fixtures['target']).square().mean().sqrt()); distances=torch.cdist(out,out); mask=~torch.eye(out.shape[0],dtype=torch.bool); min_d=float(distances[mask].min()); ok=torch.isfinite(out).all().item() and rms<0.005 and min_d>0.5; return (ok, {'rms_to_target':rms,'min_distance':min_d})
