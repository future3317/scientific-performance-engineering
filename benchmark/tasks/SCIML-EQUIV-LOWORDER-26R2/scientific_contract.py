from __future__ import annotations
import torch

def equivariance_preserved(solution,fixtures):
    m=solution.build_model(fixtures); dev=next(m.parameters()).device; p=fixtures['eval_positions'].to(dev)
    q=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],device=dev)
    with torch.no_grad(): y=m(p); yr=m(p@q.T); expected=y@q.T
    err=float((yr-expected).abs().max().cpu()); return (err<2e-5, {'max_equivariance_error':err})
