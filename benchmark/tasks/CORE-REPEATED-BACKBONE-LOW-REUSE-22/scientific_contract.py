from __future__ import annotations
import torch

def output_equivalence(solution,fixtures):
    m=solution.build_model(fixtures); x1,x2,_,_=fixtures['batch']; dev=next(m.parameters()).device
    with torch.no_grad(): a,b=m.forward_pair(x1.to(dev),x2.to(dev))
    distinct=not torch.allclose(a,b,rtol=1e-5,atol=1e-6)
    return (distinct, {'outputs_distinct':distinct})
