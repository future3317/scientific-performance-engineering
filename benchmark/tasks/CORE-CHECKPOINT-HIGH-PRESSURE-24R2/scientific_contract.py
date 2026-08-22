from __future__ import annotations
import torch

def gradient_equivalence(solution,fixtures):
    candidate=solution.run_training(fixtures,1)
    # Full-batch reference without checkpointing, same loss normalization.
    probe=solution.build_model(fixtures); dev=next(probe.parameters()).device; x,y=fixtures['batch']; opt=torch.optim.SGD(probe.parameters(),lr=fixtures['lr']); opt.zero_grad(); import torch.nn.functional as F
    loss=F.mse_loss(probe(x.to(dev)),y.to(dev)); loss.backward(); opt.step(); ref={k:v.detach().cpu() for k,v in probe.state_dict().items()}
    max_err=max(float((candidate['state'][k]-ref[k]).abs().max()) for k in ref)
    return (max_err < 3e-5, {'max_parameter_error':max_err})
