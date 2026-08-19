from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
TASK_VARIANT = "CORE-SCALAR-SYNC-LOW-CADENCE-21"

class TinyRegressor(nn.Module):
    def __init__(self, d=16):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def build_model(fixtures):
    m=TinyRegressor(fixtures['in_dim']); m.load_state_dict(fixtures['init_state']); m.metric_step=0; return m.to(fixtures['device'])

def train_step(model,batch,optimizer):
    x,y=batch; pred=model(x.to(fixtures_device:=next(model.parameters()).device)); loss=F.mse_loss(pred,y.to(fixtures_device))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    # Four scalar reads only on the declared metric cadence. The average hot-loop
    # cost is intentionally below the applicability boundary.
    values=[]
    if model.metric_step % 8 == 0:
        metrics=torch.stack([loss.detach().cpu(), pred.detach().cpu().mean(), pred.detach().cpu().abs().mean(), pred.detach().cpu().square().mean()])
        values=[metrics[i].item() for i in range(4)]
    model.metric_step += 1
    return {'loss':loss.detach(),'metrics':values,'work_units':{'forward':1,'backward':1,'optimizer':1}}

def run_training(fixtures,steps):
    m=build_model(fixtures); opt=torch.optim.SGD(m.parameters(),lr=fixtures['lr']); out=None
    for _ in range(steps): out=train_step(m,fixtures['batch'],opt)
    return {'final_loss':out['loss'],'metrics':out['metrics']}
