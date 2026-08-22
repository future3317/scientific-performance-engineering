from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
TASK_VARIANT="CORE-H2D-OVERFANOUT-23R3"
class Model(nn.Module):
    def __init__(self,d=64): super().__init__(); self.fc=nn.Linear(d,1)
    def forward(self,x): return self.fc(x).squeeze(-1)
def build_model(fixtures):
    m=Model(fixtures['in_dim']); m.load_state_dict(fixtures['init_state']); return m.to(fixtures['device'])
def train_step(model,batch,optimizer):
    x,y=batch; dev=next(model.parameters()).device; x=x.to(dev); y=y.to(dev); pred=model(x); loss=F.mse_loss(pred,y); optimizer.zero_grad(); loss.backward(); optimizer.step(); return {'loss':loss.detach(),'work_units':{'samples':x.shape[0],'optimizer':1}}
def run_training(fixtures,steps):
    m=build_model(fixtures); o=torch.optim.SGD(m.parameters(),lr=fixtures['lr']); r=None
    for _ in range(steps): r=train_step(m,fixtures['batch'],o)
    return {'final_loss':r['loss']}
