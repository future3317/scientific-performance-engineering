from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
TASK_VARIANT="CORE-REPEATED-BACKBONE-LOW-REUSE-22"
class Model(nn.Module):
    def __init__(self,d=32,w=128):
        super().__init__(); self.backbone=nn.Sequential(nn.Linear(d,w),nn.ReLU(),nn.Linear(w,w),nn.ReLU()); self.h1=nn.Linear(w,1); self.h2=nn.Linear(w,1)
    def forward_pair(self,x1,x2): return self.h1(self.backbone(x1)).squeeze(-1), self.h2(self.backbone(x2)).squeeze(-1)
def build_model(fixtures):
    m=Model(fixtures['in_dim'],fixtures['width']); m.load_state_dict(fixtures['init_state']); return m.to(fixtures['device'])
def train_step(model,batch,optimizer):
    x1,x2,y1,y2=batch; dev=next(model.parameters()).device; p1,p2=model.forward_pair(x1.to(dev),x2.to(dev)); loss=F.mse_loss(p1,y1.to(dev))+F.mse_loss(p2,y2.to(dev)); optimizer.zero_grad(); loss.backward(); optimizer.step(); return {'loss':loss.detach(),'work_units':{'backbone':2,'optimizer':1}}
def run_training(fixtures,steps):
    m=build_model(fixtures); o=torch.optim.SGD(m.parameters(),lr=fixtures['lr']); r=None
    for _ in range(steps): r=train_step(m,fixtures['batch'],o)
    return {'final_loss':r['loss']}
