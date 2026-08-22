from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
TASK_VARIANT="SCIML-EQUIV-LOWORDER-26R2"
class Model(nn.Module):
    def __init__(self,h=48,irrep_order=1,recompute_rate=1.0):
        super().__init__(); self.radial=nn.Sequential(nn.Linear(1,h),nn.SiLU(),nn.Linear(h,1)); self.radial_calls=0; self.irrep_order=int(irrep_order); self.recompute_rate=float(recompute_rate)
    def scalar(self,pos):
        self.radial_calls+=1; r2=pos.square().sum(dim=-1,keepdim=True)
        radial_input=r2.pow(self.irrep_order)*(1.0+0.01*self.recompute_rate)
        return self.radial(radial_input).squeeze(-1)
    def forward(self,pos):
        comps=[]
        for axis in range(3): comps.append(pos[:,axis]*self.scalar(pos))
        return torch.stack(comps,dim=-1)
def build_model(fixtures): m=Model(fixtures['hidden'],fixtures['irrep_order'],fixtures['recompute_rate']); m.load_state_dict(fixtures['init_state']); return m.to(fixtures['device'])
def train_step(model,batch,optimizer):
    pos,target=batch; dev=next(model.parameters()).device; pred=model(pos.to(dev)); loss=F.mse_loss(pred,target.to(dev)); optimizer.zero_grad(); loss.backward(); optimizer.step(); return {'loss':loss.detach(),'work_units':{'nodes':pos.shape[0],'optimizer':1}}
def run_training(fixtures,steps):
    m=build_model(fixtures); o=torch.optim.SGD(m.parameters(),lr=fixtures['lr']); r=None
    for _ in range(steps): r=train_step(m,fixtures['batch'],o)
    return {'final_loss':r['loss']}
