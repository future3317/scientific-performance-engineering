from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
TASK_VARIANT="CORE-CHECKPOINT-HIGH-PRESSURE-24"
MICROBATCH=16
USE_CHECKPOINT=True
class Model(nn.Module):
    def __init__(self,d=64,w=96,blocks=4):
        super().__init__(); self.inp=nn.Linear(d,w); self.blocks=nn.ModuleList(nn.Linear(w,w) for _ in range(blocks)); self.out=nn.Linear(w,1); self.checkpoint_calls=0
    def _block(self,layer,h): return F.gelu(layer(h))+h
    def forward(self,x):
        h=F.gelu(self.inp(x))
        for layer in self.blocks:
            self.checkpoint_calls += 1
            h=checkpoint(lambda t, layer=layer: self._block(layer,t), h, use_reentrant=False)
        return self.out(h).squeeze(-1)
def build_model(fixtures):
    m=Model(fixtures['in_dim'],fixtures['width'],fixtures['blocks']); m.load_state_dict(fixtures['init_state']); return m.to(fixtures['device'])
def _forward(model,x): return model(x)
def train_step(model,batch,optimizer):
    x,y=batch; dev=next(model.parameters()).device; n=x.shape[0]; optimizer.zero_grad(); total=torch.zeros((),device=dev)
    for start in range(0,n,MICROBATCH):
        xb=x[start:start+MICROBATCH].to(dev); yb=y[start:start+MICROBATCH].to(dev); pred=_forward(model,xb); loss=F.mse_loss(pred,yb,reduction='sum')/n; loss.backward(); total=total+loss.detach()
    optimizer.step(); return {'loss':total,'work_units':{'samples':n,'optimizer':1}}
def run_training(fixtures,steps):
    m=build_model(fixtures); o=torch.optim.SGD(m.parameters(),lr=fixtures['lr']); r=None
    for _ in range(steps): r=train_step(m,fixtures['batch'],o)
    return {'final_loss':r['loss'],'state':{k:v.detach().cpu() for k,v in m.state_dict().items()}}
