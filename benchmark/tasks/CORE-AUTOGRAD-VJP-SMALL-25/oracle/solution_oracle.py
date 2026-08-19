from __future__ import annotations
import torch
import torch.nn as nn
TASK_VARIANT="CORE-AUTOGRAD-VJP-SMALL-25"
class Model(nn.Module):
    def __init__(self,d=64,o=2): super().__init__(); self.net=nn.Sequential(nn.Linear(d,48),nn.Tanh(),nn.Linear(48,o))
    def forward(self,x): return self.net(x)
def build_model(fixtures):
    m=Model(fixtures['input_dim'],fixtures['output_count']); m.load_state_dict(fixtures['init_state']); return m.to(fixtures['device'])
def jacobian_features(model,x):
    x=x.detach().requires_grad_(True); y=model(x).mean(dim=0); grads=[]
    for i in range(y.numel()): grads.append(torch.autograd.grad(y[i],x,retain_graph=True,create_graph=True)[0])
    return torch.stack(grads)
def train_step(model,batch,optimizer):
    x=batch[0].to(next(model.parameters()).device); optimizer.zero_grad(); j=jacobian_features(model,x); y=model(x); loss=y.square().mean()+0.1*j.square().mean(); loss.backward(); optimizer.step(); return {'loss':loss.detach(),'work_units':{'vjp':y.shape[-1],'optimizer':1}}
def run_training(fixtures,steps):
    m=build_model(fixtures); o=torch.optim.SGD(m.parameters(),lr=fixtures['lr']); r=None
    for _ in range(steps): r=train_step(m,fixtures['batch'],o)
    return {'final_loss':r['loss']}
