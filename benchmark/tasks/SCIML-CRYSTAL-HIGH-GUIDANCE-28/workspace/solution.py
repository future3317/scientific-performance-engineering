from __future__ import annotations
import torch
TASK_VARIANT="SCIML-CRYSTAL-HIGH-GUIDANCE-28"
class Sampler:
    def __init__(self,target,noise,guidance): self.target=target; self.noise=noise; self.guidance=guidance; self.step_calls=0
def build_sampler(fixtures): return Sampler(fixtures['target'].clone(),fixtures['noise'].clone(),fixtures['guidance_scale'])
def sample_step(sampler,state,step_index):
    sampler.step_calls+=1; drift=0.02*sampler.guidance*(sampler.target-state); return state+drift+0.001*sampler.noise[step_index]
def sample(sampler,fixtures,num_steps):
    state=fixtures['initial'].clone()
    for i in range(num_steps): state=sample_step(sampler,state,i)
    return state
