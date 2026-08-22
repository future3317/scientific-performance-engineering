from __future__ import annotations
import torch
TASK_VARIANT="SCIML-CRYSTAL-STATIC-SAMPLING-29R2"
class Sampler:
    def __init__(self,cutoff,variation): self.cutoff=cutoff; self.variation=variation; self.rebuild_count=0; self.cached_edges=None
def build_sampler(fixtures):
    config=fixtures['config']
    return Sampler(config['cutoff'],config['geometry_variation'])
def _slow_edges(pos,cutoff):
    pairs=[]; n=pos.shape[0]
    for i in range(n):
        for j in range(i+1,n):
            if float(torch.linalg.vector_norm(pos[i]-pos[j]))<cutoff: pairs.append((i,j)); pairs.append((j,i))
    return torch.tensor(pairs,dtype=torch.long).T if pairs else torch.empty(2,0,dtype=torch.long)
def sample_step(sampler,state,step_index):
    pos=state['positions']
    if sampler.cached_edges is None or step_index%2==0: sampler.cached_edges=_slow_edges(pos,sampler.cutoff); sampler.rebuild_count+=1
    edges=sampler.cached_edges; delta=torch.zeros_like(pos)
    if edges.numel():
        src,dst=edges; delta.index_add_(0,src,0.0005*(1.0+sampler.variation)*(pos[dst]-pos[src]))
    return {'positions':pos+delta,'edges':edges}
def sample(sampler,fixtures,num_steps):
    state={'positions':fixtures['initial'].clone(),'edges':None}
    for i in range(num_steps): state=sample_step(sampler,state,i)
    return state['positions']
