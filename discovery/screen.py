"""Preserved exploratory CPU screen, not a controlled density-matched ablation."""
from pathlib import Path
import hashlib,json
from itertools import combinations
import numpy as np
from pong.relearn.model import Reference
class ScreenReference(Reference):
    def forward(self,x,label):
        before=self.ram[:];p=super().forward(x,label)
        for j in range(self.u):
            b=2+j*self.stride
            if self.ram[b+4:b+self.stride]!=before[b+4:b+self.stride]:self.ram[b+3]=(p[j]<<16)|(self.ram[b+3]&65535)
        return p

def main():
    raw=np.asarray([[int(((h>>k)&1)==v) for k in range(4) for v in range(2) for _ in range(32)] for h in range(16)],'u1')
    projected=np.asarray([[int(all(((h>>k)&1)==((v>>b)&1) for b,k in enumerate(c))) for c in combinations(range(4),3) for v in range(8) for _ in range(8)] for h in range(16)],'u1')
    results=[];arrays={}
    for seed in [113101,113201,113301,113401,113501]:
        rng=np.random.default_rng(seed);w=np.zeros((256,32),'u1');ram=[0,0]
        for j in range(32):
            w[rng.choice(256,32,False),j]=1
            ram.extend([0,0,0,16]+[sum(int(w[32*k+b,j])<<b for b in range(32)) for k in range(8)])
        order=rng.permutation(16)
        for name,codes in [('raw',raw),('third_order',projected)]:
            r=ScreenReference(ram,n=256,u=32,nw=32,classes=1);rows=[];states=[]
            for stage,h in enumerate(order):
                for _ in range(12):r.forward(codes[h],0);states.append(r.ram[:])
                bits=[sum(1<<int(k) for k in np.flatnonzero(codes[q])) for q in order[:stage+1]]
                active=np.asarray([[(ww&b).bit_count()>27 for ww in r.weights] for b in bits])
                rows.append(dict(histories=stage+1,active_counts=active.sum(1).tolist(),distinct_masks=len(np.unique(active,axis=0))))
            key=f's{seed}_{name}';arrays[key]=np.asarray(states,'<u4');results.append(dict(seed=seed,basis=name,input_active=int(codes[0].sum()),stages=rows))
    base=Path('results/20261024-context-discovery-screen');np.savez_compressed(base.with_suffix('.npz'),**arrays)
    base.with_suffix('.json').write_text(json.dumps(dict(status='CPU_EXPLORATION_PRESERVED',physical_device_opened=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),archive_sha256=hashlib.sha256(base.with_suffix('.npz').read_bytes()).hexdigest(),limits='Exploratory32-record screen: input activity differs128 vs32; not a density-matched causal ablation or hardware result. No fixed history helpers in this earlier standalone screen.',rows=results),indent=2)+'\n')
if __name__=='__main__':main()
