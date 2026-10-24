"""General competition cap around the independently reconstructed native rule."""
import math
import numpy as np
from pong.relearn.model import Reference as PriorReference
N=256;U=8;SEEDS=[111101,111201,111301,111401,111501]
ARMS=['native','frozen','disconnected','baseline'];EPOCHS=[1,2,4,12];H=16
class Reference(PriorReference):
    def __init__(self,ram,competition):
        super().__init__(ram,n=N,u=U,nw=32,classes=U);self.cap=math.floor(32*(1-competition)+.5)
    def forward(self,x,label):
        before=self.ram[:];p=super().forward(x,label)
        for j in range(U):
            b=2+j*self.stride
            if before[b+4:b+self.stride]!=self.ram[b+4:b+self.stride]:self.ram[b+3]=(min(p[j],self.cap)<<16)|(self.ram[b+3]&65535)
        return p
    def infer(self,x):
        bits=sum(1<<int(k) for k in np.flatnonzero(x));p=[(w&bits).bit_count() for w in self.weights]
        self.rot=(self.rot+sum(((v>>2)&7)|((v&3)<<3) for v in p))&31
        return p
def design(seed):
    rng=np.random.default_rng(seed);f=rng.permutation(N);codes=np.zeros((8,N),'u1');partial=codes.copy()
    for j in range(8):codes[j,f[:16]]=1;codes[j,f[16+16*j:32+16*j]]=1;partial[j,f[:16]]=1;partial[j,rng.choice(f[16+16*j:32+16*j],12,False)]=1
    ring=rng.permutation(8);old=np.empty(8,'<i4');old[ring]=np.roll(ring,-1);changed=np.sort(rng.choice(8,2,False)).astype('<i4');new=old.copy();new[changed]=new[changed[::-1]]
    return codes,partial,old,new,changed
def simulate(ram,seed,arm,epochs,extra_queries=0,full_queries=False):
    codes,partial,old,new,changed=design(seed);ref=Reference(ram,.25 if arm=='baseline' else .75);events=[];xs=[];ys=[];states=[];pots=[];table=old.copy();table_ys=[]
    def step(x,label,kind,epoch,source=-1,context=-1):
        x=x.copy();i=len(events);label=int(label);learn=label>=0
        bits=sum(1<<int(k) for k in np.flatnonzero(x));p=ref.forward(x,label) if learn else ref.infer(x);active=np.flatnonzero(np.asarray(p)>24);y=np.maximum.reduce(codes[active],axis=0) if len(active) else np.zeros(256,'u1')
        # Independent task table is keyed by observed context, not native record.
        matches=codes.astype('i4')@x.astype('i4');s=int(np.argmax(matches));assert matches[s]>=28
        ty=codes[table[s]].copy()
        if label>=0:table[s]=label
        events.append(dict(index=i,kind=kind,epoch=epoch,label=label,source=source,context=int(context)));xs.append(x);ys.append(y);states.append(ref.ram[:]);pots.append(p);table_ys.append(ty);return i,y
    for s in range(8):step(codes[s],old[s],'old_example',-1,context=s)
    for s in range(8):step(partial[s],-1,'old_query',-1,context=s)
    current=partial[changed[0]];carry=-1
    for epoch in range(epochs):
        # The environment flips after the previous burst. Teaching is an
        # independent two-context observation stream, not a CPU weight repair.
        truth=new if epoch%2==0 else old
        pre,current=step(current,-1,'pre_prediction',epoch,source=carry)
        saved=current.copy()
        if epoch==1:
            for _ in range(extra_queries):step(codes[0] if full_queries else partial[0],-1,'selector_query',epoch,context=0)
        for repeat in range(2):
            for s in changed:step(codes[s],truth[s] if arm not in ['frozen','disconnected'] else -1,'fresh_example',epoch,context=s)
        for s in range(8):step(partial[s],-1,'query',epoch,context=s)
        current=saved
        for t in range(H):carry,current=step(current,-1,'feedback',epoch,source=pre if t==0 else carry)
    return dict(events=events,inputs=np.asarray(xs,'u1'),outputs=np.asarray(ys,'u1'),states=np.asarray(states,'<u4'),potentials=np.asarray(pots,'<i4'),table_outputs=np.asarray(table_ys,'u1'))
