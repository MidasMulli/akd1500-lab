"""Unlabelled native prototype allocation over a fixed third-order history basis.

The feature basis and shift register are engineered. Prototype identities are not.
No outcome predictor is included in this qualification study.
"""
from itertools import combinations
import numpy as np
from pong.relearn.model import Reference as PriorReference
N=128;FEATURES=256;U=40;LEARNED=32;SEEDS=[113101,113201,113301,113401,113501]
STAGES=[1,2,4,12];ARMS=['native','frozen','unlabelled','severed','unseen','unseen_frozen']
CYCLE=[int(c) for c in '0000100110101111']

def matrices():
    encoder=np.zeros((N,FEATURES),'i1');j=0
    for triple in combinations(range(4),3):
        for value in range(8):
            for copy in range(8):
                for b,k in enumerate(triple):encoder[k*16+((value>>b)&1)*8+copy,j]=1
                j+=1
    relay=np.zeros((U,N),'i1')
    for j in range(LEARNED):relay[j,64+j]=1
    for k in range(1,4):
        for value in range(2):relay[LEARNED+k*2+value,(k-1)*16+value*8:(k-1)*16+value*8+8]=1
    return encoder,relay

def weights(seed):
    rng=np.random.default_rng(seed);w=np.zeros((FEATURES,U),'i1')
    for j in range(LEARNED):w[rng.choice(FEATURES,32,False),j]=1
    triples=list(combinations(range(4),3))
    # Fixed helper records decode each raw bit from one triple group.
    # Each has32 weights, potential8 or0, threshold7, learning threshold32.
    for k in range(4):
        g=next(i for i,t in enumerate(triples) if k in t);b=triples[g].index(k)
        for value in range(2):
            for v in range(8):
                if ((v>>b)&1)==value:w[g*64+v*8:g*64+v*8+8,LEARNED+k*2+value]=1
    return w

def packed_ram(w):
    ram=[0,0]
    for j in range(U):ram.extend([0,0,0,16 if j<LEARNED else (32<<16)|16]+[sum(int(w[k*32+b,j])<<b for b in range(32)) for k in range(8)])
    return ram

class Reference(PriorReference):
    def __init__(self,ram):super().__init__(ram,n=FEATURES,u=U,nw=32,classes=1)
    def forward(self,x,label):
        if label<0:
            bits=sum(1<<int(k) for k in np.flatnonzero(x));p=[(w&bits).bit_count() for w in self.weights]
            # Explicitly disabled native learning mode: no learning-score accumulation.
            return p
        assert label==0
        old=self.ram[:];p=super().forward(x,0)
        for j in range(U):
            b=2+j*self.stride
            if old[b+4:b+self.stride]!=self.ram[b+4:b+self.stride]:self.ram[b+3]=(p[j]<<16)|(self.ram[b+3]&65535)
        return p

def schedule(cycles,arm):
    rows=[]
    def add(bits,phase,learn):
        for bit in bits:rows.append(dict(index=len(rows),bit=bit,phase=phase,label=0 if learn else -1,sever=int(arm=='severed' and phase=='probe')))
    add([0]*3,'warmup',False)
    add([0,0,0,1]*4,'before',False)
    add([0,0,0,1]*16,'bootstrap',arm!='unlabelled')
    add([0,0,0,1]*4,'old_probe',False)
    for epoch in range(cycles):add(CYCLE,'train',arm in ['native','severed','unseen'])
    add(CYCLE*2,'probe',False)
    if arm in ['unseen','unseen_frozen']:add(np.random.default_rng(113901).integers(0,2,128).tolist(),'heldout',False)
    return rows

def simulate(ram,cycles,arm):
    enc,relay=matrices();ref=Reference(ram);rows=schedule(cycles,arm);last=np.zeros(N,'u1');last[np.r_[0:8,16:24,32:40]]=1;inputs=[];features=[];outputs=[];pots=[];states=[];seen=set();history=[];cpu={};cpu_y=[]
    for e in rows:
        x=last.copy()
        if e['sever']:x[:48]=0
        x[48:64]=0;x[48+e['bit']*8:56+e['bit']*8]=1
        z=(x.astype('i4')@enc.astype('i4')>2).astype('u1')
        p=np.asarray(ref.forward(z,e['label']),'<i4');active=p>np.r_[np.full(LEARNED,27),np.full(8,7)]
        last=(active.astype('i4')@relay.astype('i4')>0).astype('u1')
        # Independent raw-symbol CPU baseline; no native-record IDs or feature vector.
        history.append(e['bit']);history=history[-4:];key=sum(b<<k for k,b in enumerate(history)) if len(history)==4 else -1
        if e['label']==0 and key>=0:cpu.setdefault(key,len(cpu))
        cpu_y.append(cpu.get(key,-1))
        e['history']=key
        if e['label']==0 and key>=0:seen.add(key)
        inputs.append(x);features.append(z);outputs.append(last);pots.append(p);states.append(ref.ram[:])
    return dict(events=rows,inputs=np.asarray(inputs,'u1'),features=np.asarray(features,'u1'),outputs=np.asarray(outputs,'u1'),potentials=np.asarray(pots,'<i4'),states=np.asarray(states,'<u4'),cpu=np.asarray(cpu_y,'<i4'),seen=sorted(seen))

def assess(a):
    rows=a['events'];out=a['outputs'][:,64:96];ids={};ambiguous=0;missing=0;primary={e['index'] for e in rows if e['phase']=='probe'};primary=set(sorted(primary)[-16:])
    for e,y in zip(rows,out):
        if e['index'] not in primary:continue
        active=np.flatnonzero(y);key=e['history']
        if len(active)==1:ids[key]=int(active[0])
        elif not len(active):missing+=1
        else:ambiguous+=1
    return dict(histories=len(ids),unique_records=len(set(ids.values())),missing=missing,ambiguous=ambiguous,assignments=ids,learned_records=sum(any(a['states'][i,2+j*12+4:2+j*12+12].tolist()!=a['states'][0,2+j*12+4:2+j*12+12].tolist() for i in range(len(rows))) for j in range(U)))
