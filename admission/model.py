"""Own fixed policy, two native banks, and independent bit-set replay.

The repeat-admission policy and key representation are designed, not learned.
Each bank's competing records and their binary connections are learned.
"""
from pathlib import Path
import numpy as np
from pong.relearn.model import Reference as Base

N=128; N0=1024; N1=1024; U0=48; U1=32; NW=32
KEYS=12; RAW=KEYS*32; QUERY=RAW*2
SEEDS=[114101,114201,114301,114401,114501]
STAGES=['gate','prefix','pressure','heldout']
ARMS=['native','open','frozen','closed','severed']
P0=list(range(4))+list(range(24,28))
P1=list(range(4))+list(range(16,20))
H0=[j for j in range(U0) if j not in P0]
H1=[j for j in range(U1) if j not in P1]
OUTCOME=[0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1]

def matrices(arm):
    decode=np.zeros((N,64),'i1');dt=np.full(64,32,'i4')
    for key in range(KEYS):
        for b in range(4):
            decode[32+b*8+((key>>b)&1)*4,key]=1
            decode[b*8+((key>>b)&1)*4,16+key]=1
        dt[key]=dt[16+key]=3
    decode[64,32]=1;dt[32]=0
    decode[:,33:33+KEYS]=decode[:,:KEYS];dt[33:33+KEYS]=dt[:KEYS]
    policy=np.zeros((64,N0),'i1');pt=np.zeros(N0,'i4')
    for key in range(KEYS):
        policy[key,key*32:(key+1)*32]=1
        a=slice(RAW+key*32,RAW+(key+1)*32)
        policy[key,a]=policy[33+key,a]=1
        if arm=='open':pt[a]=0
        else:
            policy[32,a]=1;pt[a]=2
            if arm!='closed':policy[16+key,a]=1
    policy[32,QUERY:QUERY+32]=1
    bridge=np.zeros((U0,N1),'i1');bt=np.zeros(N1,'i4')
    for key in range(KEYS):bridge[H0[KEYS+key],key*32:(key+1)*32]=1
    for p,j in enumerate(P0):
        a=slice(RAW+p*32,RAW+(p+1)*32)
        bridge[j,a]=bridge[H0[2*KEYS],a]=1;bt[a]=1
    relay=np.zeros((U1,N),'i1')
    for key in range(KEYS):
        for b in range(4):relay[H1[key],b*8+((key>>b)&1)*4:b*8+((key>>b)&1)*4+4]=1
        relay[H1[key],96+key]=1
    for p,j in enumerate(P1):relay[j,64+p]=1
    for p,j in enumerate(H1[KEYS:KEYS+8]):relay[j,72+p]=1
    return [(decode,dt),(policy,pt),(bridge,bt),(relay,np.zeros(N,'i4'))]

def weights(seed,bank):
    n,u,plastic,helpers=(N0,U0,P0,H0) if bank==0 else (N1,U1,P1,H1)
    w=np.zeros((n,u),'i1');rng=np.random.default_rng(seed+bank*17)
    for j in plastic:w[rng.choice(RAW,32,False),j]=1
    if bank==0:
        for key in range(KEYS):
            w[key*32:(key+1)*32,helpers[key]]=1
            w[RAW+key*32:RAW+(key+1)*32,helpers[KEYS+key]]=1
        w[QUERY:QUERY+32,helpers[KEYS*2]]=1
        for j in helpers[KEYS*2+1:]:w[N0-32:,j]=1
    else:
        for key in range(KEYS):w[key*32:(key+1)*32,helpers[key]]=1
        for p,j in enumerate(helpers[KEYS:KEYS+8]):w[RAW+p*32:RAW+(p+1)*32,j]=1
        for j in helpers[KEYS+8:]:w[N1-32:,j]=1
    return w

def packed(w,bank):
    n,u=w.shape;plastic=P0 if bank==0 else P1;ram=[0,0]
    for j in range(u):
        ram += [0,0,0,16 if j in plastic else (32<<16)|16]
        ram += [sum(int(w[k*32+b,j])<<b for b in range(32)) for k in range(n//32)]
    return ram

class Reference(Base):
    def forward(self,x,label):
        if label<0:
            bits=sum(1<<int(k) for k in np.flatnonzero(x))
            return [(w&bits).bit_count() for w in self.weights]
        return super().forward(x,label)

def schedule(stage,arm):
    observations=[]
    def add(keys,phase,teach=True):
        for key in keys:observations.append(dict(key=int(key),phase=phase,teach=bool(teach)))
    if stage=='gate':
        add([0,0,0,1,1,2,2,3,4,4,4,5,6,6,7,7],'gate')
    else:
        add([0]*12+[1]*12,'bootstrap')
        add([0,1]*2,'old_probe',False)
        if stage in ['pressure','heldout']:
            # Six competing keys, none consecutive. Same actual outcomes in every arm.
            add(list(range(2,8))*8,'distractors')
            add([k for k in range(8,12) for _ in range(16)],'new_useful')
            add([0,1,8,9,10,11]*4,'final_probe',False)
        if stage=='heldout':
            add(np.random.default_rng(114901).choice([0,1,8,9,10,11],96),'heldout',False)
    rows=[]
    for obs in observations:
        i=len(rows);rows.append(dict(index=i,**obs,teacher=False,label=-1,observation=i//2))
        if obs['teach']:
            learn=not(arm=='frozen' and obs['phase'] not in ['bootstrap','gate'])
            rows.append(dict(index=len(rows),**obs,teacher=True,label=OUTCOME[obs['key']] if learn else -1,observation=i//2))
    return rows

def active(ref,z,label,plastic):
    p=np.asarray(ref.forward(z,label),'<i4')
    thresholds=np.full(ref.u,31);thresholds[plastic]=27
    return (p>thresholds).astype('u1'),p

def simulate(rams,stage,arm):
    mats=matrices(arm);refs=[Reference(rams[k],n=n,u=u,nw=32,classes=2) for k,(n,u) in enumerate([(N0,U0),(N1,U1)])]
    rows=schedule(stage,arm);last=np.zeros(N,'u1');query_input=None
    xs=[];ys=[];s0=[];s1=[];ps0=[];ps1=[];features0=[];features1=[]
    cpu={'unbounded':{},'repeat_lru':{},'lru':{}};cpu_out=[];previous=None
    for e in rows:
        if not e['teacher']:
            x=last.copy();x[32:64]=0
            for b in range(4):x[32+b*8+((e['key']>>b)&1)*4:36+b*8+((e['key']>>b)&1)*4]=1
            x[64:68]=0;x[64]=1;query_input=x.copy()
            cpu_out.append([cpu[k].get(e['key'],-1) for k in cpu])
            for name in ['repeat_lru','lru']:
                if e['key'] in cpu[name]:cpu[name][e['key']]=cpu[name].pop(e['key'])
        else:
            x=query_input.copy();x[64:68]=0
            if arm=='severed' and e['phase'] not in ['bootstrap','gate']:x[:32]=0
            cpu_out.append([-2]*3)
            if e['label']>=0:
                for name,bank in cpu.items():
                    if name=='repeat_lru' and previous!=e['key']:continue
                    bank.pop(e['key'],None);bank[e['key']]=e['label']
                    if name!='unbounded' and len(bank)>8:del bank[next(iter(bank))]
            previous=e['key']
        z=(x.astype('i4')@mats[0][0].astype('i4')>mats[0][1]).astype('u1')
        f0=(z.astype('i4')@mats[1][0].astype('i4')>mats[1][1]).astype('u1')
        y0,p0=active(refs[0],f0,e['label'],P0)
        f1=(y0.astype('i4')@mats[2][0].astype('i4')>mats[2][1]).astype('u1')
        y1,p1=active(refs[1],f1,e['label'],P1)
        y=(y1.astype('i4')@mats[3][0].astype('i4')>mats[3][1]).astype('u1')
        if not e['teacher']:last=y.copy()
        xs.append(x);ys.append(y);s0.append(refs[0].ram[:]);s1.append(refs[1].ram[:]);ps0.append(p0);ps1.append(p1);features0.append(f0);features1.append(f1)
    return dict(events=rows,inputs=np.asarray(xs,'u1'),outputs=np.asarray(ys,'u1'),states0=np.asarray(s0,'<u4'),states1=np.asarray(s1,'<u4'),potentials0=np.asarray(ps0,'<i4'),potentials1=np.asarray(ps1,'<i4'),features0=np.asarray(features0,'u1'),features1=np.asarray(features1,'u1'),cpu=np.asarray(cpu_out,'<i4'))

def assess(a):
    result={}
    for phase in dict.fromkeys(e['phase'] for e in a['events']):
        inds=[e['index'] for e in a['events'] if e['phase']==phase and not e['teacher']]
        r={}
        for name,sl in [('protected',slice(64,72)),('always_bank',slice(72,80))]:
            hits=ambiguous=missing=wrong=0
            for i in inds:
                b=a['outputs'][i,sl].reshape(2,4).any(1);actual=OUTCOME[a['events'][i]['key']]
                hits+=int(b[actual] and not b[1-actual]);ambiguous+=int(b.all());missing+=int(not b.any());wrong+=int(b[1-actual] and not b[actual])
            r[name]=dict(correct=hits,total=len(inds),ambiguous=ambiguous,missing=missing,wrong=wrong)
        r['cpu']={name:dict(correct=sum(int(a['cpu'][i,k]==OUTCOME[a['events'][i]['key']]) for i in inds),total=len(inds)) for k,name in enumerate(['unbounded','repeat_lru','lru'])}
        result[phase]=r
    return result

if __name__=='__main__':
    import json
    for seed in SEEDS:
        rams=[packed(weights(seed,k),k) for k in range(2)]
        for arm in ['native','open','frozen','closed']:
            a=simulate(rams,'heldout',arm)
            print(json.dumps(dict(seed=seed,arm=arm,assessment=assess(a))))
