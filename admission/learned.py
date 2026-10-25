"""Native synaptic recency controls native consequence-memory admission.

The management rule is designed. Its recent-key state is encoded and updated in
one plastic native record, rather than supplied by a host history comparator.
"""
import numpy as np
from admission import model as prior
from admission.model import N,N0,N1,U0,U1,P0,P1,H0,H1,KEYS,RAW,QUERY,SEEDS,STAGES,OUTCOME,Reference,active,assess
NM=256;UM=32
ARMS=['native','open','closed','frozen','manager_frozen','severed']
DIMENSIONS=[(NM,UM,1),(N0,U0,2),(N1,U1,2)]

def matrices(arm):
    encoder=np.zeros((N,NM),'i1');et=np.full(NM,4,'i4')
    encoder[64,:25]=1;et[:25]=0
    for key in range(KEYS):
        for b in range(4):encoder[32+b*8+((key>>b)&1)*4,25+key*7:25+(key+1)*7]=1
        et[25+key*7:25+(key+1)*7]=3
    encoder[80,128:160]=encoder[65,128:160]=1;et[128:160]=1
    policy=np.zeros((UM,N0),'i1');pt=np.zeros(N0,'i4')
    for key in range(KEYS):
        policy[1+key,key*32:(key+1)*32]=1
        a=slice(RAW+key*32,RAW+(key+1)*32)
        policy[1+key,a]=policy[13+key,a]=1
        if arm=='open':pt[a]=0
        else:
            policy[26,a]=1;pt[a]=2
            if arm!='closed':policy[25,a]=1
    policy[26,QUERY:QUERY+32]=1
    policy[0,QUERY+32:QUERY+64]=1
    bridge=prior.matrices('native')[2][0].copy();bt=prior.matrices('native')[2][1].copy()
    # Query-only native manager match is carried through a read-only helper.
    bridge[H0[2*KEYS+1],RAW+8*32:RAW+9*32]=1
    relay=prior.matrices('native')[3][0].copy()
    for key in range(KEYS):
        for b in range(4):relay[H1[key],32+b*8+((key>>b)&1)*4:36+b*8+((key>>b)&1)*4]=1
    relay[H1[KEYS+8],80]=1
    return [(encoder,et),(policy,pt),(bridge,bt),(relay,np.zeros(N,'i4'))]

def weights(seed,k):
    if k:
        w=prior.weights(seed,k-1)
        if k==1:w[:,H0[2*KEYS+1]]=0;w[QUERY+32:QUERY+64,H0[2*KEYS+1]]=1
        else:w[:,H1[KEYS+8]]=0;w[RAW+8*32:RAW+9*32,H1[KEYS+8]]=1
        return w
    w=np.zeros((NM,UM),'i1');rng=np.random.default_rng(seed+91)
    w[:25,0]=1;w[rng.choice(np.arange(25,25+KEYS*7),7,False),0]=1
    for key in range(KEYS):
        for j in [1+key,13+key]:w[25+key*7:25+(key+1)*7,j]=1;w[200:225,j]=1
    w[128:160,25]=1;w[:25,26]=1;w[200:207,26]=1
    for j in range(27,UM):w[200:232,j]=1
    return w

def thresholds(k):
    n,u,_=DIMENSIONS[k];learn=np.full(u,32,'i4');threshold=np.full(u,31,'i4')
    plastic=[0] if k==0 else P0 if k==1 else P1
    learn[plastic]=0
    if k==0:threshold[1:25]=6;threshold[26]=24
    else:threshold[plastic]=27
    return learn,threshold

def packed(w,k):
    n,u=w.shape;lt,_=thresholds(k);ram=[0,0]
    for j in range(u):
        ram += [0,0,0,(int(lt[j])<<16)|16]
        ram += [sum(int(w[a*32+b,j])<<b for b in range(32)) for a in range(n//32)]
    return ram

def schedule(stage,arm):
    # Original raw workload is unchanged. Only the native controller implementation differs.
    rows=prior.schedule(stage,'frozen' if arm=='frozen' else 'native')
    for e in rows:
        e['manager_learn']=not e['teacher'] and not(arm=='manager_frozen' and e['phase'] not in ['bootstrap','gate'])
        e['labels']=[0 if e['manager_learn'] else -1,e['label'],e['label']]
    return rows

def simulate(rams,stage,arm,*,events=None):
    mats=matrices(arm);refs=[Reference(rams[k],n=n,u=u,nw=32,classes=c) for k,(n,u,c) in enumerate(DIMENSIONS)]
    rows=schedule(stage,arm) if events is None else events
    last=np.zeros(N,'u1');xs=[];ys=[];states=[[],[],[]];potentials=[[],[],[]];features=[[],[],[]]
    cpu={'unbounded':{},'repeat_lru':{},'lru':{}};cpu_out=[];previous=None
    for e in rows:
        x=last.copy()
        if not e['teacher']:
            x[32:64]=0
            for b in range(4):x[32+b*8+((e['key']>>b)&1)*4:36+b*8+((e['key']>>b)&1)*4]=1
            x[64:68]=0;x[64]=1
            cpu_out.append([cpu[name].get(e['key'],-1) for name in cpu])
            for name in ['repeat_lru','lru']:
                if e['key'] in cpu[name]:cpu[name][e['key']]=cpu[name].pop(e['key'])
        else:
            x[64:68]=0;x[65]=1
            if arm=='severed' and e['phase'] not in ['bootstrap','gate']:x[80]=0;last[80]=0
            cpu_out.append([-2]*3)
            if e['label']>=0:
                for name,bank in cpu.items():
                    if name=='repeat_lru' and previous!=e['key']:continue
                    bank.pop(e['key'],None);bank[e['key']]=e['label']
                    if name!='unbounded' and len(bank)>8:del bank[next(iter(bank))]
            previous=e['key']
        f=(x.astype('i4')@mats[0][0].astype('i4')>mats[0][1]).astype('u1');features[0].append(f)
        for k in range(3):
            p=np.asarray(refs[k].forward(f,e['labels'][k]),'<i4');v=(p>thresholds(k)[1]).astype('u1')
            states[k].append(refs[k].ram[:]);potentials[k].append(p)
            if k<2:
                w,t=mats[k+1];f=(v.astype('i4')@w.astype('i4')>t).astype('u1');features[k+1].append(f)
        y=(v.astype('i4')@mats[3][0].astype('i4')>mats[3][1]).astype('u1')
        if not e['teacher']:last=y.copy()
        xs.append(x);ys.append(y)
    a=dict(events=rows,inputs=np.asarray(xs,'u1'),outputs=np.asarray(ys,'u1'),cpu=np.asarray(cpu_out,'<i4'))
    for k in range(3):
        a[f'states{k}']=np.asarray(states[k],'<u4');a[f'potentials{k}']=np.asarray(potentials[k],'<i4');a[f'features{k}']=np.asarray(features[k],'u1')
    return a

if __name__=='__main__':
    import json
    for seed in SEEDS:
        rams=[packed(weights(seed,k),k) for k in range(3)]
        for arm in ['native','open','frozen','manager_frozen','severed']:
            a=simulate(rams,'heldout',arm);r=assess(a)
            print(json.dumps(dict(seed=seed,arm=arm,final_probe=r['final_probe'],heldout=r['heldout'])),flush=True)
