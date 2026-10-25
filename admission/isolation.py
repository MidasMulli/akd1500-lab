"""Two native correcting memories, shared clocks, disjoint context routes.

Both banks are enabled for every teacher. Only native feature routing decides
which receives a matching input. The foreign input fields are disjoint from
plastic connection support, and every inactive-bank boundary is checked.
"""
import numpy as np
from admission.recycle import N,D,KEYS,Reference,OUTCOME
SEEDS=[114101,114201,114301]
STREAMS=[117000,117101,117201,117301]
ARMS=['native','no_retire','frozen','freeze_A','freeze_B','solo_A','solo_B','open','closed']
STAGES=['truth','gate','repeat','pressure']
UA,UB=64,32
PA=list(range(4))+list(range(32,36));PB=list(range(4))+list(range(16,20))
HA=[j for j in range(UA) if j not in PA];HB=[j for j in range(UB) if j not in PB]
DIMENSIONS=[(D,UA,PA,HA),(D,UB,PB,HB)]

def matrices(arm):
    a=np.zeros((N,64),'i1');at=np.full(64,127,'i4')
    for key in range(KEYS):
        for bit in range(4):
            j=32+bit*8+((key>>bit)&1)*4;a[j,key]=a[j,16+key]=1
        a[68,key]=-1;a[68,16+key]=1;at[key]=3;at[16+key]=4
    a[64,32]=1;at[32]=0
    a[65,34:37]=1;a[66,34]=a[67,35]=1;a[84,34]=a[85,35]=-1;a[84:86,36]=1;at[34:37]=[1,1,2]
    a[24,50]=a[25,51]=1;at[50:52]=0
    if arm=='open':a[:,34:37]=0;a[65,34]=1;at[34]=0
    if arm=='closed':a[:,34:37]=0;at[34:37]=127
    b=np.zeros((64,128),'i1');bt=np.full(128,127,'i4')
    for ctx in range(2):
        for key in range(KEYS):
            base=ctx*64
            b[key,base+key]=b[32,base+key]=b[50+ctx,base+key]=1;bt[base+key]=2
            b[key,base+16+key]=b[50+ctx,base+16+key]=1;b[34:37,base+16+key]=1;bt[base+16+key]=2
            if arm!='no_retire':b[16+key,base+32+key]=b[50+ctx,base+32+key]=1;b[34:37,base+32+key]=1;bt[base+32+key]=2
    c=np.zeros((128,D),'i1');ct=np.zeros(D,'i4')
    for ctx in range(2):
        base=ctx*64;off=ctx*128
        c[base:base+12,off:off+25]=c[base+16:base+28,off:off+25]=c[base+32:base+44,off:off+25]=1
        for key in range(KEYS):
            c[[base+key,base+16+key],off+25+key*7:off+25+(key+1)*7]=1
            c[base+32+key,off+25+key*7:off+25+key*7+6]=1;c[base+32+key,off+109]=1
    bridge=np.zeros((UA,D),'i1');tbridge=np.zeros(D,'i4')
    bridge[HA[24],:25]=1;bridge[HA[25],109]=1
    for key in range(KEYS):
        bridge[HA[key],25+key*7:25+key*7+6]=1
        bridge[HA[12+key],25+key*7+6]=1
        bridge[HA[26+key],576+key*7:576+(key+1)*7]=1
    for p,j in enumerate(PA):bridge[j,256+p*32:256+(p+1)*32]=1
    bridge[HA[38],700:725]=1
    relay=np.zeros((UB,N),'i1')
    for key,j in enumerate(HB[:KEYS]):
        for bit in range(4):relay[j,32+bit*8+((key>>bit)&1)*4:36+bit*8+((key>>bit)&1)*4]=1
        relay[j,96+key]=1
    for p,j in enumerate(HB[12:20]):relay[j,p]=1;relay[j,84+p//4]=1
    for p,j in enumerate(PB):relay[j,72+p]=1;relay[j,84+p//4]=1
    relay[HB[20],24]=relay[HB[21],25]=1;relay[HB[20:22],80]=1
    return [(a,at),(b,bt),(c,ct),(bridge,tbridge),(relay,np.zeros(N,'i4'))]

def weights(seed,k):
    n,u,plastic,helpers=DIMENSIONS[k];w=np.zeros((n,u),'i1');rng=np.random.default_rng(seed+71*k)
    for j in plastic:w[:25,j]=1;w[rng.choice(np.arange(25,109),7,False),j]=1
    def helper(j,inds):
        inds=list(inds);assert len(inds)<=32 and len(set(inds))==len(inds)
        w[inds,j]=1
        if len(inds)<32:w[n-(32-len(inds)):,j]=1
    if k==0:
        for key in range(KEYS):
            helper(helpers[key],range(128+25+key*7,128+25+key*7+6))
            helper(helpers[12+key],range(128+25+key*7,128+25+(key+1)*7))
            helper(helpers[26+key],range(25+key*7,25+(key+1)*7))
        helper(helpers[24],range(128,153));helper(helpers[25],[237]);helper(helpers[38],range(25))
        used=set(helpers[:39])
    else:
        for key in range(KEYS):helper(helpers[key],[*range(25+key*7,25+(key+1)*7),*range(576+key*7,576+(key+1)*7)])
        for p,j in enumerate(helpers[12:20]):helper(j,range(256+p*32,256+(p+1)*32))
        helper(helpers[20],range(700,725));helper(helpers[21],range(25));used=set(helpers[:22])
    for j in helpers:
        if j not in used:w[n-32:,j]=1
    assert np.all(w.sum(0)==32)
    return w

def thresholds(k):
    _,u,plastic,h=DIMENSIONS[k];lt=np.full(u,32,'i4');lt[plastic]=0;t=np.full(u,31,'i4')
    if k==0:
        t[h[:12]]=5;t[h[12:24]]=6;t[h[26:38]]=6;t[h[24]]=t[h[38]]=24;t[h[25]]=0
    else:t[h[:12]]=6;t[h[20:22]]=24
    return lt,t

def packed(w,k):
    lt,_=thresholds(k);ram=[0,0]
    for j in range(w.shape[1]):
        ram += [0,0,0,(int(lt[j])<<16)|16]
        ram += [sum(int(w[p*32+v,j])<<v for v in range(32)) for p in range(D//32)]
    return ram

def schedule(stage,arm,stream=117000):
    rows=[];rng=np.random.default_rng(stream)
    def row(ctx,key,teacher,actual,phase,erase=False,override=False,p0=0,p1=0):
        label=(1-actual if erase else actual) if teacher else -1
        modes=[label,label]
        if phase!='bootstrap':
            if arm=='frozen':modes=[-1,-1]
            if arm=='freeze_A':modes[0]=-1
            if arm=='freeze_B':modes[1]=-1
        rows.append(dict(index=len(rows),context=ctx,key=key,teacher=teacher,actual=actual,label=label,labels=modes,erase=erase,phase=phase,override=override,p0=p0,p1=p1))
    def add(ctx,key,actual,phase,teach=True):
        row(ctx,key,False,actual,phase)
        if teach:row(ctx,key,True,actual,phase);row(ctx,key,True,actual,phase,True)
    if stage=='truth':
        for ctx in range(2):
            for p0,p1 in [(0,0),(1,0),(0,1),(1,1)]:
                for actual in range(2):
                    for teach,erase in [(False,False),(True,False),(True,True)]:row(ctx,0,teach,actual,'truth',erase,True,p0,p1)
        for e in rows:e['labels']=[-1,-1]
    else:
        for rep in range(3):
            order=[(ctx,key) for ctx in range(2) for key in range(4)]
            if stream!=117000:rng.shuffle(order)
            for ctx,key in order:add(ctx,key,OUTCOME[key]^ctx,'bootstrap')
        for ctx,key in [(c,k) for c in range(2) for k in range(4)]*4:add(ctx,key,OUTCOME[key]^ctx,'old_probe',False)
        if stage=='pressure':
            for key in range(4,12):
                for _ in range(6):add(0,key,OUTCOME[key],'pressure')
            order=[(0,k) for k in [2,3,8,9,10,11] for _ in range(16)]+[(1,k) for k in range(4) for _ in range(24)]
            rng.shuffle(order)
            for ctx,key in order:add(ctx,key,OUTCOME[key]^ctx,'capacity_probe',False)
        flips=[0,0]
        for cycle in range(1,0 if stage=='pressure' else 2 if stage=='gate' else 13):
            ctx=(cycle-1)%2;flips[ctx]^=1
            for rep in range(2):
                keys=[0,1]
                if stream!=117000:rng.shuffle(keys)
                for key in keys:add(ctx,key,OUTCOME[key]^ctx^flips[ctx],f'change_{cycle}')
            order=[(c,k) for c in range(2) for k in range(4) for _ in range(12)]
            rng.shuffle(order)
            for c,k in order:add(c,k,OUTCOME[k]^c^(flips[c] if k<2 else 0),f'probe_{cycle}',False)
    if arm in ['solo_A','solo_B']:
        rows=[e for e in rows if e['context']==int(arm=='solo_B')]
        for i,e in enumerate(rows):e['index']=i
    return rows

def simulate(rams,stage,arm,stream=117000):
    mats=matrices(arm);refs=[Reference(rams[k],n=D,u=DIMENSIONS[k][1],nw=32,classes=2) for k in range(2)]
    rows=schedule(stage,arm,stream);last=np.zeros(N,'u1');xs=[];ys=[];ss=[[],[]];ps=[[],[]];fs=[[],[]];cpu=[];dictionary={};aliased={};lru=[{},{}]
    for e in rows:
        x=last.copy()
        if not e['teacher'] or e['override']:
            x[24:28]=0;x[24+e['context']]=1;x[32:64]=0
            for bit in range(4):x[32+bit*8+((e['key']>>bit)&1)*4:36+bit*8+((e['key']>>bit)&1)*4]=1
        x[64:72]=0;x[64+int(e['teacher'])]=1;x[68]=e['erase']
        if e['teacher']:x[66+e['actual']]=1
        if e['override']:x[84:86]=[e['p0'],e['p1']]
        key=(e['context'],e['key']);cache=lru[e['context']]
        cpu.append([dictionary.get(key,-1),aliased.get(e['key'],-1),cache.get(e['key'],-1)] if not e['teacher'] else [-2,-2,-2])
        if not e['teacher'] and e['key'] in cache:cache[e['key']]=cache.pop(e['key'])
        if e['teacher'] and not e['erase'] and stage!='truth':
            dictionary[key]=aliased[e['key']]=e['actual'];cache.pop(e['key'],None);cache[e['key']]=e['actual']
            if len(cache)>8:del cache[next(iter(cache))]
        z=x.astype('i4')
        for w,t in mats[:3]:z=(z@w.astype('i4')>t).astype('u1')
        for k in range(2):
            fs[k].append(z.copy());p=np.asarray(refs[k].forward(z,e['labels'][k]),'<i4');v=(p>thresholds(k)[1]).astype('u1')
            ps[k].append(p);ss[k].append(refs[k].ram[:]);w,t=mats[3+k];z=(v.astype('i4')@w.astype('i4')>t).astype('u1')
        if not e['teacher']:last=z.copy()
        xs.append(x);ys.append(z)
    a=dict(events=rows,inputs=np.asarray(xs,'u1'),outputs=np.asarray(ys,'u1'),cpu=np.asarray(cpu,'<i4'))
    for k in range(2):a[f'states{k}']=np.asarray(ss[k],'<u4');a[f'potentials{k}']=np.asarray(ps[k],'<i4');a[f'features{k}']=np.asarray(fs[k],'u1')
    return a

def assess(a):
    out={}
    for phase in dict.fromkeys(e['phase'] for e in a['events']):
        ii=[e['index'] for e in a['events'] if e['phase']==phase and not e['teacher']];p=a['outputs'][ii,84:86];actual=np.asarray([a['events'][i]['actual'] for i in ii]);correct=np.asarray([int(v[t] and not v[1-t]) for v,t in zip(p,actual)])
        out[phase]=dict(correct=int(correct.sum()),queries=len(ii),ambiguous=int(p.all(1).sum()),missing=int((~p.any(1)).sum()),cpu=int((a['cpu'][ii,0]==actual).sum()),aliased_cpu=int((a['cpu'][ii,1]==actual).sum()),cpu_lru8=int((a['cpu'][ii,2]==actual).sum()),
            by_context={str(ctx):dict(correct=int(correct[[a['events'][i]['context']==ctx for i in ii]].sum()),total=sum(a['events'][i]['context']==ctx for i in ii)) for ctx in range(2)})
    return out

if __name__=='__main__':
    import json
    for seed in SEEDS:
        for stream in STREAMS:
            a=simulate([packed(weights(seed,k),k) for k in range(2)],'repeat','native',stream);r=assess(a)
            print(json.dumps(dict(seed=seed,stream=stream,old=r['old_probe'],correct=sum(v['correct'] for k,v in r.items() if k.startswith('probe_')),total=1152)))
