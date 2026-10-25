"""Prospective native correction screen after the error-only study's failures.

Engineered common features keep records eligible. A contradictory teacher shifts
one key connection to a non-query sentinel, retiring that class association.
The native winner selects a matching record; the host never chooses/edits it.
"""
import numpy as np
from admission.error import N,D,U,PLASTIC,HELPERS,KEYS,SEEDS,OUTCOME,Reference
ARMS=['native','no_retire','frozen','cut','open','closed']
STAGES=['truth','correction','repeat']

def matrices(arm):
    a=np.zeros((N,64),'i1');at=np.full(64,127,'i4')
    for key in range(KEYS):
        for b in range(4):
            j=32+b*8+((key>>b)&1)*4;a[j,key]=a[j,16+key]=1
        a[68,key]=-1;a[68,16+key]=1;at[key]=3;at[16+key]=4
    a[64,32]=1;at[32]=0
    a[65,34:37]=1;a[66,34]=a[67,35]=1;a[84,34]=a[85,35]=-1
    a[84:86,36]=1;at[34:37]=[1,1,2]
    if arm=='open':a[:,34:37]=0;a[65,34]=1;at[34]=0
    if arm=='closed':a[:,34:37]=0;at[34:37]=127
    b=np.zeros((64,64),'i1');bt=np.full(64,127,'i4')
    for key in range(KEYS):
        b[key,key]=b[32,key]=1;bt[key]=1
        b[key,16+key]=1;b[34:37,16+key]=1;bt[16+key]=1
        if arm!='no_retire':b[16+key,32+key]=1;b[34:37,32+key]=1;bt[32+key]=1
    c=np.zeros((64,D),'i1');ct=np.zeros(D,'i4')
    c[:12,:25]=c[16:28,:25]=c[32:44,:25]=1
    for key in range(KEYS):
        sl=slice(25+key*7,25+(key+1)*7);c[[key,16+key],sl]=1
        c[32+key,25+key*7:25+key*7+6]=1;c[32+key,109]=1
    relay=np.zeros((U,N),'i1')
    for key,j in enumerate(HELPERS[:KEYS]):
        for bit in range(4):relay[j,32+bit*8+((key>>bit)&1)*4:36+bit*8+((key>>bit)&1)*4]=1
        relay[j,96+key]=1
    relay[HELPERS[KEYS],80]=1
    for p,j in enumerate(PLASTIC):relay[j,72+p]=1;relay[j,84+p//4]=1
    return [(a,at),(b,bt),(c,ct),(relay,np.zeros(N,'i4'))]

def weights(seed):
    w=np.zeros((D,U),'i1');rng=np.random.default_rng(seed)
    for j in PLASTIC:w[:25,j]=1;w[rng.choice(np.arange(25,109),7,False),j]=1
    for key,j in enumerate(HELPERS[:KEYS]):w[25+key*7:25+(key+1)*7,j]=1;w[D-25:,j]=1
    w[:25,HELPERS[KEYS]]=1;w[D-7:,HELPERS[KEYS]]=1
    for j in HELPERS[KEYS+1:]:w[D-32:,j]=1
    return w

def thresholds():
    lt=np.full(U,32,'i4');lt[PLASTIC]=0;t=np.full(U,31,'i4')
    t[HELPERS[:KEYS]]=6;t[HELPERS[KEYS]]=24
    return lt,t

def packed(w):
    lt,_=thresholds();ram=[0,0]
    for j in range(U):
        ram += [0,0,0,(int(lt[j])<<16)|16]
        ram += [sum(int(w[k*32+v,j])<<v for v in range(32)) for k in range(D//32)]
    return ram

def schedule(stage,arm):
    rows=[]
    def row(key,teacher,actual,phase,erase=False,override=False,p0=0,p1=0):
        label=(1-actual if erase else actual) if teacher else -1
        if arm=='frozen' and phase!='bootstrap':label=-1
        rows.append(dict(index=len(rows),key=key,teacher=teacher,actual=actual,label=label,erase=erase,
            phase=phase,override=override,p0=p0,p1=p1,cut=arm=='cut' and teacher))
    def add(key,actual,phase,teach=True):
        row(key,False,actual,phase)
        if teach:row(key,True,actual,phase);row(key,True,actual,phase,True)
    if stage=='truth':
        for p0,p1 in [(0,0),(1,0),(0,1),(1,1)]:
            for actual in range(2):
                for teacher,erase in [(False,False),(True,False),(True,True)]:row(0,teacher,actual,'truth',erase,True,p0,p1)
        for e in rows:e['label']=-1
    else:
        for _ in range(3):
            for k in [0,1,2,3]:add(k,OUTCOME[k],'bootstrap')
        for k in [0,1,2,3]*4:add(k,OUTCOME[k],'old_probe',False)
        for cycle in range(1,2 if stage=='correction' else 13):
            flip=cycle%2
            for _ in range(2):
                for k in [0,1]:add(k,OUTCOME[k]^flip,f'change_{cycle}')
            for k in np.random.default_rng(116000+cycle).choice([0,1,2,3],48).tolist():
                add(k,OUTCOME[k]^(flip if k<2 else 0),f'probe_{cycle}',False)
    return rows

def simulate(ram,stage,arm):
    mats=matrices(arm);ref=Reference(ram,n=D,u=U,nw=32,classes=2);last=np.zeros(N,'u1')
    xs=[];ys=[];ss=[];ps=[];fs=[];cpu=[];bank={};rows=schedule(stage,arm)
    for e in rows:
        x=last.copy()
        if not e['teacher'] or e['override']:
            x[32:64]=0
            for bit in range(4):x[32+bit*8+((e['key']>>bit)&1)*4:36+bit*8+((e['key']>>bit)&1)*4]=1
        x[64:72]=0;x[64+int(e['teacher'])]=1;x[68]=e['erase']
        if e['teacher']:x[66+e['actual']]=1
        if e['override']:x[84:86]=[e['p0'],e['p1']]
        if e['cut']:x[84:86]=0;last[84:86]=0
        cpu.append([bank.get(e['key'],-1) if not e['teacher'] else -2])
        if e['teacher'] and not e['erase'] and stage!='truth':bank[e['key']]=e['actual']
        z=x.astype('i4')
        for w,t in mats[:3]:z=(z@w.astype('i4')>t).astype('u1')
        fs.append(z.copy());p=np.asarray(ref.forward(z,e['label']),'<i4');v=(p>thresholds()[1]).astype('u1');y=(v.astype('i4')@mats[3][0]>mats[3][1]).astype('u1')
        if not e['teacher']:last=y.copy()
        xs.append(x);ys.append(y);ss.append(ref.ram[:]);ps.append(p)
    return dict(events=rows,inputs=np.asarray(xs,'u1'),outputs=np.asarray(ys,'u1'),states=np.asarray(ss,'<u4'),potentials=np.asarray(ps,'<i4'),features=np.asarray(fs,'u1'),cpu=np.asarray(cpu,'<i4'))

def assess(a):
    result={}
    for phase in dict.fromkeys(e['phase'] for e in a['events']):
        ii=[e['index'] for e in a['events'] if e['phase']==phase and not e['teacher']];pred=a['outputs'][ii,84:86];actual=np.asarray([a['events'][i]['actual'] for i in ii])
        result[phase]=dict(queries=len(ii),correct=sum(int(v[t] and not v[1-t]) for v,t in zip(pred,actual)),ambiguous=int(pred.all(1).sum()),missing=int((~pred.any(1)).sum()),cpu=int((a['cpu'][ii,0]==actual).sum()),
            stable_correct=sum(int(v[t] and not v[1-t]) for v,t,i in zip(pred,actual,ii) if a['events'][i]['key']>=2),stable_total=sum(a['events'][i]['key']>=2 for i in ii))
    return result

if __name__=='__main__':
    import json
    for seed in SEEDS:
        for arm in ['native','no_retire','frozen']:
            a=simulate(packed(weights(seed)),'repeat',arm);r=assess(a)
            print(json.dumps(dict(seed=seed,arm=arm,old=r['old_probe'],correct=sum(v['correct'] for k,v in r.items() if k.startswith('probe_')),total=576,stable=sum(v['stable_correct'] for k,v in r.items() if k.startswith('probe_')))))
