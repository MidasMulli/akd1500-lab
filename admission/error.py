"""Prediction-error admission: native saved prediction, delayed external outcome.

The objective and comparator are engineered. Only consequence connections learn.
No importance, correctness, or admission bit is provided by the actor's host.
"""
import numpy as np
from admission.model import N, N1 as D, U1 as U, P1 as PLASTIC, H1 as HELPERS, KEYS, Reference

SEEDS = [114101, 114201, 114301]
ARMS = ['native', 'open', 'closed', 'frozen', 'cut']
STAGES = ['truth', 'learn', 'reversal', 'capacity']
OUTCOME = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]

def matrices(arm):
    # Signed fixed inference gates; the learning layer remains strictly binary.
    a = np.zeros((N, 64), 'i1'); at = np.full(64, 127, 'i4')
    for key in range(KEYS):
        for b in range(4): a[32+b*8+((key>>b)&1)*4, key] = 1
        at[key] = 3
    a[64,32] = 1; at[32] = 0
    # teacher AND actual0 AND NOT predicted0; analogous class1; ambiguity.
    a[65,34:37] = 1; a[66,34] = a[67,35] = 1
    a[84,34] = a[85,35] = -1
    a[84:86,36] = 1; at[34:37] = [1,1,2]
    if arm == 'open':
        a[:,34:37] = 0; a[65,34] = 1; at[34] = 0
    elif arm == 'closed': a[:,34:37] = 0; at[34:37] = 127
    b = np.zeros((64, 64), 'i1'); bt = np.full(64,127,'i4')
    for key in range(KEYS):
        b[key,key] = b[32,key] = 1; bt[key] = 1
        b[key,16+key] = 1; b[34:37,16+key] = 1; bt[16+key] = 1
    c = np.zeros((64,D),'i1'); ct = np.zeros(D,'i4')
    for key in range(KEYS): c[[key,16+key],key*32:(key+1)*32] = 1
    relay = np.zeros((U,N),'i1')
    for key,j in enumerate(HELPERS[:KEYS]):
        for bit in range(4): relay[j,32+bit*8+((key>>bit)&1)*4:36+bit*8+((key>>bit)&1)*4] = 1
        relay[j,96+key] = 1
    for p,j in enumerate(PLASTIC):
        relay[j,72+p] = 1; relay[j,84+(p//4)] = 1
    return [(a,at),(b,bt),(c,ct),(relay,np.zeros(N,'i4'))]

def weights(seed):
    w=np.zeros((D,U),'i1'); rng=np.random.default_rng(seed)
    for j in PLASTIC: w[rng.choice(KEYS*32,32,False),j]=1
    for key,j in enumerate(HELPERS[:KEYS]): w[key*32:(key+1)*32,j]=1
    for j in HELPERS[KEYS:]: w[D-32:,j]=1
    return w

def thresholds():
    lt=np.full(U,32,'i4'); lt[PLASTIC]=0
    t=np.full(U,31,'i4'); t[PLASTIC]=27
    return lt,t

def packed(w):
    lt,_=thresholds(); ram=[0,0]
    for j in range(U):
        ram += [0,0,0,(int(lt[j])<<16)|16]
        ram += [sum(int(w[k*32+v,j])<<v for v in range(32)) for k in range(D//32)]
    return ram

def schedule(stage,arm):
    rows=[]
    def row(key,teacher,actual,phase,override=False,p0=0,p1=0):
        label = actual if teacher else -1
        if arm=='frozen' and phase!='bootstrap': label=-1
        rows.append(dict(index=len(rows),key=key,teacher=teacher,actual=actual,label=label,
                         phase=phase,override=override,p0=p0,p1=p1,cut=arm=='cut' and teacher))
    def add(keys,phase,teach=True,flip=False):
        for key in keys:
            actual=OUTCOME[key] ^ int(flip)
            row(int(key),False,actual,phase)
            if teach: row(int(key),True,actual,phase)
    if stage=='truth':
        for p0,p1 in [(0,0),(1,0),(0,1),(1,1)]:
            for actual in range(2):
                for teacher in [False,True]: row(0,teacher,actual,'truth',True,p0,p1)
        for e in rows: e['label']=-1
    else:
        add([0,1]*8,'bootstrap')
        add([0,1]*4,'old_probe',False)
        if stage=='learn':
            add([2,3]*8,'novel')
            add(np.random.default_rng(115901).choice([0,1,2,3],48).tolist(),'heldout',False)
        elif stage=='reversal':
            add([0,1]*8,'reversal',flip=True)
            add([0,1]*24,'heldout',False,True)
        else:
            add([k for k in range(2,8) for _ in range(6)],'distractions')
            add([k for k in range(8,12) for _ in range(8)],'later')
            add(np.random.default_rng(115902).choice([0,1,8,9,10,11],96).tolist(),'heldout',False)
    return rows

def simulate(ram,stage,arm):
    mats=matrices(arm); ref=Reference(ram,n=D,u=U,nw=32,classes=2)
    last=np.zeros(N,'u1'); rows=schedule(stage,arm)
    xs=[]; ys=[]; states=[]; ps=[]; fs=[]; cpu=[]
    banks={'dictionary':{},'lru8':{}}; boot_end=False
    for e in rows:
        x=last.copy()
        if not e['teacher'] or e['override']:
            x[32:64]=0
            for bit in range(4): x[32+bit*8+((e['key']>>bit)&1)*4:36+bit*8+((e['key']>>bit)&1)*4]=1
        x[64:68]=0; x[64+int(e['teacher'])]=1
        if e['teacher']: x[66+e['actual']]=1
        if e['override']: x[84:86]=[e['p0'],e['p1']]
        if e['cut']: x[84:86]=0; last[84:86]=0
        cpu.append([bank.get(e['key'],-1) if not e['teacher'] else -2 for bank in banks.values()])
        if e['teacher'] and stage!='truth':
            for name,bank in banks.items():
                bank.pop(e['key'],None); bank[e['key']]=e['actual']
                if name=='lru8' and len(bank)>8: del bank[next(iter(bank))]
        else:
            if e['key'] in banks['lru8']: banks['lru8'][e['key']]=banks['lru8'].pop(e['key'])
        z=x.astype('i4')
        for w,t in mats[:3]: z=(z@w.astype('i4')>t).astype('u1')
        fs.append(z.copy()); p=np.asarray(ref.forward(z,e['label']),'<i4')
        v=(p>thresholds()[1]).astype('u1'); y=(v.astype('i4')@mats[3][0]>mats[3][1]).astype('u1')
        if not e['teacher']: last=y.copy()
        xs.append(x);ys.append(y);states.append(ref.ram[:]);ps.append(p)
    return dict(events=rows,inputs=np.asarray(xs,'u1'),outputs=np.asarray(ys,'u1'),
                states=np.asarray(states,'<u4'),potentials=np.asarray(ps,'<i4'),features=np.asarray(fs,'u1'),cpu=np.asarray(cpu,'<i4'))

def assess(a):
    result={}
    for phase in dict.fromkeys(e['phase'] for e in a['events']):
        ii=[e['index'] for e in a['events'] if e['phase']==phase and not e['teacher']]
        pred=a['outputs'][ii,84:86]; actual=np.asarray([a['events'][i]['actual'] for i in ii])
        result[phase]=dict(queries=len(ii),correct=sum(int(v[t] and not v[1-t]) for v,t in zip(pred,actual)),
            ambiguous=int(pred.all(1).sum()),missing=int((~pred.any(1)).sum()),
            cpu={name:int((a['cpu'][ii,k]==actual).sum()) for k,name in enumerate(['dictionary','lru8'])})
    teachers=[e for e in a['events'] if e['teacher']]
    result['admission']=dict(teachers=len(teachers),passed=sum(int(a['features'][e['index']].any()) for e in teachers),
        blocked=sum(int(not a['features'][e['index']].any()) for e in teachers))
    return result

def screen():
    for arm in ['native','open','closed']:
        a=simulate(packed(weights(SEEDS[0])),'truth',arm)
        for e,x,y in zip(a['events'],a['inputs'],a['outputs']):
            expected=not e['teacher'] or (arm=='open' or arm!='closed' and
                (not x[84+e['actual']] or x[84] and x[85]))
            assert bool(y[96])==bool(expected),(arm,e,y[96],expected)
    return {f'{seed}_{arm}_{stage}':assess(simulate(packed(weights(seed)),stage,arm))
        for seed in SEEDS for arm in ['native','open','frozen','cut'] for stage in ['learn','reversal','capacity']}

if __name__=='__main__':
    import json
    print(json.dumps(screen(),indent=2))
