"""Two learned source cohorts protect and revise resident student memories.

Agreement is an engineered native rule, not a learned truth detector. External
evidence grounds the two class partitions independently. No source outcome is
provided to student transfer; both students receive constant wire class zero.
"""
import itertools
import numpy as np
from admission import transfer as base, revision
from admission.transfer import N,D,KEYS,PS,PR,HS,HR,OUTCOME,Reference

SEEDS=[124101,124201,124301]
ARMS=['native','follow_b','students_frozen','teacher_cut']
STAGES=['gate','repeat','correlated']
ROLES=revision.ROLES
GROUPS=[list(range(c*32,c*32+4)) for c in range(4)]
PT=sum(GROUPS,[]);HT=[j for j in range(128) if j not in PT]
DIMENSIONS=[(128,4,PT,HT),(64,2,PS,HS),(128,2,PR,HR)]

def matrices(arm):
    mats=revision.matrices('native');enc,et=mats[0]
    cmp=np.zeros((128,64),'i1');ct=np.full(64,127,'i4')
    for k in range(KEYS):cmp[HT[k],k]=1;ct[k]=0
    cmp[HT[12],32]=1;ct[32]=0
    for c,g in enumerate(GROUPS):cmp[g,28+c]=1;ct[28+c]=0
    for val in range(2):
        own=GROUPS[val]+GROUPS[2+val];other=GROUPS[1-val]+GROUPS[3-val]
        threshold=1
        if arm=='follow_b':own=GROUPS[2+val];other=GROUPS[3-val];threshold=0
        cmp[own,24+val]=1;cmp[other,24+val]=-8;ct[24+val]=threshold
        prev,opp=HT[14+val],HT[15-val]
        for col in [16+val*2,17+val*2]:
            cmp[own,col]=1;cmp[other,col]=-8;cmp[HT[13],col]=1
            if arm=='teacher_cut':cmp[HT[16],col]=-8
        cmp[prev,16+val*2]=-1;ct[16+val*2]=threshold+1
        cmp[[prev,opp],17+val*2]=1;ct[17+val*2]=threshold+3
    mats[1]=(cmp,ct)
    code,tt=mats[2];bridge,bt=mats[3]
    for c in range(4):
        sl=slice(224+c*32,256+c*32)
        code[32,sl]=code[28+c,sl]=1;tt[sl]=1
        bridge[HS[17+c],128+c*32:160+c*32]=1
    return mats

def weights(seed,k):
    if k:
        w=base.weights(seed,k)
        if k==1:
            for c in range(4):w[:,HS[17+c]]=0;w[224+c*32:256+c*32,HS[17+c]]=1
        else:
            for c,j in enumerate([86,87,89,90]):w[:,j]=0;w[128+c*32:160+c*32,j]=1
        return w
    w=np.zeros((D,128),'i1');rng=np.random.default_rng(seed)
    for j in PT:w[:25,j]=1;w[rng.choice(np.arange(25,109),7,False),j]=1
    used=set()
    def helper(j,inds):
        inds=list(inds);w[inds,j]=1
        if len(inds)<32:w[D-(32-len(inds)):,j]=1
        used.add(j)
    for key in range(KEYS):helper(HT[key],[*range(25+key*7,32+key*7),*range(128+key*7,135+key*7)])
    for j,start in zip(HT[12:17],[256,288,320,352,384]):helper(j,range(start,start+32))
    for j in HT:
        if j not in used:w[D-32:,j]=1
    assert np.all(w.sum(0)==32)
    return w

def thresholds(k):
    if k:return base.thresholds(k)
    lt=np.full(128,32,'i4');lt[PT]=0;t=np.full(128,31,'i4');t[HT[:12]]=6;t[HT[15]]=3
    return lt,t

def packed(w,k):
    lt,_=thresholds(k);ram=[0,0]
    for j in range(w.shape[1]):
        ram += [0,0,0,(int(lt[j])<<16)|16]
        ram += [sum(int(w[p*32+b,j])<<b for b in range(32)) for p in range(D//32)]
    return ram

def schedule(stage,arm):
    rows=[];truth=OUTCOME.copy();sources=[{},{}]
    def add(key,role,phase,cohort=0,fact=None,block=1):
        wire=(cohort*2+(1-fact if role=='ground_negative' else fact)) if role in ['ground','ground_negative'] else 0
        labels=[wire,-1,-1] if role in ['ground','ground_negative'] else [-1,0,0] if role=='transfer' else [-1,-1,-1]
        if arm=='students_frozen' and block:labels[1:]=[-1,-1]
        rows.append(dict(index=len(rows),key=key,role=role,phase=phase,labels=labels,wire=wire,actual=truth[key],cohort=cohort,fact=fact,block=block))
    def ground(keys,cohorts,phase,values,block=1):
        for _ in range(2):
            for c in cohorts:
                for key in keys:
                    fact=values[key];add(key,'ground',phase,c,fact,block);add(key,'ground_negative',phase,c,fact,block)
    def transfer(keys,phase,block=1):
        for _ in range(3):
            for key in keys:add(key,'query',phase,block=block);add(key,'transfer',phase,block=block)
    def probe(keys,phase,exam=True,repeat=12,block=1):
        order=np.tile(keys,repeat);np.random.default_rng(125000+len(rows)).shuffle(order)
        for key in order.tolist():add(key,'exam' if exam else 'query',phase,block=block)
    ground(range(4),[0,1],'bootstrap_source',truth,0)
    probe(range(4),'bootstrap_source_probe',False,4,0)
    transfer(range(4),'bootstrap_transfer',0)
    probe(range(4),'bootstrap_exam',True,12,0)
    cycles=1 if stage in ['gate','correlated'] else 6
    for cycle in range(1,cycles+1):
        suffix=f'_{cycle}'
        false=truth.copy()
        for key in [0,1]:false[key]^=1
        if stage=='correlated':
            ground([0,1],[0,1],'correlated_ground'+suffix,false)
            transfer(range(4),'correlated_transfer'+suffix)
            probe(range(4),'correlated_exam'+suffix)
            break
        ground([0,1],[1],'bad_source'+suffix,false)
        probe(range(4),'disagree_source'+suffix,False,4)
        transfer(range(4),'blocked_transfer'+suffix)
        probe(range(4),'protected_exam'+suffix)
        # Restore B, then let A observe a legitimate change before B sees it.
        ground([0,1],[1],'restore_b'+suffix,truth)
        for key in [0,1]:truth[key]^=1
        ground([0,1],[0],'legitimate_a'+suffix,truth)
        transfer(range(4),'pending_transfer'+suffix)
        probe(range(4),'pending_exam'+suffix)
        ground([0,1],[1],'legitimate_b'+suffix,truth)
        probe(range(4),'agree_source'+suffix,False,4)
        transfer(range(4),'accepted_transfer'+suffix)
        probe(range(4),'updated_exam'+suffix)
    if stage=='repeat':
        ground([4,5],[0,1],'new_source',truth)
        transfer(range(6),'new_transfer')
        probe(range(6),'new_exam',True,16)
    return rows

def simulate(rams,stage,arm):
    mats=matrices(arm);refs=[Reference(rams[k],n=D,u=u,nw=32,classes=c) for k,(u,c,_,_) in enumerate(DIMENSIONS)]
    rows=schedule(stage,arm);last=np.zeros(N,'u1');xs=[];ys=[];ss=[[],[],[]];ps=[[],[],[]];fs=[[],[],[]];cpu=[];teachers=[{},{}];student={};naive={}
    for e in rows:
        role=e['role'];x=last.copy()
        if role!='transfer':
            x[32:64]=0
            for bit in range(4):x[32+bit*8+((e['key']>>bit)&1)*4:36+bit*8+((e['key']>>bit)&1)*4]=1
        x[64:72]=0;x[65 if role in ['ground','ground_negative'] else 69 if role=='transfer' else 64]=1
        x[68]=int(role=='ground_negative');x[70]=int(role=='exam');x[71]=e['block']
        key=e['key']
        if role=='ground':teachers[e['cohort']][key]=e['fact']
        if role=='transfer':
            a,b=[t.get(key,-1) for t in teachers]
            if a==b and a>=0:student[key]=a
            if b>=0:naive[key]=b
        cpu.append([student.get(key,-1),naive.get(key,-1)])
        z=(x.astype('i4')@mats[0][0].astype('i4')>mats[0][1]).astype('u1')
        for k in range(3):
            fs[k].append(z.copy());p=np.asarray(refs[k].forward(z,e['labels'][k]),'<i4');z=(p>thresholds(k)[1]).astype('u1');ps[k].append(p);ss[k].append(refs[k].ram[:])
            if k<2:
                for w,t in mats[1:3] if k==0 else mats[3:4]:z=(z.astype('i4')@w.astype('i4')>t).astype('u1')
        if role!='transfer':last=z.copy()
        xs.append(x);ys.append(z)
    a=dict(events=rows,inputs=np.asarray(xs,'u1'),outputs=np.asarray(ys,'u1'),cpu=np.asarray(cpu,'i4'))
    for k in range(3):a[f'states{k}']=np.asarray(ss[k],'<u4');a[f'potentials{k}']=np.asarray(ps[k],'<i4');a[f'features{k}']=np.asarray(fs[k],'u1')
    return a

def assess(a):
    out={}
    for phase in dict.fromkeys(e['phase'] for e in a['events']):
        ii=[e['index'] for e in a['events'] if e['phase']==phase and e['role'] in ['query','exam']]
        if not ii:continue
        y=a['outputs'][ii];truth=np.asarray([a['events'][i]['actual'] for i in ii]);p=np.stack([y[:,88],y[:,PR].any(1)],1);coh=y[:,[86,87,89,90]].reshape(-1,2,2)
        correct=np.asarray([bool(v[c] and not v[1-c]) for v,c in zip(p,truth)])
        counts=np.stack([(a['potentials0'][ii][:,g]>31).sum(1) for g in GROUPS],1)
        out[phase]=dict(queries=len(ii),correct=int(correct.sum()),ambiguous=int(p.all(1).sum()),missing=int((~p.any(1)).sum()),changed_correct=sum(int(v) for v,i in zip(correct,ii) if a['events'][i]['key']<2),changed_queries=sum(a['events'][i]['key']<2 for i in ii),stable_correct=sum(int(v) for v,i in zip(correct,ii) if a['events'][i]['key'] in [2,3]),stable_queries=sum(a['events'][i]['key'] in [2,3] for i in ii),source_agreement=int(((coh[:,0]==coh[:,1]).all(1)&(coh[:,0].sum(1)==1)).sum()),source_disagreement=int(((coh.sum(2)==1).all(1)&(coh[:,0]!=coh[:,1]).any(1)).sum()),source_silent=int((~coh.any((1,2))).sum()),max_records_per_source_class=int(counts.max(initial=0)),cpu_agreement_correct=int((a['cpu'][ii,0]==truth).sum()),cpu_follow_b_correct=int((a['cpu'][ii,1]==truth).sum()))
    return out

def predicate_screen():
    w,t=matrices('native')[1];rows=[]
    # Explicit scope: zero or one responding record per class. Trace admission
    # rejects any observed violation; no robustness claim for duplicate matches.
    for tc in itertools.product(range(2),repeat=4):
        for prev in itertools.product(range(2),repeat=2):
            for teach,block in itertools.product(range(2),repeat=2):
                v=np.zeros(128,'i4')
                for c,g in enumerate(GROUPS):v[g[0]]=tc[c]
                v[HT[13]]=teach;v[HT[14]]=prev[0];v[HT[15]]=prev[1];v[HT[16]]=teach*block
                got=(v@w>t)[16:20];want=np.zeros(4,bool)
                if teach and sum(tc[:2])==sum(tc[2:])==1 and tc[:2]==tc[2:]:
                    val=tc[:2].index(1)
                    if not prev[val]:want[2*val]=1
                    elif all(prev):want[2*val+1]=1
                assert np.array_equal(got,want),(tc,prev,teach,got,want)
                rows.append(dict(source=tc,student=prev,transfer=teach,post_prefix=block,error=got.tolist()))
    return rows

if __name__=='__main__':
    import json
    predicate_screen()
    for seed in SEEDS:
        for arm in ARMS:
            a=simulate([packed(weights(seed,k),k) for k in range(3)],'repeat',arm)
            print(json.dumps(dict(seed=seed,arm=arm,events=len(a['events']),assessment=assess(a))),flush=True)
