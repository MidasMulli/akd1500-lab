"""Revise native student memories from a correcting learned source.

Builds on the qualified transfer circuit. External positive/complement evidence
updates the source. Constant-class native transfer revises both valence banks.
Matched ablations start from the same successfully learned prefix.
"""
import numpy as np
from admission import transfer as base
from admission.transfer import N,D,KEYS,PT,PS,PR,HT,HS,HR,DIMENSIONS,OUTCOME,Reference
SEEDS=[122101,122201,122301]
ARMS=['native','students_frozen','teacher_cut','no_source_retire']
STAGES=['gate','repeat']
ROLES=['query','ground','transfer','exam','ground_negative']
weights_base=base.weights
packed=base.packed

def matrices(arm):
    mats=base.matrices('native');enc,et=mats[0]
    for key in range(KEYS):enc[68,25+key*7+6]=-1
    enc[65,109]=enc[68,109]=1;et[109]=1
    enc[69,384:416]=enc[71,384:416]=1;et[384:416]=1
    if arm=='teacher_cut':mats[1][0][HT[16],16:20]=-1
    return mats

def weights(seed,k):
    w=weights_base(seed,k)
    if k==0:w[:,HT[16]]=0;w[384:416,HT[16]]=1
    return w

thresholds=base.thresholds

def schedule(stage,arm):
    rows=[dict(e,block=0) for e in base.schedule('gate','native')]
    def add(key,role,phase,actual):
        wire=(1-actual if role=='ground_negative' else actual) if role in ['ground','ground_negative'] else 0
        labels=[wire,-1,-1] if role in ['ground','ground_negative'] else [-1,0,0] if role=='transfer' else [-1,-1,-1]
        if arm=='students_frozen':labels[1:]=[-1,-1]
        if arm=='no_source_retire' and role=='ground_negative':labels[0]=-1
        rows.append(dict(index=len(rows),key=key,role=role,phase=phase,labels=labels,wire=wire,actual=actual,block=1))
    def probe(keys,repeat,phase,cycle,exam):
        order=np.tile(keys,repeat);np.random.default_rng(123000+len(rows)).shuffle(order)
        for key in order.tolist():add(key,'exam' if exam else 'query',phase,OUTCOME[key]^(cycle%2 if key<2 else 0))
    for cycle in range(1,2 if stage=='gate' else 13):
        for _ in range(2):
            for key in [0,1]:
                actual=OUTCOME[key]^(cycle%2);add(key,'ground',f'ground_{cycle}',actual);add(key,'ground_negative',f'ground_{cycle}',actual)
        probe(range(4),4,f'source_probe_{cycle}',cycle,False)
        probe(range(4),4,f'before_transfer_{cycle}',cycle,True)
        for _ in range(3):
            for key in range(4):
                actual=OUTCOME[key]^(cycle%2 if key<2 else 0);add(key,'query',f'transfer_{cycle}',actual);add(key,'transfer',f'transfer_{cycle}',actual)
        probe(range(4),24,f'exam_{cycle}',cycle,True)
    return rows

def simulate(rams,stage,arm):
    mats=matrices(arm);refs=[Reference(rams[k],n=D,u=u,nw=32,classes=c) for k,(u,c,_,_) in enumerate(DIMENSIONS)];rows=schedule(stage,arm)
    last=np.zeros(N,'u1');xs=[];ys=[];ss=[[],[],[]];ps=[[],[],[]];fs=[[],[],[]];cpu=[];teacher={};student={}
    for e in rows:
        role=e['role'];x=last.copy()
        if role!='transfer':
            x[32:64]=0
            for bit in range(4):x[32+bit*8+((e['key']>>bit)&1)*4:36+bit*8+((e['key']>>bit)&1)*4]=1
        x[64:72]=0;x[65 if role in ['ground','ground_negative'] else 69 if role=='transfer' else 64]=1
        x[68]=int(role=='ground_negative');x[70]=int(role=='exam');x[71]=e['block']
        if role=='ground':teacher[e['key']]=e['actual']
        if role=='transfer' and e['key'] in teacher:student[e['key']]=teacher[e['key']]
        cpu.append([teacher.get(e['key'],-1),student.get(e['key'],-1)])
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
    out=base.assess(a)
    for phase,v in out.items():
        ii=[e['index'] for e in a['events'] if e['phase']==phase and e['role'] in ['query','exam']]
        y=a['outputs'][ii];p=np.stack([y[:,88],y[:,PR].any(1)],1);t=y[:,84:86]
        for keyname,keep in [('changed',lambda k:k<2),('stable',lambda k:k in [2,3])]:
            jj=[j for j,i in enumerate(ii) if keep(a['events'][i]['key'])]
            v[keyname+'_queries']=len(jj);v[keyname+'_correct']=sum(int(p[j,e['actual']] and not p[j,1-e['actual']]) for j in jj for e in [a['events'][ii[j]]])
        v['teacher_ambiguous']=int(t.all(1).sum());v['teacher_missing']=int((~t.any(1)).sum())
    return out

if __name__=='__main__':
    import json
    for seed in SEEDS:
        for arm in ARMS:
            a=simulate([packed(weights(seed,k),k) for k in range(3)],'repeat',arm);r=assess(a);ex=[v for p,v in r.items() if p.startswith('exam_')]
            print(json.dumps(dict(seed=seed,arm=arm,events=len(a['events']),initial=r['silent_exam'],exam_correct=sum(v['student_correct'] for v in ex),exam_queries=sum(v['queries'] for v in ex),source_correct=sum(v['teacher_correct'] for p,v in r.items() if p.startswith('source_probe_')),source_queries=192,assessment=r)),flush=True)
