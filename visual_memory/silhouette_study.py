"""Sealed image/source stream, exact-rule forecast, and task-level CPU control."""
from pathlib import Path
import argparse,json,time
import numpy as np
from visual_memory import silhouette as m
from visual_memory.data import save,sha

def rows(kind='support',arm='native',seed=27102703):
    rng=np.random.default_rng(seed);pp=m.patterns();truth=pp.copy();rr=[]
    def add(role,phase,x,mode=-1,label=0,**kw):
        r=dict(index=len(rr),role=role,phase=phase,input=x,mode=mode,label=label,source_kind=0,**kw)
        if arm=='manager_frozen' and phase!='bootstrap' and role=='policy_teacher':r['mode']=-1
        if arm=='content_frozen' and phase!='bootstrap' and role in ['content_teacher','presence_teacher']:r['mode']=-1
        rr.append(r)
    def feedback(s,good,phase):
        for _ in range(2):
            add('policy_teacher',phase,m.encode(source=s,aux=False),0,4+good,source=s,retire=False,good=good)
            add('policy_teacher',phase,m.encode(source=s,retire=True,aux=False),0,5-good,source=s,retire=True,good=good)
        add('policy_probe',phase,m.encode(source=s,aux=True),source=s,good=good)
    def teach(i,p,s,phase,initial=False,good=True):
        add('policy_probe',phase,m.encode(source=s,aux=True),source=s,good=int(good))
        for b in range(96):
            for retire in ([False,False] if initial else [False,False,True]):
                x=m.encode(pp[i],source=s,retire=retire,aux=False,opened=arm=='open',cut=arm=='cut' and phase!='bootstrap',value=int(p[b]))
                add('content_teacher',phase,x,1,i*96+b,pattern=i,bit=b,retire=retire,source=s,good=int(good),candidate=p.tolist())
        for _ in range(2):add('presence_teacher',phase,m.encode(pp[i],source=s,aux=False,opened=arm=='open',cut=arm=='cut' and phase!='bootstrap',presence=True),1,384+i,pattern=i,source=s,good=int(good))
        if good:truth[i]=p
    def queries(phase,ids,n=16):
        for i in ids:
            add('query',phase,m.encode(pp[i],query=True),pattern=i,erase=[],expected=truth[i].tolist())
            for _ in range(n):
                mask=sorted(map(int,rng.choice(96,24,False)));add('query',phase,m.encode(pp[i],erase=mask,query=True),pattern=i,erase=mask,expected=truth[i].tolist())
        for _ in range(32 if kind!='support' else 16):
            q=rng.integers(0,2,96,'u1');add('unknown',phase,m.encode(q,query=True),bitmap=q.tolist())
    feedback(0,1,'bootstrap');feedback(1,0,'bootstrap')
    for i in range(4):
        for _ in range(2):add('cue_teacher','bootstrap',m.encode(pp[i],aux=False),0,i,pattern=i)
    for i in range(4 if kind=='support' else 2):teach(i,pp[i],0,'bootstrap',True)
    if kind=='support':queries('support_probe',range(4),16);return rr
    queries('old_probe',range(2),8)
    for i in range(2):teach(i,m.poisoned(i),1,'misleading_old',good=False)
    queries('protected_old',range(2),8)
    for i in [2,3]:teach(i,pp[i],0,'new_images',True)
    queries('four_resident',range(4),16)
    feedback(0,0,'quality_reversal');feedback(1,1,'quality_reversal')
    for i in range(2):teach(i,m.revised()[i],1,'legitimate_revision')
    queries('after_revision',range(4),16)
    for i in [2,3]:teach(i,m.poisoned(i),0,'misleading_new',good=False)
    for i in [0,2]:teach(i,m.poisoned(i),2,'unknown_source',good=False)
    queries('final',range(4),32)
    # Same native int32 buffer may be handed back unchanged on read-only rows.
    for i in range(4):
        add('query','feedback',m.encode(pp[i],query=True),pattern=i,erase=[],expected=truth[i].tolist())
        for _ in range(2):add('feedback','feedback',None,pattern=i,erase=[],expected=truth[i].tolist());rr[-1]['source_kind']=1
    return rr

class TaskCPU:
    """Observed cue exemplars + learned source status + literal pixel dictionary.

    No target, future truth, candidate-cache lookup, or native state input.
    """
    def __init__(self):self.cues={};self.values={};self.sources={}
    def step(self,x,e):
        z=x.reshape(-1,4)[:,0];pix=z[:288:3];known=(pix|z[288:384]).astype(bool);src=int(np.argmax([int(z[k].sum()) for k in m.KEYS]))
        role=e['role'];label=e['label'];ret=bool(z[m.RET] or (role=='policy_teacher' and z[m.SENT]))
        if role=='policy_teacher' and not ret:self.sources[src]=label-4
        if role=='cue_teacher':self.cues[label]=pix.copy()
        distances={i:int(np.count_nonzero(c[known]!=pix[known])) for i,c in self.cues.items()};best=min(distances.values()) if distances else 999
        ids=[i for i,d in distances.items() if d==best]
        i=ids[0] if len(ids)==1 and best<=int(known.sum())//8 else None
        if role=='content_teacher' and not ret and self.sources.get(src,0) and i is not None:
            if i not in self.values:self.values[i]=np.full(96,-1,'i4')
            self.values[i][label%96]=int(z[487])
        return self.values[i].copy() if i is not None and i in self.values and np.all(self.values[i]>=0) else None

def assess(plan,outputs,states=None,task=None):
    phases={};gated=[];native=0;queries=0;unknown=0;rejected=0;full=[];partial=[];policy=[]
    for i,(e,y) in enumerate(zip(plan,outputs)):
        if e['role'] in ['query','feedback']:
            got=y[:288:3];ok=bool(y[472] and np.array_equal(got,e['expected']));p=phases.setdefault(e['phase'],dict(correct=0,queries=0,full_correct=0,full=0,partial_correct=0,partial=0,cpu_correct=0 if task is not None else None));p['correct']+=ok;p['queries']+=1;queries+=1;native+=ok
            field='partial' if e['erase'] else 'full';p[field]+=1;p[field+'_correct']+=ok;(partial if e['erase'] else full).append(ok)
            if task is not None:p['cpu_correct']+=bool(task[i] is not None and np.array_equal(task[i],e['expected']))
        if e['role']=='unknown':unknown+=1;rejected+=not bool(y[472])
        if e['role']=='policy_probe':
            v=y[448:452];allow=bool(v[2:].any() and not v[:2].any());policy.append(dict(index=i,source=e['source'],good=e['good'],allow=allow,correct=allow==bool(e['good'])))
        if e['role'] in ['content_teacher','presence_teacher'] and not e['good'] and states is not None:
            before=m.initial(1) if i==0 else states[i-1];gated.append(bool(np.array_equal(before,states[i])))
    pct=lambda a:100*sum(a)/len(a) if a else None
    r=dict(phases=phases,queries=queries,correct=native,full_recall_pct=pct(full),quarter_erasure_recall_pct=pct(partial),unknown_queries=unknown,unknown_rejected=rejected,unknown_rejection_pct=100*rejected/max(unknown,1),policy_probes=policy,rejected_teaching_rows=len(gated),rejected_teaching_RAM_unchanged=all(gated) if gated else None)
    r['support_gate']=all(full) and pct(partial)>=90 and r['unknown_rejection_pct']>=95 and all(x['correct'] for x in policy)
    if 'final' in phases:
        f=phases['final'];r['primary_quality_gate']=r['support_gate'] and 100*f['correct']/f['queries']>=95 and all(gated)
    return r

def run(out,kind='support',arm='native',seed=27102703,fixture=None):
    out.mkdir(exist_ok=False);plan=rows(kind,arm,seed);rams=[np.fromfile(fixture.parent/'prepared'/f'bank{k}.ram','<u4') for k in range(2)] if fixture else None;c=m.Circuit(rams);cpu=TaskCPU();inputs=[];ys=[];ss=[[],[]];task=[];last=None;start=time.monotonic()
    for e in plan:
        x=e['input'] if not e['source_kind'] else last.astype('<u4').view('u1');inputs.append(x);task.append(cpu.step(x,e));v=c.step(x,e['mode'],e['label']);last=v['output'];ys.append(last)
        for k in range(2):ss[k].append(v['states'][k])
    inputs=np.asarray(inputs,'u1');outputs=np.asarray(ys,'u1');states=[np.asarray(s,'<u4') for s in ss];r=assess(plan,outputs,states[1],task);r.update(status='CPU_FORECAST_ONLY',kind=kind,arm=arm,events=len(plan),elapsed_s=time.monotonic()-start,physical_device_opened=False)
    np.savez_compressed(out/'TRACE.npz',inputs=inputs,outputs=outputs,states0=states[0],states1=states[1],modes=np.array([e['mode'] for e in plan],'<i4'),labels=np.array([e['label'] for e in plan],'<i4'),source_kind=np.array([e['source_kind'] for e in plan],'<i4'))
    clean=[{k:v for k,v in e.items() if k!='input'} for e in plan];save(out/'ROWS.json',clean);save(out/'RESULT.json',r)
    if fixture:
        import shutil
        fixture.mkdir(exist_ok=False)
        for p in Path(fixture.parent/'prepared').iterdir():
            if p.suffix in ['.bin','.ram','.infer','.learn','.threshold','.weights','.step'] or p.name=='fixed_count.txt':shutil.copy2(p,fixture/p.name)
        inputs.tofile(fixture/'inputs.bin')
        for field in ['modes','labels','source_kind']:np.array([e[field[:-1] if field in ['modes','labels'] else field] for e in plan],'<i4').tofile(fixture/(field+'.bin'))
        shutil.copy2(out/'TRACE.npz',fixture/'TRACE.npz');shutil.copy2(out/'ROWS.json',fixture/'ROWS.json');save(fixture/'FIXTURE.json',dict(events=len(plan),kind=kind,arm=arm,seed=seed,hashes={p.name:sha(p) for p in fixture.iterdir() if p.is_file()},physical_device_opened=False))
    return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--kind',default='support',choices=['support','primary','confirmation']);p.add_argument('--arm',default='native',choices=['native','manager_frozen','open','content_frozen','cut']);p.add_argument('--seed',default=27102703,type=int);p.add_argument('--fixture',type=Path);a=p.parse_args();print(json.dumps(run(a.output,a.kind,a.arm,a.seed,a.fixture),indent=2))
