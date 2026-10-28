"""Finite source-frozen learned-policy admissions and raw-derived numeric audit."""
from pathlib import Path
import datetime as dt, fcntl, json, subprocess, sys
import numpy as np
from admission import learned_policy as model
from admission.run import now, save, sha, masks

P=Path(__file__).resolve().parents[1]; S=P/'studies/20261025-learned-policy'; F=S/'fixtures01'; RAW=S/'raw02'
CELLS=[(model.SEEDS[0],'native','support',0)]
CELLS += [(model.SEEDS[0],arm,'pilot',0) for arm in model.ARMS]
CELLS += [(seed,arm,'sealed',1) for seed in model.SEEDS for arm in ['native','manager_frozen','open']]
CELLS += [(seed,'native','sealed',0) for seed in model.SEEDS[1:]]


def name(mode,seed,arm,stage,orientation):return f'{mode}_s{seed}_{arm}_{stage}_o{orientation}'
def fixture(seed,arm):return F/f's{seed}_{arm}'
def dtype(key):return 'u1' if key in ['inputs','scores'] else '<i4' if key.startswith('potentials') or key=='learning_modes' else '<u4'


def check(r,get,fg):
    seed,arm,stage,orientation=[r[k] for k in ['seed','arm','stage','orientation']]
    initial=[fg(f'bank{k}.ram') for k in range(3)]
    a=model.simulate(initial,stage,arm,orientation); rows=a['events']; n=len(rows)
    x=get('inputs').reshape(n,model.N); y=get('scores').reshape(n,model.N)
    assert np.array_equal(x,a['inputs']),'raw input / native carried responses'
    assert np.array_equal(y,a['outputs']),'returned decisions and recall'
    edits=[]; blocked=[]; prefix=next((e['index'] for e in rows if e['block']),n)
    for k,(u,classes,plastic,helpers) in enumerate(model.DIMENSIONS):
        state=get(f'states{k}').reshape(n,-1)
        assert np.array_equal(state,a[f'states{k}']),'all visible native SRAM boundaries'
        assert np.array_equal(get(f'potentials{k}').reshape(n,u),a[f'potentials{k}']),'derived potentials'
        assert np.array_equal(masks(initial[k],model.D,u).T,model.weights(seed,k)),'initialized connection mask'
        before=initial[k]; old=masks(before,model.D,u); changes=[]; additions=removals=0; stride=4+model.D//32
        for e in rows:
            i=e['index']; after=state[i]; mm=masks(after,model.D,u)
            assert np.all(mm.sum(-1)==32),'fixed connection budget'
            for j in helpers:
                sl=slice(2+j*stride,2+(j+1)*stride)
                assert np.array_equal(after[sl],initial[k][sl]),'fixed helper record changed'
            if e['labels'][k]<0 or not a[f'features{k}'][i,:25].any():
                assert np.array_equal(after,before),'query / disabled / empty bank changed'
            if k==0 and e['role'].startswith('feedback'):
                assert a['features0'][i].sum()==32,'feedback field contamination'
                target=e['wire']
                for j in range(u):
                    if j//(u//classes)!=target:
                        sl=slice(2+j*stride,2+(j+1)*stride)
                        assert np.array_equal(after[sl],before[sl]),'foreign manager class edited'
            add=int(((mm==1)&(old==0)).sum()); remove=int(((mm==0)&(old==1)).sum())
            assert add==remove,'unbalanced connection replacement'
            additions+=add;removals+=remove
            if add:changes.append(dict(event=i,added=add,removed=remove))
            if k and arm=='native' and e['role']=='transfer' and e['phase'].startswith(('bad_proposals','unknown_source')):
                assert np.array_equal(after,before),'rejected proposal changed student RAM'
                blocked.append(dict(bank=k,event=i,unchanged=True))
            before=after;old=mm
        if arm=='manager_frozen' and k==0 and prefix<n:assert np.all(state[prefix:]==state[prefix-1]),'manager control changed'
        if arm in ['students_frozen','cut'] and k and prefix<n:assert np.all(state[prefix:]==state[prefix-1]),'student control changed'
        edits.append(dict(bank=k,connection_additions=additions,connection_removals=removals,changed_events=changes))
    # Qualification of the fixed predicate is explicitly zero/one response per class.
    qi=[e['index'] for e in rows if e['role'] in ['query','transfer']]
    for group in [model.PT[:8],model.PT[8:]]:
        assert np.all((a['potentials0'][qi][:,group]>31).sum(1)<=1),'manager duplicate match outside predicate support'
    for e in rows:
        i=e['index']
        if e['role']=='transfer':assert e['wire']==0 and not x[i,65:69].any(),'no correctness outcome during transfer'
        if e['role']=='exam':assert not a['potentials0'][i,model.PT].any() and not y[i,84:86].any(),'manager-silent memory exam'
    assessment=model.assess(dict(a,outputs=y)); useful=True
    for phase,v in assessment.items():
        if 'exam' in phase:useful &= v['memory_correct']==v['queries'] and not v['ambiguous']
        if 'policy' in phase:useful &= v['policy_correct']==v['queries']
    if r['mode']=='native':
        assert r['exit_code']==0 and r['device_free'],'native cleanup'
        for k in range(3):assert np.array_equal(get(f'initial{k}'),initial[k])
        for k in range(4):
            for tag in ['initial','final']:assert np.array_equal(get(f'fixed{k}.{tag}'),fg(f'fixed{k}.ram')),'fixed routing RAM'
        modes=get('learning_modes').reshape(n,3)
        assert np.array_equal(modes,[[int(v>=0) for v in e['labels']] for e in rows]),'fixed role clocks'
        desc=get('descriptors').reshape(n,16); inp=get('input_addresses'); out=get('output_addresses')
        assert np.array_equal(desc[:,1],inp) and np.array_equal(desc[:,7],out),'descriptor source/destination'
        previous_query=None
        for i,e in enumerate(rows):
            assert inp[i]!=out[i]+32,'source/destination alias'
            assert ((int(desc[i,8])>>16)&1023)==e['wire']+1,'wire class'
            packet=get(f'packet{i}'); assert packet.shape==(40,)
            assert np.array_equal(packet.view('u1')[32:],y[i]),'saved native output packet'
            if previous_query is None:assert not x[i,:32].any()
            else:
                assert inp[i]==out[previous_query]+32,'direct native carry pointer'
                assert np.array_equal(x[i,:32],y[previous_query,:32]),'student response not host-rebuilt'
            if e['role']!='transfer':previous_query=i
    return dict(status='SAVED_ARRAY_PASS',mode=r['mode'],seed=seed,arm=arm,stage=stage,orientation=orientation,
                events=n,assessment=assessment,edits=edits,blocked_student_boundaries=len(blocked),
                useful_gate=bool(useful),manager_class_partition_isolation=True,unique_manager_response_bound=True)


def audit(directory):
    d=Path(directory);r=json.loads((d/'RECEIPT.json').read_text())
    assert r['exit_code']==0,(d/'stderr').read_text()
    for k,v in r['hashes'].items():assert sha(d/k)==v,('raw hash',k)
    freeze=json.loads((d/'FROZEN.json').read_text())
    assert sha(d/'FROZEN.json')==r['freeze_sha256']
    for k,v in freeze['hashes'].items():assert sha(P/k)==v,('acting source/fixture drift',k)
    f=fixture(r['seed'],r['arm'])
    a=check(r,lambda k:np.fromfile(d/('result.'+k),dtype(k)),lambda k:np.fromfile(f/k,'<u4'))
    save(d/'AUDIT.json',a);return a


def run(mode,seed,arm,stage,orientation):
    cell=(seed,arm,stage,orientation); assert cell in CELLS and mode in ['cpu','native']
    auth,state=[json.loads((S/k).read_text()) for k in ['AUTHORITY.json','STATE.json']]
    cutoff=dt.datetime.fromisoformat(auth['scientific_cutoff_utc'])
    assert not state['closed'] and not state.get('failure') and not state.get('scientific_stop') and now()<cutoff
    for k,v in json.loads((S/'SOURCE_FREEZE.json').read_text())['hashes'].items():assert sha(P/k)==v,('source freeze',k)
    f=fixture(seed,arm);assert json.loads((f/'BUILD.json').read_text())['status']=='VIRTUAL_COMPILER_PASS'
    d=RAW/name(mode,*cell);assert not d.exists(),'immutable attempts, no automatic retry'
    if mode=='native':
        cpu=RAW/name('cpu',*cell)/'AUDIT.json';assert cpu.exists()
        control=json.loads(cpu.read_text());assert control['status']=='SAVED_ARRAY_PASS'
        if arm=='native':assert control['useful_gate'],'CPU quality gate blocks physical admission'
        for prev in CELLS[:CELLS.index(cell)]:
            result=RAW/name('native',*prev)/'AUDIT.json';assert result.exists(),prev
            a=json.loads(result.read_text());assert a['status']=='SAVED_ARRAY_PASS'
            if prev[1]=='native':assert a['useful_gate']
    events=len(model.schedule(stage,arm,orientation));owners=[json.loads(p.read_text()) for p in RAW.glob('native_*/OWNER.json')]
    assert len(owners)+int(mode=='native')<=auth['maximum_physical_sessions']
    assert sum(v['reserved_events'] for v in owners)+(events if mode=='native' else 0)<=auth['maximum_forwards']
    d.mkdir(parents=True);files=[*f.glob('*'),S/'AUTHORITY.json',S/'PROTOCOL.md',S/'SOURCE_FREEZE.json']
    files += [P/k for k in json.loads((S/'SOURCE_FREEZE.json').read_text())['hashes']]
    save(d/'FROZEN.json',dict(utc=now().isoformat(),hashes={str(p.relative_to(P)):sha(p) for p in files if p.is_file()}))
    receipt=dict(mode=mode,seed=seed,arm=arm,stage=stage,orientation=orientation,start_utc=now().isoformat(),reserved_events=events,freeze_sha256=sha(d/'FROZEN.json'))
    save(d/'OWNER.json',receipt)
    def execute():
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            proc=subprocess.Popen(['taskset','-c','2','build/native_learned_policy',mode,str(f),arm,f'{stage}_o{orientation}',str(d/'result')],cwd=P,stdout=out,stderr=err)
            receipt['pid']=proc.pid;save(d/'OWNER.json',receipt)
            state['owned_jobs']=[dict(pid=proc.pid,cell=d.name)];save(S/'STATE.json',state)
            try:code=proc.wait(timeout=min(auth['maximum_seconds_per_cell'],max(1,(cutoff-now()).total_seconds())))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:code=proc.wait(timeout=2)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
        state['owned_jobs']=[]
        receipt.update(exit_code=code,finish_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,
                       hashes={p.name:sha(p) for p in d.iterdir() if p.is_file()})
        save(d/'RECEIPT.json',receipt)
        try:result=audit(d)
        except Exception:state['failure']=d.name;save(S/'STATE.json',state);raise
        state['completed'].append(d.name)
        if mode=='native' and arm=='native' and not result['useful_gate']:state['scientific_stop']=d.name
        save(S/'STATE.json',state);return result
    if mode=='native':
        with open('/tmp/akd1500-research.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
            result=execute()
    else:result=execute()
    print(json.dumps({k:v for k,v in result.items() if k!='edits'}),flush=True);return result


if __name__=='__main__':
    if sys.argv[1]=='audit':print(json.dumps(audit(RAW/sys.argv[2])))
    else:run(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4],int(sys.argv[5]))
