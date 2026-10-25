"""Finite staged admission and raw-derived audit of source agreement memory."""
from pathlib import Path
import datetime as dt,json,subprocess,sys
import numpy as np
from admission.consensus import *
from admission.run import now,save,sha,masks
P=Path(__file__).resolve().parents[1];S=P/'studies/20261025-consensus-memory';F=S/'fixtures02'
def fixture(seed,arm):return F/f's{seed}_{arm}'
def passed(mode,seed,arm,stage):return (S/'raw'/f'{mode}_s{seed}_{arm}_{stage}'/'AUDIT.json').exists()

def check(r,get,fg):
    seed,arm,stage=[r[k] for k in ['seed','arm','stage']]
    initial=[fg(f'bank{k}.ram') for k in range(3)];a=simulate(initial,stage,arm);rows=a['events'];n=len(rows)
    x=get('inputs').reshape(n,N);y=get('scores').reshape(n,N)
    assert np.array_equal(x,a['inputs']),'raw cue/role and prior native student response'
    assert np.array_equal(y,a['outputs']),'native class flags and student output'
    edits=[];states=[];blocked=[];stable=[]
    prefix=next(e['index'] for e in rows if e['block'])
    for k,(u,classes,plastic,helpers) in enumerate(DIMENSIONS):
        state=get(f'states{k}').reshape(n,-1);states.append(state)
        assert np.array_equal(state,a[f'states{k}']),'complete native RAM trajectory'
        assert np.array_equal(get(f'potentials{k}').reshape(n,u),a[f'potentials{k}']),'derived packed potentials'
        assert np.array_equal(masks(initial[k],D,u).T,weights(seed,k)),'own initialized weights'
        before=initial[k];old=masks(before,D,u);replacements=0;changed=[];stride=4+D//32
        for e in rows:
            i=e['index'];v=state[i];mm=masks(v,D,u)
            assert np.all(mm.sum(-1)==32) and not mm[plastic,110:].any(),'bounded native plastic support'
            for j in helpers:assert np.array_equal(v[2+j*stride:2+(j+1)*stride],initial[k][2+j*stride:2+(j+1)*stride]),'fixed helper'
            if e['labels'][k]<0 or not a[f'features{k}'][i,:25].any():assert np.array_equal(v,before),'disabled/empty/query bank changed'
            if k==0 and e['labels'][0]>=0:
                target=e['wire'];foreign=[j for j in range(u) if j//32!=target]
                for j in foreign:assert np.array_equal(v[2+j*stride:2+(j+1)*stride],before[2+j*stride:2+(j+1)*stride]),'source class-partition interference'
            delta=int(((mm==1)&(old==0)).sum());replacements+=delta
            if delta:changed.append(i)
            before=v;old=mm
        edits.append(dict(bank=k,connection_replacements=replacements,changed_events=changed))
        if k and arm in ['students_frozen','teacher_cut']:assert np.all(state[prefix:]==state[prefix-1]),'trained control must remain unchanged'
        if k and arm=='native':
            for phase in dict.fromkeys(e['phase'] for e in rows):
                if not phase.startswith(('blocked_transfer_','pending_transfer_')):continue
                ii=[e['index'] for e in rows if e['phase']==phase]
                same=bool(np.all(state[ii]==state[ii[0]-1]));assert same,'native disagreement must protect complete student RAM'
                blocked.append(dict(bank=k,phase=phase,events=len(ii),unchanged=same))
        if stage!='pressure':
            for key in [2,3]:
                at=next(e['index'] for e in rows if e['phase']==('bootstrap_source_probe' if k==0 else 'bootstrap_exam') and e['key']==key)
                pools=[GROUPS[OUTCOME[key]],GROUPS[2+OUTCOME[key]]] if k==0 else [plastic] if k==(1 if OUTCOME[key]==0 else 2) else []
                for pool in pools:
                    ids=[j for j in pool if a[f'potentials{k}'][at,j]>31];assert len(ids)==1,'one original stable record'
                    j=ids[0];sl=slice(2+j*stride,2+(j+1)*stride);same=bool(np.all(state[prefix:,sl]==state[prefix-1,sl]));assert same,'unrelated original payload changed'
                    stable.append(dict(bank=k,key=key,record=j,words=stride,unchanged=same))
    result=assess(dict(a,outputs=y));main=[v for p,v in result.items() if p.startswith(('protected_exam_','updated_exam_'))]
    qi=[e['index'] for e in rows if e['role'] in ['query','transfer']]
    bound=all(np.all((a['potentials0'][qi][:,g]>31).sum(1)<=1) for g in GROUPS)
    assert bound,'duplicate source match outside predicate qualification'
    initial_ok=result['bootstrap_exam']['correct']==48
    useful=initial_ok and all(v['correct']==v['queries'] and v['ambiguous']==0 for v in main)
    if stage=='pressure':useful=useful and result['new_exam']['correct']==96
    if stage=='correlated':useful=initial_ok # corruption is an expected limitation, reported separately
    for e in rows:
        i=e['index']
        if e['role'] in ['query','exam']:assert np.array_equal(y[i,32:64],x[i,32:64]),'native raw-key relay'
        if e['role']=='transfer':assert x[i,65:69].sum()==0 and e['wire']==0,'constant class and no external outcome'
        if e['role']=='exam':assert not a['potentials0'][i,PT].any() and not y[i,[84,85,86,87,89,90]].any(),'teacher-silent exam'
    if r['mode']=='native':
        assert r['exit_code']==0 and r['device_free']
        for k in range(3):assert np.array_equal(get(f'initial{k}'),initial[k])
        for k in range(4):
            for tag in ['initial','final']:assert np.array_equal(get(f'fixed{k}.{tag}'),fg(f'fixed{k}.ram')),'fixed native route'
        modes=get('learning_modes').reshape(n,3);assert modes.tolist()==[[int(v>=0) for v in e['labels']] for e in rows]
        assert np.array_equal(modes[:,1],modes[:,2]),'equal student mode clocks'
        desc=get('descriptors').reshape(n,16);inp=get('input_addresses');addr=get('output_addresses')
        assert np.array_equal(desc[:,1],inp) and np.array_equal(desc[:,7],addr)
        previous=None
        for i,e in enumerate(rows):
            assert ((int(desc[i,8])>>16)&1023)==e['wire']+1,'external source class or zero transfer class'
            assert np.array_equal(get(f'packet{i}').view('u1')[32:],y[i]),'captured output packet'
            if previous is None:expected=np.zeros(N,'u1')
            else:assert inp[i]==addr[previous]+32,'direct card-SRAM carry pointer';expected=y[previous].copy()
            if e['role']!='transfer':expected[32:64]=x[i,32:64]
            expected[64:72]=0;expected[65 if e['role'] in ['ground','ground_negative'] else 69 if e['role']=='transfer' else 64]=1
            expected[68]=int(e['role']=='ground_negative');expected[70]=int(e['role']=='exam');expected[71]=e['block']
            assert np.array_equal(x[i],expected),'host restricted to raw cue and fixed phase'
            if e['role']!='transfer':previous=i
        report=r.get('stdout_report') or json.loads((Path(r['directory'])/'stdout').read_text())
        assert report['program_calls']==1 and report['weight_restore_calls']==report['host_teacher_decodes']==report['host_student_selections']==0
    return dict(status='SAVED_ARRAY_PASS',mode=r['mode'],seed=seed,arm=arm,stage=stage,events=n,assessment=result,edits=edits,stable_records=stable,blocked_student_ram=blocked,useful_gate=bool(useful),source_partition_isolation=True,source_unique_response_bound=True)

def audit(d):
    d=Path(d);r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0,(d/'stderr').read_text()
    for k,v in r['hashes'].items():assert sha(d/k)==v,('receipt hash',k)
    r['directory']=str(d);f=fixture(r['seed'],r['arm'])
    def get(k):return np.fromfile(d/('result.'+k),'u1' if k in ['inputs','scores'] else '<i4' if k.startswith('potentials') or k=='learning_modes' else '<u4')
    a=check(r,get,lambda k:np.fromfile(f/k,'<u4'));save(d/'AUDIT.json',a);return a

def run(mode,seed,arm,stage):
    assert mode in ['cpu','native'] and seed in SEEDS and arm in ARMS and stage in STAGES
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text());cut=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert not state['closed'] and not state.get('failure') and not state.get('scientific_stop') and now()<cut
    f=fixture(seed,arm);assert json.loads((f/'BUILD.json').read_text())['status']=='VIRTUAL_COMPILER_PASS'
    if mode=='native':
        assert passed('cpu',seed,arm,stage)
        if stage=='gate':
            assert seed==SEEDS[0]
            if arm!='native':assert passed('native',seed,'native','gate')
        elif stage=='repeat':
            assert all(passed('native',SEEDS[0],q,'gate') for q in ARMS)
            if seed!=SEEDS[0] or arm!='native':assert passed('native',SEEDS[0],'native','repeat')
            if arm!='native':assert passed('native',seed,'native','repeat')
        else:
            assert arm=='native' and all(passed('native',s,q,'repeat') for s in SEEDS for q in ARMS)
    physical=[json.loads(p.read_text()) for p in S.glob('raw*/native_*/OWNER.json')];events=len(schedule(stage,arm))
    assert len(physical)+int(mode=='native')<=auth['maximum_physical_sessions'] and sum(p['reserved_events'] for p in physical)+(events if mode=='native' else 0)<=auth['maximum_forwards']
    d=S/'raw'/f'{mode}_s{seed}_{arm}_{stage}';assert not d.exists(),'no automatic retry';d.mkdir(parents=True)
    files=[*f.glob('*'),*S.glob('*.md'),S/'AUTHORITY.json',P/'build/native_consensus',*[P/'admission'/n for n in ['consensus.py','consensus_run.py','prepare_consensus.py','consensus_bench.cpp','build_consensus.sh','revision.py','transfer.py','prepare_transfer.py','recycle.py','error.py','prepare.py','prepare_recycle.py','model.py','run.py']],*[P/n for n in ['pong/relearn/model.py','src/readout_bench.cpp','src/mmio_driver.hpp','tools/prepare_reduction.py','tools/program_codec.py','tools/prepare_network.py','tools/prepare.py','metal/prepare.py']]]
    save(d/'FROZEN.json',dict(utc=now().isoformat(),hashes={str(q.relative_to(P)):sha(q) for q in files if q.is_file()}));r=dict(mode=mode,seed=seed,arm=arm,stage=stage,start_utc=now().isoformat(),reserved_events=events,freeze_sha256=sha(d/'FROZEN.json'));save(d/'OWNER.json',r)
    def execute():
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            proc=subprocess.Popen(['taskset','-c','2','build/native_consensus',mode,str(f),arm,stage,str(d/'result')],cwd=P,stdout=out,stderr=err);r['pid']=proc.pid;save(d/'OWNER.json',r)
            try:code=proc.wait(timeout=min(90,max(1,(cut-now()).total_seconds())))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:code=proc.wait(timeout=2)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
        r.update(exit_code=code,finish_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,hashes={p.name:sha(p) for p in d.iterdir() if p.is_file()});save(d/'RECEIPT.json',r)
        try:a=audit(d)
        except Exception:state['failure']=d.name;save(S/'STATE.json',state);raise
        state['completed'].append(d.name)
        if mode=='native' and arm=='native' and not a['useful_gate']:state['scientific_stop']=d.name
        save(S/'STATE.json',state);return a
    if mode=='native':
        import fcntl
        with open('/tmp/akd1500-research.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1;a=execute()
    else:a=execute()
    print(json.dumps(a),flush=True);return a
if __name__=='__main__':
    if sys.argv[1]=='audit':print(json.dumps(audit(S/'raw'/sys.argv[2])))
    else:run(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4])
