"""Finite admission, captured-source lineage and independent transfer audit."""
from pathlib import Path
import datetime as dt,json,subprocess,sys
import numpy as np
from admission.transfer import *
from admission.run import now,save,sha,masks
P=Path(__file__).resolve().parents[1];S=P/'studies/20261025-native-transfer';F=S/'fixtures05'
def fixture(seed,arm):return F/f's{seed}_{arm}'
def passed(mode,seed,arm,stage):return (S/'raw02'/f'{mode}_s{seed}_{arm}_{stage}'/'AUDIT.json').exists()

def check(r,get,fg):
    seed,arm,stage=[r[k] for k in ['seed','arm','stage']];initial=[fg(f'bank{k}.ram') for k in range(3)];a=simulate(initial,stage,arm);rows=a['events'];n=len(rows)
    x=get('inputs').reshape(n,N);y=get('scores').reshape(n,N)
    assert np.array_equal(x,a['inputs']),'raw role/cue and direct native prior student prediction'
    assert np.array_equal(y,a['outputs']),'all teacher/student responses and returned context'
    edits=[]
    for k,(u,classes,plastic,helpers) in enumerate(DIMENSIONS):
        state=get(f'states{k}').reshape(n,-1);assert np.array_equal(state,a[f'states{k}']),'each complete native RAM boundary'
        assert np.array_equal(get(f'potentials{k}').reshape(n,u),a[f'potentials{k}']),'derived packed potentials'
        assert np.array_equal(masks(initial[k],D,u).T,weights(seed,k)),'own initialized weights'
        before=initial[k];old=masks(before,D,u);changes=[];replacements=0
        for e in rows:
            i=e['index'];v=state[i];mm=masks(v,D,u);assert np.all(mm.sum(-1)==32)
            assert not mm[plastic,110:].any(),'plastic support cannot include query-only carrier fields'
            for j in helpers:assert np.array_equal(v[2+j*(4+D//32):2+(j+1)*(4+D//32)],initial[k][2+j*(4+D//32):2+(j+1)*(4+D//32)]),'fixed helper changed'
            if e['labels'][k]<0:assert np.array_equal(v,before),'disabled bank/query changed RAM'
            if not a[f'features{k}'][i,:25].any():assert np.array_equal(v,before),'no admitted feature / foreign route changed RAM'
            if e['phase']=='unknown_transfer' and k>0:assert np.array_equal(v,before),'unknown source taught student'
            delta=int(((mm==1)&(old==0)).sum());replacements+=delta
            if delta:changes.append(i)
            before=v;old=mm
        edits.append(dict(bank=k,connection_replacements=replacements,changed_events=changes))
    result=assess(dict(a,outputs=y));assert result['source_probe']['teacher_correct']==16,'qualified initial source knowledge'
    for phase in ['silent_exam','silent_final_exam']:
        if phase not in result:continue
        q=result[phase];assert q['teacher_silent']==q['queries'],'teacher answer must be absent during exams'
        if arm=='native':assert q['student_correct']==q['queries'] and q['ambiguous']==0 and q['missing']==0,'native transfer exam gate'
        else:assert q['student_correct']==0,'cut/disabled students must not pass'
    for e in rows:
        i=e['index']
        if e['role'] in ['query','exam']:assert np.array_equal(y[i,32:64],x[i,32:64]),'raw key native relay'
        if e['role']=='transfer':assert x[i,65:69].sum()==0,'no external actual during transfer'
        if e['role']=='exam':assert not a['potentials0'][i,PT].any(),'source plastic comparisons are zero in teacher-silent exam'
    if r['mode']=='native':
        assert r['exit_code']==0 and r['device_free']
        for k in range(3):assert np.array_equal(get(f'initial{k}'),initial[k])
        for k in range(4):
            for tag in ['initial','final']:assert np.array_equal(get(f'fixed{k}.{tag}'),fg(f'fixed{k}.ram')),'fixed native route changed'
        modes=get('learning_modes').reshape(n,3);assert modes.tolist()==[[int(v>=0) for v in e['labels']] for e in rows]
        assert np.array_equal(modes[:,1],modes[:,2]),'student clocks must be identical'
        desc=get('descriptors').reshape(n,16);inp=get('input_addresses');addr=get('output_addresses');assert np.array_equal(desc[:,1],inp) and np.array_equal(desc[:,7],addr)
        previous=None
        for i,e in enumerate(rows):
            assert ((int(desc[i,8])>>16)&1023)==e['wire']+1,'descriptor raw ground label / constant zero transfer label'
            assert np.array_equal(get(f'packet{i}').view('u1')[32:],y[i]),'native packet payload'
            if previous is None:expected=np.zeros(N,'u1')
            else:assert inp[i]==addr[previous]+32,'direct native query output pointer';expected=y[previous].copy()
            if e['role']!='transfer':expected[32:64]=x[i,32:64]
            expected[64:72]=0;expected[65 if e['role']=='ground' else 69 if e['role']=='transfer' else 64]=1
            if e['role']=='exam':expected[70]=1
            assert np.array_equal(x[i],expected),'host may only supply raw cue and fixed role'
            if e['role']!='transfer':previous=i
        report=r.get('stdout_report') or json.loads((Path(r['directory'])/'stdout').read_text())
        assert report['program_calls']==1 and report['weight_restore_calls']==report['host_teacher_decodes']==report['host_student_selections']==0
    return dict(status='SAVED_ARRAY_PASS',mode=r['mode'],seed=seed,arm=arm,stage=stage,events=n,assessment=result,edits=edits)

def audit(d):
    d=Path(d);r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0,(d/'stderr').read_text()
    for k,v in r['hashes'].items():assert sha(d/k)==v,('receipt hash',k)
    r['directory']=str(d);f=fixture(r['seed'],r['arm'])
    def get(k):return np.fromfile(d/('result.'+k),'u1' if k in ['inputs','scores'] else '<i4' if k.startswith('potentials') or k=='learning_modes' else '<u4')
    a=check(r,get,lambda k:np.fromfile(f/k,'<u4'));save(d/'AUDIT.json',a);return a

def run(mode,seed,arm,stage):
    assert mode in ['cpu','native'] and seed in SEEDS and arm in ARMS and stage in STAGES
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text());cut=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert not state['closed'] and not state.get('failure') and now()<cut
    f=fixture(seed,arm);assert json.loads((f/'BUILD.json').read_text())['status']=='VIRTUAL_COMPILER_PASS'
    if mode=='native':
        assert passed('cpu',seed,arm,stage)
        if stage=='gate':
            assert seed==SEEDS[0]
            if arm!='native':assert passed('native',seed,'native','gate')
        else:
            assert all(passed('native',SEEDS[0],q,'gate') for q in ARMS)
            if seed!=SEEDS[0] or arm!='native':assert passed('native',SEEDS[0],'native','extended')
            if arm!='native':assert passed('native',seed,'native','extended')
    physical=[json.loads(p.read_text()) for p in S.glob('raw*/native_*/OWNER.json')];events=len(schedule(stage,arm))
    assert len(physical)+int(mode=='native')<=auth['maximum_physical_sessions'] and sum(p['reserved_events'] for p in physical)+(events if mode=='native' else 0)<=auth['maximum_forwards']
    d=S/'raw02'/f'{mode}_s{seed}_{arm}_{stage}';assert not d.exists(),'no automatic retry';d.mkdir(parents=True)
    files=[*f.glob('*'),*S.glob('*.md'),S/'AUTHORITY.json',P/'build/native_transfer',*[P/'admission'/n for n in ['transfer.py','transfer_run.py','prepare_transfer.py','transfer_bench.cpp','build_transfer.sh','recycle.py','error.py','prepare.py','prepare_recycle.py','model.py','run.py']],*[P/n for n in ['pong/relearn/model.py','src/readout_bench.cpp','src/mmio_driver.hpp','tools/prepare_reduction.py','tools/program_codec.py','tools/prepare_network.py','tools/prepare.py','metal/prepare.py']]]
    save(d/'FROZEN.json',dict(utc=now().isoformat(),hashes={str(q.relative_to(P)):sha(q) for q in files if q.is_file()}));r=dict(mode=mode,seed=seed,arm=arm,stage=stage,start_utc=now().isoformat(),reserved_events=events,freeze_sha256=sha(d/'FROZEN.json'));save(d/'OWNER.json',r)
    def execute():
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            proc=subprocess.Popen(['taskset','-c','2','build/native_transfer',mode,str(f),arm,stage,str(d/'result')],cwd=P,stdout=out,stderr=err);r['pid']=proc.pid;save(d/'OWNER.json',r)
            try:code=proc.wait(timeout=min(90,max(1,(cut-now()).total_seconds())))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:code=proc.wait(timeout=2)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
        r.update(exit_code=code,finish_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,hashes={p.name:sha(p) for p in d.iterdir() if p.is_file()});save(d/'RECEIPT.json',r)
        try:a=audit(d)
        except Exception:state['failure']=d.name;save(S/'STATE.json',state);raise
        state['completed'].append(d.name);save(S/'STATE.json',state);return a
    if mode=='native':
        import fcntl
        with open('/tmp/akd1500-research.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1;a=execute()
    else:a=execute()
    print(json.dumps(a),flush=True);return a
if __name__=='__main__':
    if sys.argv[1]=='audit':print(json.dumps(audit(S/'raw02'/sys.argv[2])))
    else:run(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4])
