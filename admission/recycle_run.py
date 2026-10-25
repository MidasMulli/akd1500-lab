"""Finite device owner, sequential admissions and independent saved-array audit."""
from pathlib import Path
import datetime as dt,json,subprocess,sys
import numpy as np
from admission.recycle import *
from admission.run import now,save,sha,masks
P=Path(__file__).resolve().parents[1];S=P/'studies/20261025-native-correction';F=S/'fixtures01'
def fixture(seed,arm): return F/f's{seed}_{arm}'
def passed(mode,seed,arm,stage):
    return any(json.loads(p.read_text()).get('status')=='SAVED_ARRAY_PASS' for p in (S/'raw').glob(f'{mode}_s{seed}_{arm}_{stage}*/AUDIT.json'))

def check(r,get,fg):
    seed,arm,stage=r['seed'],r['arm'],r['stage'];initial=fg('bank.ram');a=simulate(initial,stage,arm);rows=a['events'];count=len(rows)
    x=get('inputs').reshape(count,N);y=get('scores').reshape(count,N);state=get('states').reshape(count,len(initial))
    assert np.array_equal(x,a['inputs']),'raw cue, delayed actual, saved native prediction'
    assert np.array_equal(y,a['outputs']),'every output and admission helper'
    assert np.array_equal(state,a['states']),'every native learning RAM boundary'
    assert np.array_equal(get('potentials').reshape(count,U),a['potentials']),'derived bit-set potentials'
    assert np.array_equal(masks(initial,D,U).T,weights(seed)),'own initialized connections'
    before=initial;previous=masks(initial,D,U);replacements=0;changes=[];blocked=admitted=0
    for e in rows:
        i=e['index'];v=state[i];mm=masks(v,D,U);stride=4+D//32
        assert np.all(mm.sum(-1)==32),'all neurons keep 32 connections'
        for j in HELPERS: assert np.array_equal(v[2+j*stride:2+(j+1)*stride],initial[2+j*stride:2+(j+1)*stride]),'fixed helper changed'
        if e['label']<0: assert np.array_equal(v,before),'query or disabled bank changed RAM'
        if e['teacher']:
            if a['features'][i].any(): admitted+=1
            else:
                blocked+=1;assert np.array_equal(v,before),'refused teacher changed memory'
        delta=int(((mm==1)&(previous==0)).sum());replacements+=delta
        if delta:changes.append(i)
        assert bool(y[i,80])==bool(a['features'][i,:25].all()),'native admitted-key helper'
        before=v;previous=mm
    if stage=='truth':
        for e in rows:
            want=not e['teacher'] or arm=='open' or arm!='closed' and (not e['p0' if e['actual']==0 else 'p1'] or e['p0'] and e['p1'])
            assert bool(y[e['index'],80])==bool(want),'all comparator truth cases'
    summary=assess(dict(a,outputs=y))
    if stage!='truth':
        if arm in ['native','no_retire','frozen']:assert summary['old_probe']['correct']==16
        if arm=='native':
            assert all(v['correct']==48 and v['ambiguous']==0 for k,v in summary.items() if k.startswith('probe_'))
            start=next(e['index'] for e in rows if e['phase']=='old_probe')
            old=masks(state[start],D,U);mm=masks(state,D,U)
            for key in [2,3]:
                i=next(e['index'] for e in rows if e['phase']=='old_probe' and e['key']==key)
                ids=np.flatnonzero(y[i,72:80]);assert len(ids)==1;j=PLASTIC[int(ids[0])]
                assert np.all(mm[start:,j]==old[j]),'unrelated learned association must stay byte-exact'
    if r['mode']=='native':
        assert r['exit_code']==0 and r['device_free'];assert np.array_equal(get('initial'),initial)
        for k in range(4):
            for tag in ['initial','final']:assert np.array_equal(get(f'fixed{k}.{tag}'),fg(f'fixed{k}.ram')),'fixed comparator/group RAM'
        assert get('learning_modes').tolist()==[int(e['label']>=0) for e in rows]
        desc=get('descriptors').reshape(count,16);addr=get('output_addresses');inp=get('input_addresses')
        assert np.array_equal(desc[:,1],inp) and np.array_equal(desc[:,7],addr)
        previous_query=None;mutated=False
        for i,e in enumerate(rows):
            assert ((int(desc[i,8])>>16)&1023)==((1-e['actual'] if e['erase'] else e['actual'])+1 if e['teacher'] else 1),'actual label, never error metadata'
            assert inp[i]!=addr[i]+32
            assert np.array_equal(get(f'packet{i}').view('u1')[32:],y[i]),'native packet payload'
            if previous_query is not None:
                assert inp[i]==addr[previous_query]+32,'previous native output pointer directly reused'
                expected=y[previous_query].copy()
                if mutated:expected[84:88]=0
            else: expected=np.zeros(N,'u1')
            if not e['teacher'] or e['override']:expected[32:64]=x[i,32:64]
            expected[64:72]=0;expected[64+int(e['teacher'])]=1;expected[68]=e['erase']
            if e['teacher']:expected[66+e['actual']]=1
            if e['override']:expected[84:88]=[e['p0'],e['p1'],0,0]
            if e['cut']:expected[84:88]=0;mutated=True
            assert np.array_equal(x[i],expected),'no host reconstructed correctness or permission'
            if not e['teacher']:previous_query=i;mutated=False
        report=json.loads((Path(r['directory'])/'stdout').read_text())
        assert report['synthetic_prediction_writes']==(count if stage=='truth' else 0)
        assert report['host_error_decisions']==report['weight_restore_calls']==0
    return dict(status='SAVED_ARRAY_PASS',mode=r['mode'],seed=seed,arm=arm,stage=stage,events=count,
        assessment=summary,connection_replacements=replacements,changed_events=changes,admitted_teachers=admitted,blocked_teachers=blocked,
        useful_gate=all(v['correct']==48 and v['ambiguous']==0 for k,v in summary.items() if k.startswith('probe_')) if stage!='truth' else None)

def audit(d):
    d=Path(d);r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0,(d/'stderr').read_text()
    for name,h in r['hashes'].items():assert sha(d/name)==h,('receipt hash',name)
    r['directory']=str(d);f=fixture(r['seed'],r['arm'])
    def get(k):return np.fromfile(d/('result.'+k),'u1' if k in ['inputs','scores'] else '<i4' if k in ['potentials','learning_modes'] else '<u4')
    q=check(r,get,lambda k:np.fromfile(f/k,'<u4'));save(d/'AUDIT.json',q);return q

def run(mode,seed,arm,stage):
    assert mode in ['cpu','native'] and seed in SEEDS and arm in ARMS and stage in STAGES
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text());cut=dt.datetime.fromisoformat(auth['scientific_cutoff_utc'])
    assert not state['closed'] and not state.get('failure') and now()<cut,'closed/failure/deadline'
    f=fixture(seed,arm);assert json.loads((f/'BUILD.json').read_text())['status']=='VIRTUAL_COMPILER_PASS'
    if mode=='native':
        assert passed('cpu',seed,arm,stage),'independent CPU gate first'
        if stage=='truth':
            assert seed==SEEDS[0] and arm in ['native','open','closed']
            if arm!='native':assert passed('native',seed,'native','truth')
        else:
            assert all(passed('native',SEEDS[0],a,'truth') for a in ['native','open','closed'])
            if stage=='correction':
                assert seed==SEEDS[0] and arm in ['native','no_retire','frozen','cut']
                if arm!='native':assert passed('native',seed,'native',stage)
            else:
                assert stage=='repeat' and arm in ['native','no_retire','frozen']
                assert all(passed('native',SEEDS[0],q,'correction') for q in ['native','no_retire','frozen','cut'])
                if seed!=SEEDS[0] or arm!='native':assert passed('native',SEEDS[0],'native','repeat')
                if seed!=SEEDS[0]:assert all(passed('native',SEEDS[0],q,'repeat') for q in ['native','no_retire','frozen'])
    physical=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json') if json.loads(p.read_text())['mode']=='native'];events=len(schedule(stage,arm))
    assert len(physical)+int(mode=='native')<=auth['maximum_physical_sessions']
    assert sum(r['reserved_events'] for r in physical)+(events if mode=='native' else 0)<=auth['maximum_forwards']
    name=f'{mode}_s{seed}_{arm}_{stage}';d=S/'raw'/name;assert not d.exists(),'no automatic retry'
    d.mkdir(parents=True,exist_ok=False)
    files=[P/'admission/error.py',P/'admission/recycle.py',P/'admission/prepare_recycle.py',P/'admission/recycle_bench.cpp',P/'admission/recycle_run.py',P/'admission/build_recycle.sh',P/'admission/prepare.py',P/'admission/model.py',P/'admission/run.py',P/'pong/relearn/model.py',P/'src/readout_bench.cpp',P/'src/mmio_driver.hpp',P/'tools/prepare_reduction.py',P/'tools/program_codec.py',P/'tools/prepare.py',P/'tools/prepare_network.py',P/'metal/prepare.py',P/'build/recycle_admission',*f.glob('*'),*S.glob('*.md'),S/'AUTHORITY.json']
    save(d/'FROZEN.json',dict(utc=now().isoformat(),hashes={str(q.relative_to(P)):sha(q) for q in files if q.is_file()}))
    r=dict(mode=mode,seed=seed,arm=arm,stage=stage,start_utc=now().isoformat(),reserved_events=events,freeze_sha256=sha(d/'FROZEN.json'));save(d/'OWNER.json',r)
    def execute():
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            proc=subprocess.Popen(['taskset','-c','2','build/recycle_admission',mode,str(f),arm,stage,str(d/'result')],cwd=P,stdout=out,stderr=err);r['pid']=proc.pid;save(d/'OWNER.json',r)
            try:code=proc.wait(timeout=min(60,max(1,(cut-now()).total_seconds())))
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
    if sys.argv[1]=='audit':print(json.dumps(audit(S/'raw'/sys.argv[2])))
    else:run(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4])
