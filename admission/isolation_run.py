"""Finite two-bank study: shared-clock admissions and saved-array verification."""
from pathlib import Path
import datetime as dt,json,subprocess,sys
import numpy as np
from admission.isolation import *
from admission.run import now,save,sha,masks
P=Path(__file__).resolve().parents[1];S=P/'studies/20261025-isolated-correction';F=S/'fixtures01'
def fixture(seed,arm):return F/f's{seed}_{arm}'
def passed(mode,seed,arm,stage,stream=117000):return (S/'raw'/f'{mode}_s{seed}_{arm}_{stage}_w{stream}'/'AUDIT.json').exists()

def check(r,get,fg):
    seed,arm,stage,stream=[r[k] for k in ['seed','arm','stage','stream']];initial=[fg(f'bank{k}.ram') for k in range(2)];a=simulate(initial,stage,arm,stream);rows=a['events'];count=len(rows)
    x=get('inputs').reshape(count,N);y=get('scores').reshape(count,N)
    assert np.array_equal(x,a['inputs']),'raw cue/context, delayed actual and native saved prediction'
    assert np.array_equal(y,a['outputs']),'every native response and admission/context relay'
    states=[];edits=[];stable=[]
    for k,(n,u,plastic,helpers) in enumerate(DIMENSIONS):
        state=get(f'states{k}').reshape(count,len(initial[k]));states.append(state)
        assert np.array_equal(state,a[f'states{k}']),'every native learning-RAM boundary'
        assert np.array_equal(get(f'potentials{k}').reshape(count,u),a[f'potentials{k}']),'derived packed potentials'
        assert np.array_equal(masks(initial[k],n,u).T,weights(seed,k)),'own initialized weights'
        before=initial[k];old=masks(before,n,u);changes=[];swaps=foreign=refused=0
        for e in rows:
            i=e['index'];v=state[i];mm=masks(v,n,u);assert np.all(mm.sum(-1)==32),'32 connections per record'
            assert not mm[plastic,110:].any(),'plastic connections cannot enter foreign routing support'
            for j in helpers:assert np.array_equal(v[2+j*36:2+(j+1)*36],initial[k][2+j*36:2+(j+1)*36]),'helper record changed'
            if e['labels'][k]<0:assert np.array_equal(v,before),'query/disabled bank changed RAM'
            if e['context']!=k:
                foreign+=1;assert np.array_equal(v,before),'foreign context modified bank'
            if e['teacher'] and not a[f'features{k}'][i,:25].any():
                refused+=1;assert np.array_equal(v,before),'refused/foreign teacher modified bank'
            delta=int(((mm==1)&(old==0)).sum());swaps+=delta
            if delta:changes.append(i)
            before=v;old=mm
        edits.append(dict(bank=k,connection_replacements=swaps,changed_events=changes,foreign_boundaries_unchanged=foreign,refused_teachers_unchanged=refused))
    if stage=='truth':
        for e in rows:
            expected=not e['teacher'] or arm=='open' or arm!='closed' and (not e['p0' if e['actual']==0 else 'p1'] or e['p0'] and e['p1'])
            assert bool(y[e['index'],80])==bool(expected),'context/phase/error truth table'
    summary=assess(dict(a,outputs=y))
    if stage!='truth' and arm in ['native','frozen','freeze_A','freeze_B','solo_A','solo_B']:
        assert summary['old_probe']['correct']==summary['old_probe']['queries'],'bootstrap context separation'
    if stage in ['gate','repeat'] and arm in ['native','solo_A','solo_B']:
        assert all(v['correct']==v['queries'] and v['ambiguous']==0 for phase,v in summary.items() if phase.startswith('probe_')),'native correction qualification'
        start=next(e['index'] for e in rows if e['phase']=='old_probe')
        for ctx in range(2):
            for key in [2,3]:
                candidates=[e for e in rows if e['phase']=='old_probe' and e['context']==ctx and e['key']==key]
                if not candidates:continue
                i=candidates[0]['index'];sl=slice(0,8) if ctx==0 else slice(72,80);ids=np.flatnonzero(y[i,sl]);assert len(ids)==1
                j=DIMENSIONS[ctx][2][int(ids[0])];b=2+j*36;assert np.all(states[ctx][start:,b:b+36]==states[ctx][start,b:b+36]),'complete unrelated record changed'
                stable.append(dict(context=ctx,key=key,record=j,words=36))
    for e in rows:
        if not e['teacher']:
            expected=np.zeros(4,'u1');expected[e['context']]=1
            assert np.array_equal(y[e['index'],24:28],expected),'raw context must survive native route'
            assert np.array_equal(y[e['index'],32:64],x[e['index'],32:64]),'raw cue must survive native route'
    if r['mode']=='native':
        assert r['exit_code']==0 and r['device_free']
        for k in range(2):assert np.array_equal(get(f'initial{k}'),initial[k])
        for k in range(5):
            for tag in ['initial','final']:assert np.array_equal(get(f'fixed{k}.{tag}'),fg(f'fixed{k}.ram')),'fixed inference RAM changed'
        modes=get('learning_modes').reshape(count,2);assert modes.tolist()==[[int(label>=0) for label in e['labels']] for e in rows]
        if arm in ['native','no_retire','solo_A','solo_B'] and stage!='truth':assert modes.tolist()==[[int(e['teacher'])]*2 for e in rows],'host must enable both banks equally'
        desc=get('descriptors').reshape(count,16);addr=get('output_addresses');inp=get('input_addresses');assert np.array_equal(desc[:,1],inp) and np.array_equal(desc[:,7],addr)
        previous=None
        for i,e in enumerate(rows):
            assert inp[i]!=addr[i]+32
            assert ((int(desc[i,8])>>16)&1023)==((1-e['actual'] if e['erase'] else e['actual'])+1 if e['teacher'] else 1),'raw actual/complement descriptor class'
            assert np.array_equal(get(f'packet{i}').view('u1')[32:],y[i]),'native packet payload'
            if previous is None:expected=np.zeros(N,'u1')
            else:assert inp[i]==addr[previous]+32,'direct native query-output source';expected=y[previous].copy()
            if not e['teacher'] or e['override']:expected[24:28]=x[i,24:28];expected[32:64]=x[i,32:64]
            expected[64:72]=0;expected[64+int(e['teacher'])]=1;expected[68]=e['erase']
            if e['teacher']:expected[66+e['actual']]=1
            if e['override']:expected[84:88]=[e['p0'],e['p1'],0,0]
            assert np.array_equal(x[i],expected),'no host-computed prediction/error/bank input'
            if not e['teacher']:previous=i
        report=r.get('stdout_report') or json.loads((Path(r['directory'])/'stdout').read_text())
        assert report['program_calls']==1 and report['weight_restore_calls']==report['host_error_decisions']==report['host_bank_selections']==0
        assert report['synthetic_prediction_writes']==(count if stage=='truth' else 0)
    return dict(status='SAVED_ARRAY_PASS',mode=r['mode'],seed=seed,arm=arm,stage=stage,stream=stream,events=count,assessment=summary,edits=edits,stable_records=stable,
        useful_capacity_gate=summary['capacity_probe']['by_context']['0']['correct']>=87 and summary['capacity_probe']['by_context']['1']['correct']==96 if stage=='pressure' else None)

def audit(d):
    d=Path(d);r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0,(d/'stderr').read_text()
    for k,v in r['hashes'].items():assert sha(d/k)==v,('receipt hash',k)
    r['directory']=str(d);f=fixture(r['seed'],r['arm'])
    def get(k):return np.fromfile(d/('result.'+k),'u1' if k in ['inputs','scores'] else '<i4' if k.startswith('potentials') or k=='learning_modes' else '<u4')
    a=check(r,get,lambda k:np.fromfile(f/k,'<u4'));save(d/'AUDIT.json',a);return a

def run(mode,seed,arm,stage,stream):
    assert mode in ['cpu','native'] and seed in SEEDS and arm in ARMS and stage in STAGES and stream in STREAMS
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text());cut=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert not state['closed'] and not state.get('failure') and now()<cut
    f=fixture(seed,arm);assert json.loads((f/'BUILD.json').read_text())['status']=='VIRTUAL_COMPILER_PASS'
    if mode=='native':
        assert passed('cpu',seed,arm,stage,stream)
        if stage=='truth':
            assert seed==SEEDS[0] and stream==STREAMS[0] and arm in ['native','open','closed']
            if arm!='native':assert passed('native',seed,'native','truth')
        else:
            assert all(passed('native',SEEDS[0],q,'truth') for q in ['native','open','closed'])
            if stage=='gate':
                assert seed==SEEDS[0] and stream==STREAMS[0] and arm in ['native','no_retire','frozen']
                if arm!='native':assert passed('native',seed,'native','gate')
            else:
                assert all(passed('native',SEEDS[0],q,'gate') for q in ['native','no_retire','frozen'])
                if stage=='repeat':
                    assert arm in ['native','no_retire','frozen','freeze_A','freeze_B','solo_A','solo_B']
                    assert seed==SEEDS[0] or stream==STREAMS[0]
                    if seed!=SEEDS[0] or stream!=STREAMS[0] or arm!='native':assert passed('native',SEEDS[0],'native','repeat')
                    if arm in ['freeze_A','freeze_B','solo_A','solo_B']:assert seed==SEEDS[0] and stream==STREAMS[0]
                    elif arm!='native':assert passed('native',seed,'native','repeat',stream)
                    if seed!=SEEDS[0] or stream!=STREAMS[0]:assert all(passed('native',SEEDS[0],q,'repeat') for q in ['native','no_retire','frozen'])
                else:
                    assert stage=='pressure' and seed==SEEDS[0] and stream==STREAMS[0] and arm in ['native','no_retire','frozen']
                    assert all(passed('native',s,'native','repeat') for s in SEEDS) and all(passed('native',SEEDS[0],'native','repeat',w) for w in STREAMS)
                    if arm!='native':assert passed('native',seed,'native','pressure')
    physical=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json') if json.loads(p.read_text())['mode']=='native'];events=len(schedule(stage,arm,stream))
    assert len(physical)+int(mode=='native')<=auth['maximum_physical_sessions'] and sum(p['reserved_events'] for p in physical)+(events if mode=='native' else 0)<=auth['maximum_forwards']
    d=S/'raw'/f'{mode}_s{seed}_{arm}_{stage}_w{stream}';assert not d.exists(),'no automatic retry';d.mkdir(parents=True)
    files=[*f.glob('*'),*S.glob('*.md'),S/'AUTHORITY.json',P/'build/isolated_admission',*[P/'admission'/n for n in ['isolation.py','isolation_run.py','prepare_isolation.py','isolation_bench.cpp','build_isolation.sh','recycle.py','error.py','prepare.py','prepare_recycle.py','model.py','run.py']],*[P/n for n in ['pong/relearn/model.py','src/readout_bench.cpp','src/mmio_driver.hpp','tools/prepare_reduction.py','tools/program_codec.py','tools/prepare_network.py','tools/prepare.py','metal/prepare.py']]]
    save(d/'FROZEN.json',dict(utc=now().isoformat(),hashes={str(q.relative_to(P)):sha(q) for q in files if q.is_file()}))
    r=dict(mode=mode,seed=seed,arm=arm,stage=stage,stream=stream,start_utc=now().isoformat(),reserved_events=events,freeze_sha256=sha(d/'FROZEN.json'));save(d/'OWNER.json',r)
    def execute():
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            proc=subprocess.Popen(['taskset','-c','2','build/isolated_admission',mode,str(f),arm,f'{stage}_w{stream}',str(d/'result')],cwd=P,stdout=out,stderr=err);r['pid']=proc.pid;save(d/'OWNER.json',r)
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
    if sys.argv[1]=='audit':print(json.dumps(audit(S/'raw'/sys.argv[2])))
    else:run(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4],int(sys.argv[5]) if len(sys.argv)>5 else STREAMS[0])
