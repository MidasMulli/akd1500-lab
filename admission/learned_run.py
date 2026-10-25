"""Locked, staged three-bank study and portable semantic verifier."""
from pathlib import Path
import datetime as dt,json,subprocess,sys
import numpy as np
from admission.learned import *
from admission.run import now,save,sha,masks
P=Path(__file__).resolve().parents[1];S=P/'studies/20261025-learned-admission';F=S/'fixtures01'
def fixture(seed,arm):return F/f's{seed}_{arm}'
def passed(mode,seed,arm,stage):return any(json.loads(p.read_text()).get('status')=='SAVED_ARRAY_PASS' for p in (S/'raw').glob(f'{mode}_s{seed}_{arm}_{stage}*/AUDIT.json'))

def check(run,get,fg):
    seed,arm,stage=run['seed'],run['arm'],run['stage'];initial=[fg(f'bank{k}.ram') for k in range(3)];a=simulate(initial,stage,arm);rows=a['events'];count=len(rows)
    x=get('inputs').reshape(count,N);y=get('scores').reshape(count,N);states=[get(f'states{k}').reshape(count,len(initial[k])) for k in range(3)]
    assert np.array_equal(x,a['inputs']),'raw observations and native permission inputs'
    assert np.array_equal(y,a['outputs']),'all native output/permission bits'
    summary=assess(dict(a,outputs=y));edits=[]
    for k,(n,u,c) in enumerate(DIMENSIONS):
        assert np.array_equal(states[k],a[f'states{k}']),'every native learning-RAM boundary'
        assert np.array_equal(get(f'potentials{k}').reshape(count,u),a[f'potentials{k}']),'derived packed potentials'
        assert np.array_equal(masks(initial[k],n,u).T,weights(seed,k)),'initial own seeded weights'
        plastic=[0] if k==0 else P0 if k==1 else P1;before=initial[k];m0=masks(before,n,u);changes=0;changed_events=[];blocked=admitted=0
        for i,e in enumerate(rows):
            v=states[k][i];mm=masks(v,n,u);assert np.all(mm.sum(-1)==32),'connection budget'
            stride=4+n//32
            for j in range(u):
                if j not in plastic:assert np.array_equal(v[2+j*stride:2+(j+1)*stride],initial[k][2+j*stride:2+(j+1)*stride]),'fixed forwarding helper RAM changed'
            if e['labels'][k]<0:assert np.array_equal(v,before),'disabled native bank changed'
            if k==0 and e['manager_learn']:
                expected=np.zeros(NM,'u1');expected[:25]=1;expected[25+e['key']*7:25+(e['key']+1)*7]=1
                assert np.array_equal(mm[0],expected),'manager synapses must learn current raw key in one update'
            if k==2 and e['teacher'] and e['label']>=0:
                if not a['features2'][i].any():blocked+=1;assert np.array_equal(v,before),'refused teacher modified protected RAM'
                else:admitted+=1
            delta=int(((mm==1)&(m0==0)).sum());changes+=delta
            if delta:changed_events.append(i)
            before=v;m0=mm
        edits.append(dict(bank=k,connection_replacements=changes,changed_events=changed_events,blocked_teachers=blocked,admitted_teachers=admitted))
    assert not matrices(arm)[0][0][:32].any(),'previous-key field must be disconnected'
    if stage=='gate':
        if arm=='native':assert edits[2]['blocked_teachers']==edits[2]['admitted_teachers']==8 and edits[2]['connection_replacements']>0
        if arm=='open':assert edits[2]['blocked_teachers']==0 and edits[2]['admitted_teachers']==16 and edits[2]['connection_replacements']>0
        if arm=='closed':assert edits[2]['blocked_teachers']==16 and edits[2]['connection_replacements']==0
    old={};assignments={};retained_weights=0
    if stage!='gate':
        for e in rows:
            if e['phase']=='old_probe' and not e['teacher']:
                ids=np.flatnonzero(y[e['index'],64:72]);assert len(ids)==1 or arm=='closed','old context record'
                if len(ids)==1:old[e['key']]=int(ids[0])
        if arm!='closed':assert summary['old_probe']['protected']['correct']==4
        if stage in ['pressure','heldout']:
            first_probe=next(e['index'] for e in rows if e['phase']=='old_probe')
            for key,j in old.items():
                record=P1[j];stride=4+N1//32;b=2+record*stride
                assert np.all(states[2][first_probe:,b:b+stride]==states[2][first_probe,b:b+stride]),'old useful record must stay unchanged'
                retained_weights+=1
            if arm=='native':
                assert summary['final_probe']['protected']['correct']==24
                for e in rows:
                    if e['phase']=='final_probe' and not e['teacher']:
                        ids=np.flatnonzero(y[e['index'],64:72]);assert len(ids)==1;assignments[e['key']]=int(ids[0])
                assert len(assignments)==len(set(assignments.values()))==6 and all(assignments[k]==j for k,j in old.items()),'six learned records with original identities'
                if stage=='heldout':assert summary['heldout']['protected']['correct']==96
            if arm in ['open','frozen','manager_frozen','severed']:assert summary['final_probe']['protected']['correct']<24
    if run['mode']=='native':
        assert run['exit_code']==0 and run['device_free']
        for k in range(3):assert np.array_equal(get(f'initial{k}'),initial[k])
        for k in range(4):
            for tag in ['initial','final']:assert np.array_equal(get(f'fixed{k}.{tag}'),fg(f'fixed{k}.ram'))
        assert get('learning_modes').reshape(count,3).tolist()==[[int(v>=0) for v in e['labels']] for e in rows],'uniform phase clocks only'
        desc=get('descriptors').reshape(count,16);addr=get('output_addresses');inp=get('input_addresses');assert np.array_equal(desc[:,1],inp) and np.array_equal(desc[:,7],addr)
        previous_query=None;mutated=False
        for i,e in enumerate(rows):
            assert ((int(desc[i,8])>>16)&1023)==(OUTCOME[e['key']]+1 if e['teacher'] else 1),'constant manager label or delayed actual outcome'
            assert np.array_equal(get(f'packet{i}').view('u1')[32:],y[i]),'saved native packet payload'
            assert int(inp[i])!=int(addr[i])+32
            if e['teacher']:
                assert previous_query is not None and inp[i]==addr[previous_query]+32,'teacher must use native query output directly'
                expected=y[previous_query].copy();expected[64:68]=0;expected[65]=1
                if arm=='severed' and e['phase'] not in ['bootstrap','gate']:expected[80]=0;mutated=True
                assert np.array_equal(x[i],expected),'no host-decoded acting permission'
                assert addr[i]!=addr[previous_query]
            else:
                if previous_query is not None:
                    assert inp[i]==addr[previous_query]+32,'query retained native source'
                    expected=y[previous_query].copy();expected[32:64]=x[i,32:64];expected[64:68]=0;expected[64]=1
                    if mutated:expected[80]=0
                    assert np.array_equal(x[i],expected),'host writes only raw current key and uniform phase'
                previous_query=i;mutated=False
    return dict(status='SAVED_ARRAY_PASS',mode=run['mode'],seed=seed,arm=arm,stage=stage,events=count,assessment=summary,edits=edits,old_assignments=old,final_assignments=assignments,old_records_unchanged=retained_weights)

def audit(d):
    r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0,(d/'stderr').read_text()
    for k,v in r['hashes'].items():assert sha(d/k)==v,('receipt hash',k)
    f=fixture(r['seed'],r['arm'])
    def get(k):return np.fromfile(d/('result.'+k),'u1' if k in ['inputs','scores'] else '<i4' if k.startswith('potentials') or k=='learning_modes' else '<u4')
    a=check(r,get,lambda k:np.fromfile(f/k,'<u4'));save(d/'AUDIT.json',a);return a

def run(mode,seed,arm,stage):
    assert mode in ['cpu','native'] and seed in SEEDS and arm in ARMS and stage in STAGES
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text());cut=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert not state['closed'] and not state.get('failure') and now()<cut
    f=fixture(seed,arm);assert json.loads((f/'BUILD.json').read_text())['status']=='VIRTUAL_COMPILER_PASS'
    if mode=='native':
        assert passed('cpu',seed,arm,stage)
        if stage=='gate':
            assert seed==SEEDS[0] and arm in ['native','open','closed']
            if arm!='native':assert passed('native',seed,'native','gate')
        elif stage=='prefix':assert seed==SEEDS[0] and arm=='native' and all(passed('native',seed,q,'gate') for q in ['native','open','closed'])
        elif stage=='pressure':
            assert seed==SEEDS[0] and arm in ['native','open','frozen'] and passed('native',seed,'native','prefix')
            if arm!='native':assert passed('native',seed,'native','pressure')
        else:
            assert arm in ['native','open','frozen','manager_frozen','severed'] and all(passed('native',SEEDS[0],q,'pressure') for q in ['native','open','frozen'])
            if arm!='native' or seed!=SEEDS[0]:assert passed('native',SEEDS[0],'native','heldout')
            if seed!=SEEDS[0]:assert arm in ['native','open','frozen'] and all(passed('native',SEEDS[0],q,'heldout') for q in ['native','open','frozen','manager_frozen','severed'])
    physical=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json') if json.loads(p.read_text())['mode']=='native'];events=len(schedule(stage,arm))
    assert len(physical)+(mode=='native')<=auth['maximum_physical_sessions']
    assert sum(r['reserved_events'] for r in physical)+(events if mode=='native' else 0)<=auth['maximum_forwards']
    name=f'{mode}_s{seed}_{arm}_{stage}';d=S/'raw'/name
    if d.exists():d=d.with_name(name+'_attempt'+str(len(list((S/'raw').glob(name+'*')))+1))
    d.mkdir(parents=True,exist_ok=False)
    files=[*(P/'admission').glob('*'),*f.glob('*'),*S.glob('*.md'),S/'AUTHORITY.json',P/'build/learned_admission',P/'src/readout_bench.cpp',P/'src/mmio_driver.hpp',P/'pong/relearn/model.py',P/'tools/prepare_reduction.py',P/'tools/prepare_network.py',P/'tools/program_codec.py',P/'tools/prepare.py']
    freeze=dict(utc=now().isoformat(),hashes={str(q.relative_to(P)):sha(q) for q in files if q.is_file()});save(d/'FROZEN.json',freeze)
    r=dict(mode=mode,seed=seed,arm=arm,stage=stage,start_utc=now().isoformat(),reserved_events=events,freeze_sha256=sha(d/'FROZEN.json'));save(d/'OWNER.json',r)
    def execute():
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            proc=subprocess.Popen(['taskset','-c','2','build/learned_admission',mode,str(f),arm,stage,str(d/'result')],cwd=P,stdout=out,stderr=err);r['pid']=proc.pid;save(d/'OWNER.json',r)
            try:code=proc.wait(timeout=min(60,(cut-now()).total_seconds()))
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
    print(json.dumps(a),flush=True)
    return a
if __name__=='__main__':
    if sys.argv[1]=='audit':print(json.dumps(audit(S/'raw'/sys.argv[2])))
    else:run(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4])
