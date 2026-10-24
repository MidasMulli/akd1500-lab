"""Bounded physical admission and saved-array verification; failures stop expansion."""
from pathlib import Path
import datetime as dt,fcntl,hashlib,json,subprocess,sys
import numpy as np
from discovery.model import *
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-context-discovery';F=S/'fixtures05'
def now():return dt.datetime.now(dt.timezone.utc)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,a):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(a,indent=2)+'\n')
def freeze():
    p=S/'evidence'/f'FROZEN{len(list((S/"evidence").glob("FROZEN*.json")))+1:02d}.json'
    files=[*(P/'discovery').glob('*'),*F.glob('*/*'),P/'src/readout_bench.cpp',P/'src/mmio_driver.hpp',P/'pong/relearn/model.py',P/'tools/prepare_reduction.py',P/'tools/prepare.py',P/'tools/program_codec.py',P/'tools/prepare_network.py',P/'coprocessor/resident_prepare.py',P/'metal/prepare.py',P/'build/context_discovery',S/'PROTOCOL.md',S/'AMENDMENT01.md',S/'AMENDMENT02.md',S/'AUTHORITY.json',*S.glob('AMENDMENT*.md')]
    save(p,dict(utc=now().isoformat(),hashes={str(q.relative_to(P)):sha(q) for q in files if q.is_file()}))
def passed(mode,seed,arm,cycles):
    return any(json.loads(p.read_text())['status']=='SAVED_ARRAY_PASS' for p in (S/'raw').glob(f'{mode}_s{seed}_{arm}_c{cycles}*/AUDIT.json'))
def audit(d):
    r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0,(d/'stderr').read_text()
    for k,v in r['hashes'].items():assert sha(d/k)==v,('receipt hash',k)
    seed,arm,cycles=r['seed'],r['arm'],r['cycles'];f=F/f's{seed}';rd=lambda k,t:np.fromfile(d/('result.'+k),t)
    initial=np.fromfile(f/'initial.ram','<u4');a=simulate(initial,cycles,arm);events=a['events'];count=len(events)
    y=rd('scores','u1').reshape(count,N);x=rd('inputs','u1').reshape(count,N);states=rd('states','<u4').reshape(count,482)
    assert np.array_equal(y,a['outputs']),'saved outputs'
    assert np.array_equal(x,a['inputs']),'saved carried inputs'
    assert np.array_equal(states,a['states']) and np.array_equal(rd('ram','<u4'),a['states'][-1]),'saved learning RAM'
    assert np.array_equal(rd('potentials','<i4').reshape(count,U),a['potentials']),'derived potentials'
    assert np.array_equal(rd('cpu','<i4'),a['cpu']),'independent raw-symbol CPU dictionary'
    for j in range(LEARNED,U):assert np.all(states[:,2+j*12:2+(j+1)*12]==initial[2+j*12:2+(j+1)*12]),'fixed helper record changed'
    probe_start=next(e['index'] for e in events if e['phase']=='probe');assert np.all(states[probe_start:]==states[probe_start-1]),'probe-time RAM edit'
    old={}
    if arm!='unlabelled':
        for e in events:
            if e['phase']=='old_probe':
                ids=np.flatnonzero(y[e['index'],64:96]);assert len(ids)==1,'bootstrap four-history qualification';old[e['history']]=int(ids[0])
        assert len(old)==len(set(old.values()))==4
    result=assess(dict(a,outputs=y,states=states))
    if arm in ['native','unseen'] and cycles==12:
        assert result['unique_records']==16 and result['ambiguous']==result['missing']==0 and result['learned_records']>=16,'native allocation gate'
        for k,j in old.items():assert result['assignments'][k]==j,'old record retention'
        for e in events:
            if e['phase']=='probe':assert y[e['index'],64:96].sum()==1 and y[e['index'],64+result['assignments'][e['history']]]==1,'stable probe assignment'
    if arm in ['frozen','unseen_frozen']:
        assert result['unique_records']==4 and result['missing']==12 and result['ambiguous']==0
        assert result['assignments']==old,'frozen known histories retained'
    if arm in ['severed','unlabelled']:assert not y[probe_start:,64:96].any()
    if arm=='unlabelled':assert np.all(states==initial),'unlabelled no updates'
    if arm in ['native','unseen'] and cycles==12:
        assert len({int(a['cpu'][e['index']]) for e in events if e['phase']=='probe'})==16,'strong CPU discovers histories'
    if arm in ['unseen','unseen_frozen']:
        held=[e for e in events if e['phase']=='heldout'];assert len(held)==128 and len({e['history'] for e in held})==16
        for e in held:
            j=result['assignments'].get(e['history']);v=y[e['index'],64:96]
            assert (v.sum()==1 and v[j]==1) if j is not None else not v.any(),'heldout-order recall'
        result['heldout_correct']=sum(int(y[e['index'],64:96].sum()==1) for e in held);result['heldout_total']=128
    if r['mode']=='native':
        out=json.loads((d/'stdout').read_text());assert r['device_free'] and out['program_calls']==1 and out['weight_restore_calls']==out['threshold_resets']==0 and out['cleanup']=='unprogrammed' and out['current_symbol_writes']==count
        assert out['history_ablation_writes']==sum(e['sever'] for e in events)
        assert np.array_equal(rd('initial.ram','<u4'),initial)
        for j in range(2):
            for tag in ['initial','final']:assert np.array_equal(rd(f'fixed{j}.'+tag,'<u4'),np.fromfile(f/f'fixed{j}.ram','<u4'))
        assert rd('learning_modes','<i4').tolist()==[int(e['label']>=0) for e in events],'explicit query learning disable'
        desc=rd('descriptors','<u4').reshape(count,16);addr=rd('output_addresses','<u4');inp=rd('input_addresses','<u4');assert np.array_equal(desc[:,1],inp) and np.array_equal(desc[:,7],addr)
        for i,e in enumerate(events):
            assert ((int(desc[i,8])>>16)&1023)==(1 if e['label']==0 else 0),'only constant training label0'
            current=np.zeros(16,'u1');current[e['bit']*8:e['bit']*8+8]=1;assert np.array_equal(x[i,48:64],current)
            if i:
                assert inp[i]==addr[i-1]+32 and addr[i]!=addr[i-1]
                if e['sever']:assert not x[i,:48].any() and np.array_equal(x[i,64:],y[i-1,64:])
                else:assert np.array_equal(x[i,:48],y[i-1,:48]) and np.array_equal(x[i,64:],y[i-1,64:]),'native output carried without reconstruction'
            assert np.array_equal(rd('packet'+str(i),'<u4').view('u1')[32:],y[i]),'saved native packet payload'
    result.update(status='SAVED_ARRAY_PASS',name=d.name,mode=r['mode'],seed=seed,arm=arm,cycles=cycles,events=count,old_assignments=old);save(d/'AUDIT.json',result);return result

def run(mode,seed,arm,cycles):
    assert mode in ['cpu','native'] and seed in SEEDS and arm in ARMS and cycles in STAGES
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text());cut=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert not state['closed'] and not state.get('failure') and now()<cut
    frozen=sorted((S/'evidence').glob('FROZEN*.json'))[-1]
    for p,v in json.loads(frozen.read_text())['hashes'].items():assert sha(P/p)==v,('source drift',p)
    if mode=='native':
        assert passed('cpu',seed,arm,cycles)
        if arm in ['unseen','unseen_frozen']:
            assert cycles==12 and (arm=='unseen' or seed==SEEDS[0]) and all(passed('native',s,a,12) for s in SEEDS for a in ['native','frozen','unlabelled'])
        elif seed==SEEDS[0] and arm=='native':
            if cycles>1:assert passed('native',seed,arm,{2:1,4:2,12:4}[cycles])
        elif seed==SEEDS[0]:assert cycles==12 and passed('native',seed,'native',12)
        else:assert cycles==12 and all(passed('native',SEEDS[0],q,12) for q in ARMS)
    physical=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json') if json.loads(p.read_text())['mode']=='native'];assert len(physical)+(mode=='native')<=auth['maximum_physical_sessions'];assert sum(131+16*r['cycles']+(128 if r['arm'] in ['unseen','unseen_frozen'] else 0) for r in physical)+(131+16*cycles+(128 if arm in ['unseen','unseen_frozen'] else 0) if mode=='native' else 0)<=auth['maximum_forwards']
    d=S/'raw'/f'{mode}_s{seed}_{arm}_c{cycles}'
    if d.exists():d=d.with_name(d.name+'_attempt'+str(len(list((S/'raw').glob(d.name+'*')))+1))
    d.mkdir(parents=True,exist_ok=False);r=dict(mode=mode,seed=seed,arm=arm,cycles=cycles,start_utc=now().isoformat(),freeze_sha256=sha(frozen));save(d/'OWNER.json',r)
    def execute():
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            proc=subprocess.Popen(['taskset','-c','2','build/context_discovery',mode,str(F/f's{seed}'),arm,str(cycles),str(d/'result')],cwd=P,stdout=out,stderr=err);r['pid']=proc.pid;save(d/'OWNER.json',r)
            try:code=proc.wait(timeout=min(40,(cut-now()).total_seconds()))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:code=proc.wait(timeout=2)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
        r.update(exit_code=code,finish_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,hashes={p.name:sha(p) for p in d.iterdir() if p.is_file()});save(d/'RECEIPT.json',r)
        try:result=audit(d)
        except Exception:state['failure']=d.name;save(S/'STATE.json',state);raise
        state['completed'].append(d.name);save(S/'STATE.json',state);return result
    if mode=='native':
        with open('/tmp/akd1500-research.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1;result=execute()
    else:result=execute()
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='audit':print(json.dumps(audit(S/'raw'/sys.argv[2])))
    else:run(sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
