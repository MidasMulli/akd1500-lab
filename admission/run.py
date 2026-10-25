"""Finite, locked physical admissions; independently re-read every saved array."""
from pathlib import Path
import datetime as dt,hashlib,json,subprocess,sys
import numpy as np
from admission.model import *
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-native-admission';F=S/'fixtures06'
def now():return dt.datetime.now(dt.timezone.utc)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def fixture(seed,arm):return F/f's{seed}_{arm}'
def passed(mode,seed,arm,stage):
    return any(json.loads(p.read_text()).get('status')=='SAVED_ARRAY_PASS' for p in (S/'raw').glob(f'{mode}_s{seed}_{arm}_{stage}*/AUDIT.json'))
def masks(ram,n,u):
    stride=4+n//32
    return np.unpackbits(np.asarray(ram,dtype='<u4')[...,2:].reshape(*ram.shape[:-1],u,stride)[...,4:].copy().view('u1'),axis=-1,bitorder='little')

def check(run,get,fg):
    seed,arm,stage=run['seed'],run['arm'],run['stage'];initial=[fg('bank0.ram'),fg('bank1.ram')];a=simulate(initial,stage,arm);rows=a['events'];count=len(rows)
    x=get('inputs').reshape(count,N);y=get('scores').reshape(count,N);states=[get(f'states{k}').reshape(count,len(initial[k])) for k in range(2)]
    assert np.array_equal(x,a['inputs']),'native raw key/history/phase'
    assert np.array_equal(y,a['outputs']),'native predictions/history response'
    out=assess(dict(a,outputs=y));edits=[]
    for k,(n,u,plastic) in enumerate([(N0,U0,P0),(N1,U1,P1)]):
        assert np.array_equal(states[k],a[f'states{k}']),'every stored native learning boundary'
        assert np.array_equal(get(f'potentials{k}').reshape(count,u),a[f'potentials{k}']),'derived packed CPU potentials'
        assert np.array_equal(masks(initial[k],n,u).T,weights(seed,k)),'own seeded initial connections'
        before=initial[k];m0=masks(before,n,u);changes=0;changed_events=[];blocked=admitted=0
        for i,e in enumerate(rows):
            after=states[k][i];mm=masks(after,n,u)
            assert np.all(mm.sum(-1)==32),'fixed binary connection budget'
            for j in range(u):
                if j not in plastic:assert np.array_equal(after[2+j*(4+n//32):2+(j+1)*(4+n//32)],initial[k][2+j*(4+n//32):2+(j+1)*(4+n//32)]),'read-only forwarding helper changed'
            if e['label']<0:assert np.array_equal(after,before),'read-only query/frozen teacher changed RAM'
            if k==1 and e['teacher'] and e['label']>=0:
                if not a['features1'][i].any():
                    blocked+=1;assert np.array_equal(after,before),'rejected teacher changed protected RAM'
                else:admitted+=1
            c=int(((mm==1)&(m0==0)).sum());changes+=c
            if c:changed_events.append(i)
            before=after;m0=mm
        edits.append(dict(bank=k,connection_replacements=changes,changed_events=changed_events,blocked_teachers=blocked,admitted_teachers=admitted))
    if stage=='gate':
        if arm=='native':assert edits[1]['blocked_teachers']==edits[1]['admitted_teachers']==8 and edits[1]['connection_replacements']>0,'native selective admission gate'
        if arm=='open':assert edits[1]['blocked_teachers']==0 and edits[1]['admitted_teachers']==16 and edits[1]['connection_replacements']>0,'open policy control'
        if arm=='closed':assert edits[1]['blocked_teachers']==16 and edits[1]['connection_replacements']==0,'closed policy control'
    if stage!='gate':
        if arm!='closed':assert out['old_probe']['protected']['correct']==4,'initial useful memories must qualify'
        assert out['old_probe']['always_bank']['correct']==4
    if stage in ['pressure','heldout']:
        if arm=='native':assert out['final_probe']['protected']['correct']==24,'adaptation and original recall gate'
        if arm in ['open','frozen']:assert out['final_probe']['protected']['correct']<24,'matched control must distinguish admission'
        if stage=='heldout' and arm=='native':assert out['heldout']['protected']['correct']==96,'sealed read-only ordering'
    if run['mode']=='native':
        assert run['exit_code']==0 and run['device_free'],'owned execution/cleanup'
        for k in range(2):assert np.array_equal(get(f'initial{k}'),initial[k])
        for k in range(4):
            for tag in ['initial','final']:assert np.array_equal(get(f'fixed{k}.{tag}'),fg(f'fixed{k}.ram')),'fixed policy/route RAM'
        assert get('learning_modes').tolist()==[int(e['label']>=0) for e in rows],'shared host query/teacher clock only'
        desc=get('descriptors').reshape(count,16);addr=get('output_addresses');inp=get('input_addresses');assert np.array_equal(desc[:,1],inp) and np.array_equal(desc[:,7],addr)
        previous_query=None;last_query=None
        for i,e in enumerate(rows):
            assert int(inp[i])!=int(addr[i])+32,'no input/output alias'
            assert ((int(desc[i,8])>>16)&1023)==(e['label']+1 if e['label']>=0 else 0),'actual outcome label only'
            assert np.array_equal(get(f'packet{i}').view('u1')[32:],y[i]),'saved native packet'
            if e['teacher']:
                assert last_query is not None and inp[i]==inp[last_query],'teacher reuses the preceding raw query input'
                expected=x[last_query].copy();expected[64:68]=0
                if arm=='severed' and e['phase'] not in ['bootstrap','gate']:expected[:32]=0
                assert np.array_equal(x[i],expected),'teacher changes phase only, plus declared ablation'
                assert addr[i]!=addr[last_query],'teacher cannot erase retained query output'
            else:
                if previous_query is None:assert not x[i,:32].any()
                else:
                    assert inp[i]==addr[previous_query]+32,'native carry pointer'
                    assert np.array_equal(x[i,:32],y[previous_query,:32]),'native carried history, no host reconstruction'
                previous_query=last_query=i
    return dict(status='SAVED_ARRAY_PASS',mode=run['mode'],seed=seed,arm=arm,stage=stage,events=count,assessment=out,edits=edits)

def audit(d):
    r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0,(d/'stderr').read_text()
    for k,v in r['hashes'].items():assert sha(d/k)==v,('receipt hash',k)
    f=fixture(r['seed'],r['arm'])
    def get(k):return np.fromfile(d/('result.'+k),'u1' if k in ['inputs','scores'] else '<i4' if k in ['potentials0','potentials1','learning_modes'] else '<u4')
    q=check(r,get,lambda k:np.fromfile(f/k,'<u4'));save(d/'AUDIT.json',q);return q

def run(mode,seed,arm,stage):
    assert mode in ['cpu','native'] and seed in SEEDS and arm in ARMS and stage in STAGES
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text());cut=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert not state['closed'] and not state.get('failure') and now()<cut
    f=fixture(seed,arm);assert json.loads((f/'BUILD.json').read_text())['status']=='VIRTUAL_COMPILER_PASS'
    if mode=='native':
        assert passed('cpu',seed,arm,stage),'independent CPU saved-array gate'
        if stage=='gate':
            assert seed==SEEDS[0] and arm in ['native','open','closed']
            if arm!='native':assert passed('native',seed,'native','gate')
        elif stage=='prefix':assert seed==SEEDS[0] and arm=='native' and all(passed('native',seed,a,'gate') for a in ['native','open','closed'])
        elif stage=='pressure':
            assert seed==SEEDS[0] and arm in ['native','open','frozen'] and passed('native',seed,'native','prefix')
            if arm!='native':assert passed('native',seed,'native','pressure')
        else:
            assert arm in ['native','open','frozen','severed'] and all(passed('native',SEEDS[0],a,'pressure') for a in ['native','open','frozen'])
            if seed!=SEEDS[0]:assert all(passed('native',SEEDS[0],a,'heldout') for a in ['native','open','frozen'])
    physical=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json') if json.loads(p.read_text())['mode']=='native'];events=len(schedule(stage,arm))
    assert len(physical)+(mode=='native')<=auth['maximum_physical_sessions']
    assert sum(r['reserved_events'] for r in physical)+(events if mode=='native' else 0)<=auth['maximum_forwards']
    name=f'{mode}_s{seed}_{arm}_{stage}';d=S/'raw'/name
    if d.exists():d=d.with_name(name+'_attempt'+str(len(list((S/'raw').glob(name+'*')))+1))
    d.mkdir(parents=True,exist_ok=False)
    files=[*(P/'admission').glob('*'),*f.glob('*'),*S.glob('*.md'),S/'AUTHORITY.json',P/'build/native_admission',P/'src/readout_bench.cpp',P/'src/mmio_driver.hpp',P/'pong/relearn/model.py',P/'tools/prepare_reduction.py',P/'tools/prepare_network.py',P/'tools/program_codec.py',P/'tools/prepare.py']
    freeze=dict(utc=now().isoformat(),hashes={str(q.relative_to(P)):sha(q) for q in files if q.is_file()});save(d/'FROZEN.json',freeze)
    r=dict(mode=mode,seed=seed,arm=arm,stage=stage,start_utc=now().isoformat(),reserved_events=events,freeze_sha256=sha(d/'FROZEN.json'));save(d/'OWNER.json',r)
    def execute():
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            proc=subprocess.Popen(['taskset','-c','2','build/native_admission',mode,str(f),arm,stage,str(d/'result')],cwd=P,stdout=out,stderr=err);r['pid']=proc.pid;save(d/'OWNER.json',r)
            try:code=proc.wait(timeout=min(60,(cut-now()).total_seconds()))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:code=proc.wait(timeout=2)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
        r.update(exit_code=code,finish_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,hashes={p.name:sha(p) for p in d.iterdir() if p.is_file()});save(d/'RECEIPT.json',r)
        try:q=audit(d)
        except Exception:state['failure']=d.name;save(S/'STATE.json',state);raise
        state['completed'].append(d.name);save(S/'STATE.json',state);return q
    if mode=='native':
        import fcntl
        with open('/tmp/akd1500-research.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1;q=execute()
    else:q=execute()
    print(json.dumps(q),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='audit':print(json.dumps(audit(S/'raw'/sys.argv[2])))
    else:run(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4])
