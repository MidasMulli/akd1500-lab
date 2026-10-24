"""Fail-closed physical stages with independent saved-array checks."""
from pathlib import Path
import datetime as dt,fcntl,hashlib,json,subprocess,sys
import numpy as np
from temporal.model import *
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-temporal-context';F=S/'fixtures01'
def now():return dt.datetime.now(dt.timezone.utc)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,a):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(a,indent=2)+'\n')
def freeze():
    p=S/'evidence'/f'FROZEN{len(list((S/"evidence").glob("FROZEN*.json")))+1:02d}.json'
    files=[*(P/'temporal').glob('*'),*F.glob('*/*'),P/'src/readout_bench.cpp',P/'src/mmio_driver.hpp',P/'plasticity/model.py',P/'pong/relearn/model.py',P/'tools/prepare_reduction.py',P/'tools/prepare.py',P/'tools/program_codec.py',P/'tools/prepare_network.py',P/'coprocessor/resident_prepare.py',P/'build/temporal_context',S/'PROTOCOL.md',S/'AUTHORITY.json']
    save(p,dict(utc=now().isoformat(),hashes={str(q.relative_to(P)):sha(q) for q in files if q.is_file()}))
def passed(mode,seed,arm,epochs):
    for p in (S/'raw').glob(f'{mode}_s{seed}_{arm}_e{epochs}*/AUDIT.json'):
        if json.loads(p.read_text())['status']=='SAVED_ARRAY_PASS':return True
    return False
def audit(d):
    r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0,(d/'stderr').read_text()
    for k,v in r['hashes'].items():assert sha(d/k)==v
    seed,arm,epochs=r['seed'],r['arm'],r['epochs'];f=F/f's{seed}';rd=lambda k,t:np.fromfile(d/('result.'+k),t)
    initial=np.fromfile(f/'initial.ram','<u4');a=simulate(initial,seed,arm,epochs);events=a['events'];count=len(events)
    codes,cues,w,old,new,changed=design(seed);y=rd('scores','u1').reshape(count,256);x=rd('inputs','u1').reshape(count,256);states=rd('states','<u4').reshape(count,98)
    assert np.array_equal(y,a['outputs']) and np.array_equal(x,a['inputs']) and np.array_equal(states,a['states']) and np.array_equal(rd('ram','<u4'),a['states'][-1])
    rows=[]
    for phase in ['old_query','teacher','probe']:
      for variant in ['full','short','long']:
        es=[e for e in events if e['phase']==phase and e['variant']==variant];pred=[e for e in es if e['cue']>=4];ctx=[e for e in es if e['cue']<4]
        if not es:continue
        correct=lambda q:sum(np.array_equal(y[e['index']],codes[e['target']]) for e in q)
        changed_es=[e for e in pred if e['context'] in changed];stable=[e for e in pred if e['context'] not in changed]
        assert correct(ctx)==len(ctx),'context code retention'
        if phase=='probe' and arm=='native':assert correct(pred)==len(pred),'native temporal prediction gate'
        if phase=='probe' and arm in ['frozen','unlabelled']:assert correct(stable)==len(stable),'stable association retention'
        if phase=='probe' and arm=='severed':assert not y[[e['index'] for e in pred]].any(),'history ablation produces no response'
        rows.append(dict(phase=phase,variant=variant,context_correct=correct(ctx),context_total=len(ctx),changed_correct=correct(changed_es),changed_total=len(changed_es),stable_correct=correct(stable),stable_total=len(stable)))
    if r['mode']=='native':
        out=json.loads((d/'stdout').read_text());assert r['device_free'] and out['program_calls']==1 and out['weight_restore_calls']==out['threshold_resets']==0 and out['cleanup']=='unprogrammed' and out['current_half_writes']==count
        assert out['history_ablation_writes']==sum(arm=='severed' and e['phase']=='probe' and e['cue']>=4 for e in events)
        assert np.array_equal(rd('initial.ram','<u4'),initial)
        for tag in ['initial','final']:assert np.array_equal(rd('fixed.'+tag,'<u4'),np.fromfile(f/'fixed.ram','<u4'))
        desc=rd('descriptors','<u4').reshape(count,16);addr=rd('output_addresses','<u4');inp=rd('input_addresses','<u4');delays=rd('delays_ns','<i8');assert np.array_equal(desc[:,1],inp) and np.array_equal(desc[:,7],addr)
        for i,e in enumerate(events):
            label=e['label'] if arm in ['native','severed'] else -1;assert ((int(desc[i,8])>>16)&1023)==(label+1 if label>=0 else 0)
            assert delays[i]>=e['delay_ms']*1_000_000
            assert np.array_equal(x[i,128:],cues[e['cue'],128:])
            if i:
                assert inp[i]==addr[i-1]+32 and addr[i]!=addr[i-1]
                if arm=='severed' and e['phase']=='probe' and e['cue']>=4:assert not x[i,:128].any()
                else:assert np.array_equal(x[i,:128],y[i-1,:128]),'previous native output is actual history'
            else:assert not x[i,:128].any()
            assert np.array_equal(rd('packet'+str(i),'<u4').view('u1')[32:],y[i])
    else:
        assert np.array_equal(rd('task_scores','u1').reshape(count,256),a['task_outputs'])
        assert np.array_equal(rd('stateless_scores','u1').reshape(count,256),a['stateless_outputs'])
        if arm=='native':
            for e in events:
                if e['phase']=='probe':assert np.array_equal(a['task_outputs'][e['index']],codes[e['target']]),'strong CPU control'
    result=dict(status='SAVED_ARRAY_PASS',name=d.name,mode=r['mode'],seed=seed,arm=arm,epochs=epochs,events=count,rows=rows);save(d/'AUDIT.json',result);return result
def run(mode,seed,arm,epochs):
    assert mode in ['cpu','native'] and seed in SEEDS and arm in ARMS and epochs in STAGES
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text());cut=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert not state['closed'] and not state.get('failure') and now()<cut
    frozen=sorted((S/'evidence').glob('FROZEN*.json'))[-1]
    for p,v in json.loads(frozen.read_text())['hashes'].items():assert sha(P/p)==v,('source drift',p)
    if mode=='native':
        assert passed('cpu',seed,arm,epochs)
        if seed==SEEDS[0] and arm=='native':
            if epochs:assert passed('native',seed,arm,{1:0,2:1,8:2}[epochs])
        elif seed==SEEDS[0]:assert epochs==8 and passed('native',seed,'native',8)
        else:assert epochs==8 and all(passed('native',SEEDS[0],q,8) for q in ARMS)
    owners=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')];physical=[r for r in owners if r['mode']=='native'];assert len(physical)+(mode=='native')<=auth['maximum_physical_sessions'];assert sum(8+32*r['epochs'] for r in physical)+(8+32*epochs if mode=='native' else 0)<=auth['maximum_forwards']
    d=S/'raw'/f'{mode}_s{seed}_{arm}_e{epochs}'
    if d.exists():d=d.with_name(d.name+'_attempt'+str(len(list((S/'raw').glob(d.name+'*')))+1))
    d.mkdir(parents=True,exist_ok=False);r=dict(mode=mode,seed=seed,arm=arm,epochs=epochs,start_utc=now().isoformat(),freeze_sha256=sha(frozen));save(d/'OWNER.json',r)
    def execute():
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            proc=subprocess.Popen(['taskset','-c','2','build/temporal_context',mode,str(F/f's{seed}'),arm,str(epochs),str(d/'result')],cwd=P,stdout=out,stderr=err);r['pid']=proc.pid;save(d/'OWNER.json',r)
            try:code=proc.wait(timeout=min(30,(cut-now()).total_seconds()))
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
