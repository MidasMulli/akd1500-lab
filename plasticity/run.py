"""Sequential resident-adaptation admissions with raw-derived independent checks."""
from pathlib import Path
import datetime as dt,fcntl,hashlib,json,subprocess,sys
import numpy as np
from plasticity.model import SEEDS,ARMS,EPOCHS,H,simulate,design
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-resident-adaptation';F=S/'fixtures02'
def now():return dt.datetime.now(dt.timezone.utc)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,a):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(a,indent=2)+'\n')
def name(mode,seed,arm,epochs):return f'{mode}_s{seed}_{arm}_e{epochs}'
def passed(mode,seed,arm,epochs,extra_queries=0,full_queries=False):
    for p in (S/'raw').glob(name(mode,seed,arm,epochs)+'*/AUDIT.json'):
        a=json.loads(p.read_text());r=json.loads((p.parent/'RECEIPT.json').read_text())
        if a['status']=='SAVED_ARRAY_PASS' and a['epochs']==epochs and r.get('extra_queries',0)==extra_queries and r.get('full_queries',False)==full_queries and (mode!='cpu' or a.get('reference_model')=='unlabelled_accumulator_v2'):return True
    return False

def freeze():
    p=S/'evidence'/f'FROZEN{len(list((S/"evidence").glob("FROZEN*.json")))+1:02d}.json';files=[*(P/'plasticity').glob('*'),*F.glob('*/*'),P/'src/readout_bench.cpp',P/'src/mmio_driver.hpp',P/'pong/relearn/model.py',P/'tools/prepare_reduction.py',P/'tools/prepare.py',P/'tools/program_codec.py',P/'tools/prepare_network.py',P/'coprocessor/resident_prepare.py',P/'build/resident_adaptation',S/'PROTOCOL.md',S/'PROTOCOL-AMENDMENT01.md',S/'PROTOCOL-AMENDMENT02.md',S/'evidence/SELECTOR_DIAGNOSTIC01.json',S/'AUTHORITY.json'];save(p,dict(utc=now().isoformat(),hashes={str(q.relative_to(P)):sha(q) for q in files if q.is_file()}))
def audit(d):
    r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0,(d/'stderr').read_text()
    for k,v in r['hashes'].items():assert sha(d/k)==v
    seed,arm,epochs=r['seed'],r['arm'],r['epochs'];f=F/(f's{seed}'+(f'_q{r.get("extra_queries",0)}'+('full' if r.get('full_queries',False) else '') if r.get('extra_queries',0) else ''));initial=np.fromfile(f/f'{arm}.ram','<u4');a=simulate(initial,seed,arm,epochs,r.get('extra_queries',0),r.get('full_queries',False));events=a['events'];count=len(events);rd=lambda n,t:np.fromfile(d/('result.'+n),t)
    y=rd('scores','u1').reshape(count,256);x=rd('inputs','u1').reshape(count,256);states=rd('states','<u4').reshape(-1,98);ids=rd('state_ids','<i4');assert np.array_equal(y,a['outputs']),'scalar output replay';assert np.array_equal(x,a['inputs']),'scalar native feedback input replay';assert np.array_equal(states,a['states'][ids]),'all observed SRAM boundaries';assert np.array_equal(rd('ram','<u4'),a['states'][-1])
    codes,partial,old,new,changed=design(seed);same=np.setdiff1d(range(8),changed);rows=[]
    for epoch in range(epochs):
        truth=new if epoch%2==0 else old;q=[e['index'] for e in events if e['kind']=='query' and e['epoch']==epoch];fb=[e['index'] for e in events if e['kind']=='feedback' and e['epoch']==epoch];correct=np.asarray([np.array_equal(y[i],codes[truth[events[i]['context']]]) for i in q]);consistency=0
        for i in fb:
            context=next((j for j in range(8) if np.array_equal(x[i],codes[j])),-1);assert context>=0;consistency+=np.array_equal(y[i],codes[truth[context]])
        assert int(correct[same].sum())==6,'retention gate'
        if arm=='native':assert correct.all(),'native repeated-change gate'
        b=a['states'][q[-1]][2:].reshape(8,12);rows.append(dict(epoch=epoch,changed_recalled=int(correct[changed].sum()),changed_total=2,retained=int(correct[same].sum()),retained_total=6,feedback_consistent=int(consistency),feedback_total=H,learning_thresholds=(b[:,3]>>16).tolist(),swap_budgets=(b[:,3]&65535).tolist()))
    if r['mode']=='native':
        assert r['device_free'];out=json.loads((d/'stdout').read_text());assert out['program_calls']==1 and out['weight_restore_calls']==out['learning_parameter_resets']==0 and out['cleanup']=='unprogrammed'
        assert np.array_equal(rd('initial.ram','<u4'),initial)
        for tag in ['initial','final']:assert np.array_equal(rd('fixed.'+tag,'<u4'),np.fromfile(f/f'{arm}.fixed','<u4'))
        desc=rd('descriptors','<u4').reshape(count,16);addr=rd('output_addresses','<u4');inp=rd('input_addresses','<u4');assert np.array_equal(desc[:,7],addr) and np.array_equal(desc[:,1],inp)
        for i,e in enumerate(events):
            assert ((int(desc[i,8])>>16)&1023)==(e['label']+1 if e['label']>=0 else 0)
            if e['source']>=0:
                src=e['source'];assert inp[i]==addr[src]+32 and i-src<64
                assert all(addr[j]!=addr[src] for j in range(src+1,i+1)),'carry buffer survives intervening observations'
            assert np.array_equal(rd('packet'+str(i),'<u4').view('u1')[32:],y[i])
        assert sum(e['source']<0 for e in events if e['kind'] in ['pre_prediction','feedback'])==1
        q=rd('quiet','<i8').reshape(epochs,9);assert np.all(q[:,0]==np.arange(epochs)) and np.all(q[:,3]-q[:,2]>=5_000_000) and np.all(q[:,5]==0) and np.all(q[:,6]==1) and np.all(q[:,8]>>16==q[:,7]);assert all(np.array_equal(a['states'][e['index']],a['states'][e['index']-1]) for e in events if e['kind']=='feedback')
    result=dict(reference_model='unlabelled_accumulator_v2',status='SAVED_ARRAY_PASS',name=d.name,mode=r['mode'],seed=seed,arm=arm,epochs=epochs,events=count,changed_recalled=sum(a['changed_recalled'] for a in rows),changed_total=2*epochs,retained=sum(a['retained'] for a in rows),retained_total=6*epochs,feedback_consistent=sum(a['feedback_consistent'] for a in rows),feedback_total=H*epochs,observed_sram_boundaries=len(ids),rows=rows);save(d/'AUDIT.json',result);return result
def run(mode,seed,arm,epochs,extra_queries=0,full_queries=False):
    assert not full_queries or extra_queries==8
    assert extra_queries in [0,1,8] and (not extra_queries or seed==SEEDS[0] and arm=='native' and epochs==2)
    assert mode in ['cpu','native'] and seed in SEEDS and arm in ARMS and epochs in EPOCHS;auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text());cut=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert not state['closed'] and not state.get('failure') and now()<cut
    frozen=sorted((S/'evidence').glob('FROZEN*.json'))[-1]
    for p,v in json.loads(frozen.read_text())['hashes'].items():assert sha(P/p)==v,('source drift',p)
    if mode=='native':
        assert passed('cpu',seed,arm,epochs,extra_queries,full_queries)
        if extra_queries:assert passed('native',seed,'native',12)
        elif seed==SEEDS[0]:
            if arm=='baseline':
                if epochs==2:assert passed('native',seed,arm,1)
                elif epochs==12:assert passed('native',seed,'native',12)
                else:assert epochs==1
            elif arm=='native':assert passed('native',seed,'baseline',2) if epochs==1 else passed('native',seed,arm,{2:1,4:2,12:4}[epochs])
            else:assert epochs==12 and passed('native',seed,'native',12)
        else:assert epochs==12 and all(passed('native',SEEDS[0],a,12) for a in ARMS)
    owners=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')];physical=[r for r in owners if r['mode']=='native'];assert len(physical)+(mode=='native')<=auth['maximum_physical_sessions'];assert sum(16+29*r['epochs']+r.get('extra_queries',0) for r in physical)+(16+29*epochs+extra_queries if mode=='native' else 0)<=auth['maximum_forwards']
    d=S/'raw'/(name(mode,seed,arm,epochs)+(f'_q{extra_queries}'+('full' if full_queries else '') if extra_queries else ''))
    if d.exists():d=S/'raw'/(d.name+'_attempt'+str(len(list((S/'raw').glob(d.name+'*')))+1))
    d.mkdir(parents=True,exist_ok=False);r=dict(full_queries=full_queries,extra_queries=extra_queries,mode=mode,seed=seed,arm=arm,epochs=epochs,start_utc=now().isoformat(),freeze_sha256=sha(frozen));save(d/'OWNER.json',r)
    def execute():
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            proc=subprocess.Popen(['taskset','-c','2','build/resident_adaptation',mode,str(F/(f's{seed}'+(f'_q{extra_queries}'+('full' if full_queries else '') if extra_queries else ''))),arm,str(epochs),str(d/'result')],cwd=P,stdout=out,stderr=err);r['pid']=proc.pid;save(d/'OWNER.json',r)
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
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='audit':print(json.dumps(audit(S/'raw'/sys.argv[2])))
    else:run(sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4]),int(sys.argv[5]) if len(sys.argv)>5 else 0,len(sys.argv)>6 and sys.argv[6]=='full')
