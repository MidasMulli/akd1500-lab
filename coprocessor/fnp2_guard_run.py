"""Scale verified FNP2 records in fixed, independently gated capacity cells."""
import datetime as dt,fcntl,json,subprocess,sys
import numpy as np
from coprocessor.run import P,digest,save,now
from pong.relearn.model import Reference
S=P/'studies/20261024-fnp2-guard'
WIDTHS=[11264]
def freeze():
    target=S/'evidence/FROZEN.json';assert not target.exists()
    files=[* (S/'fixtures01').rglob('*'),P/'coprocessor/fnp2_guard.cpp',P/'coprocessor/fnp2_guard_prepare.py',P/'coprocessor/fnp2_guard_run.py',P/'coprocessor/record_write_guard.hpp',P/'coprocessor/record_range.hpp',P/'src/readout_bench.cpp',P/'src/mmio_driver.hpp',P/'pong/relearn/model.py',P/'build/fnp2_guard',S/'AUTHORITY.json',S/'PROTOCOL.md']
    save(target,dict(status='FROZEN_BEFORE_EXECUTION',utc=now().isoformat(),hashes={str(p.relative_to(P)):digest(p) for p in files if p.is_file()}))
def passed(name):
    p=S/'raw'/name/'AUDIT.json';return p.exists() and json.loads(p.read_text())['status']=='INDEPENDENT_FNP2_CAPACITY_PASS'
def trajectory(initial,x,u,learn):
    ref=Reference([0,0]+list(map(int,initial)),n=512,u=u,nw=32,classes=2);ys=[];states=[]
    for i,v in enumerate(x):
        if learn:y=ref.forward(v,i%2)
        else:
            bits=sum(1<<int(k) for k in np.flatnonzero(v));y=[(w&bits).bit_count() for w in ref.weights]
        ys.append(y);states.append(ref.ram[2:].copy())
    return np.asarray(ys,'<i4'),np.asarray(states,'<u4')
def audit(d):
    rec=json.loads((d/'RECEIPT.json').read_text());assert rec['exit_code']==0
    for name,h in rec['hashes'].items():assert digest(d/name)==h
    u,n=rec['units'],rec['count'];f=S/'fixtures01'/f'u{u}'/'learner';initial=np.fromfile(f.with_suffix('.raw'),'<u4');x=np.fromfile(f.with_suffix('.inputs'),np.uint8).reshape(64,512)[:n];ys,states=trajectory(initial,x,u,rec['learn'])
    assert np.array_equal(np.fromfile(d/'result.scores','<i4').reshape(n,u),ys),'independent capacity potentials'
    assert np.array_equal(np.fromfile(d/'result.raw','<u4'),states[-1]),'independent capacity final state'
    if rec['mode']=='native':
        assert np.array_equal(np.fromfile(d/'result.snapshots','<u4').reshape(n,u*20),states),'independent capacity every-step state'
        address,size,base,extent=map(int,np.fromfile(d/'result.storage','<u4'));assert size==u*80 and base+256<=address and address+size<=base+extent,'capacity storage domain'
        summary=json.loads((d/'stdout').read_text());assert summary['records']==u and summary['host_DDR_used'] is False and summary['memory_usage']<extent
        assert summary['guarded_host_record_write_attempts']==0 and summary['guarded_host_write_calls']>0,'guarded writes must exclude the full record buffer'
    t=np.fromfile(d/'result.latency_ns','<i8');c=np.fromfile(d/'result.caller_ns','<i8');assert len(t)==len(c)==n*(5 if rec['mode']=='cpu' else 1) and np.all(t>0)
    result=dict(status='INDEPENDENT_FNP2_CAPACITY_PASS',units=u,mode=rec['mode'],count=n,learn=rec['learn'],complete_snapshots=n if rec['mode']=='native' else 0,changed_words=int(np.sum(states[-1]!=initial)),median_us=float(np.median(t)/1000),median_caller_us=float(np.median(c)/1000),timing='unwarmed eight-event screen, not performance qualification')
    save(d/'AUDIT.json',result);return result
def run(mode,u,learn):
    assert mode in ['cpu','native'] and u in WIDTHS and learn in [0,1]
    a=json.loads((S/'AUTHORITY.json').read_text());st=json.loads((S/'STATE.json').read_text());assert not st['closed'] and not st.get('failure')
    assert (dt.datetime.fromisoformat(a['scientific_cutoff_utc'])-now()).total_seconds()>125
    for name,h in json.loads((S/'evidence/FROZEN.json').read_text())['hashes'].items():assert digest(P/name)==h,name
    name=f'{mode}_u{u}_l{learn}';d=S/'raw'/name;assert not d.exists()
    if mode=='native':
        assert passed(f'cpu_u{u}_l{learn}')
        if learn:assert passed(f'native_u{u}_l0')
        elif u!=11264:assert passed(f'native_u{WIDTHS[WIDTHS.index(u)-1]}_l1')
        contract=P/'studies/20261024-fnp2-contract/raw/native_n64_l1/AUDIT.json';assert json.loads(contract.read_text())['status']=='INDEPENDENT_FNP2_PASS'
    prior=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')];physical=[p for p in prior if p['mode']=='native'];assert len(physical)+(mode=='native')<=2 and len(physical)*8+(8 if mode=='native' else 0)<=16
    rec=dict(mode=mode,units=u,count=8,learn=bool(learn),source_freeze_sha256=digest(S/'evidence/FROZEN.json'),admitted_utc=now().isoformat())
    def execute():
        d.mkdir(parents=True);save(d/'OWNER.json',rec)
        f=S/'fixtures01'/f'u{u}'/'learner'
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            p=subprocess.Popen(['taskset','-c','2','build/fnp2_guard',mode,str(f),str(d/'result'),'8',str(learn)],cwd=P,stdout=out,stderr=err);rec['pid']=p.pid;save(d/'OWNER.json',rec)
            try:code=p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:code=p.wait(timeout=3)
                except subprocess.TimeoutExpired:p.kill();code=p.wait()
        rec.update(exit_code=code,finish_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1);rec['hashes']={p.name:digest(p) for p in d.iterdir() if p.is_file()};save(d/'RECEIPT.json',rec)
        try:
            assert code==0 and rec['device_free'],(d/'stderr').read_text();result=audit(d)
        except Exception:st['failure']=name;save(S/'STATE.json',st);raise
        st.setdefault('completed',[]).append(name);save(S/'STATE.json',st);return result
    if mode=='native':
        with open('/tmp/akd1500-research.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1;result=execute()
    else:result=execute()
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:run(sys.argv[1],*map(int,sys.argv[2:]))
