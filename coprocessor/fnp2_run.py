"""One staged FNP2 admission. Reject state mismatch before any expansion."""
import datetime as dt,fcntl,json,subprocess,sys
from pathlib import Path
import numpy as np
from coprocessor.run import P,digest,save,now
from pong.relearn.model import Reference
S=P/'studies/20261024-fnp2-contract';F=S/'fixtures02/learner'
def freeze():
    target=S/'evidence/FROZEN.json';assert not target.exists()
    files=[*F.parent.iterdir(),P/'coprocessor/fnp2_bench.cpp',P/'coprocessor/fnp2_prepare.py',P/'coprocessor/fnp2_run.py',P/'coprocessor/build_fnp2.sh',P/'src/readout_bench.cpp',P/'src/mmio_driver.hpp',P/'pong/relearn/model.py',P/'build/fnp2_contract',S/'AUTHORITY.json',S/'PROTOCOL.md']
    save(target,dict(status='FROZEN_BEFORE_EXECUTION',utc=now().isoformat(),hashes={str(p.relative_to(P)):digest(p) for p in files if p.is_file()}))
def passed(name):
    p=S/'raw'/name/'AUDIT.json';return p.exists() and json.loads(p.read_text())['status']=='INDEPENDENT_FNP2_PASS'
def replay(initial,x,learn):
    ref=Reference([0,0]+list(map(int,initial)),n=512,u=544,nw=32,classes=2);ys=[];rs=[]
    for i,v in enumerate(x):
        if learn:y=ref.forward(v,i%2)
        else:
            bits=sum(1<<int(k) for k in np.flatnonzero(v));y=[(w&bits).bit_count() for w in ref.weights]
        ys.append(y);rs.append(ref.ram[2:].copy())
    return np.asarray(ys,'<i4'),np.asarray(rs,'<u4')
def audit(d):
    rec=json.loads((d/'RECEIPT.json').read_text());assert rec['exit_code']==0
    for name,h in rec['hashes'].items():assert digest(d/name)==h
    n=rec['count'];initial=np.fromfile(F.with_suffix('.raw'),'<u4');x=np.fromfile(F.with_suffix('.inputs'),np.uint8).reshape(64,512)[:n];ys,states=replay(initial,x,rec['learn'])
    got=np.fromfile(d/'result.scores','<i4').reshape(n,544);assert np.array_equal(got,ys),'independent FNP2 score replay'
    assert np.array_equal(np.fromfile(d/'result.raw','<u4'),states[-1]),'independent FNP2 final state'
    if rec['mode']=='native':
        assert np.array_equal(np.fromfile(d/'result.snapshots','<u4').reshape(n,10880),states),'independent FNP2 every-step full record state'
        address,size,base,extent=map(int,np.fromfile(d/'result.storage','<u4'));assert size==43520 and base+256<=address and address+size<=base+extent,'card SRAM storage address'
    t=np.fromfile(d/'result.latency_ns','<i8');c=np.fromfile(d/'result.caller_ns','<i8');assert len(t)==len(c)==n*(5 if rec['mode']=='cpu' else 1) and np.all(t>0)
    if n==64:
        if rec['mode']=='cpu':t=t.reshape(5,n)[:,16:].reshape(-1);c=c.reshape(5,n)[:,16:].reshape(-1)
        else:t=t[16:];c=c[16:]
    result=dict(status='INDEPENDENT_FNP2_PASS',mode=rec['mode'],count=n,learn=rec['learn'],full_record_snapshots=n if rec['mode']=='native' else 0,median_us=float(np.median(t)/1000),median_caller_us=float(np.median(c)/1000),changed_words=int(np.sum(states[-1]!=initial)),host_DDR_used=False if rec['mode']=='native' else None)
    save(d/'AUDIT.json',result);return result
def run(mode,count,learn):
    assert mode in ['cpu','native'] and count in [8,64] and learn in [0,1]
    auth=json.loads((S/'AUTHORITY.json').read_text());st=json.loads((S/'STATE.json').read_text());assert not st['closed'] and not st.get('failure')
    assert (dt.datetime.fromisoformat(auth['scientific_cutoff_utc'])-now()).total_seconds()>125
    for name,h in json.loads((S/'evidence/FROZEN.json').read_text())['hashes'].items():assert digest(P/name)==h,name
    name=f'{mode}_n{count}_l{learn}';d=S/'raw'/name;assert not d.exists()
    if mode=='cpu':assert count==64
    else:
        assert passed(f'cpu_n64_l{learn}')
        if count==8 and learn:assert passed('native_n8_l0')
        if count==64:assert passed('native_n8_l1')
    prior=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')];physical=[p for p in prior if p['mode']=='native'];assert len(physical)+(mode=='native')<=4 and sum(p['count'] for p in physical)+(count if mode=='native' else 0)<=144
    rec=dict(mode=mode,count=count,learn=bool(learn),source_freeze_sha256=digest(S/'evidence/FROZEN.json'),admitted_utc=now().isoformat())
    def execute():
        d.mkdir(parents=True);save(d/'OWNER.json',rec)
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            p=subprocess.Popen(['taskset','-c','2','build/fnp2_contract',mode,str(F),str(d/'result'),str(count),str(learn)],cwd=P,stdout=out,stderr=err);rec['pid']=p.pid;save(d/'OWNER.json',rec)
            try:code=p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:code=p.wait(timeout=3)
                except subprocess.TimeoutExpired:p.kill();code=p.wait()
        rec.update(exit_code=code,finish_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1);rec['hashes']={p.name:digest(p) for p in d.iterdir() if p.is_file()};save(d/'RECEIPT.json',rec)
        try:
            assert code==0 and rec['device_free'],(d/'stderr').read_text();result=audit(d)
        except Exception:
            st['failure']=name;save(S/'STATE.json',st);raise
        st.setdefault('completed',[]).append(name);save(S/'STATE.json',st);return result
    if mode=='native':
        with open('/tmp/akd1500-research.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1;result=execute()
    else:result=execute()
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:run(sys.argv[1],*map(int,sys.argv[2:]))
