"""Prospective transport-only follow-up, one audited execution per invocation."""
import datetime as dt
import fcntl
import json
import subprocess
import sys
from coprocessor.run import P,S as BASE,F,digest,save,audit,passed,now

S=P/'studies/20261024-pcie-transport'
def freeze():
    target=S/'evidence/FROZEN.json';assert not target.exists()
    paths=[*F.glob('*'),P/'coprocessor/transport.cpp',P/'coprocessor/bench.cpp',P/'coprocessor/run.py',P/'coprocessor/transport_run.py',P/'src/readout_bench.cpp',P/'src/mmio_driver.hpp',P/'build/pcie_transport',S/'PROTOCOL.md',S/'AUTHORITY.json']
    save(target,dict(status='FROZEN_BEFORE_EXECUTION',utc=now().isoformat(),hashes={str(p.relative_to(P)):digest(p) for p in paths if p.is_file()}))
def run(profile,learn,count,repeat):
    assert profile in [-1,0,1,2,3] and learn in [0,1] and count in [0,8,96] and repeat in [0,1,2]
    a=json.loads((S/'AUTHORITY.json').read_text());st=json.loads((S/'STATE.json').read_text());assert not st['closed'] and not st.get('failure')
    assert (dt.datetime.fromisoformat(a['scientific_cutoff_utc'])-now()).total_seconds()>125
    for name,h in json.loads((S/'evidence/FROZEN.json').read_text())['hashes'].items():assert digest(P/name)==h,('source drift',name)
    assert passed('native_rgb32_c128_l1_b8_n96_r2') and passed('native_rgb32_c128_l0_b8_n96_r2')
    name='copy_gate' if profile==-1 else f'p{profile}_l{learn}_n{count}_r{repeat}'
    def good(n):
        p=S/'raw'/n/'AUDIT.json';return p.exists() and json.loads(p.read_text())['status'] in ['INDEPENDENT_SAVED_ARRAY_PASS','TRANSPORT_COPY_PASS']
    if profile==-1:assert (learn,count,repeat)==(0,0,0)
    else:
        assert good('copy_gate')
        if count==8:assert profile in [1,2,3] and learn==1 and repeat==0
        else:
            assert count==96
            if profile:assert good(f'p{profile}_l1_n8_r0')
            if repeat:
                assert all(good(f'p{p}_l{l}_n96_r0') for p in range(4) for l in [0,1])
            elif profile:assert good(f'p{profile-1}_l{learn}_n96_r0')
    d=S/'raw'/name;assert not d.exists()
    prior=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')]
    assert len(prior)<a['maximum_physical_sessions'] and sum(p['count'] for p in prior)+count<=a['maximum_forwards']
    rec=dict(name=name,mode='native',cell='rgb32_c128',learn=bool(learn),batch=1 if count==8 else 8,count=count,repeat=repeat,repeats=1,profile=profile,source_freeze_sha256=digest(S/'evidence/FROZEN.json'),admitted_utc=now().isoformat())
    with open('/tmp/akd1500-research.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
        d.mkdir(parents=True);save(d/'OWNER.json',rec)
        args=['taskset','-c','2','build/pcie_transport','transport' if profile==-1 else 'native',str(F/'rgb32_c128'),str(d/'result'),str(count or 8),str(learn),str(rec['batch']),'1',str(max(0,profile))]
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            p=subprocess.Popen(args,cwd=P,stdout=out,stderr=err);rec['pid']=p.pid;save(d/'OWNER.json',rec)
            try:code=p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:code=p.wait(timeout=3)
                except subprocess.TimeoutExpired:p.kill();code=p.wait()
        rec.update(exit_code=code,finish_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1)
        rec['hashes']={p.name:digest(p) for p in d.iterdir() if p.is_file()};save(d/'RECEIPT.json',rec)
        try:
            assert code==0 and rec['device_free'],(d/'stderr').read_text()
            if profile==-1:
                summary=json.loads((d/'stdout').read_text());assert summary['forwards']==0 and summary['bytes_verified_per_direction']==3712
                result=dict(status='TRANSPORT_COPY_PASS',**rec);save(d/'AUDIT.json',result)
            else:result=audit(d)
        except Exception:
            st['failure']=name;save(S/'STATE.json',st);raise
    st.setdefault('completed',[]).append(name);save(S/'STATE.json',st);print(json.dumps(result),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:run(*map(int,sys.argv[1:]))
