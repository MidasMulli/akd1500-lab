"""One owned zero-forward allocation/cleanup diagnostic; no learning retry."""
import datetime as dt
import fcntl
import json
import subprocess
import numpy as np
from capability.design import P
from metal.run import digest
from capability.run import save
S=P/'studies/20261024-queue-allocation'
F=P/'studies/20261024-native-primitives/fixtures-quiet/u512_s102101'


def freeze():
    files=[P/p for p in ['capability/allocation.py','capability/build_allocation.sh','src/queue_allocation.cpp','src/readout_bench.cpp','src/mmio_driver.hpp','build/queue_allocation']]
    files+=list(F.parent.glob(F.name+'.*'))+[S/'PROTOCOL.md',S/'AUTHORITY.json',S/'evidence/SOURCE_AUDIT.json']
    target=S/'evidence/FROZEN.json';assert not target.exists();save(target,dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),hashes={str(f.relative_to(P)):digest(f) for f in files}))


def run():
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text());assert not state['research_admissions_closed']
    now=dt.datetime.now(dt.timezone.utc);cutoff=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert (cutoff-now).total_seconds()>10
    for p,h in json.loads((S/'evidence/FROZEN.json').read_text())['hashes'].items():assert digest(P/p)==h,p
    d=S/'raw/allocation01'
    with open('/tmp/akd1500-research.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1;d.mkdir(parents=True,exist_ok=False)
        r=dict(status='RUNNING',physical_forwards=0,source_freeze_sha256=digest(S/'evidence/FROZEN.json'),start_utc=now.isoformat());save(d/'OWNER.json',r)
        with (d/'native.stdout').open('w') as out,(d/'native.stderr').open('w') as err:
            p=subprocess.Popen(['taskset','-c','2','build/queue_allocation',str(F),str(d/'native')],cwd=P,stdout=out,stderr=err);r['native_pid']=p.pid;save(d/'OWNER.json',r)
            try:code=p.wait(timeout=min(120,int((cutoff-now).total_seconds())))
            except subprocess.TimeoutExpired:
                p.terminate()
                try:code=p.wait(timeout=3)
                except subprocess.TimeoutExpired:p.kill();code=p.wait()
        r.update(status='PASS' if code==0 else 'FAILED_PRESERVED',exit_code=code,device_released=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,finish_utc=dt.datetime.now(dt.timezone.utc).isoformat());r['hashes']={p.name:digest(p) for p in d.iterdir() if p.is_file()};save(d/'RECEIPT.json',r)
        state.update(research_admissions_closed=True,stage='CLOSED_PASS' if code==0 else 'CLOSED_FAIL',owned_jobs_remaining=0);save(S/'STATE.json',state)
        assert code==0 and r['device_released'],(d/'native.stderr').read_text()
        pairs=np.fromfile(d/'native.requested_effective','<u4').reshape(6,2);assert np.array_equal(pairs,np.array([[1,1],[9,9],[33,33],[65,65],[128,128],[129,128]],np.uint32))
        expected=[np.fromfile(F.parent/f'{F.name}.bank{j}.ram','<u4') for j in range(3)]+[np.fromfile(F.parent/f'{F.name}.fixed{j}.ram','<u4') for j in range(3)]
        for j,w in enumerate(expected):
            for suffix in ['initial','final']:assert np.array_equal(w,np.fromfile(d/f'native.packet{j}.{suffix}','<u4')),'unchanged complete native memory'
        result=dict(status='INDEPENDENT_ALLOCATION_CLEANUP_PASS',physical_sessions=1,physical_forwards=0,requested_effective=pairs.tolist(),parameter_memory_unchanged=True,device_released=True)
        save(d/'AUDIT.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    import sys
    freeze() if sys.argv[1]=='freeze' else run()
