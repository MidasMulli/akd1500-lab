"""Finite, locked host-service cells; original studies remain closed."""
from pathlib import Path
import argparse,datetime as dt,fcntl,hashlib,json,os,subprocess,sys
from coprocessor.service_audit import S,P,sha,save,audit

MODES=['native','cpu-dense','cpu-sparse','cpu-parallel','cpu-blas1','cpu-blas4']
def now():return dt.datetime.now(dt.timezone.utc)
def good(name):
    p=S/'raw'/name/'AUDIT.json'
    return p.exists() and json.loads(p.read_text())['status']=='EXACT_SERVICE_PASS'
def fixture(side,seed,contract):return f's{side}_s{seed}_{contract}'
def environment():
    result={'loadavg':os.getloadavg()}
    for p in [*Path('/sys/devices/system/cpu/cpu2/cpufreq').glob('*freq'),*Path('/sys/class/thermal').glob('thermal_zone*/temp')]:
        if p.is_file():
            try:result[str(p)]=p.read_text().strip()
            except OSError as e:result[str(p)]=f'OS_ERROR_{e.errno}'
    return result
def cell(mode,side,seed,contract,arm,count,repeat,sleep,batch,profile,full):
    return f'{mode}_{fixture(side,seed,contract)}_a{arm}_n{count}_r{repeat}_w{sleep}_b{batch}_p{profile}_f{full}'
def freeze():
    paths=[P/x for x in ['coprocessor/service_bench.cpp','coprocessor/service_prepare.py','coprocessor/service_run.py','coprocessor/service_audit.py','coprocessor/extent_blas.hpp','coprocessor/resident_archive.py','coprocessor/extent_audit.py','src/readout_bench.cpp','src/mmio_driver.hpp','build/host_service']]
    paths += [p for sub in ['fixtures','reference'] for p in (S/sub).rglob('*') if p.is_file()]
    paths += [S/'PROTOCOL.md',S/'AUTHORITY.json']
    target=S/'evidence/FROZEN.json';assert not target.exists(),'freeze is append-only; use a prospective amendment for repair'
    private=[P/'build/runtime-ec24d995d1b6fc38b4f0f7a9.a',Path('/home/nicklomeli/akida-cc-venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-0f683016.so')]
    save(target,{'utc':now().isoformat(),'status':'FROZEN_BEFORE_DEVICE_EXECUTION','hashes':{str(p.relative_to(P)):sha(p) for p in paths},'external_runtime_hashes':{str(p):sha(p) for p in private}})
def run(a):
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text())
    assert not state['closed'] and not state['failure'],'closed/failed; preserve attempt and prospectively repair'
    cutoff=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert (cutoff-now()).total_seconds()>125
    assert a.mode in MODES and a.side in auth['sides'] and a.seed in auth['seeds'] and a.contract in auth['output_contracts']
    assert a.count in [8,64] and a.arm in [0,1] and a.repeat in [0,1,2] and a.sleep in auth['sleep_us'] and a.batch in auth['batches'] and a.profile in auth['transport_profiles'] and a.full in [0,1]
    frozen=json.loads((S/'evidence/FROZEN.json').read_text())
    for p,h in frozen['hashes'].items():assert sha(P/p)==h,('source drift',p)
    for p,h in frozen['external_runtime_hashes'].items():assert sha(Path(p))==h,('external runtime drift',p)
    name=cell(a.mode,a.side,a.seed,a.contract,a.arm,a.count,a.repeat,a.sleep,a.batch,a.profile,a.full)
    native=a.mode=='native'
    if native:
        for mode in MODES[1:]:assert good(cell(mode,a.side,a.seed,a.contract,a.arm,64,0,0,1,1,1)),'all strong CPU controls first'
        if a.count==64:assert good(cell('native',a.side,a.seed,a.contract,0,8,0,0,1,1,1)),'read-only smoke first'
        if a.side==64:assert all(good(cell('native',32,107501,'potentials',arm,64,0,0,1,1,1)) for arm in [0,1]),'side32 pair first'
        if a.contract=='mask':assert all(good(cell('native',a.side,a.seed,'potentials',arm,64,0,0,1,1,1)) for arm in [0,1]),'potential pair before compact output'
        if a.sleep or a.batch!=1 or a.profile!=1 or not a.full:
            assert all(good(cell('native',a.side,a.seed,a.contract,arm,64,0,0,1,1,1)) for arm in [0,1]),'baseline pair before transport/completion changes'
        if a.seed!=107501 or a.repeat:
            assert all(good(cell('native',64,107501,'potentials',arm,64,0,0,1,1,1)) for arm in [0,1]),'primary correctness before confirmation'
    owners=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')]
    assert sum(v['physical'] for v in owners)+native<=auth['maximum_physical_sessions']
    assert sum(v['count'] for v in owners if v['physical'])+(a.count if native else 0)<=auth['maximum_forwards']
    d=S/'raw'/name;d.mkdir(parents=True,exist_ok=False)
    args=['taskset','-c','2-5' if a.mode in ['cpu-parallel','cpu-blas4'] else '2','build/host_service',a.mode,str(S/'fixtures'/fixture(a.side,a.seed,a.contract)/'cell'),str(d/'result'),str(a.count),str(a.arm),'1',str(a.sleep),str(a.batch),str(a.profile),str(a.full)]
    r=dict(mode=a.mode,fixture=fixture(a.side,a.seed,a.contract),physical=native,side=a.side,seed=a.seed,contract=a.contract,arm=a.arm,count=a.count,repeat=a.repeat,sleep_us=a.sleep,batch=a.batch,profile=a.profile,full_audit=bool(a.full),repeats=1,admitted_utc=now().isoformat(),command=args,source_freeze_sha256=sha(S/'evidence/FROZEN.json'))
    r['environment_before']=environment();save(d/'OWNER.json',r)
    lock=None
    try:
        if native:
            lock=open('/tmp/akd1500-research.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            env=dict(os.environ,AKD_OPENBLAS_LIBRARY='/home/nicklomeli/akida-cc-venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-0f683016.so')
            proc=subprocess.Popen(args,cwd=P,stdout=out,stderr=err,env=env);r['pid']=proc.pid;save(d/'OWNER.json',r)
            try:code=proc.wait(timeout=min(120,(cutoff-now()).total_seconds()))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:code=proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
        r.update(exit_code=code,finish_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,environment_after=environment())
        r['hashes']={p.name:sha(p) for p in d.iterdir() if p.is_file()};save(d/'RECEIPT.json',r)
        result=audit(d);state['completed'].append(name);save(S/'STATE.json',state);print(json.dumps(result),flush=True)
    except Exception:
        state['failure']=name;save(S/'STATE.json',state);raise
    finally:
        if lock is not None:lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=MODES+['freeze']);p.add_argument('--side',type=int,default=32);p.add_argument('--seed',type=int,default=107501);p.add_argument('--contract',choices=['potentials','mask'],default='potentials');p.add_argument('--arm',type=int,default=1);p.add_argument('--count',type=int,default=64);p.add_argument('--repeat',type=int,default=0);p.add_argument('--sleep',type=int,default=0);p.add_argument('--batch',type=int,default=1);p.add_argument('--profile',type=int,default=1);p.add_argument('--full',type=int,default=1);a=p.parse_args();freeze() if a.mode=='freeze' else run(a)
