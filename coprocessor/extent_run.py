"""Prospectively bounded reference/scale admissions; preserve every attempt."""
from pathlib import Path
import argparse, datetime as dt, fcntl, hashlib, json, subprocess, sys
import numpy as np

P=Path(__file__).resolve().parents[1]; S=P/'studies/20261025-reference-scale'; F=S/'extent_fixtures'
MODEL=Path('/home/nicklomeli/akd1500-reference-vendor-20261025/assets/akidanet_vww_qat.fbz')
WIDTHS=[32,64,96,128]; SEEDS=[107501,107601,107701]
def now(): return dt.datetime.now(dt.timezone.utc)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,r): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2)+'\n')
def passed(p): return (p/'AUDIT.json').exists() and json.loads((p/'AUDIT.json').read_text())['status'] in ['SAVED_ARRAY_PASS','REFERENCE_SAVED_ARRAY_PASS']
def environment():
    paths=list(Path('/sys/devices/system/cpu/cpu2/cpufreq').glob('*freq'))+list(Path('/sys/class/thermal').glob('thermal_zone*/temp'))
    result={}
    for p in paths:
        if p.is_file():
            try:result[str(p)]=p.read_text().strip()
            except OSError as error:result[str(p)]=f'UNREADABLE_OS_ERROR_{error.errno}'
    return result
def audit_reference(d):
    r=json.loads((d/'RECEIPT.json').read_text()); assert r['exit_code']==0
    result=json.loads((d/'RESULT.json').read_text()); assert result['status']=='REFERENCE_EXACT_PASS'
    assert sha(d/'result.npz')==result['result_sha256']
    expected=np.load(S/'reference/expected.npy')
    with np.load(d/'result.npz',allow_pickle=False) as z:
        assert np.array_equal(z['outputs'],expected[np.arange(r['count'])%16])
        assert np.all(z['latency_ns']>0) and np.all(z['cpu_ns']>0)
        report=dict(status='REFERENCE_SAVED_ARRAY_PASS',mode=r['mode'],count=r['count'],
                    exact_output_bytes=int(z['outputs'].nbytes),median_us=float(np.median(z['latency_ns'][min(8,r['count']//2):])/1000),
                    median_process_cpu_us=float(np.median(z['cpu_ns'][min(8,r['count']//2):])/1000),
                    device_free=r['device_free'],model_setup_ms=result['setup_ns']/1e6)
    if r['physical']: assert r['device_free'] and result['passes']==[1] and result['external_memory_bytes']==0
    save(d/'AUDIT.json',report);return report
def execute(kind,mode,c=0,seed=0,arm=0,count=64,repeat=0):
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text())
    assert not state['closed'] and not state.get('failure'), 'study closed/failed; inspect before repair'
    cutoff=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']); assert now()<cutoff
    physical=mode=='native' or kind=='reference' and mode!='software'
    if kind=='reference':
        assert mode in ['software','Minimal','AllNps'] and count in [8,64] and repeat==0
        root=S/'reference_raw';name=f'{mode}_n{count}'
        if mode=='Minimal': assert passed(root/('software_n64' if count==8 else 'Minimal_n8'))
        if mode=='AllNps': assert passed(root/('Minimal_n64' if count==8 else 'AllNps_n8'))
        cmd=[sys.executable,'-m','coprocessor.deployment','run',str(MODEL),'--mode',mode,'--count',str(count),'--out',str(root/name)]
        fixture_paths=list((S/'reference').glob('*'))+[MODEL]
    else:
        assert mode in ['native','cpu-dense','cpu-sparse','cpu-parallel','cpu-blas1','cpu-blas4'] and c in WIDTHS and seed in SEEDS and arm in [0,1] and count in [8,64] and repeat in [0,1,2]
        assert passed(S/'reference_raw/AllNps_n64'), 'reference before experimental scale'
        root=S/'raw';cell=f's{c}_s{seed}';name=f'{mode}_{cell}_a{arm}_n{count}_r{repeat}'
        f=F/cell/'cell'; assert json.loads((f.parent/'BUILD.json').read_text())['status']=='VIRTUAL_COMPILER_PASS'
        if physical:
            for ctrl in ['cpu-dense','cpu-sparse','cpu-parallel','cpu-blas1','cpu-blas4']: assert passed(root/f'{ctrl}_{cell}_a{arm}_n64_r0')
            if c==32 and seed==107501 and count==64: assert passed(root/'native_s32_s107501_a1_n8_r0')
            if c!=32 and seed==107501:
                prev=WIDTHS[WIDTHS.index(c)-1]
                for a in [0,1]:assert passed(root/f'native_s{prev}_s107501_a{a}_n64_r0')
            if seed!=107501:
                for a in [0,1]:assert passed(root/f'native_s{c}_s107501_a{a}_n64_r0')
            if repeat: assert passed(root/f'native_{cell}_a{arm}_n64_r0')
        else: assert count==64
        reps=1 if physical else 3
        cmd=['taskset','-c','2-5' if mode in ['cpu-parallel','cpu-blas4'] else '2','build/extent_spatial',mode,str(f),str(root/name/'result'),str(count),str(arm),str(reps)]
        fixture_paths=list(f.parent.glob('*'))+[P/'build/extent_spatial', Path('/home/nicklomeli/akida-cc-venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-0f683016.so')]
    if kind=='reference': cmd=['taskset','-c','2',*cmd]
    owners=[json.loads(p.read_text()) for sub in ['raw','reference_raw'] for p in (S/sub).glob('*/OWNER.json')]
    owners += [json.loads(p.read_text()) for p in (S/'failed_attempts').glob('**/OWNER.json')]
    assert sum(r['physical'] for r in owners)+physical<=auth['maximum_physical_sessions']
    assert sum(r['count'] for r in owners if r['physical'])+(count if physical else 0)<=auth['maximum_forwards']
    d=root/name; d.mkdir(parents=True,exist_ok=False)
    sources=[P/x for x in ['coprocessor/extent_run.py','coprocessor/extent_prepare.py','coprocessor/extent_bench.cpp','coprocessor/extent_blas.hpp','coprocessor/extent_audit.py','coprocessor/deployment.py','coprocessor/resident_prepare.py','coprocessor/resident_run.py','coprocessor/resident_bench.cpp','src/mmio_driver.hpp','src/readout_bench.cpp','coprocessor/prepare.py','tools/prepare.py','tools/program_codec.py','metal/prepare.py']]
    sources += [S/'PROTOCOL.md',S/'EXTENT_AMENDMENT.md',S/'AUTHORITY.json',*fixture_paths]
    r=dict(kind=kind,mode=mode,count=count,physical=physical,cell=f's{c}_s{seed}',arm=arm,repeats=1 if physical or kind=='reference' else 3,
           admitted_utc=now().isoformat(),command=cmd,source_hashes={str(p):sha(p) for p in sources if p.is_file()},environment_before=environment())
    save(d/'OWNER.json',r)
    def go():
        for p,h in r['source_hashes'].items(): assert sha(Path(p))==h
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            import os
            env=dict(os.environ,AKD_OPENBLAS_LIBRARY='/home/nicklomeli/akida-cc-venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-0f683016.so')
            proc=subprocess.Popen(cmd,cwd=P,stdout=out,stderr=err,env=env);r['pid']=proc.pid;save(d/'OWNER.json',r)
            try: code=proc.wait(timeout=min(120,(cutoff-now()).total_seconds()))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:code=proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
        r.update(exit_code=code,finished_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,environment_after=environment())
        r['hashes']={p.name:sha(p) for p in d.iterdir() if p.is_file()};save(d/'RECEIPT.json',r)
        try:
            if kind=='reference': result=audit_reference(d)
            else:
                from coprocessor.extent_audit import audit_directory
                result=audit_directory(d)
        except Exception:
            state['failure']=str(d.relative_to(S));save(S/'STATE.json',state);raise
        state['completed'].append(str(d.relative_to(S)));save(S/'STATE.json',state)
        return result
    if physical:
        with open('/tmp/akd1500-research.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
            result=go()
    else:result=go()
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['reference','scale']);p.add_argument('mode');p.add_argument('--filters',type=int,default=32);p.add_argument('--seed',type=int,default=107501);p.add_argument('--arm',type=int,default=1);p.add_argument('--count',type=int,default=64);p.add_argument('--repeat',type=int,default=0)
    a=p.parse_args();execute(a.kind,a.mode,a.filters,a.seed,a.arm,a.count,a.repeat)
