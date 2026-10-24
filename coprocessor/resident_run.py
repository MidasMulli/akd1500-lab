"""Fresh bounded study; saved-array audit gates each physical expansion."""
from pathlib import Path
import argparse,datetime as dt,fcntl,hashlib,json,subprocess
import numpy as np
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-resident-spatial';F=S/'fixtures03'
def now():return dt.datetime.now(dt.timezone.utc)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,r):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2)+'\n')
def freeze():
    assert not (S/'evidence/FROZEN.json').exists()
    paths=[*F.glob('*/*'),P/'coprocessor/resident_prepare.py',P/'coprocessor/resident_run.py',P/'coprocessor/resident_bench.cpp',P/'coprocessor/build_resident.sh',P/'src/mmio_driver.hpp',P/'src/readout_bench.cpp',P/'tools/prepare.py',P/'tools/program_codec.py',P/'coprocessor/prepare.py',P/'metal/prepare.py',P/'build/resident_spatial',S/'PROTOCOL.md',S/'AUTHORITY.json']
    save(S/'evidence/FROZEN.json',dict(utc=now().isoformat(),hashes={str(p.relative_to(P)):digest(p) for p in paths if p.is_file()}))
def passed(name):
    p=S/'raw'/name/'AUDIT.json';return p.exists() and json.loads(p.read_text())['status']=='SAVED_ARRAY_PASS'
def audit(d):
    r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0
    for name,h in r['hashes'].items():assert digest(d/name)==h
    f=F/r['cell']/'cell';n=r['count'];initial=np.fromfile(f.with_suffix('.ram'),'<u4')
    y=np.fromfile(d/'result.scores','<i4').reshape(n,128);ram=np.fromfile(d/'result.ram','<u4');snaps=np.fromfile(d/'result.snapshots','<u4').reshape(n,1026)
    assert len(initial)==1026 and np.array_equal(ram,snaps[-1]);assert np.all(snaps[:8]==initial)
    if r['arm'] and n>8:assert np.any(snaps[39]!=initial),'no native learning'
    if n>40:assert np.all(snaps[40:]==snaps[39]),'query changed learned memory'
    if not r['arm']:assert np.all(snaps==initial),'frozen changed memory'
    if r['mode']=='native':
        for cpu in ['cpu-dense','cpu-sparse','cpu-parallel']:
            p=S/'raw'/f'{cpu}_{r["cell"]}_a{r["arm"]}_n64_r0';assert passed(p.name)
            assert np.array_equal(y,np.fromfile(p/'result.scores','<i4').reshape(64,128)[:n]),'saved outputs / independent CPU'
            assert np.array_equal(snaps,np.fromfile(p/'result.snapshots','<u4').reshape(64,1026)[:n]),'complete per-step memory / CPU'
        packets=int(f.with_suffix('.layout').read_text().split()[0]);assert r['device_free']
        for j in range(packets):
            expected=np.fromfile(f.with_suffix(f'.packet{j}.ram'),'<u4')
            for tag in ['initial','final']:assert np.array_equal(np.fromfile(d/f'result.packet{j}.{tag}','<u4'),expected)
    else:
        for cpu in ['cpu-dense','cpu-sparse','cpu-parallel']:
            p=S/'raw'/f'{cpu}_{r["cell"]}_a{r["arm"]}_n64_r0'
            if p!=d and passed(p.name):
                assert np.array_equal(y,np.fromfile(p/'result.scores','<i4').reshape(64,128));assert np.array_equal(snaps,np.fromfile(p/'result.snapshots','<u4').reshape(64,1026))
    expected=np.fromfile(f.with_suffix('.expected'),'<i4').reshape(64,128)
    assert np.array_equal(y[:8],expected[:8])
    if not r['arm']:assert np.array_equal(y,expected[:n])
    times=np.fromfile(d/'result.latency_ns','<i8').reshape(r['repeats'],n);callers=np.fromfile(d/'result.caller_ns','<i8').reshape(r['repeats'],n)
    assert np.all(times>0) and np.all(callers>0)
    skip=8 if n>8 else 0
    # Compare the same mixed schedule; teacher-dependent native updates included.
    a=dict(status='SAVED_ARRAY_PASS',cell=r['cell'],mode=r['mode'],arm=r['arm'],count=n,repeats=r['repeats'],median_us=float(np.median(times[:,skip:])/1000),p95_us=float(np.percentile(times[:,skip:],95)/1000),median_process_cpu_us=float(np.median(callers[:,skip:])/1000),changed_final_ram_words=int(np.sum(ram!=initial)),initial_and_frozen_query_RAM_exact=True)
    observer=np.fromfile(d/'result.observer_ns','<i8')
    if len(observer):a['median_observer_us_outside_timing']=float(np.median(observer)/1000)
    save(d/'AUDIT.json',a);return a
def run(mode,c,seed,arm,count,repeat):
    assert mode in ['native','cpu-dense','cpu-sparse','cpu-parallel'] and c in [8,128,256,512] and seed in [107501,107601,107701] and arm in [0,1] and count in [8,40,64] and repeat in [0,1,2]
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text());assert not state['closed'] and not state.get('failure')
    cutoff=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert now()<cutoff
    for name,h in json.loads((S/'evidence/FROZEN.json').read_text())['hashes'].items():assert digest(P/name)==h,('source drift',name)
    cell=f'c{c}_s{seed}';f=F/cell/'cell';assert json.loads((f.parent/'BUILD.json').read_text())['status']=='VIRTUAL_COMPILER_PASS'
    name=f'{mode}_{cell}_a{arm}_n{count}_r{repeat}';d=S/'raw'/name;assert not d.exists()
    if mode=='native':
        for cpu in ['cpu-dense','cpu-sparse','cpu-parallel']:assert passed(f'{cpu}_{cell}_a{arm}_n64_r0'),'CPU controls first'
        if c==8 and seed==107501:
            if count==40:assert passed('native_c8_s107501_a1_n8_r0')
            if count==64:assert passed('native_c8_s107501_a1_n40_r0')
        else:
            assert count==64
            prev={128:8,256:128,512:256}[c] if seed==107501 else 512
            for a in [0,1]:assert passed(f'native_c{prev}_s107501_a{a}_n64_r0'),'prior stage both arms'
        if repeat:assert passed(f'native_{cell}_a{arm}_n64_r0')
    else:assert count==64 and repeat==0
    prior=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')];physical=[q for q in prior if q['mode']=='native']
    assert len(physical)+(mode=='native')<=auth['maximum_physical_sessions'];assert sum(q['count'] for q in physical)+(count if mode=='native' else 0)<=auth['maximum_forwards']
    d.mkdir(parents=True);r=dict(cell=cell,mode=mode,arm=arm,count=count,repeat=repeat,repeats=1 if mode=='native' else 5,admitted_utc=now().isoformat(),source_freeze_sha256=digest(S/'evidence/FROZEN.json'));save(d/'OWNER.json',r)
    def execute():
        affinity='2-5' if mode=='cpu-parallel' else '2';cmd=['taskset','-c',affinity,'build/resident_spatial',mode,str(f),str(d/'result'),str(count),str(arm),str(r['repeats'])]
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            p=subprocess.Popen(cmd,cwd=P,stdout=out,stderr=err);r['pid']=p.pid;save(d/'OWNER.json',r)
            try:code=p.wait(timeout=min(120,(cutoff-now()).total_seconds()))
            except subprocess.TimeoutExpired:
                p.terminate()
                try:code=p.wait(timeout=3)
                except subprocess.TimeoutExpired:p.kill();code=p.wait()
        r.update(exit_code=code,finish_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1)
        r['hashes']={p.name:digest(p) for p in d.iterdir() if p.is_file()};save(d/'RECEIPT.json',r)
        try:a=audit(d)
        except Exception:
            state['failure']=name;save(S/'STATE.json',state);raise
        state.setdefault('completed',[]).append(name);save(S/'STATE.json',state);return a
    if mode=='native':
        with open('/tmp/akd1500-research.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
            result=execute()
    else:result=execute()
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    import sys
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='audit':print(json.dumps(audit(S/'raw'/sys.argv[2])))
    else:run(sys.argv[1],*map(int,sys.argv[2:]))
