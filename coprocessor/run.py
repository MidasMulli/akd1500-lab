"""One bounded admission at a time. Independent saved-array checks gate expansion."""
from pathlib import Path
import datetime as dt
import fcntl
import hashlib
import json
import subprocess
import sys
import numpy as np

P=Path(__file__).resolve().parents[1]
S=P/'studies/20261024-pcie-fabric'
F=S/'fixtures03'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def now():return dt.datetime.now(dt.timezone.utc)
def freeze():
    target=S/'evidence/FROZEN.json';assert not target.exists()
    files=[*F.glob('*'),* (P/'coprocessor').glob('*.py'),P/'coprocessor/bench.cpp',P/'coprocessor/build.sh',P/'src/readout_bench.cpp',P/'src/mmio_driver.hpp',P/'build/pcie_fabric',S/'PROTOCOL.md',S/'AUTHORITY.json']
    save(target,dict(utc=now().isoformat(),status='FROZEN_BEFORE_EXECUTION',hashes={str(p.relative_to(P)):digest(p) for p in files if p.is_file()}))
def check():
    a=json.loads((S/'AUTHORITY.json').read_text());assert now()<dt.datetime.fromisoformat(a['scientific_cutoff_utc'])
    st=json.loads((S/'STATE.json').read_text());assert not st['closed'] and not st.get('failure')
    for name,h in json.loads((S/'evidence/FROZEN.json').read_text())['hashes'].items():assert digest(P/name)==h,('freeze drift',name)
    return a,st
def passed(name):
    p=S/'raw'/name/'AUDIT.json';return p.exists() and json.loads(p.read_text())['status']=='INDEPENDENT_SAVED_ARRAY_PASS'
def audit(d):
    rec=json.loads((d/'RECEIPT.json').read_text());assert rec['exit_code']==0
    for name,h in rec['hashes'].items():assert digest(d/name)==h
    f=F/rec['cell'];n=rec['count'];learn=rec['learn']
    scores=np.fromfile(d/'result.scores','<i4').reshape(n,128);ram=np.fromfile(d/'result.ram','<u4');initial=np.fromfile(f.with_suffix('.ram'),'<u4')
    assert ram.shape==initial.shape==(2562,)
    if rec['mode'].startswith('cpu'):
        other='cpu-sparse' if rec['mode']=='cpu-dense' else 'cpu-dense'
        path=S/'raw'/f'{other}_{rec["cell"]}_l{int(learn)}'
        if path.exists():
            assert np.array_equal(scores,np.fromfile(path/'result.scores','<i4').reshape(n,128))
            assert np.array_equal(ram,np.fromfile(path/'result.ram','<u4'))
    else:
        for mode in ['cpu-dense','cpu-sparse']:
            p=S/'raw'/f'{mode}_{rec["cell"]}_l{int(learn)}'
            assert passed(p.name)
            expected=np.fromfile(p/'result.scores','<i4').reshape(96,128)[:n]
            assert np.array_equal(scores,expected),'native saved scores / both CPU controls'
            if n==96:assert np.array_equal(ram,np.fromfile(p/'result.ram','<u4')),'final full RAM / CPU'
        snaps=np.fromfile(d/'result.snapshots','<u4').reshape(n//rec['batch'],2562)
        assert np.array_equal(snaps[-1],ram)
    if not learn:
        assert np.array_equal(ram,initial),'frozen RAM changed'
        assert np.array_equal(scores,np.fromfile(f.with_suffix('.expected'),'<i4').reshape(96,128)[:n])
    else:assert np.any(ram!=initial),'no native or CPU learning state change'
    times=np.fromfile(d/'result.latency_ns','<i8');callers=np.fromfile(d/'result.caller_ns','<i8')
    assert len(times)==len(callers) and np.all(times>0) and np.all(callers>0)
    if rec['mode'].startswith('cpu'):
        times=times.reshape(rec['repeats'],n)[:,16:].reshape(-1)
        callers=callers.reshape(rec['repeats'],n)[:,16:].reshape(-1)
    elif n>16:
        times=times[16//rec['batch']:]/rec['batch'];callers=callers[16//rec['batch']:]/rec['batch']
    else:times=times/rec['batch'];callers=callers/rec['batch']
    report=dict(status='INDEPENDENT_SAVED_ARRAY_PASS',name=d.name,count=n,mode=rec['mode'],learn=learn,batch=rec['batch'],changed_ram_words=int(np.sum(ram!=initial)),median_us=float(np.median(times)/1000),p95_us=float(np.percentile(times,95)/1000),median_caller_us=float(np.median(callers)/1000),warmup_excluded=n>16,physical_device_opened=rec['mode']=='native')
    save(d/'AUDIT.json',report);return report
def run(mode,width,learn,batch,count,repeat):
    auth,state=check();assert width in [16,64,128] and learn in [0,1] and batch in [1,8] and repeat in [0,1,2] and count in [8,96]
    assert mode in ['cpu-dense','cpu-sparse','native']
    cell=f'rgb32_c{width}';name=f'{mode}_{cell}_l{learn}'
    if mode=='native':
        name+=f'_b{batch}_n{count}_r{repeat}'
        for cpu in ['cpu-dense','cpu-sparse']:assert passed(f'{cpu}_{cell}_l{learn}'),'CPU admission first'
        if count==8:
            assert width==16 and learn==1 and batch==1 and repeat==0
        else:
            assert passed('native_rgb32_c16_l1_b1_n8_r0'),'smoke first'
            if width!=16:assert passed(f'native_rgb32_c{[16,64,128][[16,64,128].index(width)-1]}_l1_b8_n96_r0'),'prior width gate'
            if batch==8:assert passed(f'native_{cell}_l{learn}_b1_n96_r0'),'serial before queue'
            if repeat:assert passed(f'native_{cell}_l{learn}_b{batch}_n96_r0')
    else:assert count==96 and batch==1 and repeat==0
    d=S/'raw'/name;assert not d.exists()
    prior=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')]
    physical=[p for p in prior if p['mode']=='native']
    assert len(physical)+(mode=='native')<=auth['maximum_physical_sessions']
    assert sum(p['count'] for p in physical)+(count if mode=='native' else 0)<=auth['maximum_forwards']
    d.mkdir(parents=True);rec=dict(name=name,mode=mode,cell=cell,learn=bool(learn),batch=batch,count=count,repeat=repeat,repeats=5 if mode!='native' else 1,admitted_utc=now().isoformat(),source_freeze_sha256=digest(S/'evidence/FROZEN.json'))
    save(d/'OWNER.json',rec)
    def execute():
        cmd=['taskset','-c','2','build/pcie_fabric',mode,str(F/cell),str(d/'result'),str(count),str(learn),str(batch),str(rec['repeats'])]
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            p=subprocess.Popen(cmd,cwd=P,stdout=out,stderr=err);rec['pid']=p.pid;save(d/'OWNER.json',rec)
            try:code=p.wait(timeout=min(120,int((dt.datetime.fromisoformat(auth['scientific_cutoff_utc'])-now()).total_seconds())))
            except subprocess.TimeoutExpired:
                p.terminate()
                try:code=p.wait(timeout=3)
                except subprocess.TimeoutExpired:p.kill();code=p.wait()
        rec.update(exit_code=code,finish_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1)
        rec['hashes']={p.name:digest(p) for p in d.iterdir() if p.is_file()}
        save(d/'RECEIPT.json',rec)
        try:
            assert code==0,(d/'stderr').read_text()
            if mode=='native':assert rec['device_free']
            result=audit(d)
        except Exception:
            state['failure']=name;save(S/'STATE.json',state);raise
        state.setdefault('completed',[]).append(name);save(S/'STATE.json',state);return result
    if mode=='native':
        with open('/tmp/akd1500-research.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
            result=execute()
    else:result=execute()
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='audit':print(json.dumps(audit(S/'raw'/sys.argv[2])))
    else:run(sys.argv[1],*map(int,sys.argv[2:]))
