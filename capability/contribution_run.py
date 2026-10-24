"""Owned staged CNP output-suppression checks; no learning or firmware edits."""
import datetime as dt
import fcntl
import json
import subprocess
import sys
import numpy as np
from capability.design import P,inputs,weights,smoke_indices,DENSITIES
from capability.run import save,arrays
from metal.run import digest
S=P/'studies/20261024-cnp-contribution'


def freeze():
    files=[P/p for p in ['capability/contribution_prepare.py','capability/contribution_run.py','capability/design.py','capability/run.py','src/primitive_bench.cpp','src/readout_bench.cpp','src/mmio_driver.hpp','build/primitive_bench']]+list((S/'fixtures01').iterdir())+[S/'AUTHORITY.json',S/'PROTOCOL.md']
    assert not (S/'evidence/FROZEN.json').exists();save(S/'evidence/FROZEN.json',dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),hashes={str(p.relative_to(P)):digest(p) for p in files if p.is_file()}))


def recount(rec,z):
    x=z['inputs'];th=np.asarray(rec['fixture_info']['thresholds']);w=weights();p=x.astype(np.int32)@w.astype(np.int32);y=(p>th).reshape(len(x),2,1024).sum(2);baseline=(p>15).reshape(len(x),2,1024).sum(2)
    assert np.array_equal(y,z['scores']),'causal native output'
    idle=z['idle_counter'];assert len(idle)==6 and len(set(map(int,idle[:4])))==len(set(map(int,idle[4:])))==1
    delta=(z['counter_bounds'][:,1].astype(np.uint64)-z['counter_bounds'][:,0].astype(np.uint64))&0xffffffff
    assert np.array_equal(delta,z['cycles']) and np.all(z['cycles']>0) and np.all(z['config_bounds'][:,0]==z['config_bounds'][:,1])
    for j in range(len(rec['fixture_info']['packets'])):
        assert np.array_equal(z[f'packet{j}.initial'],z[f'packet{j}.expected']) and np.array_equal(z[f'packet{j}.final'],z[f'packet{j}.expected']),'causal whole fixed memory'
    for key in ['serial_suffix','burst_suffix']:assert np.array_equal(z[key],np.tile(y[-8:][None,:,:],(3,1,1)))
    ii=np.flatnonzero(x.sum(1)==512);drop=baseline[ii]-z['scores'][ii];assert len(ii)==8 and np.all(drop==drop[0])
    block=rec['fixture_info']['suppressed_block']
    if block:
        lo,hi=block;want=np.array([max(0,min(hi,1024)-lo),max(0,hi-max(lo,1024))]);assert np.array_equal(drop[0],want) and int(drop[0].sum())==hi-lo
    else:assert np.all(drop==0)
    return dict(status='INDEPENDENT_CONTRIBUTION_PASS',samples=40,forwards=88,suppressed_block=block,changed_packets=rec['fixture_info']['changed_packets'],full_active_output_drop=drop[0].tolist(),median_cycles_by_density={str(d):float(np.median(z['cycles'][x.sum(1)==d])) for d in DENSITIES})


def run(name):
    state=json.loads((S/'STATE.json').read_text());assert not state['research_admissions_closed'] and not state['halted'];assert name in ['baseline']+[f'suppress{i}' for i in range(11)]
    if name!='baseline':
        assert (S/'raw/baseline/AUDIT.json').exists()
        if name!='suppress0':assert (S/'raw/suppress0/AUDIT.json').exists()
    assert name not in state['admitted_runs'];f=S/'fixtures01'/name;info=json.loads(f.with_suffix('.json').read_text());assert info['status']=='COMPILER_LOCALIZATION_PASS'
    frozen=S/'evidence/FROZEN.json'
    for p,h in json.loads(frozen.read_text())['hashes'].items():assert digest(P/p)==h,p
    auth=json.loads((S/'AUTHORITY.json').read_text());now=dt.datetime.now(dt.timezone.utc);cut=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert (cut-now).total_seconds()>10
    assert len(list((S/'raw').glob('*/OWNER.json')))<12;d=S/'raw'/name
    x,rows=inputs();ii=smoke_indices(rows);x=x[ii]
    with open('/tmp/akd1500-research.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
        state['admitted_runs'].append(name);save(S/'STATE.json',state);d.mkdir(exist_ok=False);stream=d/'stream';stream.with_suffix('.inputs').write_bytes(x.tobytes());stream.with_suffix('.labels').write_bytes(np.zeros(40,'<i4').tobytes());stream.with_suffix('.modes').write_bytes(bytes(40));stream.with_suffix('.expected').write_bytes(np.fromfile(f.with_suffix('.expected'),'<i4').reshape(160,2)[ii].tobytes())
        rec=dict(name=name,kind='fixed',fixture=str(f.relative_to(P)),fixture_info=info,samples=40,forwards=88,source_freeze_sha256=digest(frozen),start_utc=now.isoformat());save(d/'OWNER.json',rec)
        with (d/'native.stdout').open('w') as out,(d/'native.stderr').open('w') as err:
            p=subprocess.Popen(['taskset','-c','2','build/primitive_bench',str(f),str(stream),str(d/'native'),'40','fixed'],cwd=P,stdout=out,stderr=err);rec['native_pid']=p.pid;save(d/'OWNER.json',rec)
            try:code=p.wait(timeout=min(120,int((cut-now).total_seconds())))
            except subprocess.TimeoutExpired:
                p.terminate()
                try:code=p.wait(timeout=3)
                except subprocess.TimeoutExpired:p.kill();code=p.wait()
        rec.update(status='PASS' if code==0 else 'FAILED_PRESERVED',exit_code=code,device_released=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1);rec['hashes']={p.name:digest(p) for p in d.iterdir() if p.is_file()};save(d/'RECEIPT.json',rec)
        try:
            assert code==0 and rec['device_released'],(d/'native.stderr').read_text();result=recount(rec,arrays(d,rec));save(d/'AUDIT.json',result)
        except Exception:state['halted']=True;save(S/'STATE.json',state);raise
    print(json.dumps(result),flush=True)


if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else run(sys.argv[1])
