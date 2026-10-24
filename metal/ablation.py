"""Source-frozen, owned NPU ablations and public numerical export/recount."""
from pathlib import Path
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import subprocess
import numpy as np
from metal.run import P, digest
from metal.validate_ablation import validate

S=P/'studies/20261023-metal-ablation'
BASE=P/'results/2026-10-23-metal-ablation'


def freeze():
    assert validate(S)['status']=='PASS'
    files=[P/p for p in ['metal/ablation.py','metal/ablation_prepare.py','metal/run.py',
        'metal/validate_ablation.py','src/ablation_bench.cpp','src/readout_bench.cpp','src/mmio_driver.hpp','tools/build.sh','build/ablation_bench']]
    files+=list((S/'fixtures-ablation02').glob('*'))
    files+=[S/'PROTOCOL.md',S/'AUTHORITY.json',S/'evidence/FULL_PROGRAM_CHECK.json']
    f=S/'evidence/FROZEN.json';assert not f.exists()
    f.write_text(json.dumps(dict(status='FROZEN_BEFORE_PHYSICAL',hashes={str(p.relative_to(P)):digest(p) for p in files}),indent=2)+'\n')
    print('frozen',len(files),'files')


def run(name):
    state=json.loads((S/'STATE.json').read_text());assert not state['research_admissions_closed'] and name in state['admitted']
    auth=json.loads((S/'AUTHORITY.json').read_text());now=dt.datetime.now(dt.timezone.utc);cutoff=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert now<cutoff
    frozen=S/'evidence/FROZEN.json'
    for p,h in json.loads(frozen.read_text())['hashes'].items():assert digest(P/p)==h,('source drift',p)
    prior=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')]
    assert len(prior)<auth['maximum_physical_sessions'] and sum(r['samples'] for r in prior)+64<=auth['maximum_forwards']
    d=S/'raw'/name;d.mkdir(exist_ok=False);f=S/'fixtures-ablation02'/name
    r=dict(name=name,samples=64,fixture=str(f.relative_to(P)),start_utc=now.isoformat(),source_freeze_sha256=digest(frozen))
    (d/'OWNER.json').write_text(json.dumps(r,indent=2)+'\n')
    with open('/tmp/akd1500-research.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
        with (d/'native.stdout').open('w') as out,(d/'native.stderr').open('w') as err:
            proc=subprocess.Popen(['taskset','-c','2',str(P/'build/ablation_bench'),str(f),str(d/'native')],stdout=out,stderr=err)
            r['native_pid']=proc.pid;(d/'OWNER.json').write_text(json.dumps(r,indent=2)+'\n')
            try:code=proc.wait(timeout=min(180,int((cutoff-now).total_seconds())))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:code=proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
        r.update(status='PASS' if code==0 else 'FAILED_PRESERVED',exit_code=code,device_released=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,finish_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        r['hashes']={p.name:digest(p) for p in d.iterdir() if p.is_file() and p.name!='RECEIPT.json'}
        (d/'RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
        assert code==0 and r['device_released'],(d/'native.stderr').read_text()


def export():
    arrays={};runs=[]
    for name in ['baseline','suppress_A','suppress_B','suppress_both']:
        d=S/'raw'/name;r=json.loads((d/'RECEIPT.json').read_text());assert r['status']=='PASS'
        for p,h in r['hashes'].items():assert digest(d/p)==h
        f=P/r['fixture'];meta=json.loads(f.with_suffix('.json').read_text());key=name+'__'
        arrays[key+'inputs']=np.fromfile(f.with_suffix('.inputs'),np.uint8).reshape(64,512)
        arrays[key+'outputs']=np.fromfile(d/'native.scores','<i4').reshape(64,2)
        for j in range(5):
            for when in ['initial','final']:arrays[key+f'packet{j}_{when}']=np.fromfile(d/f'native.packet{j}.{when}','<u4')
        runs.append(dict(name=name,fixture=meta,receipt_sha256=digest(d/'RECEIPT.json')))
    np.savez_compressed(BASE.with_suffix('.npz'),**arrays)
    report=dict(status='NATIVE_NPU_CAUSAL_ABLATION_PASSED',physical_sessions=4,native_forwards=256,
        runs=runs,archive_sha256=digest(BASE.with_suffix('.npz')),
        compiler=json.loads((S/'fixtures-ablation02/BUILD.json').read_text()),
        limitation='Threshold suppression proves native contribution and memory localization, not per-cycle simultaneity')
    BASE.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(recount(report,arrays)))


def recount(report,data):
    assert [r['name'] for r in report['runs']]==['baseline','suppress_A','suppress_B','suppress_both'], 'complete causal treatment set'
    rng=np.random.default_rng(98120);w=np.zeros((512,256),np.int32)
    for j in range(256):w[rng.choice(512,32,replace=False),j]=1
    for run in report['runs']:
        name=run['name'];key=name+'__';x=data[key+'inputs'];out=data[key+'outputs']
        assert x.shape==(64,512) and out.shape==(64,2) and np.all((x==0)|(x==1))
        thresholds=np.full(256,15)
        if name in ['suppress_A','suppress_both']:thresholds[:128]=32
        if name in ['suppress_B','suppress_both']:thresholds[128:]=32
        z=x.astype(np.int32)@w>thresholds
        expected=np.stack([z[:,:128].sum(axis=1),z[:,128:].sum(axis=1)],axis=1)
        assert np.array_equal(out,expected),('causal native output',name)
        if name!='baseline':assert np.array_equal(x,data['baseline__inputs'])
        changed=[]
        for j,packet in enumerate(run['fixture']['packets']):
            a=data[key+f'packet{j}_initial'];b=data[key+f'packet{j}_final']
            assert np.array_equal(a,b),('physical memory changed',name,j)
            header=int(a[0]);mask=(header>>8)&15
            assert not mask&(mask-1)
            assert [header>>24,(header>>16)&255,mask.bit_length()-1]==packet['np'],('physical target',name,j)
            assert hashlib.sha256(a.astype('<u4').tobytes()).hexdigest()==packet['sha256'],('initial physical packet',name,j)
            base=data[f'baseline__packet{j}_initial'];ids=np.flatnonzero(a!=base)
            if len(ids):changed.append((packet['np'],ids.tolist()))
        expected_nps=[]
        if name in ['suppress_A','suppress_both']:expected_nps.append([1,2,1])
        if name in ['suppress_B','suppress_both']:expected_nps.append([1,2,2])
        assert [p for p,_ in changed]==expected_nps,('ablation locality',name)
        for _,ids in changed:assert ids==list(range(2,258,2)),('only threshold words',name)
    return dict(status='PASS',native_forwards=256,native_output_values=512,
                physical_memory_regions=20,initial_and_final_region_readbacks=40)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command');a=p.parse_args()
    if a.command=='freeze':freeze()
    elif a.command=='export':export()
    elif a.command=='audit-public':
        report=json.loads(BASE.with_suffix('.json').read_text());assert digest(BASE.with_suffix('.npz'))==report['archive_sha256']
        with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:print(json.dumps(recount(report,{k:z[k] for k in z.files})))
    else:run(a.command)
