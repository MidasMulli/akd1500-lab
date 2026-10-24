"""Isolated fixed-communication admissions, physical checks and public recount."""
import datetime as dt
import fcntl
import json
import subprocess
import numpy as np
from resident.run import P,save,digest

S=P/'studies/20261023-fixed-bridge'
BASE=P/'results/2026-10-23-fixed-bridge'
ORDER=['all_fnp_smoke','all_fnp_full','direct_cnp_smoke','direct_cnp_full','mixed_smoke','mixed_full']


def freeze():
    files=[P/f for f in ['resident/bridge.py','resident/bridge_prepare.py','resident/prepare.py','resident/design.py',
        'resident/run.py','src/fixed_bridge.cpp','src/readout_bench.cpp','src/mmio_driver.hpp','tools/build.sh',
        'tools/program_codec.py','tools/prepare_network.py','metal/ablation_prepare.py','build/fixed_bridge']]
    files+=list((S/'fixtures01').iterdir())+[S/'PROTOCOL.md',S/'AUTHORITY.json']
    p=S/'evidence/FROZEN.json';assert not p.exists();save(p,dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),hashes={str(f.relative_to(P)):digest(f) for f in files}))


def admit(name):
    state=json.loads((S/'STATE.json').read_text());assert not state['research_admissions_closed'] and not state.get('halted')
    assert name in ORDER and len(state['admitted_runs'])==ORDER.index(name)
    if ORDER.index(name):assert json.loads((S/'raw'/ORDER[ORDER.index(name)-1]/'CHECK.json').read_text())['status']=='EXACT_FIXED_BRIDGE_PASS'
    state['admitted_runs'].append(name);save(S/'STATE.json',state)


def inspect(meta,z):
    n=meta['fixture']['input_features'];x=z['inputs'];assert x.shape==(128,n)
    assert np.all(x[:,64:]==0) and np.all((x==0)|(x==1))
    support=x[:,:64].reshape(128,4,16).sum(2);truth=np.zeros((128,4),np.int32);truth[np.arange(128),support.argmax(1)]=128
    assert np.array_equal(z['expected'],truth), 'independent bridge truth table'
    requested=32 if meta['name'].endswith('smoke') else 128;assert meta['requested_forwards']==requested
    output=z['outputs'];assert output.ndim==2 and output.shape[1]==4 and 0<len(output)<=requested
    bad=np.flatnonzero((output!=truth[:len(output)]).any(1));first=int(bad[0]) if len(bad) else None
    if first is None:assert len(output)==requested and meta['exit_code']==0
    else:assert len(output)==first+1 and meta['exit_code']!=0
    assert meta['observed_forwards']==len(output)
    for j,pkt in enumerate(meta['fixture']['packets']):
        assert digest_bytes(z[f'packet{j}.initial'])==pkt['sha256']
        assert np.array_equal(z[f'packet{j}.initial'],z[f'packet{j}.final']), 'fixed bridge memory changed'
    return dict(status='EXACT_FIXED_BRIDGE_PASS' if first is None else 'FIXED_BRIDGE_OUTPUT_DIVERGENCE',
        name=meta['name'],observed_forwards=len(output),first_divergence_event_1_based=None if first is None else first+1,
        initial_final_fixed_memory_exact=True)


def digest_bytes(a):
    import hashlib
    return hashlib.sha256(a.astype('<u4').tobytes()).hexdigest()


def arrays(d,rec):
    kind=rec['name'].rsplit('_',1)[0];f=S/'fixtures01'/kind
    z=dict(inputs=np.fromfile(f.with_suffix('.inputs'),'u1').reshape(128,rec['fixture']['input_features']),
        expected=np.fromfile(f.with_suffix('.expected'),'<i4').reshape(128,4),outputs=np.fromfile(d/'native.scores','<i4').reshape(-1,4))
    for j in range(len(rec['fixture']['packets'])):
        for when in ['initial','final']:z[f'packet{j}.{when}']=np.fromfile(d/f'native.packet{j}.{when}','<u4')
    return z


def run(name):
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text())
    assert not state['research_admissions_closed'] and not state.get('halted') and name in state['admitted_runs']
    frozen=S/'evidence/FROZEN.json'
    for f,h in json.loads(frozen.read_text())['hashes'].items():assert digest(P/f)==h,('source drift',f)
    now=dt.datetime.now(dt.timezone.utc);cutoff=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert (cutoff-now).total_seconds()>5
    kind=name.rsplit('_',1)[0];f=S/'fixtures01'/kind;count=32 if name.endswith('smoke') else 128
    prior=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')]
    assert len(prior)<6 and sum(r['requested_forwards'] for r in prior)+count<=480
    d=S/'raw'/name
    with open('/tmp/akd1500-research.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
        d.mkdir(exist_ok=False);rec=dict(name=name,requested_forwards=count,start_utc=now.isoformat(),fixture=json.loads(f.with_suffix('.json').read_text()),source_freeze_sha256=digest(frozen))
        save(d/'OWNER.json',rec)
        with (d/'native.stdout').open('w') as out,(d/'native.stderr').open('w') as err:
            proc=subprocess.Popen([str(P/'build/fixed_bridge'),str(f),str(d/'native'),str(count)],stdout=out,stderr=err)
            try:code=proc.wait(timeout=min(120,int((cutoff-now).total_seconds())))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:code=proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
        rec.update(exit_code=code,device_released=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,
            finish_utc=dt.datetime.now(dt.timezone.utc).isoformat(),observed_forwards=(d/'native.scores').stat().st_size//16 if (d/'native.scores').exists() else 0)
        rec['hashes']={p.name:digest(p) for p in d.iterdir() if p.is_file()};save(d/'RECEIPT.json',rec)
        try:
            assert rec['device_released'];result=inspect(rec,arrays(d,rec));save(d/'CHECK.json',result)
            if result['status']!='EXACT_FIXED_BRIDGE_PASS':state.update(halted=True,research_admissions_closed=True,failure=name)
        except Exception:
            state.update(halted=True,research_admissions_closed=True,failure=name);save(S/'STATE.json',state);raise
        save(S/'STATE.json',state);print(json.dumps(result),flush=True)


def export():
    assert json.loads((S/'STATE.json').read_text())['research_admissions_closed'];data={};runs=[]
    dirs=sorted((S/'raw').iterdir(),key=lambda d:ORDER.index(d.name));assert [d.name for d in dirs]==ORDER[:len(dirs)]
    for d in dirs:
        rec=json.loads((d/'RECEIPT.json').read_text())
        for p,h in rec['hashes'].items():assert digest(d/p)==h
        z=arrays(d,rec);check=inspect(rec,z);rec.pop('hashes');rec['check']=check;rec['stderr']=(d/'native.stderr').read_text();runs.append(rec)
        for k,a in z.items():data[d.name+'__'+k]=a
    np.savez_compressed(BASE.with_suffix('.npz'),**data)
    save(BASE.with_suffix('.json'),dict(status='FIXED_BRIDGE_DIAGNOSTIC',physical_sessions=len(runs),observed_native_forwards=sum(r['observed_forwards'] for r in runs),runs=runs,
        archive_sha256=digest(BASE.with_suffix('.npz')),compiler=json.loads((S/'fixtures01/BUILD.json').read_text()),learning_sessions=0,
        executed_sources={f:h for f,h in json.loads((S/'evidence/FROZEN.json').read_text())['hashes'].items() if f in ['resident/bridge.py','resident/bridge_prepare.py','src/fixed_bridge.cpp']},
        boundary='Compiler-produced fixed inference only; separate from failed learning adapter. No learning or efficiency result.'))
    return audit_public()


def audit_public():
    meta=json.loads(BASE.with_suffix('.json').read_text());assert digest(BASE.with_suffix('.npz'))==meta['archive_sha256']
    runs=meta['runs'];assert [r['name'] for r in runs]==ORDER[:len(runs)] and 0<len(runs)<=6
    assert meta['learning_sessions']==0 and meta['physical_sessions']==len(runs)
    checks=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for i,r in enumerate(runs):
            check=inspect(r,{k.split('__',1)[1]:z[k] for k in z.files if k.startswith(r['name']+'__')});assert check==r['check'];checks.append(check)
            assert check['status']=='EXACT_FIXED_BRIDGE_PASS' or i==len(runs)-1, 'stop at first mismatch'
    assert meta['observed_native_forwards']==sum(r['observed_forwards'] for r in runs)
    return dict(status='FIXED_BRIDGE_RECOUNT_PASS',physical_sessions=len(runs),observed_native_forwards=meta['observed_native_forwards'],checks=checks)


if __name__=='__main__':
    import sys
    cmd=sys.argv[1]
    if cmd=='freeze':freeze()
    elif cmd=='admit':admit(sys.argv[2])
    elif cmd=='run':run(sys.argv[2])
    elif cmd=='export':print(json.dumps(export()))
    elif cmd=='audit-public':print(json.dumps(audit_public()))
    else:raise ValueError(cmd)
