"""Vector-level staged bridge diagnostics; transport and semantics are separate."""
import datetime as dt
import fcntl
import json
import shutil
import subprocess
import numpy as np
from resident.contract_design import P,S,KINDS,data,expected
from resident.run import save,digest


def name_for(kind,split,smoke):return f'{kind}_{split}_{"smoke" if smoke else "full"}'


def freeze():
    p=S/'evidence/FROZEN01.json';assert not p.exists()
    files=[P/f for f in ['resident/contract.py','resident/contract_design.py','resident/contract_prepare.py','resident/prepare.py',
        'src/contract_probe.cpp','src/readout_bench.cpp','src/mmio_driver.hpp','tools/prepare_network.py','tools/build.sh',
        'tools/program_codec.py','metal/ablation_prepare.py','build/contract_probe']]
    files+=list((S/'fixtures01').iterdir())+[S/'AUTHORITY.json',S/'PROTOCOL.md']
    save(p,dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),hashes={str(f.relative_to(P)):digest(f) for f in files}))


def admit(kind,split,smoke,review):
    assert kind in KINDS and split=='development', 'sealed requires a prospectively frozen candidate'
    state=json.loads((S/'STATE.json').read_text());assert not state['research_admissions_closed'] and not state.get('halted')
    prereq={'all_fnp':'source','direct_identity':'all_fnp','mixed_identity':'direct_identity','direct_compare':'mixed_identity','mixed_compare':'direct_compare'}
    if not smoke:
        prior=json.loads((S/'raw'/name_for(kind,split,True)/'CHECK.json').read_text());assert prior['mechanics_exact']
        assert prior['semantics_exact'] or kind.startswith('mixed')
    elif kind in prereq:
        prior=json.loads((S/'raw'/name_for(prereq[kind],split,False)/'CHECK.json').read_text());assert prior['mechanics_exact']
        assert prior['semantics_exact'] or prereq[kind].startswith('mixed')
    assert len(review)>20
    name=name_for(kind,split,smoke);assert name not in [r['name'] for r in state['admitted_runs']]
    state['admitted_runs'].append(dict(name=name,kind=kind,split=split,smoke=smoke,review=review,admitted_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    state['stage']='NATIVE_VECTOR_PROBES';save(S/'STATE.json',state);print(name)


def arrays(d,rec):
    n=rec['fixture']['input_features'];outputs=rec['fixture']['outputs'];count=rec['samples']
    z=dict(inputs=np.fromfile(d/'fixture.inputs','u1').reshape(128,n),expected=np.fromfile(d/'fixture.expected','<i4').reshape(128,outputs),
        outputs=np.fromfile(d/'native.scores','<i4').reshape(count,outputs))
    for j in range(len(rec['fixture']['packets'])):
        for when in ['initial','final']:z[f'packet{j}.{when}']=np.fromfile(d/f'native.packet{j}.{when}','<u4')
    return z


def inspect(rec,z):
    kind=rec['kind'];x=data(rec['split']);want=np.zeros_like(z['inputs']);want[:,:64]=x
    assert np.array_equal(want,z['inputs']), 'frozen vector inputs'
    truth=expected(kind,x);assert np.array_equal(z['expected'],truth), 'independent vector outputs'
    count=8 if rec['smoke'] else 128;assert rec['samples']==count and z['outputs'].shape==truth[:count].shape
    for j,pkt in enumerate(rec['fixture']['packets']):
        from hashlib import sha256
        assert sha256(z[f'packet{j}.initial'].astype('<u4').tobytes()).hexdigest()==pkt['sha256']
        assert np.array_equal(z[f'packet{j}.initial'],z[f'packet{j}.final']), 'fixed memory changed'
    exact=(z['outputs']==truth[:count]).all(1);bad=np.flatnonzero(~exact)
    return dict(status='VECTOR_CONTRACT_PASS' if exact.all() else 'VECTOR_DIAGNOSTIC_MISMATCH',mechanics_exact=True,semantics_exact=bool(exact.all()),
        name=rec['name'],samples=count,exact_vectors=int(exact.sum()),first_mismatch=None if not len(bad) else int(bad[0]),
        output_min=int(z['outputs'].min()),output_max=int(z['outputs'].max()),nonzero_outputs_per_vector=np.count_nonzero(z['outputs'],axis=1).tolist(),
        native_outputs=z['outputs'].tolist())


def run(name):
    state=json.loads((S/'STATE.json').read_text());auth=json.loads((S/'AUTHORITY.json').read_text())
    assert not state['research_admissions_closed'] and not state.get('halted')
    entries=[r for r in state['admitted_runs'] if r['name']==name];assert len(entries)==1;entry=entries[0]
    fr=S/'evidence/FROZEN01.json'
    for p,h in json.loads(fr.read_text())['hashes'].items():assert digest(P/p)==h,('source drift',p)
    now=dt.datetime.now(dt.timezone.utc);cutoff=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert (cutoff-now).total_seconds()>5
    count=8 if entry['smoke'] else 128;prior=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')]
    assert len(prior)<auth['maximum_physical_sessions'] and sum(r['samples'] for r in prior)+count<=auth['maximum_forwards']
    f=S/'fixtures01'/entry['kind'];d=S/'raw'/name
    with open('/tmp/akd1500-research.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
        d.mkdir(exist_ok=False)
        for p in f.parent.glob(entry['kind']+'.*'):
            if p.suffix in ['.bin','.layout'] or '.packet' in p.name:shutil.copy2(p,d/('fixture'+p.name[len(entry['kind']):]))
        for ext in ['inputs','expected']:shutil.copy2(f.parent/f'{entry["kind"]}.{entry["split"]}.{ext}',d/f'fixture.{ext}')
        rec=dict(**entry,samples=count,start_utc=now.isoformat(),fixture=json.loads(f.with_suffix('.json').read_text()),source_freeze_sha256=digest(fr));save(d/'OWNER.json',rec)
        with (d/'native.stdout').open('w') as out,(d/'native.stderr').open('w') as err:
            proc=subprocess.Popen([str(P/'build/contract_probe'),str(d/'fixture'),str(d/'native'),str(count)],stdout=out,stderr=err)
            try:code=proc.wait(timeout=min(auth['maximum_individual_seconds'],int((cutoff-now).total_seconds())))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:code=proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
        rec.update(exit_code=code,device_released=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,finish_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        rec['hashes']={p.name:digest(p) for p in d.iterdir() if p.is_file()};save(d/'RECEIPT.json',rec)
        try:
            assert code==0 and rec['device_released'],(d/'native.stderr').read_text()
            check=inspect(rec,arrays(d,rec));save(d/'CHECK.json',check)
            if not check['semantics_exact'] and not entry['kind'].startswith('mixed'):state.update(halted=True,failure=name)
        except Exception:
            state.update(halted=True,failure=name);save(S/'STATE.json',state);raise
        save(S/'STATE.json',state);print(json.dumps({k:v for k,v in check.items() if k not in ['native_outputs','nonzero_outputs_per_vector']}),flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='admit':admit(sys.argv[2],sys.argv[3],sys.argv[4]=='smoke',sys.argv[5])
    elif sys.argv[1]=='run':run(sys.argv[2])
    else:raise ValueError(sys.argv[1])
