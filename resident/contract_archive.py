"""Public preservation of vector collapse, failed bitwidth candidate and SDK gate."""
import json
import numpy as np
from resident.contract import arrays,inspect,name_for
from resident.contract_design import P,S,KINDS,data,expected
from resident.run import save,digest
BASE=P/'results/2026-10-23-handoff-contract'


def public_compiler_probe(probe):
    probe=json.loads(json.dumps(probe))
    for cell in probe['cells']:
        if 'traceback' in cell:
            cell['traceback']=cell['traceback'].replace('/home/nicklomeli/akida-cc-venv/','<akida-runtime>/').replace('/home/nicklomeli/akd1500-lab/','<lab>/')
    return probe


def expected_runs():
    return {name_for(k,'development',sm) for k in KINDS for sm in [True,False]}|{'mixed_identity_bits2_development_smoke'}


def recount(meta,z):
    assert meta['status']=='VECTOR_COLLAPSE_AND_FAILED_SOURCE_BITWIDTH_CANDIDATE'
    assert {r['name'] for r in meta['runs']}==expected_runs(), 'retain all diagnostics and failed candidate'
    assert meta['physical_sessions']==13 and meta['native_forwards']==824 and not meta['repair_qualified']
    checks=[]
    for r in meta['runs']:
        a={k.split('__',1)[1]:v for k,v in z.items() if k.startswith(r['name']+'__')};check=inspect(r,a);assert check==r['check'];checks.append(check)
        assert r['exit_code']==0 and r['device_released']
        if not r['kind'].startswith('mixed'):assert check['semantics_exact']
    x=data();collapsed=np.zeros_like(x);collapsed[x.any(1),63]=1
    assert np.array_equal(z['mixed_identity_development_full__outputs'],collapsed), 'channel-collapse identity model'
    assert np.array_equal(z['mixed_compare_development_full__outputs'],expected('mixed_compare',collapsed)), 'channel-collapse comparator model'
    assert np.array_equal(z['mixed_identity_bits2_development_smoke__outputs'],collapsed[:8]), 'failed declared-two-bit candidate'
    assert all(r['split']=='development' for r in meta['runs']), 'no sealed qualification was admitted'
    dist=meta['single_layer_compiler_probe'];assert dist['status']=='COMPILER_ONLY' and not dist['physical_device_opened']
    assert sum(c['status']=='FAILED_PRESERVED' for c in dist['cells'])==4
    return dict(status='HANDOFF_CHARACTERIZATION_RECOUNT_PASS',physical_sessions=13,native_forwards=824,
        mixed_identity_exact=2,mixed_identity_count=128,collapsed_comparator_exact=128,repair_qualified=False)


def export():
    assert json.loads((S/'STATE.json').read_text())['research_admissions_closed']
    assert {d.name for d in (S/'raw').iterdir()}==expected_runs()
    runs=[];z={}
    for d in sorted((S/'raw').iterdir()):
        r=json.loads((d/'RECEIPT.json').read_text())
        for f,h in r['hashes'].items():assert digest(d/f)==h,f
        aa=arrays(d,r);r.pop('hashes');r['check']=inspect(r,aa);runs.append(r)
        for k,v in aa.items():z[d.name+'__'+k]=v
    fr=json.loads((S/'evidence/FROZEN01.json').read_text())['hashes'];fr2=json.loads((S/'evidence/FROZEN02.json').read_text())['hashes']
    meta=dict(status='VECTOR_COLLAPSE_AND_FAILED_SOURCE_BITWIDTH_CANDIDATE',physical_sessions=13,native_forwards=sum(r['samples'] for r in runs),repair_qualified=False,runs=runs,
        single_layer_compiler_probe=public_compiler_probe(json.loads((P/'studies/20261023-distributed-plastic/fixtures-map01/BUILD.json').read_text())),
        compiler_only_bits4=json.loads((S/'fixtures-bits02/BUILD.json').read_text()),
        executed_sources={k:v for k,v in {**fr,**fr2}.items() if k in ['resident/contract.py','resident/contract_candidate.py','resident/contract_design.py','resident/contract_prepare.py','src/contract_probe.cpp']},
        limitations=['No sealed native candidate run; candidate failed development smoke.',
            'Collapsed-channel model explains recorded development vectors; no physical internal-wire capture.',
            'External-memory FNP2 selected under the frozen Minimal mapping policy; not a silicon capacity bound.',
            'No scaled learning, native repair, CPU or energy result in these diagnostics.'])
    recount(meta,z);np.savez_compressed(BASE.with_suffix('.npz'),**z);meta['archive_sha256']=digest(BASE.with_suffix('.npz'));save(BASE.with_suffix('.json'),meta);return audit_public()


def audit_public():
    m=json.loads(BASE.with_suffix('.json').read_text());assert digest(BASE.with_suffix('.npz'))==m['archive_sha256']
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:return recount(m,{k:z[k] for k in z.files})


if __name__=='__main__':
    import sys
    print(json.dumps(export() if sys.argv[1]=='export' else audit_public()))
