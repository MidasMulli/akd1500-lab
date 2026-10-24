"""Recount successful prefixes and preserved first divergences, without hardware."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
from resident.design import Network, schedule
from resident.run import P, S, save, digest

BASE = P / 'results/2026-10-23-resident-boundary'
CASES = {
    'smoke_b1_s99101_repair': (1, None),
    'smoke_b2_s99101_repair': (2, (128, 1)),
    'smoke_b1_s99101_repair_a02': (1, None),
    'smoke_b2_s99101_repair_a02': (2, (0, 0)),
    'smoke_b1_s99101_repair_a03': (1, None),
    'smoke_b2_s99101_repair_a03': (2, (4, 0)),
}


def inspect_case(rec, z):
    name=rec['name']; nb,expected_failure=CASES[name]
    assert rec['banks']==nb and rec['seed']==99101 and rec['arm']=='repair' and rec['smoke']
    x,rows=schedule(nb,'repair',True); assert rec['requested_forwards']==len(rows)
    assert np.array_equal(z['inputs'],x) and rec['rows']==rows, 'frozen sensory stream'
    ram=[z[f'initial{k}'].tolist() for k in range(nb)]
    for k in range(nb):
        assert hashlib.sha256(np.asarray(ram[k],'<u4').tobytes()).hexdigest()==rec['fixture']['bank_ram_sha256'][k], 'initial bank identity'
    completed=len(rows) if expected_failure is None else expected_failure[0]+1
    assert rec['observed_forwards']==completed
    expected_coords=[(i,k) for i in range(completed) for k in range(nb)
        if expected_failure is None or i<expected_failure[0] or k<=expected_failure[1]]
    coords=[tuple(a) for a in rec['snapshot_coordinates']]
    assert coords==expected_coords, 'partial capture accounting'
    got_states=z['states']; assert got_states.shape==(len(coords),1282)
    by={pair:got_states[j] for j,pair in enumerate(coords)}
    net=Network(ram);first=None;outputs=[];mismatch_words=0
    for i in range(completed):
        output,_=net.forward(x[i],rows[i]['label'],rows[i]['mask']);outputs.append(output)
        for k in range(nb):
            if (i,k) not in by:continue
            differences=np.flatnonzero(by[i,k]!=net.banks[k].ram)
            if len(differences) and first is None:
                first=(i,k);mismatch_words=len(differences)
    assert first==expected_failure, ('preserved first divergence',first,expected_failure)
    assert rec['status']==('PASS' if first is None else 'FAILED_PRESERVED')
    if 'outputs' in z:
        assert z['outputs'].shape==(completed,4)
        assert np.array_equal(z['outputs'],outputs), 'captured final output prefix'
    else:
        assert name=='smoke_b2_s99101_repair', 'missing captured output prefix'
    for k,pkt in enumerate(rec['fixture']['fixed_packets']):
        assert hashlib.sha256(z[f'fixed{k}.initial'].astype('<u4').tobytes()).hexdigest()==pkt['sha256']
        if first is None:assert np.array_equal(z[f'fixed{k}.initial'],z[f'fixed{k}.final']), 'fixed memory changed'
    if first is None:
        for k in range(nb):assert np.array_equal(z[f'final{k}'],net.banks[k].ram)
        from resident.run import recount
        # Separate complete-run auditor additionally checks timers and counter controls.
        rr=dict(name=name,banks=nb,seed=99101,arm='repair',smoke=True,samples=len(rows),fixture_info=rec['fixture'],rows=rows)
        aa={'stream.inputs':z['inputs'],'stream.labels':np.asarray([r['label'] for r in rows]),'stream.masks':np.asarray([r['mask'] for r in rows]),'native.scores':z['outputs']}
        for key in rec['diagnostic_arrays']:aa[key]=z[key]
        for k in range(nb):
            aa[f'bank{k}.initial']=z[f'initial{k}'];aa[f'bank{k}.final']=z[f'final{k}']
            aa[f'bank{k}.snapshots']=np.stack([by[i,k] for i in range(completed)])
        for k in range(len(rec['fixture']['fixed_packets'])):
            for when in ['initial','final']:aa[f'fixed{k}.{when}']=z[f'fixed{k}.{when}']
        recount(rr,aa)
    return dict(name=name,status=rec['status'],observed_forwards=completed,full_bank_captures=len(coords),
        first_divergence=list(first) if first else None,divergent_words=mismatch_words)


def recount(meta,data):
    assert meta['status']=='RESIDENT_SCALING_STOPPED_AT_MIXED_RESOURCE_STATE_GATE'
    assert {r['name'] for r in meta['runs']}==set(CASES), 'complete successful and failed attempts'
    assert meta['physical_sessions']==6 and meta['qualified_learning_banks']==1 and not meta['post_gate_passed'], 'no scaled success claim'
    checks=[]
    for r in meta['runs']:
        checks.append(inspect_case(r,{k.split('__',1)[1]:v for k,v in data.items() if k.startswith(r['name']+'__')}))
    assert meta['observed_native_forwards']==sum(c['observed_forwards'] for c in checks)==663, 'executed forward accounting'
    assert meta['requested_forward_cap_accounting']==sum(r['requested_forwards'] for r in meta['runs'])==1440
    assert meta['successful_native_forwards']==528 and meta['failed_attempt_native_forwards']==135
    assert meta['full_bank_captures']==sum(c['full_bank_captures'] for c in checks)==796
    assert meta['attempt_checks']==checks
    assert not meta['compiler_partition']['physical_device_opened']
    assert len(meta['compiler_partition']['cells'])==2
    assert all('IP version' in a['error'] for a in meta['compiler_partition']['cells']), 'merge rejection preserved'
    for build in meta['compiler_builds']:
        assert build['status']=='COMPILER_ONLY' and not build['physical_device_opened']
        for cell in build['cells']:
            assert cell['status']=='COMPILER_VIRTUAL_ADAPTER_PASS'
            if cell['banks']==6:
                assert len({tuple(p['np']) for layer in cell['mapping'] for p in layer['nps']})==17
    return dict(status='PRESERVED_BOUNDARY_RECOUNT_PASS',physical_sessions=6,observed_native_forwards=663,
        full_bank_captures=796,qualified_learning_banks=1,post_gate_passed=False)


def export():
    assert json.loads((S/'STATE.json').read_text())['research_admissions_closed']
    runs=[];data={}
    for name in CASES:
        d=S/'raw'/name;rec=json.loads((d/'RECEIPT.json').read_text())
        for f,h in rec['hashes'].items():assert digest(d/f)==h,f
        assert rec['device_released'], 'device ownership retained'
        arrays={};coords=[];states=[]
        for p in d.glob('native.snapshot_*_bank*.ram'):
            m=re.fullmatch(r'native.snapshot_(\d+)_bank(\d+)\.ram',p.name);assert m
            coords.append(tuple(map(int,m.groups())))
        coords.sort()
        for i,k in coords:states.append(np.fromfile(d/f'native.snapshot_{i}_bank{k}.ram','<u4'))
        rows=json.loads((d/'stream.rows.json').read_text());nb=rec['banks'];completed=max(i for i,k in coords)+1
        arrays['inputs']=np.fromfile(d/'stream.inputs','u1').reshape(-1,512);arrays['states']=np.stack(states)
        diag=[]
        for k in range(nb):
            arrays[f'initial{k}']=np.fromfile(d/f'native.bank{k}.initial.ram','<u4')
            if rec['status']=='PASS':arrays[f'final{k}']=np.fromfile(d/f'native.bank{k}.ram','<u4')
        for k in range(len(rec['fixture_info']['fixed_packets'])):
            for when in ['initial','final'] if rec['status']=='PASS' else ['initial']:
                arrays[f'fixed{k}.{when}']=np.fromfile(d/f'native.fixed{k}.{when}.ram','<u4')
        if (d/'native.scores').exists():arrays['outputs']=np.fromfile(d/'native.scores','<i4').reshape(-1,4)
        if rec['status']=='PASS':
            for key,typ,shape in [('native.cycles','<u4',(completed,)),('native.counter_bounds','<u4',(completed,2)),
                ('native.config_bounds','<u4',(completed,2)),('native.control_bounds','<u4',(completed,2)),('native.idle_counter','<u4',(6,)),
                ('native.latency_ns','<i8',(completed,)),('native.caller_ns','<i8',(completed,)),('native.mode_ns','<i8',(completed,)),('native.diagnostic_ns','<i8',(completed,))]:
                arrays[key]=np.fromfile(d/key,typ).reshape(shape);diag.append(key)
        r=dict(name=name,banks=nb,seed=rec['seed'],arm=rec['arm'],smoke=rec['smoke'],status=rec['status'],
            requested_forwards=rec['samples'],observed_forwards=completed,snapshot_coordinates=coords,rows=rows,
            fixture=rec['fixture_info'],source_freeze_sha256=rec['source_freeze_sha256'],receipt_sha256=digest(d/'RECEIPT.json'),
            diagnostic_arrays=diag,device_released=rec['device_released'],stderr=(d/'native.stderr').read_text())
        r['check']=inspect_case(r,arrays);runs.append(r)
        for k,v in arrays.items():data[name+'__'+k]=v
    partition=json.loads((S/'fixtures-partition01/BUILD.json').read_text())
    meta=dict(status='RESIDENT_SCALING_STOPPED_AT_MIXED_RESOURCE_STATE_GATE',physical_sessions=6,observed_native_forwards=663,
        requested_forward_cap_accounting=1440,successful_native_forwards=528,failed_attempt_native_forwards=135,
        full_bank_captures=796,qualified_learning_banks=1,post_gate_passed=False,runs=runs,attempt_checks=[r['check'] for r in runs],
        compiler_parameter_differentials={name:json.loads((S/'evidence'/name).read_text()) for name in ['COMPILER_AUX_FIELD_DIFF02.json','COMPILER_CLASS_FIELD.json']},
        executed_latest_sources={name:h for name,h in json.loads((S/'evidence/FROZEN03.json').read_text())['hashes'].items() if name.startswith('resident/') or name in ['src/resident_bench.cpp','src/readout_bench.cpp','src/mmio_driver.hpp']},
        postclose_auditor_sha256=digest(Path(__file__)),compiler_partition=partition,compiler_builds=[json.loads((S/f'fixtures-chain{i:02}/BUILD.json').read_text()) for i in [1,2,3]],
        limitations=['Six banks/17 NPUs are compiler-only; no physical runs at four or six banks.',
            'One initialization seed and one synthetic world; repeated one-bank successes are implementation checks.',
            'Earliest mixed-resource failure has no saved final-output prefix; its RAM divergence is captured.',
            'Intermediate messages are inferred from reference/state, not electrical traces.',
            'No CPU advantage, energy result, scaled learning success or new internal learning rule.'])
    recount(meta,data);np.savez_compressed(BASE.with_suffix('.npz'),**data)
    meta['archive_sha256']=digest(BASE.with_suffix('.npz'));save(BASE.with_suffix('.json'),meta)
    return audit_public()


def audit_public():
    meta=json.loads(BASE.with_suffix('.json').read_text());assert digest(BASE.with_suffix('.npz'))==meta['archive_sha256']
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:return recount(meta,{k:z[k] for k in z.files})


if __name__=='__main__':
    import sys
    print(json.dumps(export() if sys.argv[1]=='export' else audit_public()))
