"""Publish the prospective confirmation and fresh outcomes, preserving failed gates."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from bank import save,sha

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from verify import read_words,verify


def export(study,fresh_paths):
    report=json.loads((study/'evidence/RECOUNT.json').read_text());assert len(report['cohorts'])==5
    arrays={};fixtures={};runs=[]
    for run in sorted((study/'raw').iterdir()):
        rec=json.loads((run/'RECEIPT.json').read_text());audit=verify(ROOT,run);f=ROOT/rec['fixture'];key=run.name
        n,u,nw,classes=map(int,f.with_suffix('.meta').read_text().split());fixtures[key]=dict(inputs=n,neurons=u,num_weights=nw,classes=classes)
        for suffix,dtype in [('inputs','u1'),('labels','<u4'),('ram','<u4')]:arrays[f'fixture__{key}__{suffix}']=np.frombuffer(f.with_suffix('.'+suffix).read_bytes(),dtype=dtype).copy()
        for mode in ['native','cpu']:
            for suffix in ['scores','ram']:arrays[f'{key}__{mode}_{suffix}']=np.array(read_words(run/f'{mode}.{suffix}'),np.uint32)
        arrays[key+'__snapshots']=np.array([read_words(run/f'native.snapshot_{i}.ram') for i in range(rec['samples'])],np.uint32)
        row=dict(name=key,fixture_key=key,samples=rec['samples'],learning=1,audit=audit,binary_sha256=rec['binary_sha256'],program_sha256=rec['program_sha256'])
        if key!='smoke32':row.update(next(r for r in report['runs'] if r['name']==key))
        runs.append(row)
    archive=ROOT/'results/2026-10-23-mario-confirmation.npz';np.savez_compressed(archive,**arrays)
    fresh=[]
    for directory in fresh_paths:
        z=json.loads((directory/'FRESH.json').read_text());assert z['status']=='FRESH_NATIVE_SELECTIONS_MATCH';fresh.extend(z['rows'])
    assert len(fresh)==1280 and len({(r['seed'],r['arm'],r['context']) for r in fresh})==1280
    context_ids=[r['id'] for r in report['contexts']]
    for row in fresh:
        r=next(r for r in report['runs'] if r.get('seed')==row['seed'] and r.get('arm')==row['arm']);i=context_ids.index(row['context'])
        assert row['action']==r['actions'][i] and row['success']==bool(r['correct'][i])
    fresh_file=ROOT/'results/2026-10-23-mario-confirmation-fresh.json';save(fresh_file,dict(status='FRESH_NATIVE_SELECTIONS_MATCH',trials=1280,rows=fresh))
    controls=json.loads((study/'cpu_runs/controls01/REPORT.json').read_text());assert controls['physical_device_opened'] is False
    result=dict(status='SEALED_CONFIRMATION_COMPLETED_GATE_FAILED' if report['learning_effect_gate']=='FAIL' else 'SEALED_LEARNING_EFFECT_GATE_PASS',
        regime='Train first; protected test queries only; no test prequeries or teaching',
        endpoint_registration='Prospective new learning-effect confirmation after development; earlier usefulness studies remain failed',
        cohorts=report['cohorts'],learning_effect_gate=report['learning_effect_gate'],initialization_wins=report['initialization_wins'],
        learned_total=report['learned_total'],frozen_total=report['frozen_total'],best_constant_total=report['best_constant_total'],opportunities=report['opportunities'],
        all_valid_gain=report['all_valid_gain'],conditional_gain=report['conditional_gain'],conditional_block_bootstrap_95=report['block_bootstrap_95'],
        conditional_hierarchical_bootstrap_95=report['hierarchical_bootstrap_95'],data_adequacy=report['data_adequacy'],
        contexts=report['contexts'],test_truth=report['test_truth'],train_test_encoded_overlap=report['train_test_encoded_overlap'],
        fresh_trials=1280,fresh_receipt_sha256=sha(fresh_file),physical_chips=1,physical_sessions=len(runs),native_forwards=sum(r['samples'] for r in runs),
        archive_sha256=sha(archive),fixtures=fixtures,runs=runs,cpu_model_controls=controls,
        protocol_sha256=sha(study/'PROTOCOL.md'),provenance_sha256=sha(ROOT/'results/2026-10-23-mario-provenance.json'),
        usefulness='NOT_QUALIFIED; best constant exceeds learned aggregate and full physical baseline controls were not run',
        preserved_failures=['Two earlier development/usefulness gates failed; their original128 test contexts remain unopened',
            'Compiler wrapper import collision before data/device access; standalone invocation repaired it without changing the scientific pipeline',
            'Static duplicate-key construction fixed before compiler execution; original source and lock versions retained'],
        limitations=['Five initializations share128 contexts on one chip; not640 independent environments',
            'Known task families and related prior1-1 exposures; no unseen-level claim',
            'Fixed RGB/time encoder, ranking and two90-frame motor primitives; no full-game RL or frame-by-frame native policy',
            'Control-model outputs are CPU forecasts, separate from measured physical learned/frozen arms',
            'No CPU/energy advantage, complete hidden-state restore, power-loss persistence or electrical spike trace'])
    save(ROOT/'results/2026-10-23-mario-confirmation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs','fixtures','contexts','test_truth','cpu_model_controls','limitations','preserved_failures']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('fresh',nargs='+',type=Path);a=p.parse_args();export(a.study.resolve(),[f.resolve() for f in a.fresh])
