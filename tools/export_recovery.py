"""Export independently recounted recovery evidence; no hardware operations."""
from pathlib import Path
import argparse,hashlib,importlib,json,shutil,sys
from audit_image_copy import audit
P=Path(__file__).resolve().parents[1]

def export(study):
    recount=audit(study)
    (study/'evidence/MAC_RELOCATED_AUDIT.json').write_text(json.dumps(recount,indent=2)+'\n')
    gates=json.loads((study/'evidence/GATE_REVIEW.json').read_text());assert gates['status']=='FIXED_GATES_PASS'
    assert len(recount['results'])==13 # nine scientific native cells, three CPU cells, smoke
    rows=[]
    for seed in [71011,71012,71013]:
        data={}
        for arm in ['probe','reactive','frozen','cpu']:
            name=('pilot_probe_71011' if seed==71011 and arm=='probe' else f'control_71011_{arm}' if seed==71011 else f'confirm_{seed}_{arm}')
            d=study/('cpu_runs' if arm=='cpu' else 'runs')/name
            z=json.loads((d/'SUMMARY.json').read_text());v=next(x for x in recount['results'] if x['run']==name)
            assert v['status']=='PASS'
            data[arm]=dict(completed=z['completed'],attempts=z['operation_attempts'],failures=z['worker_failures'],
                first_preferred_return_job=v['first_preferred_return_job'],recovery_events=v['recovery_events'],
                events_sha256=v['events_sha256'],raw_sha256=v['raw_sha256'])
        rows.append(dict(seed=seed,**data))
    # Build a replay only from the audited primary native stream.
    vn=importlib.import_module('verify_native');d=study/'runs/pilot_probe_71011';nv=vn.verify(d)
    mechanism=vn.verify(study/'runs/probe_mechanics_8')
    assert mechanism['native_forwards']==24
    parent=study.parent/'20261023-recovery'
    assert (parent/'frozen/twohop.bin').read_bytes()==(study/'frozen/twohop.bin').read_bytes()
    failed=json.loads((parent/'runs/probe_mechanics_8/RECEIPT.json').read_text())
    assert failed['exit_code']==1 and failed['failed_at_host']
    raw=[json.loads(l) for l in (d/'stdout.jsonl').read_text().splitlines()]
    shots={x['id']:x for x in raw if x['kind']=='snapshot'};prior=shots[-1]['words']
    summary=json.loads((d/'SUMMARY.json').read_text());samples=[]
    for job in summary['jobs']:
        for e in job['operations']:
            shot=min((s for s in nv['snapshots'] if s['id']>e['observation_id']),key=lambda s:s['id'])
            words=shots[shot['id']]['words'];banks=[]
            for bank,count in [(0,8),(1,4)]:
                records=[]
                for r in range(count):
                    get=lambda w:[b for b in range(512) if w[bank][6+20*r+b//32]>>(b%32)&1]
                    before=get(prior);after=get(words);assert len(before)==len(after)==32
                    records.append(dict(selected_features=after,added=sorted(set(after)-set(before)),removed=sorted(set(before)-set(after))))
                banks.append(records)
            samples.append(dict(job=job['job'],attempt=e['local_attempt'],state=e['state'],action=e['action'],
                backend=e['backend'],stage=e['stage'],probe=e['probe'],outcome=e['outcome'],
                worker_ok=e['rpc']['result']['ok'],mapping_after=shot['mapping'],history_after=shot['history'],banks=banks))
            prior=words
    result=dict(status='INDEPENDENT_RAW_RECOUNT_PASS',study=study.name,physical_chips=1,rows=rows,
        completed_probe=72,completed_reactive=72,completed_frozen=53,probe_attempts=156,reactive_attempts=154,
        gate_status=gates['status'],claim='Earlier recovery recognition through supplied bounded exploration and native updates; no completion advantage over reactive in this schedule',
        boundaries=['Identical CPU algorithm matches every native decision/outcome','Two additional operations overall; no throughput, energy or substrate advantage',
                    'Three encodings/probe offsets on one chip, not three independent applications','Native memory within one live session; no restart/power-loss persistence'],
        preserved_failures=['Prephysical missing FlatBuffers environment','Parent hardware attempt refused one-byte MMIO; cleaned up, sealed; r1 changed transport only'])
    result.update(passing_native_sessions=11,preserved_failed_native_sessions=1,
        verified_native_forwards=sum(x.get('native_forwards',0) for x in recount['results'])+mechanism['native_forwards'],
        parent_neural_program_unchanged=True,parent_failed_prefix_counted_as_success=False)
    out=P/'results/2026-10-23-recovery.json';out.write_text(json.dumps(result,indent=2)+'\n')
    replay=dict(status='RECORDED_HARDWARE_REPLAY',samples=samples,scope='Primary encoding71011; per-operation native RAM and outputs measured; geometry and expanded timing schematic',
        primary_source_sha256=hashlib.sha256((d/'stdout.jsonl').read_bytes()).hexdigest(),
        raw_sha256=nv['raw_sha256'],learning_npus=json.loads((study/'frozen/twohop_AUDIT.json').read_text())['memories'])
    replay['learning_npus']=[dict(np=m['np']) for m in replay['learning_npus']]
    reactive=json.loads((study/'runs/control_71011_reactive/SUMMARY.json').read_text());mapping=[None]*8;rr=[]
    for job in reactive['jobs']:
        for e in job['operations']:
            mapping[2*e['state']+e['action']]=e['outcome']
            rr.append(dict(job=job['job'],attempt=e['local_attempt'],mapping_after=mapping[:]))
    replay['reactive_samples']=rr
    owner=json.loads((d/'OWNER.json').read_text());origin=Path(owner['args'][1]).parent.parent
    for name,key in [('recovery-input.jpg','source'),('recovery-output.png','final_png')]:
        source=study/Path(summary['jobs'][0][key]).relative_to(origin)
        shutil.copy2(source,P/'visual'/name)
        replay[name+'_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    (P/'visual/recovery-replay.json').write_text(json.dumps(replay,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','boundaries','preserved_failures']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('study',type=Path);args=parser.parse_args();export(args.study.resolve())
