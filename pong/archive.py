"""Export recheckable native evidence without licensed programs or runtime code."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from verify import verify,read_words
from pong.experiment import save,sha,design,schedule


def export(study):
    arrays,fixtures,runs={},{},[]
    for raw in sorted((study/'raw').iterdir()):
        rec=json.loads((raw/'RECEIPT.json').read_text()); audit=verify(ROOT,raw);key=raw.name;f=raw/'fixture'
        n,u,nw,classes=map(int,f.with_suffix('.meta').read_text().split())
        fixtures[key]=dict(inputs=n,neurons=u,num_weights=nw,classes=classes)
        for suffix,dtype in [('inputs','u1'),('labels','<u4'),('ram','<u4')]:
            arrays[f'fixture__{key}__{suffix}']=np.frombuffer(f.with_suffix('.'+suffix).read_bytes(),dtype).copy()
        for mode in ['native','cpu']:
            for suffix in ['scores','ram']:
                arrays[f'{key}__{mode}_{suffix}']=np.array(read_words(raw/f'{mode}.{suffix}'),np.uint32)
        arrays[key+'__snapshots']=np.array([read_words(raw/f'native.snapshot_{i}.ram') for i in range(rec['samples'])],np.uint32)
        runs.append(dict(name=key,fixture_key=key,samples=rec['samples'],learning=1,audit=audit,
            binary_sha256=rec['binary_sha256'],program_sha256=rec['program_sha256']))
    archive=ROOT/'results/2026-10-23-pong-physics.npz';np.savez_compressed(archive,**arrays)
    pair=json.loads((study/'evidence/PAIR_development_85201.json').read_text())
    plan=json.loads((study/'PROTOCOL.json').read_text());rows=schedule(plan)
    truth=[r['truth'] for r in rows if r['kind']=='changed_after_feedback']
    result=dict(status='STOP_PREDICTION_GATE_CONFIRMATION_UNOPENED',physical_chips=1,physical_initializations=1,
        prediction_pair=pair,changed_court_majority_constant=dict(correct=max(np.bincount(truth)),n=len(truth),
            role='evaluation-only best constant; selected after outcomes, not an acting policy'),
        physical_sessions=len(runs),verified_native_forwards=sum(r['samples'] for r in runs),
        archive_sha256=sha(archive),fixtures=fixtures,runs=runs,
        confirmation_opened=False,paddle_control_admitted=False,
        posthoc_diagnostic=json.loads((study/'evidence/DIAGNOSTIC.json').read_text()),
        workflow='host motion encoder -> native landing-band match -> host display -> observed landing -> native feedback',
        attempts=[dict(status='COMPILER_ONLY_UNSUPPORTED_FORMAT',neurons=320,physical_device_opened=False,
            error='FNP2 external-memory record not covered by FNP3 decoder'),
            dict(status='ENCODER_PREFLIGHT_REJECTED',physical_device_opened=False,
                error='Coarse position/velocity code aliased different outcomes; replaced before hardware runs'),
            dict(status='NATIVE_SMOKE_PASS',forwards=32),dict(status='PREDICTION_GATE_FAILED',seed=85201)],
        interpretation='Native updates improve the first matched comparison, but changed-physics accuracy remains only 12/48; no useful adaptive Pong controller established',
        limitations=plan['claims']['limitations']+[
            'One development seed and one fixed bank; no multi-seed or held-out confirmation claim',
            'Unchanged court also receives further correct feedback; this retention endpoint does not isolate old-memory preservation',
            'Live film is a sequential matched shadow capture of 16 serves, not continuous learned paddle play',
            'Native values are integer connection-match scores, not electrical spike measurements'])
    # Convert numpy scalars for JSON.
    result['changed_court_majority_constant']['correct']=int(result['changed_court_majority_constant']['correct'])
    save(ROOT/'results/2026-10-23-pong-physics.json',result)
    captures={}
    for arm in ['full','frozen']:
        directory=study/'runs'/f'live_85201_{arm}'
        receipt=json.loads((directory/'CAPTURE.json').read_text());audit=json.loads((directory/'AUDIT.json').read_text())
        assert audit['status']=='INDEPENDENT_LIVE_AUDIT_PASS'
        events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        events=[e for e in events if e['kind']!='preparation']
        target=ROOT/f'visual/pong-physics-{arm}-game.npz';shutil.copy2(directory/'game.npz',target)
        captures[arm]=dict(receipt=receipt,audit=audit,events=events,game_path=str(target.relative_to(ROOT)),
            game_sha256=sha(target),raw_key=f'live_85201_{arm}')
    save(ROOT/'visual/pong-physics-data.json',dict(status='RECORDED_NATIVE_SHADOW_CAPTURE',captures=captures,
        archive_sha256=sha(archive),protocol=plan,
        rendering='Reconstructed Pong serves at recorded wall times; physical runs sequential, aligned at capture starts; no future replies displayed'))
    print(json.dumps({k:result[k] for k in ['status','physical_sessions','verified_native_forwards','archive_sha256']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);export(p.parse_args().study.resolve())
