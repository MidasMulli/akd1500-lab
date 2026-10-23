"""Curate the first continuous matched pair, preserving its negative expansion gate."""
from pathlib import Path
import argparse,json,shutil,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from verify import verify,read_words
from mario.sealed.bank import save,sha


def export(study,full,frozen):
    from mario.continuous.environment import contexts,opened
    captures={};arrays={};fixtures={};runs=[];prefixes={}
    for arm,path in [('full',full),('frozen',frozen)]:
        report=json.loads((path/'REPORT.json').read_text());audit=json.loads((path/'INDEPENDENT_AUDIT.json').read_text())
        assert report['physical'] and audit['status']=='INDEPENDENT_CONTINUOUS_AUDIT_PASS'
        events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
        events=[e for e in events if e.get('phase')!='preparation']
        target=ROOT/f'visual/mario-continuous-{arm}-game.npz';shutil.copy2(path/'game.npz',target)
        captures[arm]=dict(report=report,audit=audit,events=events,game_sha256=sha(target),game_path=str(target.relative_to(ROOT)))
        frames=np.load(path/'game.npz',allow_pickle=False)['frames']
        for j,c in enumerate(contexts(report['split'])):
            env,history,sky,_,valid=opened(c)
            try:
                first=next(e['frame_id'] for e in events if e['kind']=='frame' and e['episode']==j)
                assert valid and np.array_equal(history[-1],frames[first])
                prefixes[f'{arm}__{j}__history']=np.array(history[-9:],np.uint8);prefixes[f'{arm}__{j}__sky']=sky
            finally:env.close()
    for path in sorted((study/'raw').iterdir()):
        rec=json.loads((path/'RECEIPT.json').read_text());verified=verify(ROOT,path);f=ROOT/rec['fixture'];key=path.name
        n,u,nw,classes=map(int,f.with_suffix('.meta').read_text().split());fixtures[key]=dict(inputs=n,neurons=u,num_weights=nw,classes=classes)
        for suffix,dtype in [('inputs','u1'),('labels','<u4'),('ram','<u4')]:arrays[f'fixture__{key}__{suffix}']=np.fromfile(f.with_suffix('.'+suffix),dtype)
        for mode in ['native','cpu']:
            for suffix in ['scores','ram']:arrays[f'{key}__{mode}_{suffix}']=np.array(read_words(path/f'{mode}.{suffix}'),np.uint32)
        arrays[key+'__snapshots']=np.array([read_words(path/f'native.snapshot_{i}.ram') for i in range(rec['samples'])],np.uint32)
        runs.append(dict(name=key,fixture_key=key,samples=rec['samples'],learning=1,audit=verified,program_sha256=rec['program_sha256'],binary_sha256=rec['binary_sha256']))
    archive=ROOT/'results/2026-10-23-mario-continuous.npz';np.savez_compressed(archive,**arrays)
    a=captures['full']['report'];b=captures['frozen']['report']
    regression=a['total_advance']<b['total_advance'] or a['surviving_episodes']<b['surviving_episodes']
    result=dict(status='MECHANISM_PASS_REGRESSION_STOP' if regression else 'MECHANISM_PASS_FIRST_PAIR',
        physical_sessions=len(runs),native_forwards=sum(r['samples'] for r in runs),archive_sha256=sha(archive),fixtures=fixtures,runs=runs,
        gate='STOP_DEVELOPMENT_REGRESSION_CONFIRMATION_UNOPENED' if regression else 'FIRST_PAIR_PASS',confirmation_outcomes_collected=False,
        native_pair={arm:dict(report=captures[arm]['report'],audit=captures[arm]['audit']) for arm in captures},
        total_advance=dict(learned=a['total_advance'],frozen=b['total_advance']),
        training=dict(contexts=96,valid_contexts=96,action_outcomes=192,unsafe=4,safe=188,epochs=4,presentations=1504,contradictory_encoded_inputs=0),
        cpu_constant_controls={str(i):json.loads((study/f'cpu_runs/constant{i}/REPORT.json').read_text()) for i in [0,1]},
        protocol_sha256=sha(study/'PROTOCOL.md'),design_freeze=json.loads((study/'evidence/DESIGN_FREEZE.json').read_text()),
        first_pair_gate=json.loads((study/'evidence/FIRST_PAIR_GATE.json').read_text()),source_freeze=json.loads((study/'evidence/PHYSICAL_SOURCE_FREEZE.json').read_text()),
        limitations=['One seed, one chip, four development starts; no reliability or unseen-level claim',
          'Frozen suppresses preparation and online learning; not an isolated online-update comparison after shared training',
          'New24-frame actions and outcome labels; direct accuracy comparison with the earlier90-frame task is invalid',
          'Natively chosen supplied motor primitives through authored RGB/time encoder; no learned visual perception or full-game RL',
          'Actual measured match outputs and visible RAM, not electrical spike timing or complete hidden state',
          'Matched arms were recorded sequentially on one chip; visual aligns episode starts, not simultaneous device execution',
          'All episodes ended in death; no level completion or useful learning advantage'])
    save(ROOT/'results/2026-10-23-mario-continuous.json',result)
    prefix_path=ROOT/'visual/mario-continuous-prefixes.npz';np.savez_compressed(prefix_path,**prefixes)
    display_data=dict(status='RECORDED_CONTINUOUS_PHYSICAL_PAIR',captures=captures,archive_sha256=sha(archive),prefix_sha256=sha(prefix_path),
         prefix_source='Deterministic fresh prefix replay checked against first captured frame; not live-captured prefix timing')
    (ROOT/'visual/mario-continuous-data.json').write_text(json.dumps(display_data,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs','fixtures','native_pair','source_freeze','cpu_constant_controls','limitations']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('full',type=Path);p.add_argument('frozen',type=Path);a=p.parse_args();export(a.study.resolve(),a.full.resolve(),a.frozen.resolve())
