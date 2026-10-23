"""Public recheckable relearning evidence with native and CPU forecasts distinct."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from verify import verify,read_words
from pong.experiment import save,sha


def export(study):
    arrays,fixtures,runs={},{},[]
    for raw in sorted((study/'raw').iterdir()):
        rec=json.loads((raw/'RECEIPT.json').read_text());checked=verify(ROOT,raw);key=raw.name;f=raw/'fixture'
        n,u,nw,classes=map(int,f.with_suffix('.meta').read_text().split())
        fixtures[key]=dict(inputs=n,neurons=u,num_weights=nw,classes=classes)
        for suffix,dtype in [('inputs','u1'),('labels','<u4'),('ram','<u4')]:arrays[f'fixture__{key}__{suffix}']=np.frombuffer(f.with_suffix('.'+suffix).read_bytes(),dtype).copy()
        for mode in ['native','cpu']:
            for suffix in ['scores','ram']:arrays[f'{key}__{mode}_{suffix}']=np.array(read_words(raw/f'{mode}.{suffix}'),np.uint32)
        arrays[key+'__snapshots']=np.array([read_words(raw/f'native.snapshot_{i}.ram') for i in range(rec['samples'])],np.uint32)
        arrays[key+'__native_stdout']=np.frombuffer((raw/'native.stdout').read_bytes(),'u1').copy()
        runs.append(dict(name=key,fixture_key=key,samples=rec['samples'],learning=1,audit=checked,
            binary_sha256=rec['binary_sha256'],program_sha256=rec['program_sha256']))
    # Compiler-only candidate initial states make CPU forecasts reproducible.
    for f in (study/'fixtures-relearn01').glob('*.ram'):
        arrays['forecast_initial__'+f.stem]=np.array(read_words(f),np.uint32)
    archive=ROOT/'results/2026-10-23-pong-relearning.npz';np.savez_compressed(archive,**arrays)
    triple=json.loads((study/'evidence/TRIPLE_development_nw64_s85301.json').read_text())
    screen=json.loads((study/'evidence/CPU_SCREEN.json').read_text())
    result=dict(status='DIAGNOSED_HISTORY_GAP_NO_QUALIFYING_REPAIR',physical_chips=1,physical_initializations=1,
        physical_sessions=len(runs),verified_native_forwards=sum(r['samples'] for r in runs),
        triple=triple,cpu_forecasts=screen,archive_sha256=sha(archive),fixtures=fixtures,runs=runs,
        protocol=json.loads((study/'PROTOCOL.json').read_text()),confirmation_opened=False,paddle_control_admitted=False,
        interpretation='Fresh new-rule learner46/64, relearning28/64, frozen12/64; B remains47/64 without reminders. Prior learning history impairs this development case; unique physical cause is not isolated.',
        limitations=['One physical seed; no multi-seed native qualification',
            '32/48-connection candidate results are CPU forecasts, not physical trials',
            'Fresh differs in complete native learning history; the gap is not proof of one hidden-state or stale-weight cause',
            'Timing primitives exclude observation encoding, bit packing, SSH and RAM capture; not complete system or energy comparisons',
            'CPU task baseline gets the same encoded inputs and permitted observed labels; it uses FIFO nearest-neighbour memory, not the AKD algorithm'])
    save(ROOT/'results/2026-10-23-pong-relearning.json',result)
    print(json.dumps({k:result[k] for k in ['status','physical_sessions','verified_native_forwards','archive_sha256']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);export(p.parse_args().study.resolve())
