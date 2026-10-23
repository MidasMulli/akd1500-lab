"""Publish owned numerical traces, without vendor runtime or generated programs."""
from pathlib import Path
import hashlib
import json
import numpy as np
from relay.audit import verify, words
from relay.design import gate,SEEDS

P = Path(__file__).resolve().parents[1]
S = P / 'studies/20261023-signal-relay'
TARGET = P / 'results/2026-10-23-signal-relay'


def main():
    arrays, runs = {}, []
    for d in sorted((S / 'raw').iterdir()):
        if not (d / 'RECEIPT.json').exists(): continue
        receipt = json.loads((d / 'RECEIPT.json').read_text())
        assert receipt['status'] == 'PASS', 'preserve and report failure before exporting'
        report = verify(d); name = d.name; key = name + '__'
        rows = json.loads((d / 'stream.rows.json').read_text())
        report['original_indices'] = [r['original_index'] for r in rows]
        report['receipt_sha256'] = hashlib.sha256((d / 'RECEIPT.json').read_bytes()).hexdigest()
        report['native_results'] = receipt['results']
        for ext, dtype, shape in [('inputs', np.uint8, (-1,512)), ('labels','<i4',(-1,)), ('masks',np.uint8,(-1,))]:
            arrays[key+ext] = np.frombuffer((d / ('stream.'+ext)).read_bytes(), dtype).reshape(shape)
        arrays[key+'outputs'] = np.frombuffer((d/'native.scores').read_bytes(), '<i4').reshape(-1,4)
        arrays[key+'latency_ns'] = np.frombuffer((d/'native.latency_ns').read_bytes(), '<i8')
        for k in range(2):
            arrays[key+f'bank{k}_initial'] = np.asarray(words(d/f'native.bank{k}.initial.ram'), '<u4')
            arrays[key+f'bank{k}_final'] = np.asarray(words(d/f'native.bank{k}.ram'), '<u4')
            arrays[key+f'bank{k}_snapshots'] = np.asarray([words(d/f'native.snapshot_{i}_bank{k}.ram')
                                                          for i in range(receipt['samples'])], '<u4')
        for k in range(3):
            for when in ['initial','final']:
                arrays[key+f'fixed{k}_{when}'] = np.asarray(words(d/f'native.fixed{k}.{when}.ram'), '<u4')
        runs.append(report)
    assert len(runs)==10 and sum(r['smoke'] for r in runs)==1,'do not publish an incomplete planned study'
    assert sorted({r['seed'] for r in runs if not r['smoke']})==SEEDS
    gates=[]
    for seed in sorted({r['seed'] for r in runs if not r['smoke']}):
        arms={r['arm']:r for r in runs if r['seed']==seed and not r['smoke']}
        assert set(arms)=={'recognition_repair','frozen','routing_only'}
        names=[arms[a]['run'] for a in ['recognition_repair','frozen','routing_only']]
        for name in names[1:]:
            assert np.array_equal(arrays[names[0]+'__outputs'][:640],arrays[name+'__outputs'][:640])
            for k in range(2):
                assert np.array_equal(arrays[names[0]+f'__bank{k}_snapshots'][639],arrays[name+f'__bank{k}_snapshots'][639])
        gates.append(dict(seed=seed,**gate(arms['recognition_repair'],arms['frozen']),shared_prefix_exact=True))
    np.savez_compressed(TARGET.with_suffix('.npz'), **arrays)
    result=dict(status='NATIVE_MODULAR_REPAIR_PASSED' if all(g['status']=='PASS' for g in gates) else 'FAILED_GATE_PRESERVED',
                protocol=json.loads((S/'PROTOCOL.json').read_text()),runs=runs,gates=gates,
                physical_sessions=len(runs),native_forwards=sum(r['samples'] for r in runs),
                captured_full_bank_snapshots=sum(r['measured_full_bank_snapshots'] for r in runs),
                fixtures={str(seed):json.loads((S/f'fixtures-relay03/s{seed}.json').read_text()) for seed in sorted({r['seed'] for r in runs})},
                failed_attempts=json.loads((S/'evidence/REPAIRS.json').read_text()),
                archive_sha256=hashlib.sha256(TARGET.with_suffix('.npz').read_bytes()).hexdigest(),
                boundaries=['One chip; three initializations of the same synthetic sensor world',
                            'Different teachers and phase-gated learning; no simultaneous independent sensor streams',
                            'Fixed supplied categorical messaging and readout; no native search or recurrence',
                            'Source-bound replay, not electrical spike telemetry',
                            'Exact CPU reconstruction also solves this task; no CPU speed or energy advantage'])
    TARGET.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','physical_sessions','native_forwards','captured_full_bank_snapshots','gates']}))


if __name__=='__main__':main()
