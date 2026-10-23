"""Hardware-free semantic recount of every published input, output and RAM state."""
from pathlib import Path
import hashlib
import json
import numpy as np
from relay.reference import Network
from relay.design import schedule, metrics, gate

P=Path(__file__).resolve().parents[1]
BASE=P/'results/2026-10-23-signal-relay'


def audit_contents(report, data):
    checked=0;by_seed={}
    for r in report['runs']:
        key=r['run']+'__';x,rows=schedule(r['arm']);indices=r['original_indices'];selected=[rows[i] for i in indices]
        # Decompress each array once per run, rather than once per event.
        names=['inputs','labels','masks','outputs']
        names += [f'bank{k}_{part}' for k in range(2) for part in ['initial','snapshots','final']]
        names += [f'fixed{k}_{part}' for k in range(3) for part in ['initial','final']]
        current={key+name:data[key+name] for name in names}
        assert np.array_equal(current[key+'inputs'],x[indices]),'task inputs changed'
        assert np.array_equal(current[key+'labels'],[z['label'] for z in selected]),'teacher labels changed'
        assert np.array_equal(current[key+'masks'],[z['learning_mask'] for z in selected]),'learning isolation changed'
        net=Network([current[key+f'bank{k}_initial'].tolist() for k in range(2)])
        start=None;edits=[0,0]
        for i,row in enumerate(selected):
            if row['phase']=='repair' and start is None:start=[b.ram[:] for b in net.banks]
            before=[b.ram[:] for b in net.banks]
            out,_=net.forward(current[key+'inputs'][i],row['label'],row['learning_mask'])
            assert out==current[key+'outputs'][i].tolist(),('native output recount',r['run'],i)
            for k,b in enumerate(net.banks):
                state=current[key+f'bank{k}_snapshots'][i].tolist()
                assert state==b.ram,('native full RAM recount',r['run'],i,k)
                if not row['learning_mask']&(1<<k):assert state==before[k],('isolated bank mutated',r['run'],i,k)
                edits[k]+=sum((a^b).bit_count() for j,(a,b) in enumerate(zip(before[k],state))
                              if j>=2 and (j-2)%20>=4)
            checked+=1
        for k,b in enumerate(net.banks):assert b.ram==current[key+f'bank{k}_final'].tolist()
        for k in range(3):assert np.array_equal(current[key+f'fixed{k}_initial'],current[key+f'fixed{k}_final'])
        assert r['downstream_unchanged_during_repair']==(start is not None and start[1]==net.banks[1].ram)
        assert edits==r['connection_bit_edits']
        assert r['measured_full_bank_snapshots']==2*len(selected)
        assert r['exact_native_output_values']==4*len(selected)
        if not r['smoke']:
            assert r['metrics']==metrics(selected,current[key+'outputs'])
            by_seed.setdefault(r['seed'],{})[r['arm']]=r
    for seed,arms in by_seed.items():
        full,frozen=arms['recognition_repair'],arms['frozen']
        assert next(g for g in report['gates'] if g['seed']==seed)['status']==gate(full,frozen)['status']
        for other in [arms['frozen'],arms['routing_only']]:
            a,b=full['run']+'__',other['run']+'__'
            assert np.array_equal(data[a+'outputs'][:640],data[b+'outputs'][:640])
            for k in range(2):assert np.array_equal(data[a+f'bank{k}_snapshots'][639],data[b+f'bank{k}_snapshots'][639])
    assert checked==report['native_forwards']
    return dict(status='PASS',native_forwards=checked,full_bank_snapshots=checked*2,runs=len(report['runs']))


def audit():
    report=json.loads(BASE.with_suffix('.json').read_text())
    assert hashlib.sha256(BASE.with_suffix('.npz').read_bytes()).hexdigest()==report['archive_sha256']
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as data:return audit_contents(report,data)


if __name__=='__main__':print(json.dumps(audit()))
