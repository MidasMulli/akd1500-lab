"""Lossless sparse storage of complete native bank snapshots, with raw SHA checks."""
import hashlib
import json
import numpy as np
from resident import or_run
from resident.or_design import P,S,SEEDS
from resident.run import save,digest
BASE=P/'results/2026-10-23-fnp-or-relay'


def compact(initial,states):
    previous=initial.copy();offsets=[0];indices=[];values=[];hashes=[]
    for row in states:
        ii=np.flatnonzero(previous!=row);indices.extend(ii.tolist());values.extend(row[ii].tolist());offsets.append(len(indices))
        hashes.append(np.frombuffer(hashlib.sha256(row.astype('<u4').tobytes()).digest(),np.uint8));previous=row
    return dict(offsets=np.asarray(offsets,'<u4'),indices=np.asarray(indices,'<u2'),values=np.asarray(values,'<u4'),sha256=np.stack(hashes))


def expand(initial,encoded,count):
    offsets=encoded['offsets'];indices=encoded['indices'];values=encoded['values'];hashes=encoded['sha256']
    assert offsets.shape==(count+1,) and hashes.shape==(count,32) and offsets[0]==0
    assert np.all(offsets[1:]>=offsets[:-1]) and offsets[-1]==len(indices)==len(values), 'snapshot delta boundaries'
    previous=initial.copy();states=[]
    for i in range(count):
        lo,hi=map(int,offsets[i:i+2]);ii=indices[lo:hi]
        assert np.all(ii<len(previous)) and np.all(ii[1:]>ii[:-1]), 'snapshot delta addresses'
        previous[ii]=values[lo:hi]
        assert hashlib.sha256(previous.astype('<u4').tobytes()).digest()==hashes[i].tobytes(), 'complete snapshot checksum'
        states.append(previous.copy())
    return np.stack(states)


def expand_run(rec,z):
    result={k:v for k,v in z.items() if '.delta.' not in k}
    for k in range(3):
        encoded={name:z[f'bank{k}.delta.{name}'] for name in ['offsets','indices','values','sha256']}
        result[f'bank{k}.snapshots']=expand(z[f'bank{k}.initial'],encoded,rec['samples'])
    return result


def connection_stats(rec,z):
    rows=rec['rows'];width=rec['width'];edited=[];repair_events=[];insertions=[]
    for k in range(3):
        before=z[f'bank{k}.initial'];unique=set();events=0;added=0
        for i,after in enumerate(z[f'bank{k}.snapshots']):
            aa=before[2:].reshape(width,20)[:,4:];bb=after[2:].reshape(width,20)[:,4:];ids=np.flatnonzero((aa!=bb).any(1));unique.update(map(int,ids))
            if rows[i]['phase']=='repair':
                events+=int(len(ids)>0)
                added+=sum(int(v).bit_count() for v in (bb&~aa).ravel() if v)
            before=after
        edited.append(sorted(unique));repair_events.append(events);insertions.append(added)
    return dict(unique_records_with_connection_edits=edited,repair_connection_edit_events=repair_events,repair_inserted_connection_bits=insertions,
        allocated_learning_records=3*width)


def export():
    assert json.loads((S/'STATE.json').read_text())['research_admissions_closed']
    assert {d.name for d in (S/'raw').iterdir()}==or_run.expected_runs()
    runs=[];data={}
    for d in sorted((S/'raw').iterdir()):
        or_run.audit(d);rec=json.loads((d/'RECEIPT.json').read_text());rec['rows']=json.loads((d/'stream.rows.json').read_text());raw=or_run.arrays(d,rec)
        rec['audit']=or_run.recount(rec,raw);rec['connection_stats']=connection_stats(rec,raw);rec['receipt_sha256']=digest(d/'RECEIPT.json');rec.pop('hashes');rec.pop('native_pid')
        for k in range(3):
            enc=compact(raw[f'bank{k}.initial'],raw.pop(f'bank{k}.snapshots'))
            for key,v in enc.items():raw[f'bank{k}.delta.{key}']=v
        runs.append(rec)
        for key,v in raw.items():data[d.name+'__'+key]=v
    np.savez_compressed(BASE.with_suffix('.npz'),**data)
    frozen=json.loads((S/'evidence/FROZEN.json').read_text())['hashes']
    meta=dict(status='OBSERVED_THREE_FNP_OR_POPULATIONS',physical_sessions=len(runs),native_forwards=sum(r['samples'] for r in runs),
        full_bank_snapshots=sum(r['samples']*3 for r in runs),runs=runs,archive_sha256=digest(BASE.with_suffix('.npz')),
        snapshot_encoding='Lossless changed-word deltas of complete captured RAM; every reconstructed snapshot checked against SHA256 of original bytes.',
        executed_sources={p:h for p,h in frozen.items() if p in ['resident/or_design.py','resident/or_prepare.py','resident/or_run.py','src/or_bench.cpp']},
        limitations=['Host-scheduled teachers and modes; one synthetic four-class sensory world.',
            'Three initialization seeds are not three independently sampled worlds.',
            'Class-presence OR encoding; multiple class blocks can be active, no native search or argmax claim.',
            'Allocated1536 neuron records; only a subset actually changed connections.',
            'The wrong-bank control receives no eligible repair updates; it is not a matched-plastic-work intervention.',
            'No CPU or calibrated energy advantage; host timings are descriptive and transfers overlapped later confirmations.',
            'Failed FNP-to-CNP route remains unresolved; this uses a different FNP-only architecture.'])
    BASE.with_suffix('.json').write_text(json.dumps(meta,separators=(',',':'))+'\n');return audit_public()


def recount(meta,z):
    assert meta['status']=='OBSERVED_THREE_FNP_OR_POPULATIONS'
    assert {r['name'] for r in meta['runs']}==or_run.expected_runs(), 'complete OR-relay treatment set'
    assert meta['physical_sessions']==10 and meta['native_forwards']==6672 and meta['full_bank_snapshots']==20016
    checked=[]
    for rec in meta['runs']:
        a=expand_run(rec,{k.split('__',1)[1]:v for k,v in z.items() if k.startswith(rec['name']+'__')})
        result=or_run.recount(rec,a);assert result==rec['audit'], 'saved OR relay recount'
        assert connection_stats(rec,a)==rec['connection_stats'], 'recorded connection statistics'
        mp=rec['fixture_info']['mapping'];nps={tuple(p['np']) for l in mp for p in l['nps']}
        assert len(nps)==6 and all(p['type']=='Type.FNP3' for l in mp for p in l['nps']), 'six FNP-only resources'
        checked.append(result)
    from resident.design import gate
    for seed in SEEDS:
        rr={r['arm']:r for r in checked if r['seed']==seed and not r['smoke']};assert gate(rr['repair'],rr['frozen'])['status']=='PASS'
    return dict(status='INDEPENDENT_OR_RELAY_COMPLETE_RAM_PASS',physical_sessions=10,native_forwards=6672,full_bank_snapshots=20016,
        confirmation_seeds=SEEDS,allocated_learning_records=1536,resident_npus=6)


def audit_public():
    m=json.loads(BASE.with_suffix('.json').read_text());assert digest(BASE.with_suffix('.npz'))==m['archive_sha256']
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:return recount(m,{k:z[k] for k in z.files})


if __name__=='__main__':
    import sys
    print(json.dumps(export() if sys.argv[1]=='export' else audit_public()))
