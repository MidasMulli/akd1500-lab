"""Identify newly modified records and test preservation of earlier native memories."""
from pathlib import Path
import hashlib
import json
import numpy as np
from resident.or_archive import BASE,expand,audit_public
from resident.run import P
OUT=P/'results/2026-10-23-fnp-memory-recruitment.json'


def derive(meta,z):
    rows=[]
    for rec in meta['runs']:
        if rec['smoke'] or rec['arm']!='repair':continue
        prefix=rec['name']+'__';initial=z[prefix+'bank0.initial']
        states=expand(initial,{key:z[prefix+'bank0.delta.'+key] for key in ['offsets','indices','values','sha256']},rec['samples'])
        before=initial;phase_ids={}
        for i,after in enumerate(states):
            aa=before[2:].reshape(512,20)[:,4:];bb=after[2:].reshape(512,20)[:,4:]
            ids=np.flatnonzero((aa!=bb).any(1));phase_ids.setdefault(rec['rows'][i]['phase'],set()).update(map(int,ids));before=after
        old=sorted(phase_ids['train0']);new=sorted(phase_ids['repair']);end=max(i for i,r in enumerate(rec['rows']) if r['phase']=='train0')
        positions=np.asarray([2+j*20+w for j in old for w in range(20)])
        same=bool(np.all(states[end+1:,positions]==states[end,positions]));disjoint=not bool(set(old)&set(new))
        new_positions=np.asarray([2+j*20+w for j in new for w in range(20)])
        untouched=bool(np.all(states[:end+1,new_positions]==initial[new_positions]))
        assert len(old)==len(new)==4 and disjoint and untouched and same, 'native old/new record qualification'
        rows.append(dict(seed=rec['seed'],old_connection_edited_records=old,new_connection_edited_records=new,
            new_records_not_modified_during_initial_teaching=untouched,all_old_record_words_unchanged_after_initial_teaching=same,
            new_probe_correct=rec['audit']['metrics']['changed_after']['correct'],old_retention_correct=rec['audit']['metrics']['retention']['correct']))
    assert len(rows)==3
    return dict(status='MEASURED_NATIVE_MEMORY_RECRUITMENT',seeds=rows,
        claim='Four additional existing recognition records changed connections while the four initial learned records retained all20 words exactly, in each tested initialization.',
        boundary='Host supplies class teachers and mode changes; native rule selects winning records/connections. No physical neuron growth, general forgetting immunity, or CPU/energy advantage.',
        source_archive_sha256=hashlib.sha256(BASE.with_suffix('.npz').read_bytes()).hexdigest())


def evaluate():
    audit_public();m=json.loads(BASE.with_suffix('.json').read_text())
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:return derive(m,z)


if __name__=='__main__':
    import sys
    result=evaluate()
    if '--save' in sys.argv:OUT.write_text(json.dumps(result,indent=2)+'\n')
    else:assert result==json.loads(OUT.read_text())
    print(json.dumps(result))
