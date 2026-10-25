"""Portable numeric receipts: independently recompute functional and time endpoints."""
import argparse, json
import numpy as np
from admission import latency as m

BASE=m.P/'results/20261025-consensus-latency'

def digest(v):return __import__('hashlib').sha256(v.tobytes()).hexdigest()
def units(v,unit):return {k.replace('_ns','_'+unit):x for k,x in m.stats(v).items()}

def build():
    arrays={};meta=dict(fixtures={},runs={},failures=[],timing_fields=m.FIELDS)
    for seed in m.SEEDS:
        f=m.S/'fixtures'/f's{seed}';q=dict(metadata=json.loads((f/'BUILD.json').read_text()),banks=[])
        for k in range(3):
            name=f'fixture_s{seed}_bank{k}';v=np.fromfile(f/f'bank{k}.ram','<u4');arrays[name]=v
            q['banks'].append(dict(key=name,sha256=digest(v)))
        meta['fixtures'][str(seed)]=q
    for d in sorted((m.S/'raw').iterdir()):
        if not (d/'RECEIPT.json').exists():continue
        r=json.loads((d/'RECEIPT.json').read_text())
        if not (d/'AUDIT.json').exists():
            meta['failures'].append(dict(name=d.name,receipt=r,
                 debugger=(d/'gdb.txt').read_text() if (d/'gdb.txt').exists() else None,
                 source_freeze=json.loads((d/'FROZEN.json').read_text()) if (d/'FROZEN.json').exists() else None));continue
        a=m.audit(d)
        entry=dict(receipt=r,stdout=json.loads((d/'stdout').read_text()),
                   acting_freeze=json.loads((d/'FROZEN.json').read_text()),arrays={})
        for p in sorted(d.glob('result.*')):
            ext=p.name.removeprefix('result.')
            dtype='u1' if ext=='scores' else '<i8' if ext in ['timing','clock_ns'] else '<i4' if ext=='answers' else '<u4'
            v=np.fromfile(p,dtype);key=f'{d.name}_{ext}';arrays[key]=v
            entry['arrays'][ext]=dict(key=key,sha256=digest(v))
            assert digest(v)==r['hashes'][p.name]
        meta['runs'][d.name]=entry
    meta['closed_state']=json.loads((m.S/'STATE.json').read_text())
    np.savez_compressed(BASE.with_suffix('.npz'),**arrays)
    meta['archive_sha256']=m.sha(BASE.with_suffix('.npz'))
    m.save(BASE.with_suffix('.json'),meta)
    return audit()

def check_run(r,report,get,reference):
    rows=reference['events'];n=len(rows)
    t=get('timing').reshape(n,len(m.FIELDS))
    assert (t>=0).all() and (t[:,10]>=t[:,9]).all()
    assert np.all(t[1:,9]>=t[:-1,10]),'monotonic ordered events'
    assert np.array_equal(t[:,:7].sum(1),t[:,10]-t[:,9]),'complete operational time partition'
    if r['mode']=='dictionary':
        answer=get('answers');assert np.array_equal(answer,reference['cpu'][:,0])
        assert report['agreed_update_batch_ns']>0 and report['agreed_update_batch_operations']>0
        assert not report['device_opened']
    else:
        y=get('scores').reshape(n,128);assert np.array_equal(y,reference['outputs'])
        answer=m.decode(y)
        for k in range(3):
            assert np.array_equal(get(f'final{k}'),reference[f'states{k}'][-1])
            if r['verification']=='all':assert np.array_equal(get(f'states{k}').reshape(reference[f'states{k}'].shape),reference[f'states{k}'])
    if r['mode']=='native':
        assert report['host_writes_during_execution']==0 and r['device_free']
        assert report['cleanup']=='unprogrammed' and report.get('device_opened',True)
    for prefix in ['protected_exam','updated_exam']:
        ii=[i for i,e in enumerate(rows) if e['phase'].startswith(prefix)]
        assert all(answer[i]==rows[i]['actual'] for i in ii)
    return m.convergence(rows,answer,t)

def audit():
    meta=json.loads(BASE.with_suffix('.json').read_text());assert m.sha(BASE.with_suffix('.npz'))==meta['archive_sha256']
    ancestor=json.loads((m.P/'results/20261025-consensus-memory.json').read_text())
    counts={};raw=[];summary={};cache={}
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for seed,fixture in meta['fixtures'].items():
            inherited=fixture['metadata']['inherited_files']
            freeze=ancestor['runs'][f'native_s{seed}_native_repeat']['source_freeze']['hashes']
            for name,h in inherited.items():assert freeze[f'studies/20261025-consensus-memory/fixtures02/s{seed}_native/{name}']==h,'qualified parent program/data ancestry'
            for bank in fixture['banks']:assert digest(z[bank['key']])==bank['sha256']
        for name,entry in meta['runs'].items():
            r=entry['receipt'];seed=r['seed'];stage=r['stage'];key=(seed,stage)
            if key not in cache:
                cache[key]=m.simulate([z[q['key']] for q in meta['fixtures'][str(seed)]['banks']],stage,'native')
            ref=cache[key]
            for ext,q in entry['arrays'].items():assert digest(z[q['key']])==q['sha256']==r['hashes']['result.'+ext]
            get=lambda ext:z[entry['arrays'][ext]['key']]
            con=check_run(r,entry['stdout'],get,ref)
            if stage!='repeat':continue
            mode=r['mode'];counts[mode]=counts.get(mode,0)+1
            assert r['verification']=='final'
            t=get('timing').reshape(-1,len(m.FIELDS));rows=ref['events']
            accepted=[i for i,e in enumerate(rows) if e['phase'].startswith('accepted_transfer') and e['role']=='transfer']
            revised=[c['committed_event'] for c in con]
            assert len(revised)==12 and all(c['teaching_attempts']==1 for c in con)
            for i in revised:
                if mode!='dictionary':assert any(not np.array_equal(ref[f'states{k}'][i],ref[f'states{k}'][i-1]) for k in [1,2]),'commit must change student RAM'
            row=dict(name=name,mode=mode,seed=seed,repetition=r['repetition'],convergence=con,
                accepted_timings=t[accepted],revision_timings=t[revised],clock=get('clock_ns'),report=entry['stdout'])
            raw.append(row)
        assert counts==dict(dictionary=9,native=9,replica=9),counts
        for mode in counts:
            selected=[r for r in raw if r['mode']==mode];t=np.concatenate([r['accepted_timings'] for r in selected]);v=np.concatenate([r['revision_timings'] for r in selected]);cc=sum([r['convergence'] for r in selected],[])
            assert {(r['seed'],r['repetition']) for r in selected}=={(seed,rep) for seed in m.SEEDS for rep in range(3)},'complete seed/repetition coverage'
            out=dict(runs=len(selected),accepted_transfer=m.stats(m.operational(t)),actual_revision=m.stats(m.operational(v)),
                components={name:m.stats(t[:,j]) for j,name in enumerate(m.FIELDS[:9])},
                revision_components={name:m.stats(v[:,j]) for j,name in enumerate(m.FIELDS[:9])},
                revision_bytes={name:units(v[:,j],'bytes') for j,name in [(11,'read_bytes'),(12,'write_bytes')]},
                clock_call_floor=m.stats(np.concatenate([r['clock'] for r in selected])),
                convergence={name:m.stats([c[name] for c in cc]) for name in ['learner_learning_path_ns','learner_operational_until_confirmed_ns','learner_observed_until_confirmed_ns','operational_until_confirmed_ns','observed_until_confirmed_ns','confirming_queries_ns']},
                teaching_attempts=units([c['teaching_attempts'] for c in cc],'attempts'),
                protected_recall=dict(correct=2592,queries=2592),updated_recall=dict(correct=2592,queries=2592),
                final_readback=m.stats([r['report']['final_readback_ns'] for r in selected]),
                per_run_revision=[dict(name=r['name'],**m.stats(m.operational(r['revision_timings']))) for r in selected])
            if mode=='dictionary':
                out['agreed_policy_batch_ns_per_update']=m.stats([r['report']['agreed_update_batch_ns']/r['report']['agreed_update_batch_operations'] for r in selected])
                out['mixed_stream_batch_ns_per_event']=m.stats([r['report']['dictionary_batch_ns']/(3064*r['report']['dictionary_batch_repeats']) for r in selected])
            summary[mode]=out
    primary=[entry for entry in meta['runs'].values() if entry['receipt']['stage']=='repeat']
    binary_hashes={entry['acting_freeze']['hashes']['build/consensus_latency'] for entry in primary}
    assert len(binary_hashes)==1,'all primary paths use one qualified executable'
    for name,entry in meta['runs'].items():
        if entry['receipt']['stage']!='repeat':continue
        frozen=entry['acting_freeze']['hashes']
        for path in ['admission/latency_bench.cpp','admission/latency_run.py','admission/build_latency.sh','src/readout_bench.cpp','src/mmio_driver.hpp']:
            assert m.sha(m.S/'software_versions'/__import__('pathlib').Path(path).name)==frozen[path],'preserved acting source matches primary freeze'
        assert m.sha(m.S/'software_versions/latency_executed.py')==frozen['admission/latency.py'],'preserved executed reference/audit version'
    assert meta['closed_state']['closed'] and not meta['closed_state'].get('failure')
    result=dict(status='PORTABLE_LATENCY_ARCHIVE_PASS',archive_sha256=meta['archive_sha256'],summary=summary,
        primary_binary_sha256=next(iter(binary_hashes)),
        passing_runs=len(meta['runs']),preserved_failed_or_diagnostic_attempts=len(meta['failures']),
        primary_native_sessions=9,primary_native_forwards=9*3064,
        qualification_native_sessions=3,qualification_native_forwards=3*624,
        scope='Four-cue sequential seven-FNP3 resident circuit on Nano CPU2. Same state/output behavior; no energy claim. Completion observation includes PCIe polling. CPU direct policy batches are different code from neural simulation. Primary results use final-only full-RAM audits; every output and final RAM remains exact. Stable recall means 16/16 teacher-silent probes and no later same-cycle error. Recorded verification and pure operational times remain separate.')
    m.save(BASE.with_suffix('.audit.json'),result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);a=p.parse_args()
    r=build() if a.action=='build' else audit();print(json.dumps({k:v for k,v in r.items() if k!='summary'}))
