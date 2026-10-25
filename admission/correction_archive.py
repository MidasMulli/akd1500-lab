"""Portable numeric-only evidence for error admission and its correction revision."""
from pathlib import Path
import argparse,json,tempfile
import numpy as np
from admission import error_run,recycle_run
from admission.run import sha,save
P=Path(__file__).resolve().parents[1];BASE=P/'results/20261025-native-correction'
FAMILIES={'error_only':error_run,'correction':recycle_run}
def dtype(name):return 'u1' if name in ['inputs','scores'] else '<i4' if name in ['potentials','learning_modes'] else '<u4'

def build():
    arrays={};families={}
    for family,mod in FAMILIES.items():
        fixtures={};runs={}
        for f in sorted(mod.F.glob('s*')):
            fs={}
            for p in sorted([*f.glob('*.ram'),*f.glob('*.threshold')]):
                name=f'{family}__{f.name}__{p.name.replace(".","_")}'
                arrays[name]=np.fromfile(p,'<u4');fs[p.name]=dict(key=name,sha256=sha(p))
            fixtures[f.name]=dict(arrays=fs,metadata=json.loads((f/'BUILD.json').read_text()))
        for d in sorted((mod.S/'raw').iterdir()):
            if not (d/'RECEIPT.json').exists():continue
            fs={}
            for p in sorted(d.glob('result.*')):
                name=p.name[7:];key=f'{family}__{d.name}__{name.replace(".","_")}'
                arrays[key]=np.fromfile(p,dtype(name));fs[name]=dict(key=key,sha256=sha(p))
            runs[d.name]=dict(receipt=json.loads((d/'RECEIPT.json').read_text()),verified=(d/'AUDIT.json').exists(),
                audit=json.loads((d/'AUDIT.json').read_text()) if (d/'AUDIT.json').exists() else None,
                arrays=fs,stdout=(d/'stdout').read_text(),stderr=(d/'stderr').read_text(),source_freeze=json.loads((d/'FROZEN.json').read_text()))
        families[family]=dict(fixtures=fixtures,runs=runs,closed_state=json.loads((mod.S/'STATE.json').read_text()),
            compiler_attempts=[dict(path=str(p.relative_to(mod.S)),receipt=json.loads(p.read_text())) for p in sorted(mod.S.glob('fixtures*/*/BUILD.json')) if json.loads(p.read_text())['status']!='VIRTUAL_COMPILER_PASS'])
    np.savez_compressed(BASE.with_suffix('.npz'),**arrays)
    save(BASE.with_suffix('.json'),dict(archive_sha256=sha(BASE.with_suffix('.npz')),families=families,
        provenance=dict(measured='native returned outputs, captured raw inputs, every learning-RAM boundary, initial/final fixed RAM, descriptors/output-source addresses',
            derived='fixed intermediate activations and potentials, task-level CPU dictionary/LRU controls; not hardware spike timing',
            journal='role-mode settings, counters and process receipts are own instrument/source journals'),
        scope='Designed native error comparator; later designed 25-common/7-key representation and positive/negative teaching phases. Host-paced external outcomes and fixed phase clocks, no host error decision or weight repair. Eight plastic records, 24 helpers, five sequential FNP3s; no energy/speed superiority or general importance-aware eviction claim.'))
    return audit()

def audit():
    m=json.loads(BASE.with_suffix('.json').read_text());assert sha(BASE.with_suffix('.npz'))==m['archive_sha256'];allrows={};failures=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z,tempfile.TemporaryDirectory() as td:
        for family,data in m['families'].items():
            mod=FAMILIES[family];rows=[]
            for bank in [*data['fixtures'].values(),*data['runs'].values()]:
                for v in bank['arrays'].values():
                    import hashlib
                    assert hashlib.sha256(z[v['key']].tobytes()).hexdigest()==v['sha256']
            for name,r in data['runs'].items():
                if not r['verified']:failures.append(dict(family=family,name=name,receipt=r['receipt'],stderr=r['stderr']));continue
                receipt=dict(r['receipt']);f=data['fixtures'][f"s{receipt['seed']}_{receipt['arm']}"]
                d=Path(td)/name;d.mkdir(exist_ok=True);(d/'stdout').write_text(r['stdout']);receipt['directory']=str(d)
                a=mod.check(receipt,lambda k:z[r['arrays'][k]['key']],lambda k:z[f['arrays'][k]['key']]);a['name']=name;rows.append(a)
                if receipt['mode']=='native':
                    report=json.loads(r['stdout']);assert report['program_calls']==1 and report['weight_restore_calls']==report['host_error_decisions']==0 and report['cleanup']=='unprogrammed'
            assert data['closed_state']['closed'] and not data['closed_state'].get('failure')
            allrows[family]=rows
        native={family:{(r['seed'],r['arm'],r['stage']):r for r in rows if r['mode']=='native'} for family,rows in allrows.items()}
        assert len(native['error_only'])==18 and sum(r['events'] for r in native['error_only'].values())==2304
        assert len(native['correction'])==16 and sum(r['events'] for r in native['correction'].values())==7468
        primary={}
        for arm in ['native','no_retire','frozen']:
            rs=[native['correction'][s,arm,'repeat'] for s in recycle_run.SEEDS]
            probes=[v for r in rs for k,v in r['assessment'].items() if k.startswith('probe_')]
            phases=[v for r in rs for v in r['assessment'].values()]
            primary[arm]=dict(probe_correct=sum(v['correct'] for v in probes),probe_total=sum(v['queries'] for v in probes),
                stable_correct=sum(v['stable_correct'] for v in probes),stable_total=sum(v['stable_total'] for v in probes),
                all_query_correct=sum(v['correct'] for v in phases),all_query_total=sum(v['queries'] for v in phases),
                native_connection_replacements=sum(r['connection_replacements'] for r in rs))
        stable_records=[]
        family=m['families']['correction']
        for seed in recycle_run.SEEDS:
            run=family['runs'][f'native_s{seed}_native_repeat'];f=family['fixtures'][f's{seed}_native']
            initial=z[f['arrays']['bank.ram']['key']];a=recycle_run.simulate(initial,'repeat','native')
            state=z[run['arrays']['states']['key']].reshape(-1,len(initial));y=z[run['arrays']['scores']['key']].reshape(-1,128)
            start=next(e['index'] for e in a['events'] if e['phase']=='old_probe')
            for key in [2,3]:
                i=next(e['index'] for e in a['events'] if e['phase']=='old_probe' and e['key']==key)
                j=recycle_run.PLASTIC[int(np.flatnonzero(y[i,72:80])[0])];b=2+j*36
                assert np.all(state[start:,b:b+36]==state[start,b:b+36]),'complete unrelated record unchanged'
                stable_records.append(dict(seed=seed,key=key,record=j,unchanged_words=36))
        native_rows=[native['correction'][s,'native','repeat'] for s in recycle_run.SEEDS]
        phases=[v for r in native_rows for v in r['assessment'].values()]
        cpu=dict(probe_correct=sum(v['cpu'] for r in native_rows for k,v in r['assessment'].items() if k.startswith('probe_')),
                 probe_total=1728,all_query_correct=sum(v['cpu'] for v in phases),all_query_total=sum(v['queries'] for v in phases))
        baseline={stage:{arm:native['error_only'][114101,arm,stage]['assessment'] for arm in ['native','open']} for stage in ['reversal','capacity']}
        # Fewer admitted teachers are not automatically fewer connection edits.
        # Different input sequences can change the rotating selector/swap order.
        admission_cost={arm:dict(admitted=sum(native['error_only'][s,arm,'learn']['admitted_teachers'] for s in error_run.SEEDS),
            replacements=sum(native['error_only'][s,arm,'learn']['connection_replacements'] for s in error_run.SEEDS)) for arm in ['native','open']}
    result=dict(status='NUMERIC_ARCHIVE_PASS',archive_sha256=m['archive_sha256'],
        totals={f:dict(qualified_physical_sessions=len(native[f]),native_forwards=sum(r['events'] for r in native[f].values())) for f in native},
        primary=primary,cpu=cpu,stable_records=stable_records,baseline_failures=baseline,error_only_admission_cost=admission_cost,cut_prediction_control=native['correction'][114101,'cut','correction']['assessment'],
        preserved_failures=failures,rows=allrows,scope=m['scope'])
    save(BASE.with_suffix('.audit.json'),result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);a=p.parse_args();r=build() if a.action=='build' else audit()
    print(json.dumps({k:v for k,v in r.items() if k not in ['rows','baseline_failures']}))
