"""Portable source-swapped replication; reuse the qualified raw exporter."""
import argparse,json,types
import numpy as np
from admission import consensus_archive as exporter,consensus_crossed_run as mod
from admission.consensus_archive import digest,get_run
from admission.run import save,sha
BASE=mod.P/'results/20261025-consensus-crossed'

def audit():
    m=json.loads(BASE.with_suffix('.json').read_text());assert sha(BASE.with_suffix('.npz'))==m['archive_sha256'];rows=[]
    ancestor=json.loads(exporter.BASE.with_suffix('.json').read_text())
    validate_ancestry(m,ancestor)
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for f in m['fixtures'].values():
            for p in f['arrays'].values():assert digest(z[p['key']])==p['sha256']
        for name,r in m['runs'].items():
            assert r['verified']
            for k,p in r['arrays'].items():assert digest(z[p['key']])==p['sha256']==r['receipt']['hashes']['result.'+k]
            if r['packets']:
                p=r['packets'];v=z[p['key']];assert digest(v)==p['sha256']
                for i,row in enumerate(v):assert digest(row)==r['receipt']['hashes'][f'result.packet{i}']
            _,get,fg=get_run(z,m,name);a=mod.check(dict(r['receipt'],stdout_report=json.loads(r['stdout'])),get,fg);a['name']=name;rows.append(a)
        native=[r for r in rows if r['mode']=='native'];assert len(rows)==6 and {r['seed'] for r in native}==set(mod.SEEDS) and sum(r['events'] for r in native)==2184
        assert all(r['useful_gate'] for r in native)
        endpoints={}
        for endpoint in ['protected','updated','pending']:
            q=[v for r in native for p,v in r['assessment'].items() if p.startswith(endpoint+'_exam_')]
            endpoints[endpoint]=dict(correct=sum(v['correct'] for v in q),queries=sum(v['queries'] for v in q),cpu_agreement_correct=sum(v['cpu_agreement_correct'] for v in q),cpu_follow_a_correct=sum(v['cpu_follow_a_correct'] for v in q),cpu_follow_b_correct=sum(v['cpu_follow_b_correct'] for v in q))
        assert endpoints['protected']==dict(correct=288,queries=288,cpu_agreement_correct=288,cpu_follow_a_correct=216,cpu_follow_b_correct=216)
        s=m['closed_state'];assert s['closed'] and not s.get('failure') and not s.get('scientific_stop')
    out=dict(status='NUMERIC_ARCHIVE_PASS',archive_sha256=m['archive_sha256'],parent_archive_sha256=m['parent_archive_sha256'],physical_sessions=3,native_forwards=2184,endpoints=endpoints,stable_original_payloads=sum(len(r['stable_records']) for r in native),blocked_student_ram_checks=sum(len(r['blocked_student_ram']) for r in native),scope=m['scope'],rows=rows)
    save(BASE.with_suffix('.audit.json'),out);return out

def validate_ancestry(m,ancestor):
    assert m['parent_archive_sha256']==ancestor['archive_sha256']
    for name,f in m['fixtures'].items():
        seed=f['metadata']['seed'];old=ancestor['runs'][f'native_s{seed}_native_repeat']['source_freeze']['hashes']
        prefix=f'studies/20261025-consensus-memory/fixtures02/{name}/'
        for filename,h in f['metadata']['inherited_files'].items():assert old[prefix+filename]==h,'exact reviewed ancestor program/matrix'
        for filename,entry in f['arrays'].items():assert entry['sha256']==f['metadata']['inherited_files'][filename],'inherited numeric fixture'

def build():
    def finalize():
        m=json.loads(BASE.with_suffix('.json').read_text());m['parent_archive_sha256']=json.loads(exporter.BASE.with_suffix('.json').read_text())['archive_sha256']
        m['scope']='Source-identity-swapped replication: identical parent hardware/program/initialization and two cycles, B corrupted in cycle1 and A in cycle2. CPU always-A, always-B and agreement controls receive the same raw source facts and transfer clocks. Evaluation truth never enters updates. '+m['scope']
        save(BASE.with_suffix('.json'),m);return audit()
    # Exporter has no closures or arguments. Isolated globals preserve its exact
    # reviewed array/digest/packet packing and do not mutate the parent module.
    fn=types.FunctionType(exporter.build.__code__,dict(exporter.build.__globals__,mod=mod,BASE=BASE,audit=finalize),name='crossed_export')
    return fn()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);args=p.parse_args();r=build() if args.action=='build' else audit();print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
