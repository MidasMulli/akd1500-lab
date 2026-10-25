"""Portable native correction-through-transfer evidence, including controls."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from admission import consensus_run as mod
from admission.run import save,sha
P=Path(__file__).resolve().parents[1];BASE=P/'results/20261025-consensus-memory'
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def dtype(name):return 'u1' if name in ['inputs','scores'] else '<i4' if name.startswith('potentials') or name=='learning_modes' else '<u4'

def build():
    arrays={};fixtures={};runs={}
    for f in sorted(mod.F.glob('s*')):
        ff={}
        for p in sorted([*f.glob('*.ram'),*f.glob('*.threshold')]):
            key=f'{f.name}__{p.name.replace(".","_")}';arrays[key]=np.fromfile(p,'<u4');ff[p.name]=dict(key=key,sha256=sha(p))
        fixtures[f.name]=dict(arrays=ff,metadata=json.loads((f/'BUILD.json').read_text()))
    for d in sorted(mod.S.glob('raw*/*')):
        if not (d/'RECEIPT.json').exists():continue
        ff={};packets=[];name=d.name
        for p in sorted(d.glob('result.*')):
            k=p.name[7:]
            if k.startswith('packet'):packets.append((int(k[6:]),np.fromfile(p,'<u4')));continue
            key=f'{name}__{k.replace(".","_")}';arrays[key]=np.fromfile(p,dtype(k));ff[k]=dict(key=key,sha256=sha(p))
        pm=None
        if packets:
            packets.sort(key=lambda q:q[0]);assert [i for i,_ in packets]==list(range(len(packets)));key=f'{name}__packets';arrays[key]=np.stack([v for _,v in packets]);pm=dict(key=key,sha256=digest(arrays[key]),rows=len(packets))
        runs[name]=dict(receipt=json.loads((d/'RECEIPT.json').read_text()),verified=(d/'AUDIT.json').exists(),audit=json.loads((d/'AUDIT.json').read_text()) if (d/'AUDIT.json').exists() else None,arrays=ff,packets=pm,stdout=(d/'stdout').read_text(),stderr=(d/'stderr').read_text(),source_freeze=json.loads((d/'FROZEN.json').read_text()))
    np.savez_compressed(BASE.with_suffix('.npz'),**arrays)
    save(BASE.with_suffix('.json'),dict(archive_sha256=sha(BASE.with_suffix('.npz')),fixtures=fixtures,runs=runs,closed_state=json.loads((mod.S/'STATE.json').read_text()),admission_failures=[dict(path=str(p.relative_to(mod.S)),receipt=json.loads(p.read_text())) for p in mod.S.glob('raw_admission_failed/*/ADMISSION_FAILURE.json')],compiler_attempts=[dict(path=str(p.relative_to(mod.S)),receipt=json.loads(p.read_text())) for p in mod.S.glob('fixtures*/*/BUILD.json') if json.loads(p.read_text())['status']!='VIRTUAL_COMPILER_PASS'],scope='Seven sequential FNP3s: two learned source cohorts share one four-class bank, and designed native agreement regulates two constant-class student memories. External evidence and fixed roles are host-paced; no host semantic selection, weight repair or reset. Native source class flags, student responses and stored connections measured; hidden record potentials and features derived. Synthetic cues, one-response-per-source-class qualification, no general truth inference or autonomous scheduler. CPU controls also pass. Correlated false teaching is an explicit negative; no efficiency or CPU advantage claim.'))
    return audit()

def get_run(z,m,name):
    r=m['runs'][name];f=m['fixtures'][f"s{r['receipt']['seed']}_{r['receipt']['arm']}"];packets=z[r['packets']['key']] if r['packets'] else None;cache={}
    def get(k):
        if k.startswith('packet'):return packets[int(k[6:])]
        if k not in cache:cache[k]=z[r['arrays'][k]['key']]
        return cache[k]
    return r,get,lambda k:z[f['arrays'][k]['key']]

def audit():
    m=json.loads(BASE.with_suffix('.json').read_text());assert sha(BASE.with_suffix('.npz'))==m['archive_sha256'];rows=[];failed=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for f in m['fixtures'].values():
            for p in f['arrays'].values():assert digest(z[p['key']])==p['sha256']
        for name,r in m['runs'].items():
            for k,p in r['arrays'].items():assert digest(z[p['key']])==p['sha256']==r['receipt']['hashes']['result.'+k]
            if r['packets']:
                p=r['packets'];v=z[p['key']];assert digest(v)==p['sha256']
                for i,row in enumerate(v):assert digest(row)==r['receipt']['hashes'][f'result.packet{i}']
            if not r['verified']:failed.append(dict(name=name,receipt=r['receipt'],stderr=r['stderr']));continue
            _,get,fg=get_run(z,m,name);a=mod.check(dict(r['receipt'],stdout_report=json.loads(r['stdout'])),get,fg);a['name']=name;rows.append(a)
        state=m['closed_state'];assert state['closed'] and not state.get('failure') and not state.get('scientific_stop')
        native={(r['seed'],r['arm'],r['stage']):r for r in rows if r['mode']=='native'}
        required={(mod.SEEDS[0],a,'gate') for a in mod.ARMS}|{(s,a,'repeat') for s in mod.SEEDS for a in mod.ARMS}|{(s,'native',t) for s in mod.SEEDS for t in ['pressure','correlated']}
        assert set(native)==required and len(native)==22 and sum(r['events'] for r in native.values())==27132
        primary={}
        for arm in mod.ARMS:
            rs=[native[s,arm,'repeat'] for s in mod.SEEDS];v={}
            for endpoint in ['protected','updated','pending']:
                ex=[q for r in rs for p,q in r['assessment'].items() if p.startswith(endpoint+'_exam_')]
                v[endpoint]=dict(correct=sum(q['correct'] for q in ex),queries=sum(q['queries'] for q in ex),changed_correct=sum(q['changed_correct'] for q in ex),changed_queries=sum(q['changed_queries'] for q in ex),stable_correct=sum(q['stable_correct'] for q in ex),stable_queries=sum(q['stable_queries'] for q in ex),cpu_agreement_correct=sum(q['cpu_agreement_correct'] for q in ex),cpu_follow_b_correct=sum(q['cpu_follow_b_correct'] for q in ex))
            v['connection_replacements']=[sum(r['edits'][k]['connection_replacements'] for r in rs) for k in range(3)]
            v['stable_original_record_payloads']=sum(len(r['stable_records']) for r in rs)
            v['blocked_student_ram_checks']=sum(len(r['blocked_student_ram']) for r in rs)
            primary[arm]=v
        cross={}
        for seed in mod.SEEDS:
            _,get,_=get_run(z,m,f'native_s{seed}_native_repeat');n=1912
            for arm in mod.ARMS[1:]:
                _,cg,_=get_run(z,m,f'native_s{seed}_{arm}_repeat')
                assert np.array_equal(get('states0'),cg('states0')),'identical source trajectory across all policies'
                for k in range(3):assert np.array_equal(get(f'states{k}').reshape(n,-1)[:136],cg(f'states{k}').reshape(n,-1)[:136]),'common trained prefix'
            cross[str(seed)]=dict(source_trajectory_identical=mod.ARMS,all_banks_common_prefix_identical=True)
        pressure={str(s):native[s,'native','pressure']['assessment']['new_exam'] for s in mod.SEEDS}
        correlated={str(s):native[s,'native','correlated']['assessment']['correlated_exam_1'] for s in mod.SEEDS}
        assert all(native[s,'native',t]['useful_gate'] for s in mod.SEEDS for t in ['repeat','pressure','correlated'])
    out=dict(status='NUMERIC_ARCHIVE_PASS',archive_sha256=m['archive_sha256'],physical_sessions=22,native_forwards=27132,primary=primary,pressure=pressure,correlated_false_teaching=correlated,cross_arm_checks=cross,preserved_execution_failures=failed,admission_failures=m['admission_failures'],reserved_physical_attempts=22+len(m['admission_failures']),reserved_forwards=27132+sum(p['receipt']['reserved_events'] for p in m['admission_failures']),compiler_failures=len(m['compiler_attempts']),rows=rows,scope=m['scope'])
    save(BASE.with_suffix('.audit.json'),out);return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);a=p.parse_args();r=build() if a.action=='build' else audit();print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
