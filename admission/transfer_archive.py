"""Portable readback evidence for learned-source native teaching."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from admission import transfer_run as mod
from admission.run import save,sha
P=Path(__file__).resolve().parents[1];BASE=P/'results/20261025-native-transfer'
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
        name=f'{d.parent.name}__{d.name}';ff={};packets=[]
        for p in sorted(d.glob('result.*')):
            k=p.name[7:]
            if k.startswith('packet'):packets.append((int(k[6:]),np.fromfile(p,'<u4')));continue
            key=f'{name}__{k.replace(".","_")}';arrays[key]=np.fromfile(p,dtype(k));ff[k]=dict(key=key,sha256=sha(p))
        pm=None
        if packets:
            packets.sort(key=lambda q:q[0]);assert [i for i,_ in packets]==list(range(len(packets)))
            key=f'{name}__packets';arrays[key]=np.stack([v for _,v in packets]);pm=dict(key=key,sha256=digest(arrays[key]),rows=len(packets))
        runs[name]=dict(receipt=json.loads((d/'RECEIPT.json').read_text()),verified=(d/'AUDIT.json').exists(),audit=json.loads((d/'AUDIT.json').read_text()) if (d/'AUDIT.json').exists() else None,arrays=ff,packets=pm,stdout=(d/'stdout').read_text(),stderr=(d/'stderr').read_text(),source_freeze=json.loads((d/'FROZEN.json').read_text()))
    np.savez_compressed(BASE.with_suffix('.npz'),**arrays)
    save(BASE.with_suffix('.json'),dict(archive_sha256=sha(BASE.with_suffix('.npz')),fixtures=fixtures,runs=runs,closed_state=json.loads((mod.S/'STATE.json').read_text()),compiler_attempts=[dict(path=str(p.relative_to(mod.S)),receipt=json.loads(p.read_text())) for p in mod.S.glob('fixtures*/*/BUILD.json') if json.loads(p.read_text())['status']!='VIRTUAL_COMPILER_PASS'],scope='Learned source, two valence student banks, seven sequential FNP3s. Host grounds source with external outcomes; all student transfer labels are zero and no outcomes are supplied. Native comparison/encoding determines student teaching. Teacher answer path silent during independent exams; teacher still clocked. Engineered code/rule, host-paced; CPU also passes. No efficiency or autonomous scheduling claim.'))
    return audit()

def get_run(z,m,name):
    r=m['runs'][name];f=m['fixtures'][f"s{r['receipt']['seed']}_{r['receipt']['arm']}"]
    def get(k):return z[r['packets']['key']][int(k[6:])] if k.startswith('packet') else z[r['arrays'][k]['key']]
    return r,get,lambda k:z[f['arrays'][k]['key']]

def audit():
    m=json.loads(BASE.with_suffix('.json').read_text());assert sha(BASE.with_suffix('.npz'))==m['archive_sha256'];rows=[];failures=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for f in m['fixtures'].values():
            for p in f['arrays'].values():assert digest(z[p['key']])==p['sha256']
        for name,r in m['runs'].items():
            for k,p in r['arrays'].items():
                assert digest(z[p['key']])==p['sha256'];assert p['sha256']==r['receipt']['hashes']['result.'+k]
            if r['packets']:
                p=r['packets'];assert digest(z[p['key']])==p['sha256']
                for i,row in enumerate(z[p['key']]):assert digest(row)==r['receipt']['hashes'][f'result.packet{i}']
            if not name.startswith('raw02__'):
                _,get,fg=get_run(z,m,name)
                count=len(get('scores'))//128;expected=mod.simulate([fg(f'bank{k}.ram') for k in range(3)],r['receipt']['stage'],r['receipt']['arm'])
                # Reconstruct the original query-only carry rule independently;
                # the repaired replay has identical output/RAM for this gate.
                legacy=[];last=np.zeros(128,'u1')
                for i,e in enumerate(expected['events'][:count]):
                    x=last.copy()
                    if e['role']!='transfer':
                        x[32:64]=0
                        for bit in range(4):x[32+bit*8+((e['key']>>bit)&1)*4:36+bit*8+((e['key']>>bit)&1)*4]=1
                    x[64:72]=0;x[65 if e['role']=='ground' else 69 if e['role']=='transfer' else 64]=1
                    if e['role']=='exam':x[70]=1
                    legacy.append(x)
                    if e['role']=='query':last=expected['outputs'][i].copy()
                assert np.array_equal(get('inputs').reshape(count,128),np.asarray(legacy)),'original actor input prefix'
                assert np.array_equal(get('scores').reshape(count,128),expected['outputs'][:count])
                for k in range(3):assert np.array_equal(get(f'states{k}').reshape(count,-1),expected[f'states{k}'][:count])
                if not r['verified']:
                    assert r['receipt']['exit_code']!=0 and r['receipt']['device_free']
                    failures.append(dict(name=name,captured_forwards=count,status='PRESERVED_EXECUTION_FAILURE_PREFIX_MATCHES_REPLAY',stderr=r['stderr']))
                continue # Original CPU cell/prefix are preserved, not counted twice.
            assert r['verified'],'no unreviewed post-repair failure'
            _,get,fg=get_run(z,m,name);receipt=dict(r['receipt'],stdout_report=json.loads(r['stdout']));a=mod.check(receipt,get,fg);a['name']=name;rows.append(a)
        state=m['closed_state'];assert state['closed'] and not state.get('failure')
        native={(r['seed'],r['arm'],r['stage']):r for r in rows if r['mode']=='native'}
        required={(mod.SEEDS[0],arm,'gate') for arm in mod.ARMS}|{(s,arm,'extended') for s in mod.SEEDS for arm in mod.ARMS}
        assert set(native)==required and sum(r['events'] for r in native.values())==2754
        primary={}
        for arm in mod.ARMS:
            rs=[native[s,arm,'extended'] for s in mod.SEEDS];ex=[r['assessment']['silent_final_exam'] for r in rs];all_ex=[r['assessment'][p] for r in rs for p in ['silent_exam','silent_final_exam']]
            primary[arm]=dict(final_correct=sum(q['student_correct'] for q in ex),final_queries=sum(q['queries'] for q in ex),all_exam_correct=sum(q['student_correct'] for q in all_ex),all_exam_queries=sum(q['queries'] for q in all_ex),connection_replacements=[sum(r['edits'][k]['connection_replacements'] for r in rs) for k in range(3)])
        cross={}
        for seed in mod.SEEDS:
            _,ng,_=get_run(z,m,f'raw02__native_s{seed}_native_extended')
            for arm in mod.ARMS[1:]:
                _,cg,_=get_run(z,m,f'raw02__native_s{seed}_{arm}_extended');assert np.array_equal(ng('states0'),cg('states0')),'source identical under cut/frozen controls'
                cross[f'{seed}_{arm}']=True
        cpu=sum(native[s,'native','extended']['assessment']['silent_final_exam']['cpu_student_correct'] for s in mod.SEEDS)
        source_retention={str(s):native[s,'native','extended']['assessment']['new_source_probe'] for s in mod.SEEDS}
    out=dict(status='NUMERIC_ARCHIVE_PASS',archive_sha256=m['archive_sha256'],qualified_physical_sessions=12,qualified_native_forwards=2754,attempted_physical_sessions=12+len(failures),preserved_failures=failures,compiler_failures=len(m['compiler_attempts']),primary=primary,cpu_final_correct=cpu,cpu_final_queries=288,source_trajectories_identical_in_controls=cross,source_after_new_facts=source_retention,rows=rows,scope=m['scope'])
    save(BASE.with_suffix('.audit.json'),out);return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);a=p.parse_args();r=build() if a.action=='build' else audit();print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
