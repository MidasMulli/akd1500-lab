"""Portable native correction-through-transfer evidence, including controls."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from admission import revision_run as mod
from admission.run import save,sha
P=Path(__file__).resolve().parents[1];BASE=P/'results/20261025-transfer-revision'
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
    save(BASE.with_suffix('.json'),dict(archive_sha256=sha(BASE.with_suffix('.npz')),fixtures=fixtures,runs=runs,closed_state=json.loads((mod.S/'STATE.json').read_text()),compiler_attempts=[dict(path=str(p.relative_to(mod.S)),receipt=json.loads(p.read_text())) for p in mod.S.glob('fixtures*/*/BUILD.json') if json.loads(p.read_text())['status']!='VIRTUAL_COMPILER_PASS'],scope='Seven sequential FNP3s: native correcting source and two constant-class transfer students. External positive/complement facts ground the source; student transfer uses no actual outcome and constant zero class. Engineered coding/roles; host-paced. Teacher answer path masked during exams, still clocked. Native stored edges/student outputs measured; intermediate features and teacher per-record responses derived. CPU controls also pass; no efficiency/autonomy claim.'))
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
        required={(mod.SEEDS[0],a,'gate') for a in mod.ARMS}|{(s,a,'repeat') for s in mod.SEEDS for a in mod.ARMS}
        assert set(native)==required and len(native)==16 and sum(r['events'] for r in native.values())==25408
        primary={}
        for arm in mod.ARMS:
            rs=[native[s,arm,'repeat'] for s in mod.SEEDS];ex=[v for r in rs for p,v in r['assessment'].items() if p.startswith('exam_')];before=[v for r in rs for p,v in r['assessment'].items() if p.startswith('before_transfer_')];sq=[v for r in rs for p,v in r['assessment'].items() if p.startswith('source_probe_')];online=[v for r in rs for v in r['assessment'].values()]
            primary[arm]=dict(correct=sum(v['student_correct'] for v in ex),queries=sum(v['queries'] for v in ex),changed_correct=sum(v['changed_correct'] for v in ex),changed_queries=sum(v['changed_queries'] for v in ex),stable_correct=sum(v['stable_correct'] for v in ex),stable_queries=sum(v['stable_queries'] for v in ex),before_transfer_correct=sum(v['student_correct'] for v in before),before_transfer_queries=sum(v['queries'] for v in before),source_correct=sum(v['teacher_correct'] for v in sq),source_queries=sum(v['queries'] for v in sq),all_online_correct=sum(v['student_correct'] for v in online),all_online_queries=sum(v['queries'] for v in online),connection_replacements=[sum(r['edits'][k]['connection_replacements'] for r in rs) for k in range(3)],stable_record_payloads=sum(v['unchanged'] for r in rs for v in r['stable_records']))
        cross={}
        for s in mod.SEEDS:
            _,get,_=get_run(z,m,f'native_s{s}_native_repeat');n=2028;prefix=[]
            for arm in mod.ARMS[1:]:
                _,cg,_=get_run(z,m,f'native_s{s}_{arm}_repeat')
                for k in range(3):assert np.array_equal(get(f'states{k}').reshape(n,-1)[:108],cg(f'states{k}').reshape(n,-1)[:108]),'same trained prefix'
                if arm in ['students_frozen','teacher_cut']:
                    assert np.array_equal(get('states0'),cg('states0')),'source trajectory identical in trained controls'
                    for k in [1,2]:
                        v=cg(f'states{k}').reshape(n,-1);assert np.all(v[108:]==v[107]),'trained student control remains unchanged'
                prefix.append(arm)
            cross[str(s)]=dict(common_prefix_identical=prefix,source_trajectory_identical=['students_frozen','teacher_cut'],trained_controls_students_unchanged=True)
        cpu=sum(v['cpu_student_correct'] for s in mod.SEEDS for p,v in native[s,'native','repeat']['assessment'].items() if p.startswith('exam_'))
        cpu_online=sum(v['cpu_student_correct'] for s in mod.SEEDS for v in native[s,'native','repeat']['assessment'].values())
        assert all(native[s,'native','repeat']['useful_gate'] for s in mod.SEEDS)
    out=dict(status='NUMERIC_ARCHIVE_PASS',archive_sha256=m['archive_sha256'],physical_sessions=16,native_forwards=25408,primary=primary,cpu_correct=cpu,cpu_queries=3456,cpu_all_online_correct=cpu_online,cpu_all_online_queries=5280,cross_arm_checks=cross,preserved_execution_failures=failed,compiler_failures=len(m['compiler_attempts']),rows=rows,scope=m['scope'])
    save(BASE.with_suffix('.audit.json'),out);return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);a=p.parse_args();r=build() if a.action=='build' else audit();print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
