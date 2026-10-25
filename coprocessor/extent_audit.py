"""Portable raw-array arithmetic, memory and matched-control audit."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from coprocessor.resident_archive import conv
P=Path(__file__).resolve().parents[1];S=P/'studies/20261025-reference-scale';F=S/'extent_fixtures'
BASE=P/'results/20261025-reference-scale'
CONTROLS=['cpu-dense','cpu-sparse','cpu-parallel','cpu-blas1','cpu-blas4']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,r):p.write_text(json.dumps(r,indent=2)+'\n')
def masks(r,n):
    records=r[...,2:].reshape(*r.shape[:-1],128,4+n//32)
    return np.unpackbits(records[...,4:].copy().view('u1'),axis=-1,bitorder='little')
def validate(r,fix,get,initial):
    n=fix['n'];u=fix['u'];count=r['count'];words=len(initial)
    states=get('snapshots').reshape(count,words);scores=get('scores').reshape(count,u)
    assert np.array_equal(get('ram'),states[-1]);before=np.concatenate([initial[None],states[:-1]])
    bm,am=masks(before,n),masks(states,n);features=fix['features'][:count]
    assert np.all(am.sum(axis=-1)==32) and np.all(bm.sum(axis=-1)==32)
    expected=np.einsum('nk,njk->nj',features.astype('i4'),bm.astype('i4'))
    assert np.array_equal(scores,expected),'scores from actual before-state connections'
    assert np.array_equal(scores[:8],fix['expected'][:8])
    changed=np.any(before!=states,axis=1);adds=am>bm;removes=bm>am
    assert np.all(adds.sum(axis=-1)==removes.sum(axis=-1))
    for i in range(count):
        edits=np.flatnonzero(np.any(am[i]!=bm[i],axis=-1))
        if not (r['arm'] and 8<=i<40):assert not changed[i],'query/frozen RAM changed'
        if len(edits):
            assert len(edits)==1 and edits[0]//64==fix['labels'][i]
            assert np.all(features[i][np.any(adds[i],axis=0)]==1)
            assert np.all(features[i][np.any(removes[i],axis=0)]==0)
    if r['arm'] and count>8:assert np.any(changed),'no native learning'
    else:assert np.all(states==initial)
    times=get('latency_ns').reshape(r['repeats'],count);cpu=get('caller_ns').reshape(r['repeats'],count)
    assert np.all(times>0) and np.all(cpu>0)
    data=times[:,8:] if count>8 else times
    result=dict(status='SAVED_ARRAY_PASS',cell=r['cell'],mode=r['mode'],arm=r['arm'],count=count,repeats=r['repeats'],
                median_us=float(np.median(data)/1000),p95_us=float(np.percentile(data,95)/1000),
                median_process_cpu_us=float(np.median(cpu[:,8:] if count>8 else cpu)/1000),
                connection_edit_events=int(np.any(am!=bm,axis=(1,2)).sum()),added_connections=int(adds.sum()),
                removed_connections=int(removes.sum()),changed_final_ram_words=int(np.sum(states[-1]!=initial)),
                learning_ram_bytes=words*4, feature_bits=n)
    return result
def load_fixture(d):
    f=d/'cell';meta=json.loads((d/'BUILD.json').read_text());n=meta['n'];side=meta.get('side',32);c=meta['filters'];ps=side//8
    arrays={k:np.fromfile(f.with_suffix('.'+k),dtype) for k,dtype in [('inputs','u1'),('features','u1'),('expected','<i4'),('labels','<i4'),('ram','<u4'),('w0','i1'),('w1','i1'),('w2','i1'),('t0','<i4'),('t1','<i4'),('t2','<i4')]}
    return meta,arrays
def audit_directory(d):
    r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0
    for name,h in r['hashes'].items():assert sha(d/name)==h
    meta,a=load_fixture(F/r['cell']);fix=dict(n=meta['n'],u=meta['u'],features=a['features'].reshape(64,meta['n']),expected=a['expected'].reshape(64,128),labels=a['labels'])
    get=lambda k:np.fromfile(d/('result.'+k),'<i4' if k=='scores' else '<i8' if k.endswith('_ns') else '<u4')
    result=validate(r,fix,get,a['ram'])
    for ctrl in CONTROLS:
        other=S/'raw'/f'{ctrl}_{r["cell"]}_a{r["arm"]}_n64_r0'
        if other==d:continue
        if r['mode']=='native':assert (other/'AUDIT.json').exists(), 'CPU must be admitted first'
        if (other/'AUDIT.json').exists():
            assert np.array_equal(get('scores'),np.fromfile(other/'result.scores','<i4')[:r['count']*128])
            assert np.array_equal(get('snapshots'),np.fromfile(other/'result.snapshots','<u4')[:r['count']*len(a['ram'])])
    if r['mode']=='native':
        assert r['device_free']
        for j in range(len(meta['fixed_packets'])):
            want=np.fromfile(F/r['cell']/f'cell.packet{j}.ram','<u4')
            for tag in ['initial','final']:assert np.array_equal(get(f'packet{j}.{tag}'),want)
        observer=get('observer_ns');result['median_observer_us_outside_timing']=float(np.median(observer)/1000)
    save(d/'AUDIT.json',result);return result
def build(base=BASE):
    arrays={};fixtures={};runs={}
    for d in sorted([*F.iterdir(),*(S/'fixtures03').iterdir()]):
        if json.loads((d/'BUILD.json').read_text())['status']!='VIRTUAL_COMPILER_PASS':continue
        meta,a=load_fixture(d);assert meta['status']=='VIRTUAL_COMPILER_PASS'
        meta=dict(meta,side=meta.get('side',32),control_modes=CONTROLS if d.parent==F else CONTROLS[:3])
        for j in range(len(meta['fixed_packets'])):a[f'packet{j}']=np.fromfile(d/f'cell.packet{j}.ram','<u4')
        spec={}
        for k,v in a.items():
            key=d.name+'__'+k;arrays[key]=v;spec[k]=dict(key=key,sha256=hashlib.sha256(v.tobytes()).hexdigest())
        fixtures[d.name]=dict(metadata=meta,arrays=spec)
    for d in sorted((S/'raw').iterdir()):
        if not (d/'RECEIPT.json').exists():continue
        r=json.loads((d/'RECEIPT.json').read_text());assert r['cell'] in fixtures
        if r['cell'].startswith('c'):
            from coprocessor import resident_run as legacy
            legacy.S=S;legacy.F=S/'fixtures03';a=legacy.audit(d)
        else:a=audit_directory(d)
        spec={}
        for p in d.glob('result.*'):
            k=p.name[7:];key=d.name+'__'+k.replace('.','_');dtype='<i4' if k=='scores' else '<i8' if k.endswith('_ns') else '<u4'
            arrays[key]=np.fromfile(p,dtype);spec[k]=dict(key=key,sha256=sha(p))
        runs[d.name]=dict(receipt=r,audit=a,arrays=spec)
    refs={}
    for d in sorted((S/'reference_raw').iterdir()):
        from coprocessor.scale_run import audit_reference
        a=audit_reference(d);r=json.loads((d/'RESULT.json').read_text());spec={}
        with np.load(d/'result.npz') as z:
            for k in z.files:key='reference__'+d.name+'__'+k;arrays[key]=z[k];spec[k]=dict(key=key,sha256=hashlib.sha256(z[k].tobytes()).hexdigest())
        refs[d.name]=dict(result=r,audit=a,arrays=spec)
    for k in ['inputs','expected']:
        arrays['reference__'+k]=np.load(S/'reference'/f'{k}.npy')
    base.parent.mkdir(exist_ok=True);np.savez_compressed(base.with_suffix('.npz'),**arrays)
    failures=[dict(path=str(p.relative_to(P)),**json.loads(p.read_text())) for root in [S/'fixtures03',F] for p in root.glob('*/BUILD.json') if json.loads(p.read_text())['status']=='FAILED_PRESERVED']
    repairs=json.loads((S/'evidence/REPAIRS.json').read_text())
    save(base.with_suffix('.json'),dict(archive_sha256=sha(base.with_suffix('.npz')),fixtures=fixtures,runs=runs,references=refs,compiler_failures=failures,repairs=repairs,
        timing_plan=json.loads((S/'evidence/TIMING_PLAN.json').read_text()),timing_exclusions=json.loads((S/'evidence/TIMING_EXCLUSIONS.json').read_text()),
        scope='Synthetic deployment correctness and resident native scaling; no application accuracy, pure silicon timing or calibrated energy'))
    return audit(base)
def audit(base=BASE):
    meta=json.loads(base.with_suffix('.json').read_text());assert sha(base.with_suffix('.npz'))==meta['archive_sha256'];reports=[]
    with np.load(base.with_suffix('.npz'),allow_pickle=False) as z:
        for cell,f in meta['fixtures'].items():
            get=lambda k:z[f['arrays'][k]['key']];m=f['metadata'];side=m['side'];c=m['filters'];n=m['n'];ps=side//8
            for v in f['arrays'].values():assert hashlib.sha256(z[v['key']].tobytes()).hexdigest()==v['sha256']
            a0=conv(get('inputs').reshape(64,side,side,3),get('w0').reshape(3,3,3,32),get('t0'))
            pre=conv(a0,get('w1').reshape(3,3,32,c)[::-1,::-1],None);pooled=pre.reshape(64,ps,2,ps,2,c).max(axis=(2,4))
            a1=(pooled>get('t1')).astype('i4');project=a1@get('w2').reshape(c,8).astype('i4')
            features=(project>get('t2')).transpose(0,3,2,1).reshape(64,n)
            assert np.array_equal(features,get('features').reshape(64,n)), 'independent complete spatial integer arithmetic'
        for name,run in meta['runs'].items():
            r=run['receipt'];f=meta['fixtures'][r['cell']];m=f['metadata'];n=m['n'];initial=z[f['arrays']['ram']['key']]
            for v in run['arrays'].values():assert hashlib.sha256(z[v['key']].tobytes()).hexdigest()==v['sha256']
            get=lambda k:z[run['arrays'][k]['key']];fg=lambda k:z[f['arrays'][k]['key']]
            fix=dict(n=n,u=128,features=fg('features').reshape(64,n),expected=fg('expected').reshape(64,128),labels=fg('labels'))
            report=validate(r,fix,get,initial)
            for ctrl in m['control_modes']:
                c=meta['runs'][f'{ctrl}_{r["cell"]}_a{r["arm"]}_n64_r0'];cg=lambda k:z[c['arrays'][k]['key']]
                for k,width in [('scores',128),('snapshots',len(initial))]:assert np.array_equal(get(k),cg(k)[:r['count']*width])
            if r['mode']=='native':
                for k in f['arrays']:
                    if k.startswith('packet'):
                        for tag in ['initial','final']:assert np.array_equal(get(k+'.'+tag),fg(k))
            reports.append(dict(name=name,**report))
        for name,r in meta['references'].items():
            for v in r['arrays'].values():assert hashlib.sha256(z[v['key']].tobytes()).hexdigest()==v['sha256']
            y=z[r['arrays']['outputs']['key']];assert np.array_equal(y,z['reference__expected'][np.arange(len(y))%16])
    report=dict(status='INDEPENDENT_NUMERICAL_ARCHIVE_PASS',rows=reports,archive_sha256=meta['archive_sha256'],physical_scale_sessions=sum(v['mode']=='native' for v in reports),physical_scale_forwards=sum(v['count'] for v in reports if v['mode']=='native'),reference_forwards=sum(r['audit']['count'] for r in meta['references'].values() if r['audit']['mode']!='software'),compiler_failures_preserved=len(meta['compiler_failures']))
    save(base.with_suffix('.audit.json'),report);return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);p.add_argument('--base',type=Path,default=BASE);a=p.parse_args();r=build(a.base) if a.action=='build' else audit(a.base);print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
