"""Portable numerical evidence and independent read-only audit; no SDK required."""
from pathlib import Path
import hashlib,json,argparse
import numpy as np

def conv(x,w,threshold):
    """Independent integer NHWC SAME cross-correlation, stride two."""
    n,h,width,c=x.shape;oh,ow=(h+1)//2,(width+1)//2
    ph,pw=max(0,(oh-1)*2+3-h),max(0,(ow-1)*2+3-width)
    z=np.pad(x.astype('i4'),((0,0),(ph//2,ph-ph//2),(pw//2,pw-pw//2),(0,0)))
    y=np.zeros((n,oh,ow,w.shape[-1]),'i4')
    for ky in range(3):
        for kx in range(3):y+=z[:,ky:ky+oh*2:2,kx:kx+ow*2:2]@w[ky,kx].astype('i4')
    return y if threshold is None else (y>threshold).astype('u1')

P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-resident-spatial'
DEFAULT=P/'results/20261024-resident-spatial'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):p.write_text(json.dumps(r,indent=2)+'\n')
def mask(r):return np.unpackbits(r[...,2:].reshape(*r.shape[:-1],128,8)[...,4:].copy().view('u1'),axis=-1,bitorder='little')
def build(base=DEFAULT):
    base.parent.mkdir(parents=True,exist_ok=True);arrays={};fixtures={};runs={}
    for f in sorted((S/'fixtures03').glob('*/cell.meta')):
        cell=f.parent.name;prefix=f.parent/'cell';m=json.loads((f.parent/'BUILD.json').read_text());assert m['status']=='VIRTUAL_COMPILER_PASS'
        spec={}
        for ext,dtype in [('inputs','u1'),('features','u1'),('expected','<i4'),('labels','<i4'),('ram','<u4'),('w0','i1'),('w1','i1'),('w2','i1'),('t0','<i4'),('t1','<i4'),('t2','<i4')]:
            p=prefix.with_suffix('.'+ext);key=cell+'__'+ext;arrays[key]=np.fromfile(p,dtype);spec[ext]=dict(key=key,sha256=sha(p))
        packets=int(prefix.with_suffix('.layout').read_text().split()[0])
        for j in range(packets):
            p=prefix.with_suffix(f'.packet{j}.ram');key=cell+f'__packet{j}';arrays[key]=np.fromfile(p,'<u4');spec[f'packet{j}']=dict(key=key,sha256=sha(p))
        fixtures[cell]=dict(metadata=m,arrays=spec)
    for d in sorted((S/'raw').iterdir()):
        r=json.loads((d/'RECEIPT.json').read_text());a=json.loads((d/'AUDIT.json').read_text());assert a['status']=='SAVED_ARRAY_PASS' and r['exit_code']==0
        spec={}
        for p in d.glob('result.*'):
            ext=p.name[7:];dtype='<i4' if ext=='scores' else '<i8' if ext.endswith('_ns') else '<u4'
            key=d.name+'__'+ext.replace('.','_');arrays[key]=np.fromfile(p,dtype);spec[ext]=dict(key=key,sha256=sha(p))
        runs[d.name]=dict(receipt=r,audit=a,arrays=spec)
    np.savez_compressed(base.with_suffix('.npz'),**arrays)
    failures=[]
    for p in sorted(S.glob('fixtures0[12]/*/BUILD.json')):
        r=json.loads(p.read_text())
        if r['status']=='FAILED_PRESERVED':failures.append(dict(path=str(p.relative_to(P)),**r))
    save(base.with_suffix('.json'),dict(status='NUMERICAL_ARCHIVE',archive_sha256=sha(base.with_suffix('.npz')),fixtures=fixtures,runs=runs,compiler_failures=failures,source_freeze=json.loads((S/'evidence/FROZEN.json').read_text()),scope='Native multi-CNP spatial graph and one native FNP3 learner; no registered host weight memory',unobserved=['Electrical spike timing','Individual neuron simultaneity','Calibrated energy','Physical transistor placement','Autonomous teacher generation','Application/generalization advantage']))
    return audit(base)

def audit(base=DEFAULT):
    meta=json.loads(base.with_suffix('.json').read_text());assert sha(base.with_suffix('.npz'))==meta['archive_sha256'];out=[]
    with np.load(base.with_suffix('.npz'),allow_pickle=False) as z:
        for cell,fix in meta['fixtures'].items():
            get=lambda k:z[fix['arrays'][k]['key']];c=fix['metadata']['filters']
            for r in fix['arrays'].values():assert hashlib.sha256(z[r['key']].tobytes()).hexdigest()==r['sha256']
            x=get('inputs').reshape(64,32,32,3);a0=conv(x,get('w0').reshape(3,3,3,32),get('t0'))
            zz=conv(a0,get('w1').reshape(3,3,32,c)[::-1,::-1],None);pooled=zz.reshape(64,4,2,4,2,c).max(axis=(2,4));a1=(pooled>get('t1')).astype('u1')
            projection=a1.astype('i4')@get('w2').reshape(c,8).astype('i4');features=(projection>get('t2')).transpose(0,3,2,1).reshape(64,128).astype('u1')
            assert np.array_equal(features,get('features').reshape(64,128)),'independent full spatial arithmetic'
        for name,run in meta['runs'].items():
            r=run['receipt'];fix=meta['fixtures'][r['cell']];get=lambda k:z[run['arrays'][k]['key']]
            for v in run['arrays'].values():assert hashlib.sha256(z[v['key']].tobytes()).hexdigest()==v['sha256']
            n=r['count'];initial=z[fix['arrays']['ram']['key']];states=get('snapshots').reshape(n,1026);scores=get('scores').reshape(n,128)
            before=np.concatenate([initial[None],states[:-1]]);bm=mask(before);am=mask(states);features=z[fix['arrays']['features']['key']].reshape(64,128)[:n];labels=z[fix['arrays']['labels']['key']]
            assert np.all(am.sum(axis=-1)==32) and np.all(bm.sum(axis=-1)==32),'fixed selected connection count'
            expect=np.einsum('nk,njk->nj',features.astype('i4'),bm.astype('i4'));assert np.array_equal(scores,expect),'response from actual before-state masks'
            changed=np.any(before!=states,axis=1);added=(am>bm);removed=(bm>am)
            assert np.all(added.sum(axis=-1)==removed.sum(axis=-1))
            for i in range(n):
                units=np.flatnonzero(np.any(am[i]!=bm[i],axis=-1))
                if not(r['arm'] and 8<=i<40):assert not changed[i],'query/frozen memory changed'
                if len(units):
                    assert len(units)==1 and units[0]//64==labels[i]
                    assert np.all(features[i][np.any(added[i],axis=0)]==1)
                    assert np.all(features[i][np.any(removed[i],axis=0)]==0)
            if r['arm'] and n>8:assert changed.sum()>0
            if r['mode']=='native':
                for cpu in ['cpu-dense','cpu-sparse','cpu-parallel']:
                    ctrl=meta['runs'][f'{cpu}_{r["cell"]}_a{r["arm"]}_n64_r0'];cg=lambda k:z[ctrl['arrays'][k]['key']]
                    assert np.array_equal(states,cg('snapshots').reshape(64,1026)[:n]);assert np.array_equal(scores,cg('scores').reshape(64,128)[:n])
                for k,v in fix['arrays'].items():
                    if k.startswith('packet'):
                        for tag in ['initial','final']:assert np.array_equal(get(k+'.'+tag),z[v['key']])
            out.append(dict(name=name,connection_edit_events=int(np.any(am!=bm,axis=(1,2)).sum()),added_connections=int(added.sum()),removed_connections=int(removed.sum()),**run['audit']))
    report=dict(status='INDEPENDENT_NUMERICAL_ARCHIVE_PASS',runs=len(out),physical_sessions=sum(r['mode']=='native' for r in out),physical_forwards=sum(r['count'] for r in out if r['mode']=='native'),rows=out,archive_sha256=meta['archive_sha256'],compiler_failures_preserved=len(meta['compiler_failures']))
    save(base.with_suffix('.audit.json'),report);return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);p.add_argument('--base',type=Path,default=DEFAULT);a=p.parse_args();r=build(a.base) if a.action=='build' else audit(a.base);print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
