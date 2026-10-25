"""Portable complete numeric receipts; excludes licensed programs/runtime binaries."""
from pathlib import Path
import argparse,hashlib,json,tarfile,tempfile
from unittest.mock import patch
import numpy as np
from coprocessor import service_audit as A,host_window_run as W
from coprocessor.extent_audit import load_fixture,validate
from coprocessor.resident_archive import conv
from pong.relearn.model import Reference

P=Path(__file__).resolve().parents[1];S=P/'studies/20261025-host-service';BASE=P/'results/20261025-host-service'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def pack(base=BASE):
    excluded=[];paths=[]
    for p in sorted(S.rglob('*')):
        if not p.is_file():continue
        if p.suffix=='.bin' or p.name.endswith('.private-binary') or p.name=='SOURCE_EXPORT.json':excluded.append(str(p.relative_to(S)));continue
        paths.append(p)
    target=Path(str(base)+'.tar.gz');target.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(target,'w:gz',compresslevel=6) as t:
        for p in paths:t.add(p,arcname=str(p.relative_to(S)),recursive=False)
    export=dict(format='AKD_HOST_SERVICE_NUMERIC_TAR_V1',archive_sha256=sha(target),files={str(p.relative_to(S)):sha(p) for p in paths},excluded_licensed_programs_or_private_binaries=excluded,
        scope='Exact synthetic native-learning workload, transport/completion tradeoff, repeated frozen Nano forecast co-scheduling. No energy or application-quality claim.',source_binding='Source and runtime hashes are preserved; unavailable licensed binary hashes are provenance, not a portable re-execution proof.')
    save(base.with_suffix('.json'),export);return export

def reconstruct(f):
    m,a=load_fixture(f);side=m['side'];ps=side//8;c=m['filters']
    x=a['inputs'].reshape(64,side,side,3)
    a0=conv(x,a['w0'].reshape(3,3,3,32),a['t0'])
    zz=conv(a0,a['w1'].reshape(3,3,32,c)[::-1,::-1],None)
    pooled=zz.reshape(64,ps,2,ps,2,c).max(axis=(2,4));a1=(pooled>a['t1']).astype('u1')
    features=((a1.astype('i4')@a['w2'].reshape(c,8).astype('i4'))>a['t2']).transpose(0,3,2,1).reshape(64,m['n']).astype('u1')
    assert np.array_equal(features,a['features'].reshape(64,m['n'])),'independent full convolution/features'
    return m,a

def audit(base=BASE):
    meta=json.loads(base.with_suffix('.json').read_text());archive=Path(str(base)+'.tar.gz');assert sha(archive)==meta['archive_sha256']
    rows=[];windows=[];failures=[];seen=set()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        with tarfile.open(archive,'r:gz') as t:
            for member in t.getmembers():
                rel=Path(member.name);assert not rel.is_absolute() and '..' not in rel.parts and member.isfile(),'unsafe archive member'
                assert member.name in meta['files'];dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(t.extractfile(member).read())
        assert set(meta['files'])=={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        for name,h in meta['files'].items():assert sha(root/name)==h,('portable bytes',name)
        freezes={sha(p) for p in (root/'evidence').glob('*FROZEN*.json')}
        for f in sorted((root/'fixtures').iterdir()):
            m,a=reconstruct(f)
            key=(m['side'],m['seed'])
            if key not in seen:
                seen.add(key)
                for arm in [0,1]:
                    ref=root/'reference'/f's{key[0]}_s{key[1]}_a{arm}'
                    canon={k:np.fromfile(ref.with_suffix('.'+k),'<i4' if k=='scores' else '<u4') for k in ['scores','snapshots']}
                    canon['snapshots']=A.expected_states(a['ram'],canon['snapshots'].reshape(64,-1),m['n']).reshape(-1)
                    canon['ram']=canon['snapshots'].reshape(64,-1)[-1]
                    canon['latency_ns']=np.ones(64,dtype='i8');canon['caller_ns']=np.ones(64,dtype='i8')
                    fixture=dict(n=m['n'],u=128,features=a['features'].reshape(64,m['n']),expected=a['expected'].reshape(64,128),labels=a['labels'])
                    validate(dict(count=64,arm=arm,repeats=1,cell=f.name,mode='reference'),fixture,lambda k:canon[k],a['ram'])
                    refmodel=Reference(a['ram'],n=m['n'],u=128,nw=32,classes=2)
                    state_trace=canon['snapshots'].reshape(64,-1);score_trace=canon['scores'].reshape(64,128)
                    for i,features in enumerate(fixture['features']):
                        if arm and 8<=i<40:response=refmodel.forward(features,int(a['labels'][i]))
                        else:
                            bits=sum(1<<int(k) for k in np.flatnonzero(features));response=[(w&bits).bit_count() for w in refmodel.weights]
                        assert np.array_equal(response,score_trace[i]),'independent Python learning scores'
                        assert np.array_equal(refmodel.ram,state_trace[i]),'independent Python learning state'
        with patch.object(A,'S',root),patch.object(W,'S',root):
            for d in sorted((root/'raw').iterdir()):
                receipt=json.loads((d/'RECEIPT.json').read_text());assert receipt['source_freeze_sha256'] in freezes,'acting source freeze binding';rows.append(dict(name=d.name,receipt=receipt,**A.audit(d)))
            for d in sorted((root/'window_raw').iterdir()):
                receipt=json.loads((d/'RECEIPT.json').read_text())
                assert receipt['freeze_sha256'] in freezes,'window source freeze binding'
                if receipt['status']=='FAILED_PRESERVED':
                    for name,h in receipt['hashes'].items():assert sha(d/name)==h
                    failures.append(dict(name=d.name,receipt=receipt));continue
                checked=W.audit(d)
                if receipt['busy']:
                    model=json.loads((d/'MODEL_RECEIPT.json').read_text());frame=np.load(d/'foreground-frame.npy',allow_pickle=False);timing=np.load(d/'foreground-timing.npy',allow_pickle=False)
                    assert hashlib.sha256(frame.tobytes()).hexdigest()==model['output_sha256'],'saved foreground frame binding'
                    assert len(timing)==model['calls'],'foreground call counter'
                    assert timing[0,0]==model['first_call_ns'] and timing[-1,1]==model['last_call_end_ns']
                windows.append(dict(name=d.name,receipt=receipt,**checked))
    report=dict(status='PORTABLE_HOST_SERVICE_PASS',archive_sha256=meta['archive_sha256'],exact_fixtures=8,independently_reconstructed_feature_banks=len(seen),runs=len(rows),window_runs=len(windows),
        physical_sessions=sum(r['receipt']['physical'] for r in rows+windows)+sum(r['receipt']['physical'] for r in failures),physical_forwards=sum(r['count'] for r in rows+windows if r['receipt']['physical']),preserved_failed_setups=len(failures),failed_setups=failures,rows=rows,windows=windows)
    save(base.with_suffix('.audit.json'),report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pack','audit']);p.add_argument('--base',type=Path,default=BASE);a=p.parse_args();r=pack(a.base) if a.action=='pack' else audit(a.base);print(json.dumps({k:v for k,v in r.items() if k not in ['rows','windows','files']}))
