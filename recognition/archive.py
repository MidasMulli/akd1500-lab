"""Public image/CPU evidence and an independent packed-XOR reconstruction."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from recognition.data import P,S as SMALL
from recognition.capacity import S as LARGE

BASE=P/'results/2026-10-23-image-capacity'
LUT=np.asarray([int(i).bit_count() for i in range(256)],np.uint8)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def pooled(x):
    # Independent explicit block slicing, rather than the producer's reshape/max.
    a=x.reshape(-1,32,32)
    return (a[:,::2,::2]|a[:,1::2,::2]|a[:,::2,1::2]|a[:,1::2,1::2]).reshape(-1,256)


def nearest(x,t,per_class):
    assert np.all((x==0)|(x==1)) and np.all((t==0)|(t==1)), 'binary image data'
    a=np.packbits(x,axis=1);b=np.packbits(t,axis=1);pred=[];overlap=[]
    counts=t.sum(axis=1,dtype=np.int32)
    for lo in range(0,len(a),32):
        d=LUT[a[lo:lo+32,None,:]^b[None,:,:]].sum(axis=2,dtype=np.int32)
        pred.extend((d.argmin(axis=1)//per_class).tolist())
        overlap.append((x[lo:lo+32].sum(axis=1,dtype=np.int32)[:,None]+counts[None,:]-d)//2)
    return np.asarray(pred,np.int32),np.concatenate(overlap)


def recount(meta,z):
    assert meta['status']=='REAL_IMAGE_QUALITY_PASS_NATIVE_COMPILER_STOP'
    assert meta['physical_sessions']==meta['native_forwards']==0, 'no physical result in this archive'
    parent,large=meta['parent'],meta['capacity']
    assert parent['status']=='STOP_SEALED_QUALITY_GATE' and large['status']=='SEALED_QUALITY_PASS'
    assert len(set(large['exemplar_train_indices']))==3200
    assert set(parent['exemplar_train_indices'])<=set(large['exemplar_train_indices']), 'old exemplars preserved'
    assert not set(parent['test_indices'])&set(large['test_indices']), 'fresh sealed probes'
    assert len(set(parent['test_indices']))==len(set(large['test_indices']))==1000
    for c in range(10):
        assert np.array_equal(z['parent__templates'][c*48:(c+1)*48],z['capacity__templates'][c*320:c*320+48]), 'old image exemplars preserved'
    for study,info,width in [('parent',parent,48),('capacity',large,320)]:
        template=z[study+'__templates'];assert template.shape==(10*width,1024)
        for c in range(10):
            assert np.all(z[study+'__exemplar_labels'][c*width:(c+1)*width]==c),'exemplar group labels'
        for split,key in [('development','development'),('test','sealed')]:
            x=z[study+'__'+split+'_inputs'];y=z[study+'__'+split+'_labels']
            assert x.shape==(200 if split=='development' else 1000,1024)
            for kind in ['raw','compacted']:
                features,templates=(x,template) if kind=='raw' else (pooled(x),pooled(template))
                pred,scores=nearest(features,templates,width);r=info[key][kind]
                assert pred.tolist()==r['predictions'], 'independent image decisions'
                assert r['correct']==int((pred==y).sum()) and r['count']==len(y),'image quality counts'
                assert r['per_digit']==[dict(digit=c,correct=int(((pred==y)&(y==c)).sum()),count=int((y==c).sum())) for c in range(10)], 'per-digit quality counts'
                if study=='capacity' and split=='test':
                    assert np.array_equal(scores,z[kind+'__cpu.overlap_scores']),'actual CPU overlap counts'
                    for method in range(3):
                        measured=z[f'{kind}__cpu.m{method}.predictions']
                        assert measured.shape==(3,1000) and np.all(measured==pred),'actual CPU decisions'
                        for timer in ['wall_ns','cpu_ns']:
                            a=z[f'{kind}__cpu.m{method}.{timer}']
                            assert a.shape==(3,1000) and np.all(a>0),'CPU timer shape and positivity'
    d=parent['sealed'];assert d['raw']['correct']<850 and d['compacted']['correct']<850
    d=large['sealed'];assert d['raw']['correct']>=850 and d['compacted']['correct']>=850
    assert d['compacted']['correct']>=d['raw']['correct']-20,'unchanged quality margin'
    cells=meta['compiler']['cells'];assert {c['kind'] for c in cells}=={'raw','compacted'} and len(cells)==2
    assert all(c['status']=='FAILED_PRESERVED' and 'maximum number of filters is 2048 (got 3200)' in c['error'] for c in cells),'preserved compiler rejection'
    return dict(status='INDEPENDENT_IMAGE_CPU_AND_COMPILER_RECOUNT_PASS',physical_sessions=0,
        native_forwards=0,cpu_timed_classifications=18000,capacity_quality=[d['raw']['correct'],d['compacted']['correct']])


def timing_summary(z):
    rows=[]
    for kind in ['raw','compacted']:
        for i,method in enumerate(['exact_overlap','scalar_packed','NEON_packed']):
            a=z[f'{kind}__cpu.m{i}.wall_ns'];c=z[f'{kind}__cpu.m{i}.cpu_ns']
            rows.append(dict(representation=kind,method=method,images_per_repetition=1000,repetitions=3,
                median_wall_ns=float(np.median(a)),p95_wall_ns=float(np.percentile(a,95)),
                repetition_median_wall_ns=np.median(a,axis=1).tolist(),median_thread_cpu_ns=float(np.median(c))))
    return rows


def export():
    for s in [SMALL,LARGE]:assert json.loads((s/'STATE.json').read_text())['research_admissions_closed']
    arrays={}
    for name,s in [('parent',SMALL),('capacity',LARGE)]:
        with np.load(s/'fixtures-data01/data.npz',allow_pickle=False) as z:
            for k in z.files:arrays[name+'__'+k]=z[k]
    for kind in ['raw','compacted']:
        prefix=LARGE/'cpu_runs'/kind
        arrays[kind+'__cpu.overlap_scores']=np.fromfile(str(prefix)+'.overlap_scores','<i4').reshape(1000,3200)
        for method in range(3):
            for timer,dtype in [('predictions','<i4'),('wall_ns','<i8'),('cpu_ns','<i8')]:
                arrays[f'{kind}__cpu.m{method}.{timer}']=np.fromfile(str(prefix)+f'.m{method}.{timer}',dtype).reshape(3,1000)
    build=json.loads((LARGE/'fixtures-native01/BUILD.json').read_text())
    meta=dict(status='REAL_IMAGE_QUALITY_PASS_NATIVE_COMPILER_STOP',physical_sessions=0,native_forwards=0,
        parent=json.loads((SMALL/'fixtures-data01/DATA.json').read_text()),capacity=json.loads((LARGE/'fixtures-data01/DATA.json').read_text()),
        compiler=dict(status=build['status'],physical_device_opened=False,cells=[{k:c[k] for k in ['kind','status','error']} for c in build['cells']]),
        cpu_provenance=json.loads((LARGE/'evidence/CPU_PROVENANCE.json').read_text()),cpu_timings=timing_summary(arrays),
        scientific_disposition='Original quality failure preserved; one larger-bank revision passed fresh quality but failed both compiler mappings. No physical comparison.',
        exclusions=['No new native timing, learning, power, full-chip capacity or substrate advantage claim.',
          'CPU hot paths include input packing, pooling where needed and host decode; exclude file I/O, grayscale thresholding and initial resident prototype setup.',
          'Compiler2048-filter rejection is per attempted layer, not measured silicon capacity.',
          'CPU control timing completed after compiler rejection; it is not an AKD timing result.'])
    meta['audit']=recount(meta,arrays)
    np.savez_compressed(BASE.with_suffix('.npz'),**arrays);meta['archive_sha256']=sha(BASE.with_suffix('.npz'))
    BASE.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta['audit']))


def audit_public():
    meta=json.loads(BASE.with_suffix('.json').read_text());assert sha(BASE.with_suffix('.npz'))==meta['archive_sha256']
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        audit=recount(meta,z);assert audit==meta['audit'];assert timing_summary(z)==meta['cpu_timings'],'saved CPU timing summary'
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['export','audit-public']);a=p.parse_args()
    if a.command=='export':export()
    else:print(json.dumps(audit_public()))
