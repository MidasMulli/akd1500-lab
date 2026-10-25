"""Bounded standard-SDK vendor reference, no hardware opened by prepare."""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np

P = Path(__file__).resolve().parents[1]
S = P / 'studies/20261025-reference-scale'
VENDOR_SHA = '90531d024b3359f836651748ff867aeaf0019aa0dd7067b9849fb5a3f5381b22'
VENDOR_COMMIT = 'e7e4a57de706bfabb417e38824bee186e875a6ae'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, r): p.write_text(json.dumps(r, indent=2, default=str)+'\n')
def describe(m):
    passes = [len(s.passes) for s in m.sequences]
    maps = [[dict(layer=l.name, nps=[dict(type=str(n.type), np=[n.ident.col,n.ident.row,n.ident.id]) for n in l.mapping.nps]) for l in p.layers] for s in m.sequences for p in s.passes]
    unique = [list(v) for v in sorted({tuple(n['np']) for p in maps for l in p for n in l['nps']})]
    return dict(passes=passes, mapping=maps, unique_nps=unique,
                external_memory_bytes=int(m.external_memory_size))

def prepare(model_path):
    import akida
    assert sha(model_path) == VENDOR_SHA
    d=S/'reference'; d.mkdir(exist_ok=False)
    rng=np.random.default_rng(125101)
    x=rng.integers(0,256,(16,96,96,3),np.uint8)
    x[0]=0; x[1]=255
    x[2]=np.tile(np.arange(96,dtype=np.uint8)[None,:,None]*2,(96,1,3))
    x[3]=255-x[2]
    m=akida.Model(str(model_path)); assert list(m.input_shape)==[96,96,3]
    y=m.forward(x,batch_size=1)
    np.save(d/'inputs.npy',x); np.save(d/'expected.npy',y)
    reports={}
    for mode in ['Minimal','AllNps']:
        m.map(akida.AKD1500(),hw_only=True,mode=getattr(akida.MapMode,mode))
        reports[mode]=describe(m)
        assert reports[mode]['passes']==[1] and reports[mode]['external_memory_bytes']==0
        (d/(mode+'.bin')).write_bytes(bytes(m.sequences[0].program))
        m.map(None)
    save(d/'BUILD.json',dict(status='VIRTUAL_REFERENCE_PASS',sdk=akida.__version__,
         source_commit=VENDOR_COMMIT, model_sha256=VENDOR_SHA, synthetic_inputs=True,
         input_shape=list(x.shape), output_shape=list(y.shape), output_dtype=str(y.dtype),
         physical_device_opened=False, mappings=reports))
    print(json.dumps(dict(status='VIRTUAL_REFERENCE_PASS',mappings=reports)),flush=True)

def run(model_path, mode, count, out):
    import akida
    assert mode in ['software','Minimal','AllNps'] and count in [8,64]
    assert sha(model_path)==VENDOR_SHA
    d=S/'reference'; x=np.load(d/'inputs.npy'); expected=np.load(d/'expected.npy')
    m=akida.Model(str(model_path)); setup=time.perf_counter_ns()
    report=dict(mode=mode, count=count, sdk=akida.__version__, model_sha256=VENDOR_SHA)
    try:
        if mode!='software':
            devices=akida.devices(); assert len(devices)==1
            report['device_description']=str(devices[0].desc)
            m.map(devices[0],hw_only=True,mode=getattr(akida.MapMode,mode))
            report.update(describe(m))
            save(out/'MAPPING.json',report)
            frozen=json.loads((d/'BUILD.json').read_text())['mappings'][mode]
            assert describe(m)==frozen, 'physical mapping drift'
        report['setup_ns']=time.perf_counter_ns()-setup
        outputs=[]; latency=[]; cpu=[]
        for i in range(count):
            start=time.perf_counter_ns(); cp=time.process_time_ns()
            y=m.forward(x[i%16:i%16+1],batch_size=1)
            cpu.append(time.process_time_ns()-cp); latency.append(time.perf_counter_ns()-start)
            outputs.append(y.copy()); assert np.array_equal(y,expected[i%16:i%16+1]),f'output mismatch {i}'
        np.savez(out/'result.npz', outputs=np.concatenate(outputs),latency_ns=latency,cpu_ns=cpu)
        report.update(status='REFERENCE_EXACT_PASS', metrics=m.metrics,
                      input_bytes_per_frame=96*96*3, output_bytes_per_frame=expected[0].nbytes,
                      result_sha256=sha(out/'result.npz'))
    finally:
        m.map(None)
    save(out/'RESULT.json',report);print(json.dumps(report,default=str),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('model',type=Path)
    p.add_argument('--mode',choices=['software','Minimal','AllNps']);p.add_argument('--count',type=int);p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.action=='prepare': prepare(a.model)
    else: run(a.model,a.mode,a.count,a.out)
