"""Interleaved equal-rate learner service beside unchanged T202 GPU forecasts."""
from pathlib import Path
import datetime as dt,fcntl,hashlib,json,select,subprocess,sys,time
import numpy as np
from feedback.design import P,S
from feedback.run import save,passed,name
ORDER=[['alone','cpu','native','task'],['task','native','cpu','alone'],['cpu','alone','task','native']]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
    paths=[P/'feedback/service.cpp',P/'feedback/build_service.sh',P/'feedback/gpu.py',P/'feedback/corun.py',P/'build/predictive_service',P/'streaming/nano.py',P/'streaming/design.py',P/'src/readout_bench.cpp',P/'src/mmio_driver.hpp',S/'GPU_ADDENDUM.md']
    save(S/'evidence/GPU_FREEZE.json',dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),hashes={str(p.relative_to(P)):sha(p) for p in paths},order=ORDER))
def line(p,timeout):
    assert select.select([p.stdout],[],[],timeout)[0],'owned child handshake deadline';r=p.stdout.readline();assert r,'owned child exited before handshake';return json.loads(r)
def stop(p):
    if p and p.poll() is None:
        p.terminate()
        try:p.wait(timeout=3)
        except subprocess.TimeoutExpired:p.kill();p.wait()
def audit(d):
    r=json.loads((d/'RECEIPT.json').read_text());assert r['gpu_exit']==0 and r['service_exit']==0
    gpu=json.loads((d/'gpu.json').read_text());assert gpu['receipt']['weights_unchanged'] and gpu['receipt']['checkpoint_sha256']=='3342f73cf939f689abfb1189faf3580719acbf05ca2d439001ee13887ea5bc65'
    a=np.fromfile(d/'gpu.frames_ns','<i8').reshape(-1,3);assert len(a)>50 and np.all(a[:,1]>a[:,0]);elapsed=(a[-1,1]-a[0,0])/1e9
    result=dict(status='GPU_AND_SERVICE_AUDIT_PASS',arm=r['arm'],repeat=r['repeat'],frames=len(a),gpu_forecasts_per_second=len(a)/elapsed,gpu_median_ms=float(np.median(a[:,1]-a[:,0])/1e6),gpu_p95_ms=float(np.percentile(a[:,1]-a[:,0],95)/1e6),output_sha256=gpu['output_sha256'],weights_unchanged=True)
    if r['arm']!='alone':
        s=r['service_result'];assert s['status']=='PASS' and s['events']==3072 and s['rate']==600
        batches=np.fromfile(d/'service.batch_ns','<i8').reshape(-1,4);assert batches.shape==(96,4)
        assert a[0,0]>=r['service_start']['start_ns'] and a[-1,1]<=batches[-1,2]+100000000,'service covered GPU interval'
        # Exact C++ actor replay verifies native scores/final SRAM after timing.
        y=np.fromfile(d/'service.scores','<i4');assert y.shape==(3072,)
        result.update(service_thread_cpu_ms=s['thread_cpu_ns']/1e6,service_wall_ms=s['wall_ns']/1e6,service_late_batches=s['late_batches_over_2ms'])
    save(d/'AUDIT.json',result);return result
def run(arm,repeat):
    assert repeat in [0,1,2] and arm in ORDER[repeat]
    for seed in [108101,108201,108301]:assert passed(name('native',seed,'reduced','native',768,32,5)),'all scientific gates first'
    seq=[(a,r) for r,block in enumerate(ORDER) for a in block];at=seq.index((arm,repeat))
    for a,r in seq[:at]:assert (S/f'gpu/r{r}_{a}/AUDIT.json').exists(),'frozen interleaved order'
    auth=json.loads((S/'AUTHORITY.json').read_text());assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(auth['scientific_cutoff_utc'])
    for p,h in json.loads((S/'evidence/GPU_FREEZE.json').read_text())['hashes'].items():assert sha(P/p)==h,('GPU freeze drift',p)
    d=S/f'gpu/r{repeat}_{arm}';d.mkdir(parents=True,exist_ok=False);gpu=actor=telemetry=None;r=dict(arm=arm,repeat=repeat,start_utc=dt.datetime.now(dt.timezone.utc).isoformat(),native_forwards=3072 if arm=='native' else 0);save(d/'OWNER.json',r)
    with open('/tmp/world_model_e2e_demo.lock','a') as world,open('/tmp/akd1500-research.lock','a') as chip:
        fcntl.flock(world,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if arm=='native':
            fcntl.flock(chip,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
            previous=list((S/'raw').glob('*/OWNER.json'))+list((S/'gpu').glob('*/OWNER.json'));forwards=sum(json.loads(p.read_text()).get('native_forwards',json.loads(p.read_text()).get('count',0) if json.loads(p.read_text()).get('mode')=='native' else 0) for p in previous);assert forwards<=auth['maximum_forwards']
        try:
            with (d/'gpu.stderr').open('w') as ge,(d/'service.stderr').open('w') as se,(d/'telemetry.stderr').open('w') as te:
                gpu=subprocess.Popen(['taskset','-c','0,1','/home/nicklomeli/wm-venv/bin/python','-m','feedback.gpu',str(d/'gpu')],cwd=P,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=ge,text=True,bufsize=1);r['gpu_pid']=gpu.pid;save(d/'OWNER.json',r);ready=line(gpu,60);save(d/'GPU_READY.json',ready)
                if arm!='alone':
                    actor=subprocess.Popen(['taskset','-c','2','build/predictive_service',arm,str(S/'fixtures02/s108101'),str(d/'service')],cwd=P,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=se,text=True,bufsize=1);r['service_pid']=actor.pid;save(d/'OWNER.json',r);save(d/'SERVICE_READY.json',line(actor,10))
                telemetry=subprocess.Popen(['tegrastats','--interval','200','--logfile',str(d/'tegrastats.log')],stdout=subprocess.DEVNULL,stderr=te)
                if actor:
                    actor.stdin.write('GO\n');actor.stdin.flush();r['service_start']=line(actor,5)
                gpu.stdin.write('GO\n');gpu.stdin.flush();r['gpu_result']=line(gpu,30);r['gpu_exit']=gpu.wait(timeout=10)
                r['service_result']=line(actor,10) if actor else None;r['service_exit']=actor.wait(timeout=10) if actor else 0
            result=audit_result=None
        except BaseException as e:r['failure']=repr(e);raise
        finally:
            stop(actor);stop(gpu);stop(telemetry);r['finish_utc']=dt.datetime.now(dt.timezone.utc).isoformat();r['device_free']=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
            r['hashes']={p.name:sha(p) for p in d.iterdir() if p.is_file()};save(d/'RECEIPT.json',r)
    result=audit(d);print(json.dumps(result),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='audit':print(json.dumps(audit(S/'gpu'/sys.argv[2])))
    else:run(sys.argv[1],int(sys.argv[2]))
