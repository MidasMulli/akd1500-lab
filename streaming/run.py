"""One isolated, bounded Nano+AKD owner; journals predictions before outcomes."""
import argparse,datetime,fcntl,json,os,select,shutil,signal,subprocess,time
from pathlib import Path
import numpy as np
from streaming.design import N,U,NW,QUERY,SEEDS,episodes,observation,encode,decode,band,sha,save
from streaming.nano import Nano
from pong.relearn.model import RecentNearest

ROOT=Path(__file__).resolve().parents[1]

def run(study,seed,arm,smoke=False):
    auth=json.loads((study/'AUTHORITY.json').read_text());state=json.loads((study/'STATE.json').read_text())
    assert not state['research_admissions_closed'] and not state['skipping_enabled']
    assert state['admission']==('NATIVE_SMOKE' if smoke else 'MATCHED_PILOT' if seed==SEEDS[0] else 'CONFIRMATION')
    assert seed in state['admitted_seeds'] and arm in ['learn','frozen']
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(auth['scientific_cutoff_utc'])
    freeze=json.loads((study/'evidence/SOURCE_FREEZE.json').read_text())
    for path,digest in freeze['hashes'].items():assert sha(ROOT/path)==digest,('source drift',path)
    name=('smoke_' if smoke else '')+f's{seed}_{arm}';out=study/'raw'/name;out.mkdir(exist_ok=False)
    prior=list((study/'raw').glob('*/OWNER.json'));assert len(prior)<auth['maximum_physical_sessions']
    assert sum(json.loads(p.read_text()).get('native_forwards',0) for p in prior)+1600<=auth['maximum_forwards']
    owner=dict(status='RUNNING',seed=seed,arm=arm,pid=os.getpid(),start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),skipped_inferences=0)
    save(out/'OWNER.json',owner);child=None;count=0;records=[];images={k:[] for k in ['current','previous','forecast','future']}
    lock=open('/tmp/akd1500-research.lock','a');flock=open('/tmp/world_model_e2e_demo.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(flock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('owned 180-second run cap')));signal.alarm(180)
    error=(out/'native.stderr').open('w');log=(out/'events.jsonl').open('x',buffering=1);native_log=(out/'native.stdout').open('x',buffering=1)
    def emit(kind,**v):
        row=dict(kind=kind,clock_ns=time.monotonic_ns(),**v);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno());return row
    def read():
        assert select.select([child.stdout],[],[],5)[0],'native response timeout'
        line=child.stdout.readline();assert line,'native response absent';native_log.write(line);native_log.flush();return json.loads(line)
    def forward(x,label):
        nonlocal count
        count+=1;child.stdin.write(f'F {count} {label} {x.tobytes().hex()}\n');child.stdin.flush();r=read()
        assert r['kind']=='forward' and r['id']==count
        return r
    try:
        model=Nano() # Load before starting the actor's 175-second watchdog.
        for ext in ['bin','ram','meta','json']:
            shutil.copy2(study/'fixtures-stream01'/f's{seed}.{ext}',out/f'fixture.{ext}')
        child=subprocess.Popen(['taskset','-c','2',str(ROOT/'build/stream_bench'),str(out/'fixture'),str(out/'native'),'1600'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=error,text=True,bufsize=1)
        owner['actor_pid']=child.pid;save(out/'OWNER.json',owner)
        assert read()['kind']=='ready';baseline=RecentNearest();start=time.monotonic_ns()
        for ep in episodes(smoke=smoke):
            model.begin(ep)
            for k,action in enumerate(model.blocks):
                assert time.monotonic_ns()-start<165_000_000_000
                assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(auth['scientific_cutoff_utc'])
                i=len(records);current=model.current.copy();previous=model.previous.copy()
                o=observation(previous,current,action,ep['context']);x=encode(o)
                pred,md=model.forecast(action);reply=forward(x,QUERY);prediction,score,margin=decode(reply['scores'])
                cpu=baseline.forward(x,QUERY,ep['context']);assert cpu==reply['cpu_task_prediction']
                query=emit('query',sample=i,episode=ep,step=k,action=action,observation=o,input_sha256=sha_bytes(x),
                           native=reply,prediction=prediction,score=score,margin=margin,cpu_prediction=cpu,nano_prediction=md["prediction"],**{k:v for k,v in md.items() if k!="prediction"})
                # The future has not been rendered/stepped at the durable query commit.
                future,actual=model.advance(action);truth=band(future);outcome=emit('outcome',sample=i,query_clock_ns=query['clock_ns'],truth=truth,**actual)
                teach=ep['teach'] and (arm=='learn' or ep['phase']=='prefix')
                teacher=truth if teach else QUERY;feedback=forward(x,teacher)
                baseline.forward(x,teacher,ep['context'])
                emit('feedback',sample=i,teacher=teacher,native=feedback,outcome_clock_ns=outcome['clock_ns'])
                p=band(current);velocity_y=np.clip(2*o['y']-(o['y']-o['vy']/8),0,.999999)
                record=dict(sample=i,**ep,step=k,observation=o,truth=truth,teacher=teacher,prediction=prediction,
                            score=score,margin=margin,cpu_prediction=cpu,persistence=p,velocity_prediction=min(3,int(velocity_y*4)),
                            nano_prediction=md['prediction'],model_ns=md['model_ns'],query_native_index=reply['index'],feedback_native_index=feedback['index'])
                records.append(record)
                for key,a in [('current',current),('previous',previous),('forecast',pred),('future',future.copy())]:images[key].append(a)
                if actual['terminal']:break
        child.stdin.write('X\n');child.stdin.flush();closed=read();assert closed['kind']=='closed' and child.wait(timeout=5)==0
        receipt=model.finish();np.savez_compressed(out/'frames.npz',**{k:np.asarray(v,np.uint8) for k,v in images.items()})
        save(out/'rows.json',records);save(out/'MODEL_RECEIPT.json',receipt)
        owner.update(status='EXECUTION_PASS',native_forwards=count,samples=len(records),model_calls=receipt['model_calls'],weights_unchanged=True,cleanup=closed)
    except BaseException as e:
        owner.update(status='FAILED_PRESERVED',error=repr(e),native_forwards=count,complete_samples=len(records))
        if records:save(out/'rows_partial.json',records)
        raise
    finally:
        if child is not None and child.poll() is None:
            try:child.stdin.write('X\n');child.stdin.flush();child.wait(timeout=4)
            except (BrokenPipeError,subprocess.TimeoutExpired):
                child.terminate()
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        signal.alarm(0);error.close();log.close();native_log.close()
        owner.update(finish_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),device_released=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1)
        save(out/'OWNER.json',owner)
    print(json.dumps(owner))

def sha_bytes(x):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('seed',type=int);p.add_argument('arm',choices=['learn','frozen']);p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.study.resolve(),a.seed,a.arm,a.smoke)
