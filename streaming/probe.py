"""Bounded model and pixel-interface preflight. Never imports an AKD driver."""
import argparse,fcntl,json,signal,time
from pathlib import Path
import numpy as np
from streaming.design import STUDY,episodes,observation,encode,band,save
from streaming.nano import Nano

def run(study):
    out=study/'evidence'/'model_probe';out.mkdir(exist_ok=False)
    lock=open('/tmp/world_model_e2e_demo.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('model probe cap')));signal.alarm(180)
    model=Nano();frames=[];forecasts=[];rows=[]
    try:
        for ep in episodes(probe=True):
            model.begin(ep)
            for k,a in enumerate(model.blocks):
                o=observation(model.previous,model.current,a,ep['context']);x=encode(o)
                current=model.current.copy();pred,md=model.forecast(a);query_ns=time.monotonic_ns()
                future,actual=model.advance(a);truth=band(future)
                rows.append(dict(**ep,step=k,observation=o,truth=truth,query_before_future=query_ns<time.monotonic_ns(),**md,**actual))
                frames.append(current);forecasts.append(pred)
                if actual['terminal']:break
        receipt=model.finish();hist=np.bincount([r['truth'] for r in rows],minlength=4)
        result=dict(status='MODEL_PIXEL_INTERFACE_PASS' if np.count_nonzero(hist)>=2 else 'STOP_DEGENERATE_STREAM',
                    physical_akd_opened=False,samples=len(rows),target_histogram=hist.tolist(),
                    model_accuracy=float(np.mean([r['prediction']==r['truth'] for r in rows])),
                    all_model_forecasts_retained=True,skipped_inferences=0,**receipt)
        np.savez_compressed(out/'frames.npz',current=np.asarray(frames),forecast=np.asarray(forecasts))
        save(out/'rows.json',rows);save(out/'RESULT.json',result);print(json.dumps(result))
    except BaseException as e:
        save(out/'FAILURE.json',dict(error=repr(e),completed=len(rows)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);run(p.parse_args().study.resolve())
