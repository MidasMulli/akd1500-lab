"""Unchanged T202 forecast workload. No training, skip, deployment or input change."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
from streaming.nano import Nano
from streaming.design import episodes

def main(out):
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);model=Nano()
    model.begin(episodes(smoke=True)[0]);action=model.blocks[0];first=None
    for i in range(20):
        model.b=None;model.pending=[];rgb,md=model.forecast(action)
        assert rgb.ndim==3 and rgb.shape[2]==3 and min(rgb.shape[:2])>=32 and rgb.std()>5 and rgb.max()>64,'valid nonconstant frozen-model output'
        if first is None:first=rgb.copy()
        else:assert np.array_equal(first,rgb),'deterministic fixed workload'
    print(json.dumps(dict(kind='ready',role='T202',profile=model.numeric,cpu_threads=model.W.T.get_num_threads(),output_shape=list(first.shape),output_sha256=hashlib.sha256(first.tobytes()).hexdigest())),flush=True)
    assert input()=='GO';rows=[];start=time.monotonic_ns()
    while time.monotonic_ns()-start<5_000_000_000:
        model.b=None;model.pending=[];t=time.monotonic_ns();rgb,md=model.forecast(action);end=time.monotonic_ns();assert np.array_equal(rgb,first)
        rows.append([t,end,md['model_ns']])
    np.asarray(rows,'<i8').tofile(str(out)+'.frames_ns');receipt=model.finish()
    result=dict(kind='complete',status='PASS',frames=len(rows),wall_ns=time.monotonic_ns()-start,receipt=receipt,output_sha256=hashlib.sha256(first.tobytes()).hexdigest())
    Path(str(out)+'.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();main(a.out)
