"""Isolated frozen T202 forecast worker; no changes to the live deployment."""
import json,os,sys,time,hashlib
from pathlib import Path
import numpy as np
from streaming.nano import Nano

def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
    out=Path(sys.argv[1]);stage=time.monotonic();print('MODEL_LOADING',flush=True,file=sys.stderr);model=Nano();print('MODEL_LOADED_SECONDS',time.monotonic()-stage,flush=True,file=sys.stderr);W=model.W;T=W.T
    row=dict(scene=202610249100010,context=0,command_seed=106010,reversed=False)
    model.begin(row);belief,t=model.lm.observe(dict(model.view,actions=None,elapsed=0,remaining=40),None)
    print('BELIEF_READY_SECONDS',time.monotonic()-stage,flush=True,file=sys.stderr)
    actions=[T.tensor([[.15,.5,k/40.,(40-k)/40.]],device='cuda') for k in range(5)]
    def forecast():
        with T.no_grad(),W.RT.ServicesDisabled():
            h=belief['h'].clone()
            for action in actions:h=model.m.step(h,action)
            rgb=model.m.decode(h,belief['last_rgb'])
            return (rgb.clamp(0,1)*255).round().to(T.uint8).cpu().numpy()[0]
    warm=[forecast() for _ in range(8)];print('WARMUP_READY_SECONDS',time.monotonic()-stage,flush=True,file=sys.stderr);assert all(np.array_equal(x,warm[0]) for x in warm)
    np.save(out/'foreground-frame.npy',warm[0]);np.save(out/'observed-frame.npy',model.current)
    print(json.dumps(dict(kind='ready',pid=os.getpid(),checkpoint_sha256=model.finish()['checkpoint_sha256'],output_sha256=sha(warm[0]))),flush=True)
    command=json.loads(sys.stdin.readline());start=int(command['start_ns']);end=int(command['end_ns']);assert 0<end-start<=10_000_000_000
    assert start>time.monotonic_ns();time.sleep(max(0,(start-time.monotonic_ns())/1e9))
    rows=[];initial_cpu=time.process_time_ns()
    while time.monotonic_ns()<end:
        before=time.monotonic_ns();cpu=time.process_time_ns();rgb=forecast();after=time.monotonic_ns()
        # Hash after the timed forecast, in the common workload interval.
        digest=sha(rgb);assert digest==sha(warm[0]),'foreground output changed'
        rows.append([before,after,time.process_time_ns()-cpu])
    used=time.process_time_ns()-initial_cpu
    np.save(out/'foreground-timing.npy',np.asarray(rows,dtype=np.int64))
    receipt=model.finish();receipt.update(kind='closed',start_ns=start,end_ns=end,process_cpu_ns=used,calls=len(rows),output_sha256=sha(warm[0]),pid=os.getpid(),first_call_ns=rows[0][0],last_call_end_ns=rows[-1][1])
    (out/'MODEL_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
