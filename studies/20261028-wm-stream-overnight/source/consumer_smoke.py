"""New consumer qualification: old released S0 inputs, mocked outcomes, no new science."""
import argparse, importlib.util, json, os, subprocess, threading
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('python');a=p.parse_args();b=a.base
spec=importlib.util.spec_from_file_location('peer',b/'peer-source-r2/wm_stream_r2.py');peer=importlib.util.module_from_spec(spec);spec.loader.exec_module(peer)
z=np.load(b/'S0/S0_cuda_fp32_r4.npz')
class SavedWM:
    def episode_query(self,ctx,gain):
        c=ctx[0];rg='A' if gain==1.0 else 'B';prefix=f'c{c}_{rg}_'
        return {key:z[prefix+key].tolist() for key in ['h0_pool','h5_pool','wm_logits']} | {'wm_p':z[prefix+'p'].tolist(),'wm_pT':z[prefix+'pT'].tolist()},None
    def finish(self,state):return 'D',40
plan={'events':[dict(scene_id=c,twin=0,program_id=['P4','P4','P1','P4'][c],regime=rg,stream='R') for c in range(4) for rg in ['A','B']]}
res={}
for mode in ['ensemble','stream-cpu']:
    d=b/f'qualification-{mode}-v1';out=b/f'qualification-{mode}-v1-consumer'
    logs=(b/f'qualification-{mode}-v1.log').open('w')
    cmd=[a.python,str(b/'source/live_consumer.py'),'--stream',str(d),'--out',str(out),'--prepared',str(b/'prepared-c32-v2'),'--events','8','--K','8','--mode',mode]
    if mode!='ensemble':cmd+=['--actor',str(b/'stream_actor')]
    proc=subprocess.Popen(cmd,stdout=logs,stderr=logs)
    r=peer.run(SavedWM(),plan,d)
    code=proc.wait(timeout=15);logs.close();assert r=='DONE' and code==0,(mode,r,code)
    status=json.loads((out/'STATUS.json').read_text());assert status['events']==8
    res[mode]=status
(b/'CONSUMER_SMOKE_QA.json').write_text(json.dumps(dict(status='PASS',source='old released S0 inputs; synthetic D outcomes',physical=False,runs=res),indent=2)+'\n');print(json.dumps(res))
