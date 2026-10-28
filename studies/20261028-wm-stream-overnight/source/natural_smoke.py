"""End-to-end new N consumer: released S0 inputs and explicit synthetic skips."""
import argparse,importlib.util,json,os,subprocess,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('base',type=Path);a=p.parse_args();b=a.base
s=importlib.util.spec_from_file_location('peer3',b/'peer-source-r3/wm_stream_r3.py');peer=importlib.util.module_from_spec(s);s.loader.exec_module(peer);z=np.load(b/'S0/S0_cuda_fp32_r4.npz')
class SavedWM:
    def __init__(self,skips):self.skips=set(skips)
    def episode_query(self,ctx,gain):
        if ctx[0] in self.skips:return None,('S',3)
        c=ctx[0]%4;rg='A' if gain==1 else 'B';pr=f'c{c}_{rg}_'
        q={k:z[pr+k].tolist() for k in ['h0_pool','h5_pool','wm_logits']};q.update(wm_p=z[pr+'p'].tolist(),wm_pT=z[pr+'pT'].tolist());return q,None
    def finish(self,state):return state if state else ('D',40)
plan={'events':[dict(scene_id=i,twin=0,program_id=['P4','P4','P1','P4'][i%4],regime='A' if i%2==0 else 'B',stream='N') for i in range(8)]}
result={}
for case,skips in [('all-query',[]),('mixed',[1,2,4]),('skip-first',[0,3]),('all-skip',list(range(8)))]:
    stream=b/f'qualification-N-{case}-v1';out=b/f'qualification-N-{case}-v1-consumer';log=(b/f'qualification-N-{case}-v1.log').open('w')
    env=dict(os.environ,PYTHONPATH=str(b.parents[1]),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    proc=subprocess.Popen([sys.executable,str(b/'source/natural_consumer.py'),'--stream',str(stream),'--out',str(out),'--prepared',str(b/'prepared-c32-v2'),'--episodes','8'],stdout=log,stderr=log,env=env)
    r=peer.run(SavedWM(skips),plan,stream);code=proc.wait(timeout=15);log.close();assert r=='DONE' and code==0,(case,r,code)
    st=json.loads((out/'STATUS.json').read_text());assert st['query_rows']==8-len(skips) and st['skip_rows']==len(skips)
    result[case]=st
r=dict(status='PASS',physical=False,source='old released S0 inputs with mocked outcomes and synthetic terminal skips',cases=result);(b/'N_CONSUMER_SMOKE_QA.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
