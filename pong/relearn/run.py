"""Matched hardware diagnosis with a separately bound practical CPU baseline."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pong.run import Native as PreviousNative
from pong.experiment import encode,save,sha
from pong.relearn.protocol import ARMS,schedule
from pong.relearn.model import RecentNearest


class Native(PreviousNative):
    def __init__(self,study,name,fixture,cap=4096):
        command=(f'cd /home/nicklomeli/akd1500-lab && timeout --signal=INT --kill-after=5 178 '
            f'/home/nicklomeli/akida-cc-venv/bin/python pong/relearn/serve.py studies/{study.name} {name} {fixture} --forwards {cap}')
        self.process=subprocess.Popen(['ssh','-T','jetson',command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.id=0;self.ready=self.read();assert self.ready['kind']=='ready'


def run(study,seed,arm,nw,split):
    state=json.loads((study/'STATE.json').read_text());assert not state['research_admissions_closed']
    if split=='confirmation':assert state['confirmation_opened'] and state['admission']=='RELEARN_CONFIRMATION'
    else:assert state['admission'] in ['RELEARN_BASELINE','RELEARN_CANDIDATE']
    assert nw==state['admitted_num_weights']
    plan=json.loads((study/'PROTOCOL.json').read_text());rows=schedule(plan,arm,split)
    name=f'{split}_nw{nw}_s{seed}_{arm}';out=study/'runs'/name;out.mkdir(parents=True,exist_ok=False)
    fixture=f'studies/{study.name}/fixtures-relearn01/nw{nw}_s{seed}'
    native=Native(study,name,fixture,len(rows));baseline=RecentNearest();complete=0
    with (out/'events.jsonl').open('w') as log:
        try:
            for row in rows:
                x=encode(row['observation']);reply=native.forward(x,row['label'])
                cpu=baseline.forward(x,row['label'],row['observation']['court'])
                assert reply['cpu_task_prediction']==cpu,'physical host CPU baseline differs from Python rule'
                log.write(json.dumps(dict(**row,response=reply,cpu_prediction=cpu))+'\n');log.flush();complete+=1
            save(out/'CAPTURE.json',dict(status='MATCHED_NATIVE_COMPLETE',cleanup=native.close(),seed=seed,arm=arm,nw=nw,split=split,events_sha256=sha(out/'events.jsonl')))
        except BaseException as error:
            save(out/'FAILURE.json',dict(status='FAILED_PRESERVED',error=repr(error),completed=complete));raise
        finally:
            if native.process.poll() is None:
                try:native.close()
                except Exception:native.process.terminate();native.process.wait(timeout=10)
    print(json.dumps(dict(status='MATCHED_NATIVE_COMPLETE',run=name,forwards=complete)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('seed',type=int);p.add_argument('arm',choices=ARMS)
    p.add_argument('--nw',type=int,default=64);p.add_argument('--split',choices=['development','confirmation'],default='development');a=p.parse_args()
    run(a.study.resolve(),a.seed,a.arm,a.nw,a.split)
