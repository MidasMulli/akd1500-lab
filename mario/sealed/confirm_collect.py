"""Collect the new confirmation only after its pipeline is locked."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from bank import encode, save, sha
from collect import trial
from confirm_design import design


def collect(manifest, lock, output):
    assert json.loads(manifest.read_text())==design()
    pipeline=json.loads(lock.read_text());assert pipeline['status']=='PIPELINE_FROZEN_BEFORE_TEST'
    for path,digest in pipeline['source_hashes'].items():assert sha(path)==digest,('source drift',path)
    output.mkdir(parents=True,exist_ok=False);rows=[];arrays={}
    for context in design()['contexts']:
        for action in [0,1]:
            row,observation,_=trial(context,action)
            if observation is not None:
                x=encode(**observation,action=action,encoder='projection',elapsed=context['wait']+context['start'])
                arrays[f'{context["id"]}_{action}_projection']=x
                row['projection_input_sha256']=hashlib.sha256(x.tobytes()).hexdigest()
            rows.append(row)
    np.savez_compressed(output/'inputs.npz',**arrays)
    save(output/'EXPERIENCE.json',dict(status='COLLECTED',splits=['test'],observed_clock=True,
        actor_game_RAM=False,manifest_sha256=sha(manifest),pipeline_lock_sha256=sha(lock),rows=rows))
    print(json.dumps(dict(valid_contexts=sum(r['valid'] for r in rows)//2,invalid_contexts=sum(not r['valid'] for r in rows)//2)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('lock',type=Path);p.add_argument('output',type=Path);a=p.parse_args();collect(a.manifest.resolve(),a.lock.resolve(),a.output.resolve())
