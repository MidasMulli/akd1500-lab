"""Actual emulator experience; raw pixels are the only state-encoding input."""
from pathlib import Path
import argparse,hashlib,json
import gymnasium as gym
import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from PIL import Image

STARTS={'train':[106,110,114,126,130,134],'test':[108,112,116,124,128,132,136,138]}

def encode(frame,previous,sky,action):
    def occupied(a):
        mask=np.max(np.abs(a[96:224].astype(np.int16)-sky.astype(np.int16)),axis=2)>25
        return (mask.reshape(8,16,16,16).sum(axis=(1,3))>=24).reshape(-1).astype(np.uint8)
    bits=np.concatenate([occupied(frame),occupied(previous)]);assert bits.shape==(256,)
    x=np.zeros(512,np.uint8);x[256*action:256*(action+1)]=bits;return x

def trial(start,action,folder=None):
    env=JoypadSpace(gym.make('SuperMarioBros-1-1-v0',render_mode='rgb_array'),SIMPLE_MOVEMENT)
    try:
        frame,info=env.reset(seed=81701);sky=frame[50,0].copy();history=[frame.copy()]
        for k in range(start):
            frame,_,terminated,truncated,info=env.step(1)
            assert not terminated and not truncated,'prefix already dead'
            history.append(frame.copy())
        x=encode(frame,history[-9],sky,action)
        if folder:folder.mkdir(parents=True,exist_ok=False);Image.fromarray(frame).save(folder/'start.png');Image.fromarray(history[-9]).save(folder/'previous.png')
        start_x=info['x_pos'];dead=False;delay=0 if action==0 else 16;used=0
        for k in range(90):
            frame,_,terminated,truncated,info=env.step(2 if delay<=k<delay+20 else 1);used+=1
            if info.get('death') or terminated or truncated:dead=True;break
        if folder:Image.fromarray(frame).save(folder/'outcome.png')
        result=dict(start=start,action=action,success=not dead and info['x_pos']>=350,dead=dead,
            horizon=90,frames_executed=used,start_x_evaluation_only=start_x,end_x_evaluation_only=info['x_pos'],
            actor_input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),frame_sha256=hashlib.sha256(history[-1].tobytes()).hexdigest(),
            previous_frame_sha256=hashlib.sha256(history[-9].tobytes()).hexdigest(),active_features=int(x.sum()))
        return x,result
    finally:env.close()

def main(out):
    out.mkdir(parents=True,exist_ok=False);arrays={};rows=[]
    for split,starts in STARTS.items():
        xx=[];yy=[]
        for start in starts:
            for action in [0,1]:
                x,row=trial(start,action,out/'frames'/f'{split}_{start}_{action}');row['split']=split
                xx.append(x);yy.append(int(row['success']));rows.append(row)
        arrays[split+'_inputs']=np.array(xx,np.uint8);arrays[split+'_labels']=np.array(yy,np.int32)
    # Positive repeatability control, outside the train/test input arrays.
    x,row=trial(110,0);original=next(r for r in rows if r['start']==110 and r['action']==0)
    assert row=={k:v for k,v in original.items() if k!='split'}
    aliases={}
    for row in rows:aliases.setdefault(row['actor_input_sha256'],[]).append(row)
    contradictions=[v for v in aliases.values() if len({r['success'] for r in v})>1]
    train_hash={r['actor_input_sha256'] for r in rows if r['split']=='train'}
    overlap=sum(r['actor_input_sha256'] in train_hash for r in rows if r['split']=='test')
    np.savez_compressed(out/'experience.npz',**arrays)
    report=dict(status='OBSERVABILITY_PASS' if not contradictions else 'STOP_ENCODED_STATE_ALIAS',
        actor_uses_game_RAM=False,scoring_uses_game_info=True,repeatability=True,train_test_encoded_overlap=overlap,
        opposite_label_aliases=contradictions,rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'EXPERIENCE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','opposite_label_aliases']}))
    print('test outcomes',[(r['start'],r['action'],r['success']) for r in rows if r['split']=='test'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();main(a.output.resolve())
