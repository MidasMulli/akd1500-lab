"""Isolated adapter over unchanged hash-checked Nano world-model sources."""
import os,sys,time
from pathlib import Path
import numpy as np
from streaming.design import ROLE,CHECKPOINT_SHA,band,commands

WORLD=Path('/home/nicklomeli/world_model_live_20261016')

class Nano:
    def __init__(self):
        sys.path.insert(0,str(WORLD));import demo_worker as W
        self.W=W;self.m,c,self.numeric=W.checked(ROLE)
        assert c['sha256']==CHECKPOINT_SHA
        self.digest=W.RT.EMOD.runtime_digest(self.m)
        self.lm=W.RT.LearnedModel(self.m,W.T.device('cuda'),64)
        self.calls=0
    def begin(self,row):
        W=self.W
        self.env=W.RT.SceneEnv(row['scene'],0,'K25' if row['context']==0 else 'G200',False)
        self.view=self.env.reset();self.b=None;self.pending=[];self.previous=self.view['frames'][-2].copy()
        self.current=self.view['frames'][-1].copy();self.blocks=commands(row['command_seed']);self.row=row
        return self.current
    def forecast(self,action):
        W=self.W;T=W.T;start=time.monotonic_ns()
        v=dict(self.view)
        v.update(actions=None if self.b is None else np.asarray(self.pending,np.float64),elapsed=self.env.t,remaining=40-self.env.t)
        self.b,t=self.lm.observe(v,self.b);self.pending=[]
        with T.no_grad(),W.RT.ServicesDisabled():
            h=self.b['h'].clone()
            for k in range(5):
                s=t+k;sc=T.tensor([[float(action[0]),float(action[1]),s/40.,(40-s)/40.]],device='cuda')
                h=self.m.step(h,sc)
            rgb=self.m.decode(h,self.b['last_rgb'])
            rgb=(rgb.clamp(0,1)*255).round().to(T.uint8).cpu().numpy()[0]
        self.calls+=1
        return rgb,dict(model_ns=time.monotonic_ns()-start,prediction=band(rgb,False),model_call=self.calls)
    def advance(self,action):
        applied=np.asarray(action,np.float64).copy()
        if self.row['reversed']:applied[1]*=-1
        for _ in range(5):self.env.step(applied);self.pending.append(list(map(float,action)))
        self.previous=self.current.copy();self.view=self.env.view();self.current=self.view['frames'][0].copy()
        return self.current,dict(observed_t=self.env.t,applied_action=applied.tolist(),terminal=self.env.terminal())
    def finish(self):
        assert self.W.RT.EMOD.runtime_digest(self.m)==self.digest,'Nano model weights drifted'
        return dict(checkpoint_sha256=CHECKPOINT_SHA,weights_digest_before_after=self.digest,
                    model_calls=self.calls,weights_unchanged=True,numeric_profile=self.numeric)
