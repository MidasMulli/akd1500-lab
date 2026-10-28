"""Synthetic numerical witness for a32-class primitive; no native selector claim."""
import argparse,json
from pathlib import Path
import numpy as np
from fixture_base import Recorder
from core import Layout
from pong.relearn.model import Reference

class Memory32(Reference):
    label=0
    def step(self,x,mode):
        if not mode:
            bits=sum(1<<int(k)for k in np.flatnonzero(x));return np.array([(w&bits).bit_count()for w in self.weights],'i4')
        before=self.weights.copy();scores=np.array(self.forward(x,self.label),'i4')
        for j,(a,b)in enumerate(zip(before,self.weights)):
            if a!=b:
                k=2+j*self.stride;self.ram[k+3]=(min(int(scores[j]),1 if mode==2 else 30)<<16)|(self.ram[k+3]&65535)
        return scores
class Witness(Recorder):
    teacher_class=0
    def __init__(self,prepared,out):
        super().__init__(prepared,out,'oct9');self.c.memory=Memory32(np.fromfile(out/'bank1.ram','<u4'),self.g.n1,32,32,32)
    def add(self,net,x,mode=-1,label=0,external=False,candidate=-1,phase=-1):
        if net:label=self.teacher_class;self.c.memory.label=label
        return super().add(net,x,mode,label,external,candidate,phase)
    def protocol(self,payload,write,blocked=False):
        self.role='feedback';x=self.g.encode(payload,candidate=1 if blocked else 0,truth=1,write=write)
        for phase,mode in enumerate([0,1,2,1,1,0]):
            y=self.add(0,x,external=phase==0,phase=phase)
            self.add(1,y[:self.g.P].view('u1'),mode,phase=phase)
            x=self.mail.view('u1')[4*self.g.P:4*self.g.P+1024].copy()
    def inspect(self,payload):
        self.role='query';y=self.add(0,self.g.encode(payload,candidate=0,truth=None),external=True)
        return self.add(1,y[:self.g.P].view('u1'),0)[1:33].astype(int).tolist()

p=argparse.ArgumentParser();p.add_argument('prepared',type=Path);p.add_argument('out',type=Path);p.add_argument('--arm',required=True,choices=['wildcard-retire','no-retire','permission-blocked']);a=p.parse_args();r=Witness(a.prepared,a.out);r.bootstrap();g=r.g;rng=np.random.default_rng(27102811)
def cue():return g.value_key(g.key(rng.integers(0,8,9),int(rng.integers(0,6))),int(rng.integers(0,3)))
keys=[cue()for _ in range(32)];new=[cue()for _ in range(64)]
for i,k in enumerate(keys):
    r.episode=i;r.teacher_class=i
    for _ in range(2):r.protocol(k,True)
for i,k in enumerate(keys):assert r.inspect(k)==[int(j==i)for j in range(32)]
checks=[]
for n,victim in enumerate([0,17]):
    r.episode=32+n;r.teacher_class=victim;k=new[n];before=np.array(r.c.memory.ram,'<u4').copy()
    if a.arm=='wildcard-retire':r.protocol(np.ones(g.P,'u1'),False)
    for _ in range(2):r.protocol(k,True,a.arm=='permission-blocked')
    after=np.array(r.c.memory.ram,'<u4');protected=all(np.array_equal(before[2+j*26:2+(j+1)*26],after[2+j*26:2+(j+1)*26])for j in range(32)if j!=victim);assert protected
    answer=r.inspect(k);ok=answer==[int(j==victim)for j in range(32)]
    if a.arm=='wildcard-retire':assert ok;keys[victim]=k.copy()
    else:assert not ok
    assert all(r.inspect(old)==[int(j==i)for j in range(32)]for i,old in enumerate(keys))
    checks.append(dict(victim=victim,new_recall=ok,protected31=True))
r.save(dict(source_only=True,classes=32,arm=a.arm,witness=checks,native_selector_implemented=False))
