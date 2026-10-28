"""Finite consumed-data screen for a native feature-carrying manager tail."""
from pathlib import Path
import json,time
import numpy as np
from pong.relearn.model import Reference
from visual_memory.retention_v4 import native_quality,gates
from visual_memory.data import save,sha
P=Path(__file__).resolve().parents[1];S=P/'studies/20261026-visual-admission';V=P/'studies/20261026-visual-memory-v4'

class Rule(Reference):
    def __init__(self,ram,nw):super().__init__(ram,n=512,u=220,nw=nw,classes=11)
    def forward(self,v,label,learning=True):
        if not learning:
            x=sum(1<<int(k) for k in np.flatnonzero(v));return [(w&x).bit_count() for w in self.weights]
        before=self.weights.copy();ps=super().forward(v,label)
        for j,(a,b) in enumerate(zip(before,self.weights)):
            if a!=b:self.ram[2+j*self.stride+3]=(min(ps[j],self.nw)<<16)|(self.ram[2+j*self.stride+3]&65535)
        return ps

def initial(bits,nw,seed=20261041):
    w=np.zeros((512,220),'i1');rng=np.random.default_rng(seed)
    for j in range(220):w[rng.choice(bits*2,nw,False),j]=1
    w[:,219]=0;w[400:400+nw-1,219]=1;w[511,219]=1
    ram=np.zeros(2+220*20,'<u4')
    for j in range(220):
        b=2+j*20;ram[b+3]=((0 if j<200 else nw)<<16)|int(np.floor(nw*.5+.5))
        for k in np.flatnonzero(w[:,j]):ram[b+4+k//32]|=np.uint32(1<<int(k%32))
    return ram

def run(x,y,ram,nw):
    r=Rule(ram,nw);scores=[];states=[]
    for v,label in zip(x,y):scores.append(r.forward(v,int(label),label!=10));states.append(r.ram.copy())
    return np.asarray(scores,'<i4'),np.asarray(states,'<u4')

def kernel(bits,kind):
    rng=np.random.default_rng(20261041+bits);w=np.zeros((512,bits),'i1')
    if kind=='subset':w[np.sort(rng.choice(512,bits,False)),np.arange(bits)]=1
    else:
        for j in range(bits):w[:,j]=-1;w[rng.choice(512,256,False),j]=1
    return w

def encode(features,w):
    f=(features.astype('i4')@w.astype('i4')>0).astype('u1');x=np.zeros((len(f),512),'u1');x[:,:w.shape[1]*2]=np.concatenate([f,1-f],1);x[:,511]=1;return x

def main():
    out=S/'compact-screen01';out.mkdir(exist_ok=False);start=time.monotonic();results={};passing=[]
    for bits in [128,160]:
        for kind in ['subset','balanced_projection']:
            w=kernel(bits,kind)
            for nw in [32,64]:
                name=f'{kind}{bits}_w{nw}';d=out/name;d.mkdir();np.savez_compressed(d/'KERNEL.npz',weights=w);ram=initial(bits,nw);ram.tofile(d/'initial.ram');cases={}
                for stage in ['development','fresh']:
                    src=V/(stage+'-screen');rows=json.loads((src/'ROWS.json').read_text());f=np.load(src/'OWN_FEATURES.npz')['features'];x=encode(f[[r['item'] for r in rows]],w);arms={}
                    for tag in ['adaptive','frozen']:
                        y=np.load(src/'complementary'/(tag+'.npz'))['labels'];scores,states=run(x,y,ram,nw);arms[tag]=native_quality(scores,rows);np.savez_compressed(d/(stage+'_'+tag+'.npz'),scores=scores,states=states,inputs=x,labels=y)
                    cases[stage]=dict(arms=arms,gates=gates(arms['adaptive'],arms['frozen']),evidence_role='CONSUMED_DEVELOPMENT_ONLY')
                passed=all(all(c['gates'].values()) for c in cases.values());score=min(c['arms']['adaptive'][k]['accuracy'] for c in cases.values() for k in ['old','after']);results[name]=dict(bits=bits,kind=kind,num_weights=nw,cases=cases,pass_gate=passed,selection_score=score)
                if passed:passing.append((score,name))
                print(name,passed,{s:{k:v['accuracy'] for k,v in c['arms']['adaptive'].items()} for s,c in cases.items()},flush=True)
    chosen=max(passing,key=lambda q:q[0])[1] if passing else None
    result=dict(status='COMPACT_DEVELOPMENT_PASS' if chosen else 'COMPACT_DEVELOPMENT_FAILED_STOP',chosen=chosen,candidates=results,elapsed_s=time.monotonic()-start,physical_sessions=0,encoder_fits=0,new_fresh_records_opened=0,spec_sha256=sha(S/'COMPACT_SCREEN_SPEC.json'));save(out/'RESULT.json',result);print(result['status'],chosen,flush=True)
if __name__=='__main__':main()
