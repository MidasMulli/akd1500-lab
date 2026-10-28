"""Prospective encoding comparison on own fitted features; CPU only until admitted."""
from pathlib import Path
import argparse,json,time
import numpy as np
from visual_memory.recruitment import schedule
from visual_memory.data import RESERVED,sha,save
from pong.relearn.model import Reference

P=Path(__file__).resolve().parents[1];S=P/'studies/20261026-visual-memory-v4';U=220;C=11;WIDTH=20

class NativeMemory(Reference):
    def __init__(self,ram,n):super().__init__(ram,n=n,u=U,nw=64,classes=C)
    def forward(self,v,label,learning=True):
        x=sum(1<<int(k) for k in np.flatnonzero(v));ps=[(w&x).bit_count() for w in self.weights]
        if not learning:return ps
        eligible=[j for j in range(label*WIDTH,(label+1)*WIDTH) if ps[j]>(self.ram[2+j*self.stride+3]>>16)];win=max(eligible,key=lambda j:(ps[j],-j)) if eligible else None;count=0
        if win is not None:count=min(self.ram[2+win*self.stride+3]&65535,(self.weights[win]&~x).bit_count(),(x&~self.weights[win]).bit_count())
        got=super().forward(v,label)
        if count:self.ram[2+win*self.stride+3]=(min(ps[win],64)<<16)|(self.ram[2+win*self.stride+3]&65535)
        return got

def weights(dual):
    rng=np.random.default_rng(20261030);n=1024 if dual else 512;w=np.zeros((n,U),'i1')
    for j in range(U):
        indices=rng.choice(512,64,False);polarity=np.random.default_rng(20261031+j).integers(0,2,64) if dual else np.zeros(64,'i8');w[indices+polarity*512,j]=1
    return w

def initial(dual):
    n=1024 if dual else 512;stride=4+n//32;ram=np.zeros(2+U*stride,'<u4');w=weights(dual)
    for j in range(U):
        b=2+j*stride;ram[b+3]=((64 if j>=200 else 0)<<16)|32
        for k in np.flatnonzero(w[:,j]):ram[b+4+k//32]|=np.uint32(1<<(int(k)%32))
    return ram

def simulate(x,y,ram):
    r=NativeMemory(ram,len(x[0]));scores=[];states=[]
    for v,label in zip(x,y):
        before=r.ram.copy();scores.append(r.forward(v,int(label),label!=10));states.append(r.ram.copy())
        if label==10:assert before==r.ram
    return np.array(scores,'<i4'),np.array(states,'<u4')

def quality(pred,rows):
    out={}
    for phase in ['old','before','after','retention']:
        ii=[i for i,r in enumerate(rows) if r['phase']==phase];correct=sum(int(pred[i])==rows[i]['label'] for i in ii);out[phase]=dict(correct=correct,total=len(ii),accuracy=correct/len(ii))
    return out

def native_quality(scores,rows):return quality(scores[:,:200].reshape(-1,10,20).max(2).argmax(1),rows)

def gates(a,f):
    return dict(old=a['old']['accuracy']>=.70,changed_after=a['after']['accuracy']>=.65,gain=a['after']['accuracy']-f['after']['accuracy']>=.10-1e-10,retention=a['old']['accuracy']-a['retention']['accuracy']<=.05+1e-10)

def capture(stage):
    import torch
    from visual_memory.train_v4 import EncoderV4,features
    from visual_memory.encoder_v3 import forward
    from visual_memory.task_controls import sequence
    assert stage in ['development','fresh'];assert json.loads((S/'encoder01/RESULT.json').read_text())['status']=='VIEW_STABLE_ENCODER_DEV_PASS'
    if stage=='fresh':assert json.loads((S/'DEVELOPMENT_NATIVE_REVIEW.json').read_text())['pass']
    out=S/(stage+'-screen');out.mkdir(exist_ok=False);start=time.monotonic();torch.set_num_threads(6);d=np.load('/Users/midas/Models/core50-full01/core50_full.npz');old,new=(1,2) if stage=='development' else (5,11);ids=np.flatnonzero(np.isin(d['object'],RESERVED)&np.isin(d['session'],[old,new]));x=d['images'][ids];m=EncoderV4();m.load_state_dict(torch.load(S/'encoder01/final.pt',weights_only=True,map_location='cpu')['model']);m.to('mps').eval();f=features(m,x,True).astype('u1');w=np.load(S/'encoder01/encoder.npz')
    for i in range(0,len(x),8):assert np.array_equal(forward(x[i:i+8],w),f[i:i+8]),('integer/MPS',i)
    labels=np.searchsorted(RESERVED,d['object'][ids]);s=d['session'][ids];frame=d['frame'][ids];xx,teacher,rows=schedule(f,labels,s,frame,old,new);raw=np.array([x[r['item']] for r in rows]);frozen=teacher.copy()
    for i,r in enumerate(rows):
        if r['phase']=='adapt':frozen[i]=10
    np.savez_compressed(out/'OWN_FEATURES.npz',features=f,ids=ids,labels=labels,session=s,frame=frame)
    controls={}
    for freeze in [False,True]:
        near,centroid=sequence(xx,teacher,freeze)
        for name,p in [('nearest_hamming',near),('cosine_class_mean',centroid)]:controls[name+('_frozen' if freeze else '_adaptive')]=quality(p,rows)
    result=dict(stage=stage,physical_sessions=0,integer_MPS_all400_images=True,controls=controls,encodings={});eligible=[]
    for dual,name in [(False,'positive'),(True,'complementary')]:
        inputs=np.concatenate([xx,1-xx],axis=1) if dual else xx;ram=initial(dual);arm={};dest=out/name;dest.mkdir()
        for freeze,labelvec in [(False,teacher),(True,frozen)]:
            tag='frozen' if freeze else 'adaptive';scores,states=simulate(inputs,labelvec,ram);arm[tag]=native_quality(scores,rows);np.savez_compressed(dest/(tag+'.npz'),scores=scores,ram=states,labels=labelvec)
        gate=gates(arm['adaptive'],arm['frozen']);result['encodings'][name]=dict(gates=gate,arms=arm,learner_RAM_bytes=len(ram)*4,records=U,num_weights=64,input_width=inputs.shape[1]);eligible+=([name] if all(gate.values()) else [])
        (dest/'inputs.u8').write_bytes(inputs.tobytes());(dest/'initial.ram').write_bytes(ram.tobytes())
    chosen='complementary' if 'complementary' in eligible else ('positive' if 'positive' in eligible else None)
    if stage=='fresh':
        chosen=json.loads((S/'development-screen/RESULT.json').read_text())['chosen']
        status='FIXED_FRESH_NUMERIC_SCREEN_READY'
    else:status='CPU_RETENTION_GATE_PASS' if chosen else 'CPU_RETENTION_FAILED_STOP'
    result.update(status=status,chosen=chosen,selected_quality_pass=chosen in eligible,elapsed_s=time.monotonic()-start,own_encoder_sha256=sha(S/'encoder01/encoder.npz'),feature_sha256=sha(out/'OWN_FEATURES.npz'))
    save(out/'ROWS.json',rows);(out/'raw-images.u8').write_bytes(raw.tobytes());save(out/'RESULT.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['development','fresh']);a=p.parse_args();capture(a.stage)
