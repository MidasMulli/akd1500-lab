"""One view-stability recipe; native topology reused, teacher only offline."""
from pathlib import Path
import argparse,datetime as dt,json,math,os,signal,time,traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from visual_memory.train_v3 import EncoderV3
from visual_memory.data import masks,RESERVED,sha,save

class EncoderV4(EncoderV3):
    def __init__(self):
        super().__init__();self.distill=nn.Linear(512,576,bias=False)
    def forward(self,x,quant=False,features=False,training_output=False):
        for b in self.blocks:x,logits=b(x,quant)
        f=x.flatten(1)
        if features:return f
        values=(self.head(f),self.category(f),logits.flatten(1))
        return (*values,self.distill(2*f-1),f) if training_output else values

def contrastive(f,y):
    z=F.normalize(2*f-1,dim=1);a=z@z.T/.2;diag=torch.eye(len(f),device=f.device,dtype=torch.bool);positive=y[:,None].eq(y[None,:])&~diag
    a=a.masked_fill(diag,-1e4);logp=a-torch.logsumexp(a,dim=1,keepdim=True)
    return -(logp*positive).sum(1).div(positive.sum(1).clamp_min(1)).mean()

def pair_indices(y,sessions,rng):
    classes=rng.choice(40,32,replace=False);out=[]
    for c in classes:
        ss=rng.choice([1,2,4,5,6],2,replace=False)
        for sess in ss:out.append(int(rng.choice(np.flatnonzero((y==c)&(sessions==sess)))))
    return np.array(out)

@torch.no_grad()
def features(m,x,quant):
    m.eval();out=[]
    for i in range(0,len(x),64):out.append(m(torch.from_numpy(x[i:i+64]).to('mps').permute(0,3,1,2).float()/255,quant,True).cpu().numpy())
    return np.concatenate(out)

@torch.no_grad()
def calibrate(m,x,quant):
    m.train()
    for b in m.blocks:b.bn.reset_running_stats();b.bn.momentum=None
    for i in range(0,len(x),64):m(torch.from_numpy(x[i:i+64]).to('mps').permute(0,3,1,2).float()/255,quant)
    for b in m.blocks:b.bn.momentum=.1
    m.eval()

@torch.no_grad()
def evaluate(m,x,y,vx,vy,vc,quant):
    ff,df=features(m,x,quant),features(m,vx,quant);pred=[];cat=[]
    for i in range(0,len(vx),64):
        a,b,z=m(torch.from_numpy(vx[i:i+64]).to('mps').permute(0,3,1,2).float()/255,quant);pred.extend(a.argmax(1).cpu().tolist());cat.extend(b.argmax(1).cpu().tolist())
    out={}
    for centered in [False,True]:
        a,b=(2*ff-1,2*df-1) if centered else (ff.copy(),df.copy());a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-8);b/=np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-8);means=np.stack([a[y==c].mean(0) for c in range(40)]);means/=np.maximum(np.linalg.norm(means,axis=1,keepdims=True),1e-8);p=(b@means.T).argmax(1)
        out['centered_prototype_accuracy' if centered else 'prototype_accuracy']=float((p==vy).mean())
    out.update(head_accuracy=float((np.array(pred)==vy).mean()),category_accuracy=float((np.array(cat)==vc).mean()),feature_density=float(df.mean()),total=len(vy));return out

def source(study):
    torch.manual_seed(20261030);torch.set_num_threads(6);rng=np.random.default_rng(20261030);x=rng.integers(0,256,(2,128,128,3),dtype='u1');m=EncoderV4().eval();w=m.exported();m.to('mps')
    with torch.no_grad():got=m(torch.from_numpy(x).to('mps').permute(0,3,1,2).float()/255,True,True).cpu().numpy()
    from visual_memory.encoder_v3 import forward
    want=forward(x,w);assert np.array_equal(got,want)
    y=np.repeat(np.arange(40),100);ss=np.tile(np.repeat([1,2,4,5,6],20),40);ii=pair_indices(y,ss,rng);assert np.array_equal(y[ii][::2],y[ii][1::2]) and (ss[ii][::2]!=ss[ii][1::2]).all()
    leaf=torch.randn(64,512,requires_grad=True);z=leaf.sigmoid();loss=contrastive(z,torch.tensor(y[ii]));assert torch.isfinite(loss);loss.backward();assert torch.isfinite(leaf.grad).all()
    np.savez(study/'dummy_weights.npz',**w);np.savez(study/'dummy_witness.npz',images=x,features=want)
    save(study/'SOURCE_SMOKE.json',dict(status='PASS',integer_vs_quantized_MPS=True,cross_session_pairs=True,finite_contrastive_gradient=True,physical_sessions=0,source_sha256=sha(__file__)))
    print('SOURCE_SMOKE_PASS',flush=True)

def fit(study):
    assert json.loads((study/'SOURCE_SMOKE.json').read_text())['status']=='PASS';assert json.loads((study/'TOPOLOGY.json').read_text())['status']=='VIRTUAL_MAP_PASS'
    out=study/'encoder01';out.mkdir(exist_ok=False);start=time.monotonic();owner=dict(status='FIT_RUNNING',pid=os.getpid(),owner='codex:01a11d87-0208-7613-bbc0-e5e612110c50',physical_device_opened=False,utc=dt.datetime.now(dt.timezone.utc).isoformat(),maximum_wall_s=1200);save(out/'OWNER.json',owner)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('fixed20minute fit cap')));signal.alarm(1200)
    try:
        torch.manual_seed(20261030);torch.set_num_threads(6);rng=np.random.default_rng(20261030);data=Path('/Users/midas/Models/core50-full01/core50_full.npz');teacher=Path('/Users/midas/akd1500-lab/studies/20261026-visual-memory-v2/teacher-diagnostic/features.npz');d=np.load(data);mm=masks(d);fi=np.flatnonzero(mm['fit']);di=np.flatnonzero(mm['dev']);tt=np.load(teacher);lookup={int(v):i for i,v in enumerate(tt['ids'])};tf=tt['normalized'][[lookup[int(v)] for v in fi]];mean=tf.mean(0);target=tf-mean;target/=np.maximum(np.linalg.norm(target,axis=1,keepdims=True),1e-8)
        objects=np.array([v for v in range(1,51) if v not in RESERVED]);x=d['images'][fi];y=np.searchsorted(objects,d['object'][fi]).astype('i8');c=d['labels'][fi].astype('i8');ss=d['session'][fi];vx=d['images'][di];vy=np.searchsorted(objects,d['object'][di]);vc=d['labels'][di]
        np.savez(out/'FIT_MANIFEST.npz',fit_ids=fi,dev_ids=di,teacher_fit_mean=mean);save(out/'FROZEN.json',dict(seed=20261030,epochs=240,continuous=160,quantized=80,batches_per_epoch=63,batch=64,source_sha256=sha(__file__),parent_sha256=sha(Path(__file__).with_name('train_v3.py')),protocol_sha256=sha(study/'PROTOCOL.md'),data_sha256=sha(data),teacher_sha256=sha(teacher),fresh_memory_sessions=[5,11],fresh_evaluated=False))
        m=EncoderV4().to('mps');opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);tx=torch.from_numpy(x).to('mps').permute(0,3,1,2).float()/255;ty=torch.from_numpy(y).to('mps');tc=torch.from_numpy(c).to('mps');tf=torch.from_numpy(target).to('mps');warm=final=None
        with (out/'EPOCHS.jsonl').open('x') as log:
            for epoch in range(240):
                assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,26,20,55,tzinfo=dt.timezone.utc)
                m.train();lr=.001*(epoch+1)/10 if epoch<10 else .00002+(.001-.00002)*.5*(1+math.cos(math.pi*(epoch-10)/230))
                for pg in opt.param_groups:pg['lr']=lr
                losses=[];correct=0
                for batch in range(63):
                    ii=torch.from_numpy(pair_indices(y,ss,rng)).to('mps');a=tx[ii]
                    if rng.random()<.5:a=a.flip(3)
                    a=F.pad(a,(12,12,12,12),mode='replicate');dy,dx=rng.integers(0,25,2);a=a[:,:,int(dy):int(dy)+128,int(dx):int(dx)+128];a=(((a-.5)*float(rng.uniform(.85,1.15))+.5)*float(rng.uniform(.85,1.15))).clamp(0,1)
                    opt.zero_grad(set_to_none=True);z,cat,bits,projected,f=m(a,epoch>=160,training_output=True);loss=F.cross_entropy(z,ty[ii])+.25*F.cross_entropy(cat,tc[ii])+(1-F.cosine_similarity(projected,tf[ii],dim=1)).mean()+.25*contrastive(f,ty[ii])+.05*((f.mean(0)-.5)**2).mean()*4;loss.backward();opt.step()
                    with torch.no_grad():
                        for b in m.blocks:b.bn.weight.clamp_(.01,10)
                    losses.append(float(loss.detach().cpu()));correct+=int((z.argmax(1)==ty[ii]).sum().cpu())
                row=dict(epoch=epoch+1,phase='quantized' if epoch>=160 else 'continuous',loss=float(np.mean(losses)),train_accuracy=correct/4032,lr=lr,elapsed_s=time.monotonic()-start)
                if epoch in [159,239]:
                    calibrate(m,x,epoch>=160);value=evaluate(m,x,y,vx,vy,vc,epoch>=160);row['evaluation']=value
                    if epoch==159:warm=value;torch.save(m.state_dict(),out/'warmup.pt')
                    else:final=value
                log.write(json.dumps(row)+'\n');log.flush()
                if epoch%10==9:print(json.dumps(row),flush=True)
        m.cpu().eval();torch.save(dict(model=m.state_dict()),out/'final.pt');np.savez(out/'encoder.npz',**m.exported());g=dict(dev_accuracy=final['prototype_accuracy']>=.60,quantization_drop=warm['prototype_accuracy']-final['prototype_accuracy']<=.05)
        r=dict(status='VIEW_STABLE_ENCODER_DEV_PASS' if all(g.values()) else 'VIEW_STABLE_ENCODER_FAILED_STOP',gates=g,continuous=warm,quantized=final,elapsed_s=time.monotonic()-start,physical_sessions=0,fresh_evaluated=False,teacher='Offline externalreference only; native export excludes distillation/classification heads');save(out/'RESULT.json',r);owner['status']=r['status'];print(json.dumps(r),flush=True)
    except BaseException as e:owner.update(status='FAILED_PRESERVED',error=repr(e));save(out/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
    finally:signal.alarm(0);owner['elapsed_s']=time.monotonic()-start;save(out/'OWNER.json',owner)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['source','fit']);p.add_argument('study',type=Path);a=p.parse_args();source(a.study) if a.mode=='source' else fit(a.study)
