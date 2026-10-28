"""Offline distilled own student, native integer4-bit internal activations."""
from pathlib import Path
import argparse,datetime as dt,json,os,signal,time,traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from visual_memory.data import masks,RESERVED,sha,save

class NativeBlock(nn.Module):
    def __init__(self,cin,cout,k,stride,qmax,divisor,pool=False,binary=False):
        super().__init__();self.w=nn.Parameter(torch.empty(cout,cin,k,k));nn.init.kaiming_uniform_(self.w,a=1)
        self.bn=nn.BatchNorm2d(cout);self.stride=stride;self.qmax=qmax;self.divisor=divisor;self.pool=pool;self.binary=binary
    def quantized(self):
        scale=self.w.detach().abs().amax((1,2,3),keepdim=True).clamp_min(1e-8)/self.qmax
        return (self.w/scale).round().clamp(-self.qmax,self.qmax),scale
    def folded(self,mean,var,scale):
        gamma=self.bn.weight.clamp_min(.01);sd=torch.sqrt(var+self.bn.eps);s=scale.flatten()/self.divisor
        threshold=torch.floor((mean-self.bn.bias*sd/gamma)/s)
        step=torch.ceil(6*sd/(gamma*s*15)).clamp_min(1)
        return threshold,step
    def forward(self,x,quant):
        q,s=self.quantized();w=self.w+(q*s-self.w).detach() if quant else self.w
        if self.divisor==255 and quant:x=(x*255).round()/255
        if w.shape[-1]==3:x=F.pad(x,(0,1,0,1))
        z=F.conv2d(x,w,stride=self.stride)
        if self.pool:z=F.max_pool2d(z,2)
        if self.training:mean=z.detach().mean((0,2,3));var=z.detach().var((0,2,3),unbiased=False)
        else:mean,var=self.bn.running_mean,self.bn.running_var
        logits=self.bn(z)
        soft=torch.sigmoid(logits) if self.binary else (logits/6).clamp(0,1)
        if quant:
            t,step=self.folded(mean,var,s);zi=(z/(s.flatten()/self.divisor)[None,:,None,None]).round()
            if self.binary:actual=(zi>t[None,:,None,None]).float()
            else:actual=torch.ceil((zi-t[None,:,None,None])/step[None,:,None,None]).clamp(0,15)/15
            soft=soft+(actual-soft).detach()
        return soft,logits

class EncoderV3(nn.Module):
    def __init__(self):
        super().__init__();self.blocks=nn.ModuleList([NativeBlock(3,32,3,2,127,255),NativeBlock(32,128,3,2,7,15,True),NativeBlock(128,256,3,2,7,15,True),NativeBlock(256,256,1,1,7,15,True),NativeBlock(256,256,1,1,7,15,True),NativeBlock(256,512,1,1,1,15,binary=True)])
        self.head=nn.Linear(512,40);self.category=nn.Linear(512,10)
    def forward(self,x,quant=False,features=False):
        for b in self.blocks:x,logits=b(x,quant)
        x=x.flatten(1)
        return x if features else (self.head(x),self.category(x),logits.flatten(1))
    def exported(self):
        out={}
        for i,b in enumerate(self.blocks):
            q,s=b.quantized();t,step=b.folded(b.bn.running_mean,b.bn.running_var,s)
            out[f'w{i}']=q.detach().cpu().numpy().transpose(2,3,1,0).astype('i1')
            if i and out[f'w{i}'].shape[0]==3:out[f'w{i}']=out[f'w{i}'][::-1,::-1].copy()
            out[f't{i}']=t.detach().cpu().numpy().astype('<i4');out[f's{i}']=step.detach().cpu().numpy().astype('<i4')
        return out

@torch.no_grad()
def evaluate(m,x,y,c,target,quant,fit_images,fit_labels):
    m.eval();pred=[];cat=[];agreement=[]
    for i in range(0,len(x),64):
        xx=torch.from_numpy(x[i:i+64]).to('mps').permute(0,3,1,2).float()/255;a,b,z=m(xx,quant);pred.extend(a.argmax(1).cpu().tolist());cat.extend(b.argmax(1).cpu().tolist());agreement.append((z>0).cpu().numpy()==target[i:i+64])
    def features(xx):
        rows=[]
        for i in range(0,len(xx),64):rows.append(m(torch.from_numpy(xx[i:i+64]).to('mps').permute(0,3,1,2).float()/255,quant,True).cpu().numpy())
        f=np.concatenate(rows);return f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-8)
    ff,df=features(fit_images),features(x);means=np.stack([ff[fit_labels==k].mean(0) for k in range(40)]);means/=np.maximum(np.linalg.norm(means,axis=1,keepdims=True),1e-8);proto=(df@means.T).argmax(1)
    return dict(object_correct=int((np.array(pred)==y).sum()),total=len(y),accuracy=float((np.array(pred)==y).mean()),prototype_correct=int((proto==y).sum()),prototype_accuracy=float((proto==y).mean()),category_accuracy=float((np.array(cat)==c).mean()),teacher_bit_agreement=float(np.concatenate(agreement).mean()),predictions=pred,prototype_predictions=proto.tolist())

def fit(data,teacher,study):
    assert json.loads((study/'SOURCE_SMOKE.json').read_text())['status']=='PASS'
    assert json.loads((study/'TOPOLOGY.json').read_text())['status']=='VIRTUAL_MAP_PASS'
    out=study/'encoder01';out.mkdir(exist_ok=False);t0=time.monotonic();owner=dict(pid=os.getpid(),owner='codex:01a11d87-0208-7613-bbc0-e5e612110c50',status='FIT_RUNNING',hardware_opened=False,maximum_wall_s=1800,utc=dt.datetime.now(dt.timezone.utc).isoformat());save(out/'OWNER.json',owner)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min cap')));signal.alarm(1800)
    try:
        torch.manual_seed(20261028);torch.set_num_threads(6);rng=np.random.default_rng(20261028);d=np.load(data,allow_pickle=False);mm=masks(d);tt=np.load(teacher,allow_pickle=False);lookup={int(x):i for i,x in enumerate(tt['ids'])};fi=np.flatnonzero(mm['fit']);di=np.flatnonzero(mm['dev'])
        tf=tt['normalized'][[lookup[int(i)] for i in fi]];td=tt['normalized'][[lookup[int(i)] for i in di]];mean=tf.mean(0);projection=rng.normal(size=(tf.shape[1],512)).astype('f4')/np.sqrt(tf.shape[1]);cut=np.median((tf-mean)@projection,axis=0)
        targets=((tf-mean)@projection>cut).astype('f4');devtargets=((td-mean)@projection>cut).astype('f4');np.savez(out/'teacher-transform.npz',mean=mean,projection=projection,cut=cut,fit_ids=fi,dev_ids=di,fit_targets=targets,dev_targets=devtargets)
        objects=np.array([q for q in range(1,51) if q not in RESERVED]);x=d['images'][fi];y=np.searchsorted(objects,d['object'][fi]).astype('i8');c=d['labels'][fi].astype('i8');vx=d['images'][di];vy=np.searchsorted(objects,d['object'][di]);vc=d['labels'][di]
        save(out/'FROZEN.json',dict(seed=20261028,epochs=120,continuous=80,quantized=40,data_sha256=sha(data),teacher_features_sha256=sha(teacher),teacher_transform_sha256=sha(out/'teacher-transform.npz'),source_sha256=sha(Path(__file__)),protocol_sha256=sha(study/'PROTOCOL.md'),official_test_evaluated=False))
        m=EncoderV3().to('mps');o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);tx=torch.from_numpy(x).to('mps').permute(0,3,1,2).float()/255;ty=torch.from_numpy(y).to('mps');tc=torch.from_numpy(c).to('mps');tb=torch.from_numpy(targets).to('mps');warm=final=None
        with (out/'EPOCHS.jsonl').open('x') as log:
            for epoch in range(120):
                m.train();order=rng.permutation(len(x));losses=[];correct=0
                for i in range(0,len(x),64):
                    ids=torch.from_numpy(order[i:i+64]).to('mps');a=tx[ids]
                    if rng.random()<.5:a=a.flip(3)
                    a=F.pad(a,(12,12,12,12),mode='replicate');dy,dx=rng.integers(0,25,2);a=a[:,:,int(dy):int(dy)+128,int(dx):int(dx)+128];a=(((a-.5)*float(rng.uniform(.85,1.15))+.5)*float(rng.uniform(.85,1.15))).clamp(0,1)
                    o.zero_grad(set_to_none=True);z,cat,bits=m(a,epoch>=80);loss=F.binary_cross_entropy_with_logits(bits,tb[ids])+.5*F.cross_entropy(z,ty[ids])+.5*F.cross_entropy(cat,tc[ids]);loss.backward();o.step()
                    with torch.no_grad():
                        for block in m.blocks:block.bn.weight.clamp_(.01,10)
                    losses.append(float(loss.detach().cpu()));correct+=int((z.argmax(1)==ty[ids]).sum().cpu())
                row=dict(epoch=epoch+1,phase='quantized4bit' if epoch>=80 else 'continuous',train_loss=float(np.mean(losses)),train_object_accuracy=correct/len(x),elapsed_s=time.monotonic()-t0)
                if epoch in [79,119]:
                    value=evaluate(m,vx,vy,vc,devtargets,epoch>=80,x,y);row['evaluation']=value
                    if epoch==79:warm=value;torch.save(m.state_dict(),out/'warmup.pt')
                    else:final=value
                log.write(json.dumps(row)+'\n');log.flush()
                if epoch%10==9:print(json.dumps({k:v for k,v in row.items() if k!='evaluation'}),flush=True)
        m.cpu().eval();torch.save(dict(model=m.state_dict()),out/'final.pt');np.savez(out/'encoder.npz',**m.exported())
        gates=dict(dev_object_accuracy=final['prototype_accuracy']>=.60,quantization_drop=warm['prototype_accuracy']-final['prototype_accuracy']<=.05)
        result=dict(status='DISTILLED_ENCODER_DEV_PASS' if all(gates.values()) else 'DISTILLED_ENCODER_FAILED_STOP',gates=gates,continuous=warm,quantized=final,elapsed_s=time.monotonic()-t0,official_test_evaluated=False,native_sessions=0,teacher='Offline official non-BrainChip MobileNetV3Small; own student weights start random, no pretrained weights deployed')
        save(out/'RESULT.json',result);owner['status']=result['status'];print(json.dumps({k:v for k,v in result.items() if k not in ['continuous','quantized']}),flush=True)
    except BaseException as e:owner.update(status='FAILED_PRESERVED',error=repr(e));save(out/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
    finally:signal.alarm(0);owner['elapsed_s']=time.monotonic()-t0;save(out/'OWNER.json',owner)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('teacher',type=Path);p.add_argument('study',type=Path);a=p.parse_args();fit(a.data,a.teacher,a.study)
