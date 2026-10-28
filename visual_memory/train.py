"""One fixed own-encoder fit; no hardware use, official test stays unopened."""
from pathlib import Path
import argparse, datetime as dt, hashlib, json, os, signal, time, traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from visual_memory.data import masks,sha,save

SEED=20261026

class Block(nn.Module):
    def __init__(self,cin,cout,k,stride,qmax,pool=False):
        super().__init__();self.w=nn.Parameter(torch.empty(cout,cin,k,k));nn.init.kaiming_uniform_(self.w,a=1)
        self.bn=nn.BatchNorm2d(cout);self.stride=stride;self.qmax=qmax;self.pool=pool
    def quantized(self):
        s=self.w.detach().abs().amax((1,2,3),keepdim=True).clamp_min(1e-8)/self.qmax
        q=(self.w/s).round().clamp(-self.qmax,self.qmax)
        return q,s
    def forward(self,x,quant):
        q,s=self.quantized();w=self.w+(q*s-self.w).detach() if quant else self.w
        if w.shape[-1]==3:x=F.pad(x,(0,1,0,1))
        z=F.conv2d(x,w,stride=self.stride)
        if self.pool:z=F.max_pool2d(z,2)
        z=self.bn(z);soft=torch.sigmoid(2*z)
        return soft+((z>0).float()-soft).detach() if quant else soft

class Encoder(nn.Module):
    def __init__(self):
        super().__init__();self.blocks=nn.ModuleList([Block(3,32,3,2,127),Block(32,512,3,2,7,True),Block(512,8,1,1,1)])
        self.head=nn.Linear(512,10)
    def forward(self,x,quant=False,features=False):
        for b in self.blocks:x=b(x,quant)
        x=x.transpose(2,3).flatten(1) # Native FNP axons: channel, column, row.
        return x if features else self.head(x)
    def exported(self):
        out={}
        for i,b in enumerate(self.blocks):
            q,s=b.quantized();assert bool((b.bn.weight>0).all())
            scale=s.flatten()/(255 if i==0 else 1)
            threshold=torch.floor((b.bn.running_mean-b.bn.bias*torch.sqrt(b.bn.running_var+b.bn.eps)/b.bn.weight)/scale)
            out[f'w{i}']=q.detach().cpu().numpy().transpose(2,3,1,0).astype('i1')
            out[f't{i}']=threshold.detach().cpu().numpy().astype('<i4')
        # Hardware CNP is convolution; HRC is cross correlation.
        out['w1']=out['w1'][::-1,::-1].copy()
        out['readout_w']=self.head.weight.detach().cpu().numpy().T
        out['readout_b']=self.head.bias.detach().cpu().numpy()
        return out

@torch.no_grad()
def evaluate(m,x,y,device,quant):
    m.eval();pred=[]
    for i in range(0,len(x),64):pred.extend(m(torch.from_numpy(x[i:i+64]).to(device).permute(0,3,1,2).float()/255,quant).argmax(1).cpu().tolist())
    p=np.array(pred);return dict(correct=int((p==y).sum()),total=len(y),accuracy=float((p==y).mean()),predictions=p.tolist())

def fit(data,study):
    out=study/'encoder01';out.mkdir(exist_ok=False);t0=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min owned fit cap')));signal.alarm(1800)
    owner=dict(pid=os.getpid(),owner='codex:01a11d87-0208-7613-bbc0-e5e612110c50',utc=dt.datetime.now(dt.timezone.utc).isoformat(),status='FIT_RUNNING',device='mps',maximum_wall_s=1800,hardware_opened=False)
    save(out/'OWNER.json',owner)
    try:
        assert torch.backends.mps.is_available();torch.manual_seed(SEED);np.random.seed(SEED);torch.set_num_threads(6)
        d=np.load(data,allow_pickle=False);mm=masks(d);x=np.asarray(d['images'][mm['fit']]);y=np.asarray(d['labels'][mm['fit']]).astype('i8')
        dev={k:(np.asarray(d['images'][mm[k]]),np.asarray(d['labels'][mm[k]])) for k in ['dev','reserved_dev']}
        assert x.shape==(4000,64,64,3) and set(d['session'][mm['fit']])=={1,2,4,5,6}
        save(out/'FROZEN.json',dict(seed=SEED,source_sha256=sha(Path(__file__)),data_sha256=sha(data),protocol_sha256=sha(study/'PROTOCOL.md'),
            epochs=30,warmup_epochs=20,quantized_epochs=10,batch=64,lr=.001,weight_decay=.0001,torch=torch.__version__,
            device='mps',fit_count=len(x),dev_counts={k:len(v[0]) for k,v in dev.items()},official_test_evaluated=False))
        m=Encoder().to('mps');opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(SEED)
        tx=torch.from_numpy(x).to('mps').permute(0,3,1,2).float()/255;ty=torch.from_numpy(y).to('mps')
        warm=None;final=None
        with (out/'EPOCHS.jsonl').open('x') as log:
            for epoch in range(30):
                m.train();order=rng.permutation(len(x));losses=[];good=0
                for i in range(0,len(x),64):
                    ids=torch.from_numpy(order[i:i+64]).to('mps');batch=tx[ids]
                    if rng.random()<.5:batch=batch.flip(3)
                    brightness=float(rng.uniform(.85,1.15));contrast=float(rng.uniform(.85,1.15))
                    batch=((batch-.5)*contrast+.5).mul(brightness).clamp(0,1)
                    opt.zero_grad(set_to_none=True);z=m(batch,epoch>=20);loss=F.cross_entropy(z,ty[ids]);loss.backward();opt.step()
                    with torch.no_grad():
                        for b in m.blocks:b.bn.weight.clamp_(.01,10)
                    losses.append(float(loss.detach().cpu()));good+=int((z.argmax(1)==ty[ids]).sum().cpu())
                row=dict(epoch=epoch+1,phase='quantized' if epoch>=20 else 'continuous',train_loss=float(np.mean(losses)),train_accuracy=good/len(x),elapsed_s=time.monotonic()-t0)
                if epoch in [19,29]:
                    results={k:evaluate(m,*v,'mps',epoch>=20) for k,v in dev.items()};row['evaluation']=results
                    if epoch==19:warm=results;torch.save(m.state_dict(),out/'warmup.pt')
                    else:final=results
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps({k:v for k,v in row.items() if k!='evaluation'}),flush=True)
        m.eval();torch.save(dict(model=m.cpu().state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state),out/'final.pt')
        exported=m.exported();np.savez(out/'encoder.npz',**exported)
        # Export witness only. Independent integer comparison runs as a separate stage.
        witness=dev['dev'][0][:16];got=m(torch.from_numpy(witness).permute(0,3,1,2).float()/255,True,True).detach().numpy().astype('u1')
        np.savez(out/'witness.npz',images=witness,torch_features=got)
        gates=dict(dev_accuracy=final['dev']['accuracy']>=.75,reserved_dev_accuracy=final['reserved_dev']['accuracy']>=.60,
            dev_quant_drop=warm['dev']['accuracy']-final['dev']['accuracy']<=.05,
            reserved_quant_drop=warm['reserved_dev']['accuracy']-final['reserved_dev']['accuracy']<=.05)
        result=dict(status='ENCODER_USEFULNESS_PASS' if all(gates.values()) else 'ENCODER_USEFULNESS_FAILED_STOP',gates=gates,warmup=warm,quantized=final,
            elapsed_s=time.monotonic()-t0,official_test_evaluated=False,native_physical_sessions=0,
            scope='Own offline encoder/category-readout development; not native plastic recognition or full official benchmark',
            hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()})
        save(out/'RESULT.json',result);owner['status']=result['status'];print(json.dumps({k:v for k,v in result.items() if k not in ['hashes','warmup','quantized']}),flush=True)
    except BaseException as e:
        owner.update(status='FAILED_PRESERVED',error=repr(e));save(out/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
    finally:
        signal.alarm(0);owner['elapsed_s']=time.monotonic()-t0;save(out/'OWNER.json',owner)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('study',type=Path);a=p.parse_args();fit(a.data,a.study)
