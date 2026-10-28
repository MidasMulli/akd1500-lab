"""Sealed geometry-witness or world-model replay; native windows stay on card."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
from core import Layout,Circuit
from encoder import Encoder,B

class Recorder:
    def __init__(self,prepared,out,profile):
        out.mkdir(exist_ok=False,parents=True);self.out=out;self.g=Layout(profile)
        for p in prepared.iterdir():
            if p.is_file():shutil.copy2(p,out/p.name)
        self.c=Circuit(self.g,[np.fromfile(out/f'bank{j}.ram','<u4')for j in range(2)]);self.rows=[];self.outputs=[];self.inputs=[];self.states=[[],[]];self.mail=None;self.episode=-1;self.role='bootstrap';self.answers=[]
    def add(self,net,x,mode=-1,label=0,external=False,candidate=-1,phase=-1):
        g=self.g;x=np.asarray(x,'u1');assert len(x)==(1024 if net==0 else g.n1)
        if not external:
            start=0 if net else 4*g.P;assert self.mail is not None and np.array_equal(x,self.mail.view('u1')[start:start+len(x)])
        y=self.c.source_step(x,label if mode==0 else -1)if net==0 else self.c.readout(self.c.memory.step(x,mode));yy=np.zeros(g.W,'<u4');yy[:len(y)]=y;self.outputs.append(yy);xx=np.zeros(1024,'u1');xx[:len(x)]=x;self.inputs.append(xx)
        self.rows.append(dict(net=net,mode=mode,label=label,external=int(external),episode=self.episode,role=self.role,candidate=candidate,phase=phase))
        for j,b in enumerate([self.c.source,self.c.memory]):self.states[j].append(np.asarray(b.ram,'<u4'))
        if net==0:self.mail=y.copy()
        else:self.mail[g.P+248:g.P+248+g.output1]=y
        return y
    def bootstrap(self):
        for _ in range(4):
            for a in range(3):
                for b in range(3):self.add(0,self.g.encode(candidate=a,truth=b,phase=None),0,int(a==b),True)
    def query(self,k):
        self.role='query';counts=[]
        for c in range(3):
            y=self.add(0,self.g.encode(self.g.value_key(k,c),candidate=c,truth=None),external=True,candidate=c);v=self.add(1,y[:self.g.P].view('u1'),0,candidate=c);counts.append(int(v[1:33].sum()))
        return counts.index(1)if counts.count(1)==1 and sum(counts)==1 else -1
    def feedback(self,k,wm,truth,all_admit=False,freeze=False,cut=False):
        g=self.g;self.role='feedback'
        for candidate in range(4):
            write=candidate==3;payload=g.value_key(k,truth if write else candidate);compare=0 if write and all_admit else wm if write else candidate;actual=1 if write and all_admit else truth;x=g.encode(payload,compare,actual,write=write,cut=cut)
            for phase,mode in enumerate([0,1,2,1,1,0]):
                y=self.add(0,x,external=phase==0,candidate=candidate,phase=phase);self.add(1,y[:g.P].view('u1'),0 if freeze else mode,candidate=candidate,phase=phase);x=self.mail.view('u1')[4*g.P:4*g.P+1024].copy()
    def save(self,metadata):
        o=self.out;np.array(self.inputs,'u1').tofile(o/'inputs.bin')
        for name in ['net','mode','label','external']:np.array([r[name]for r in self.rows],'<i4').tofile(o/f'{name}s.bin')
        np.savez_compressed(o/'TRACE.npz',outputs=np.array(self.outputs,'<u4'),states0=np.array(self.states[0],'<u4'),states1=np.array(self.states[1],'<u4'));(o/'ROWS.json').write_text(json.dumps(self.rows)+'\n');(o/'ANSWERS.json').write_text(json.dumps(self.answers,indent=2)+'\n');r=dict(status='SEALED_CPU_FIXTURE',forwards=len(self.rows),profile=self.g.name,physical=False,**metadata);(o/'FIXTURE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

def witness(prepared,out,profile):
    r=Recorder(prepared,out,profile);r.bootstrap();g=r.g;keys=[]
    for j in range(32):
        levels=np.zeros(g.d,'i4');levels[0]=j%g.bins;levels[1]=j//g.bins;k=g.key(levels,0);keys.append(k);r.episode=j;r.feedback(k,j%3,j%3,True)
    before=[list(r.c.source.ram),list(r.c.memory.ram)]
    for _ in range(2):
        for j,k in enumerate(keys):assert r.query(k)==j%3
    assert before==[list(r.c.source.ram),list(r.c.memory.ram)]
    r.feedback(keys[0],1,1,True)
    for j,k in enumerate(keys):assert r.query(k)==(1 if j==0 else j%3)
    r.save(dict(witness=True,all_thirty_two=True,revision_and_protected=True,readonly_queries=64))

def world(prepared,out,profile,tranche,k,arm,prefix):
    r=Recorder(prepared,out,profile);r.bootstrap();enc=Encoder(profile);data=B/('data-dev'if tranche=='DEV'else'data-held');stem=f'STREAM_{tranche}_CHALLENGE_K{k}';rows=[json.loads(l)for l in(data/f'{stem}.jsonl').read_text().splitlines()];z=np.load(data/f'{stem}.npz')
    for i,row in enumerate(rows):
        if prefix is not None and i>=prefix:break
        if not row['query']:continue
        r.episode=i;code,_=enc.encode(z['h0_pool'][i],z['h5_pool'][i],int(row['program_id'][1:]));value=r.query(code);truth={'D':0,'S':1,'C':2}[row['outcome']];wm=int(np.argmax([row['wm_pN_T'],row['wm_pG_T'],row['wm_pC_T']]));phase=int(row['phase_AUDIT_ONLY']);freeze=(arm.endswith('A-frozen')and phase>=1)or(arm.endswith('AB-frozen')and phase>=2)
        r.feedback(code,wm,truth,arm.startswith('all-'),freeze,arm.endswith('retire-cut'));r.answers.append(dict(episode=i,query=value,truth=truth,wm_class=wm,phase_AUDIT_ONLY=phase))
    if prefix is None:
        expected=np.load(B/f'CPU_{tranche}_{profile}_INFORMATIVE_K{k}'/'TRACE.npz');assert np.array_equal([v['query']for v in r.answers],expected[f'found_{arm}'])
    r.save(dict(witness=False,tranche=tranche,K=k,arm=arm,prefix=prefix))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prepared',type=Path);p.add_argument('out',type=Path);p.add_argument('profile',choices=['quad13','oct9']);p.add_argument('--witness',action='store_true');p.add_argument('--tranche',default='DEV');p.add_argument('--K',type=int,default=2);p.add_argument('--arm',default='all-normal');p.add_argument('--prefix',type=int);a=p.parse_args()
    if a.witness:witness(a.prepared,a.out,a.profile)
    else:world(a.prepared,a.out,a.profile,a.tranche,a.K,a.arm,a.prefix)
