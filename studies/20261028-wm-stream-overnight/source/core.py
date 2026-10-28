"""Magnitude-addressed native-rule memory. Geometry is explicit and bounded."""
from functools import lru_cache
import numpy as np
from pong.relearn.model import Reference

class Layout:
    def __init__(self,name):
        assert name in ['quad13','oct9'];self.name=name
        self.d,self.bins,self.P,self.n0,self.u0,self.W=(13,4,128,1024,256,384)if name=='quad13'else(9,8,176,1024,256,448)
        assert name=='oct9';self.W=512;self.n1=4*self.P;self.u1=32;self.output1=48;self.shadow=self.P
        self.error=list(range(8));self.correct=list(range(self.u0//2,self.u0//2+8));self.plastic=self.error+self.correct;self.helpers=[j for j in range(self.u0)if j not in self.plastic]
        self.program_start=self.d*self.bins*2;self.const_start=self.program_start+12;self.constant_count=31-2*self.d-2-1;self.canon=self.const_start+self.constant_count;self.swap=self.canon+1;self.value=self.swap+1
        self.variable=list(range(self.const_start))+[self.canon,self.swap]+list(range(self.value,self.value+3));self.S=len(self.variable);self.positions={j:i for i,j in enumerate(self.variable)}
        self.common=list(range(self.S,self.S+16));start=self.S+16
        self.cand=[list(range(start+8*i,start+8*i+8))for i in range(3)];start+=24
        self.truth=[list(range(start+8*i,start+8*i+8))for i in range(3)];self.phase=list(range(self.S+64,self.S+70));self.carry,self.cut,self.op=range(self.S+70,self.S+73)
        self.retire,self.error_probe,self.correct_probe=range(self.S+73,self.S+76);self.header=240;self.returns=[248]
        assert self.value+3<=self.P and self.correct_probe<self.header and self.P+256<=self.W
        self.context_targets=([[j]for j in self.common+sum(self.cand+self.truth,[])])if name=='quad13'else[self.common]+self.cand+self.truth
        self.context_inputs=[g[0]for g in self.context_targets];self.context_units=len(self.context_inputs)
        assert len(self.helpers)>=self.P+9+self.context_units
    def key(self,levels,program):
        levels=np.asarray(levels);assert levels.shape==(self.d,)and np.all((levels>=0)&(levels<self.bins))and 0<=program<6
        x=np.zeros(self.P,'u1')
        for i,bin in enumerate(levels):x[2*(i*self.bins+int(bin)):2*(i*self.bins+int(bin))+2]=1
        x[self.program_start+2*program:self.program_start+2*program+2]=1;x[self.const_start:self.canon]=1;x[self.canon]=1;assert x.sum()==31
        return x
    def value_key(self,key,value):
        x=key.copy();x[self.value:self.value+3]=0;x[self.value+value]=1;assert x.sum()==32;return x
    def encode(self,payload=None,candidate=0,truth=None,phase=0,write=False,cut=False):
        x=np.zeros(256,'<u4')
        if payload is not None:x[:self.S]=payload[self.variable]
        x[self.common]=1;x[self.cand[candidate]]=1
        if truth is not None:x[self.truth[truth]]=1
        if phase is not None:x[self.phase[phase]]=1
        x[self.op]=write;x[self.cut]=cut;return x.view('u1')
    def parameters(self,seed=27102801):
        rng=np.random.default_rng(seed);w=np.zeros((self.n0,self.u0),'i1');t=np.zeros(self.u0,'i4');lt=np.full(self.u0,32,'i4');private=sum(self.cand+self.truth,[])
        for j in self.plastic:w[4*np.array(self.common),j]=1;w[4*rng.choice(private,16,False),j]=1;t[j]=31;lt[j]=0
        dead=np.array([j for j in range(256)if j%4])[:32]
        for j in self.helpers:w[dead,j]=1
        channels=([[4*self.positions[j]]if j in self.positions else[4*self.common[0]]if self.const_start<=j<self.canon else[] for j in range(self.P)]+[[4*j]for j in self.phase]+[[4*self.carry]+[4*j for j in self.returns],[4*self.cut],[4*self.op]]+[[4*j]for j in self.context_inputs])
        for j,indices in zip(self.helpers,channels):w[:,j]=0;w[dead[:32-len(indices)],j]=1;w[indices,j]=1
        b=np.zeros((self.n1,self.u1),'i1')
        for j in range(self.u1):b[4*rng.choice(self.P,32,False),j]=1
        assert np.all(w.sum(0)==32)and np.all(b.sum(0)==32);assert not w[4*self.header:4*(self.header+8)].any()
        assert not w[[4*self.error_probe,4*self.correct_probe]].any()
        return [w,b],[t,np.full(self.u1,31,'i4')],[lt,np.zeros(self.u1,'i4')]
    @lru_cache(maxsize=1)
    def matrices(self):
        P=self.P;g=((P+11+15)//16)*16;h=g+8;w=np.zeros((self.u0,self.u0),'i1');t=np.zeros(self.u0,'i4')
        for k,j in enumerate(self.helpers[:P]):w[j,k]=1
        for k,j in enumerate(self.helpers[P:P+6]):w[j,P+k]=1
        carry,cut,op=self.helpers[P+6:P+9];w[carry,P+6]=1;w[self.error,P+7]=1;w[self.correct,P+8]=1;w[cut,P+9]=1;w[op,P+10]=1
        w[[self.helpers[P],self.helpers[P+5]],g]=1
        w[self.error+[op,self.helpers[P+1],self.helpers[P+3],self.helpers[P+4]],g+1]=1;t[g+1]=2
        w[self.error+[carry,self.helpers[P+2]],g+2]=1;w[[op,cut],g+2]=-1;t[g+2]=2
        w[self.error+[op,self.helpers[P+1],self.helpers[P+4]],g+3]=1;t[g+3]=2
        w[self.error+[op,self.helpers[P+3]],g+4]=1;t[g+4]=2
        for k,j in enumerate(self.helpers[P+9:P+9+self.context_units]):w[j,h+k]=1
        Q=P+self.S;mid=((Q+12+self.context_units+31)//32)*32;f=np.zeros((self.u0,mid),'i1');tf=np.zeros(mid,'i4')
        for j in range(P):
            if j in[self.canon,self.swap]:continue
            f[j,j]=1;f[[g,g+1,g+2],j]=1;tf[j]=1
        f[self.canon,self.canon]=1;f[[g,g+3],self.canon]=1;tf[self.canon]=1;f[[g+2,g+4],self.swap]=1
        for i,j in enumerate(self.variable):f[j,P+i]=1
        for k in range(6):f[P+k,Q+(k+1)%6]=1
        for dst,src in [(6,P+6),(7,P+9),(8,P+10),(9,P+7),(10,P+8),(11,g+2)]:f[src,Q+dst]=1
        for k in range(self.context_units):f[h+k,Q+12+k]=1
        z=np.zeros((mid,self.W),'i1')
        for j in range(P+self.S):z[j,j]=1
        for k in range(6):z[Q+k,P+self.phase[k]]=1
        for src,dst in [(6,self.carry),(7,self.cut),(8,self.op),(9,self.error_probe),(10,self.correct_probe),(11,self.retire)]:z[Q+src,P+dst]=1
        for k,destinations in enumerate(self.context_targets):
            for dst in destinations:z[Q+12+k,P+dst]=1
        return[(w,t),(f,tf),(z,np.zeros(self.W,'i4'))]

    def content_matrices(self):
        w=np.zeros((self.u1,self.output1),'i1');w[:,0]=1
        for j in range(self.u1):w[j,j+1]=1
        return[(w,np.zeros(self.output1,'i4')),(np.eye(self.output1,dtype='i1'),np.zeros(self.output1,'i4'))]

def initial(weights,lts):
    n,u=weights.shape;stride=4+n//32;r=np.zeros(2+stride*u,'<u4')
    for j in range(u):
        b=2+j*stride;r[b+3]=(int(lts[j])<<16)|16
        for a in np.flatnonzero(weights[:,j]):r[b+4+int(a)//32]|=np.uint32(1<<int(a%32))
    return r

class Content(Reference):
    def step(self,x,mode):
        if mode==0:
            b=sum(1<<int(i)for i in np.flatnonzero(x));return np.array([(v&b).bit_count()for v in self.weights],'i4')
        old=self.weights.copy();p=np.asarray(super().forward(x,0),'i4')
        for j,(a,b)in enumerate(zip(old,self.weights)):
            if a!=b:q=2+j*self.stride;self.ram[q+3]=(min(int(p[j]),1 if mode==2 else 30)<<16)|(self.ram[q+3]&65535)
        return p

class Circuit:
    def __init__(self,layout,rams=None):
        self.l=layout;ws,self.ts,lts=layout.parameters()
        if rams is None:rams=[initial(w,t)for w,t in zip(ws,lts)]
        self.source=Reference(rams[0],layout.n0,layout.u0,32,2);self.memory=Content(rams[1],layout.n1,layout.u1,32,1)
    def source_step(self,x,teacher=-1):
        if teacher<0:
            bits=sum(1<<int(i)for i in np.flatnonzero(x));p=np.array([(v&bits).bit_count()for v in self.source.weights],'i4')
        else:p=np.asarray(self.source.forward(x,teacher),'i4')
        a=(p>self.ts[0]).astype('i4')
        for w,t in self.l.matrices():a=a@w.astype('i4');a=(a>t).astype('i4')if len(t)!=self.l.W else a
        assert a.max()<=1;return a.astype('<u4')
    def bootstrap(self):
        for _ in range(4):
            for a in range(3):
                for b in range(3):self.source_step(self.l.encode(candidate=a,truth=b,phase=None),int(a==b))
        for a in range(3):
            for b in range(3):
                x=self.l.encode(candidate=a,truth=b,phase=None);bits=sum(1<<int(i)for i in np.flatnonzero(x));act=np.array([(v&bits).bit_count()for v in self.source.weights])>31
                assert int(act[self.l.error].sum())==int(a!=b)and int(act[self.l.correct].sum())==int(a==b)
    def protocol(self,payload,candidate,truth,write=False,cut=False,freeze=False):
        x=self.l.encode(payload,candidate,truth,0,write,cut)
        for mode in[0,1,2,1,1,0]:
            y=self.source_step(x);p=self.memory.step(y[:self.l.P].view('u1'),0 if freeze else mode);ret=self.readout(p);y[self.l.P+248:self.l.P+248+self.l.output1]=ret;x=y.view('u1')[4*self.l.P:4*self.l.P+self.l.n0].copy()
    def readout(self,p):
        a=(p>31).astype('i4')
        for w,t in self.l.content_matrices():a=(a@w>t).astype('i4')
        assert a[0]==int((p>31).any()) and np.array_equal(a[1:33],p>31)
        return a.astype('<u4')
    def query(self,k):
        counts=[]
        for c in range(3):
            y=self.source_step(self.l.encode(self.l.value_key(k,c),candidate=c,truth=None));p=self.memory.step(y[:self.l.P].view('u1'),0);counts.append(int((p>31).sum()))
        return counts.index(1)if counts.count(1)==1 and sum(counts)==1 else None
    def feedback(self,k,wm,truth,cut=False,freeze=False,admit_all=False):
        for c in range(3):self.protocol(self.l.value_key(k,c),c,truth,False,cut,freeze)
        self.protocol(self.l.value_key(k,truth),0 if admit_all else wm,1 if admit_all else truth,True,cut,freeze)
