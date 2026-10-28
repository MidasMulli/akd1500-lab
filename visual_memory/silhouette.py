"""Direct-pixel image content with a co-resident native admission learner.

Prospective 2026-10-27 study. Source quality and pixel labels are external;
eligibility and connection selection are native. No record-specific readout.
"""
from functools import lru_cache
import numpy as np
from pong.relearn.model import Reference

NAMES=['Key','Cup','Tree','Car']
ART=[
 ['000000000000','011100000000','110110000000','110111111110','110110010100','011100010100','000000000000','000000000000'],
 ['000000000000','011111110000','011111111110','011111110010','011111110010','011111111110','001111100000','000000000000'],
 ['000001000000','000011100000','000111110000','001111111000','011111111100','000011100000','000011100000','000011100000'],
 ['000000000000','000111110000','001111111000','011111111110','111111111111','111111111111','001100001100','000000000000']]
Q,RET,AUX,POL,OPEN,CUT=480,481,482,483,484,485
COMMON=list(range(384,433))+[POL]
KEYS=[list(range(434+14*s,448+14*s)) for s in range(3)]
SENT=476
IMAGE_CELLS=np.array([0,2,4,6],'i4')
POLICY_CELLS=np.array([8,9,10,11],'i4')
OBSERVED_CELLS=np.array([i*96+b for i in range(4) for b in range(0,96,6)],'i4')
MARKERS=[5*12+7,4*12+10]
def patterns():return np.array([[int(v) for row in a for v in row] for a in ART],'u1')
def revised():
    p=patterns().copy()
    for i,j in enumerate(MARKERS):assert p[i,j]==1;p[i,j]=0
    return p
def poisoned(i):
    p=patterns()[i].copy();p[np.arange(i,96,4)]=1-p[np.arange(i,96,4)]
    return p
def encode(record=None,erase=(),source=None,query=False,retire=False,aux=True,opened=False,cut=False,value=None,presence=False):
    z=np.zeros(512,'u1')
    if record is not None:
        z[:288]=np.repeat(record,3);z[288:384]=1-record
        for b in erase:z[3*int(b):3*int(b)+3]=0;z[288+int(b)]=0
    if source is not None:
        z[COMMON]=1;z[KEYS[source]]=1
        if retire and record is None:z[KEYS[source][-1]]=0;z[SENT]=1
    z[Q]=query;z[RET]=retire and record is not None;z[AUX]=aux;z[OPEN]=opened;z[CUT]=cut
    z[486]=value==0;z[487]=value==1;z[488]=presence
    if opened:z[POL]=0
    return z.astype('<u4').view('u1')
def dimensions(k):return (2048,32,64,16) if k==0 else (512,400,32,400)
def plastic_indices(k):return np.r_[IMAGE_CELLS,POLICY_CELLS] if k==0 else np.arange(388)
def weights(k):
    n,u,nw,_=dimensions(k);w=np.zeros((n,u),'i1');r=np.random.default_rng(27102702+k)
    if k==0:
        for j in IMAGE_CELLS:w[4*r.choice(384,64,False),j]=1
        for j in POLICY_CELLS:w[4*np.array(COMMON),j]=1;w[4*r.choice(np.r_[KEYS[0],KEYS[1]],14,False),j]=1
        dead=np.array([b for b in range(n) if b%4][:64])
        for j in set(range(u))-set(plastic_indices(k)):w[dead,j]=1
        for j,bit in [(12,Q),(13,RET),(14,AUX),(15,OPEN),(16,CUT),(17,CUT),(21,486),(22,487),(23,Q),(24,488)]:w[:,j]=0;w[dead[:63],j]=1;w[4*bit,j]=1
        for a in range(3):
            w[:,18+a]=0
            for b in range(a*32,(a+1)*32):w[4*3*b,18+a]=1;w[4*(288+b),18+a]=1
    else:
        for j in range(388):w[r.choice(128,32,False),j]=1
        for j in range(388,400):w[480:512,j]=1
        fields=[(388,448)]+[(389+i,184+i) for i in range(4)]+[(393+i,168+i) for i in range(4)]+[(397,160),(398,161)]
        for j,bit in fields:w[:,j]=0;w[480:511,j]=1;w[bit,j]=1
    assert np.all(w.sum(0)==nw)
    return w
def thresholds(k):
    if k==1:return np.r_[np.full(388,31),np.zeros(12)].astype('i4')
    t=np.full(32,64,'i4');t[IMAGE_CELLS]=0;t[POLICY_CELLS]=63;t[12:25]=0;t[18:21]=31;return t
def steps(k):
    a=np.ones(dimensions(k)[1],'f4')
    if k==0:a[IMAGE_CELLS]=4
    return a
def activate(p,k):return np.clip(np.ceil((np.array(p)-thresholds(k))/steps(k)),0,15 if k==0 else 1).astype('u1')
def learning_thresholds(k):
    n,u,nw,_=dimensions(k);t=np.full(u,nw,'i4');t[plastic_indices(k)]=0;return t
def initial(k):
    n,u,nw,_=dimensions(k);w=weights(k);stride=4+n//32;ram=np.zeros(2+u*stride,'<u4')
    for j in range(u):
        b=2+j*stride;ram[b+3]=(int(learning_thresholds(k)[j])<<16)|(nw//2)
        for bit in np.flatnonzero(w[:,j]):ram[b+4+int(bit)//32]|=np.uint32(1<<int(bit%32))
    return ram
class Memory(Reference):
    def __init__(self,k,ram=None):
        n,u,nw,c=dimensions(k);super().__init__(initial(k) if ram is None else ram,n,u,nw,c)
    def forward(self,x,label=-1):
        if label<0:
            bits=sum(1<<int(i) for i in np.flatnonzero(x));return [(w&bits).bit_count() for w in self.weights]
        return super().forward(x,label)
@lru_cache(maxsize=1)
def matrices():
    w=np.zeros((32,64),'i1');t=np.zeros(64,'i4')
    for i,j in enumerate(IMAGE_CELLS):
        others=[v for v in IMAGE_CELLS if v!=j]
        for a,other in enumerate(others):w[j,3*i+a]=1;w[other,3*i+a]=-1
        w[j,12+i]=1;w[18:21,12+i]=-1;t[12+i]=10
        for b,th in enumerate([3,7,11,14]):w[j,36+4*i+b]=1;t[36+4*i+b]=th
    for i,j in enumerate(POLICY_CELLS):w[j,16+i]=1
    for i,j in enumerate(range(12,25)):w[j,20+i]=1
    layers=[('native_pairwise_cue_competition',w,t,2)]
    w=np.zeros((64,512),'i1');t=np.zeros(512,'i4')
    for c in range(4):
        ss=slice(c*32,(c+1)*32-1);w[c*3:c*3+3,ss]=1;w[12+c,ss]=1;w[[18,19,20,23],ss]=1;w[[16,17,24,25],ss]=-1;t[ss]=4
    for col,feature in [(128,29),(129,30),(130,32)]:
        w[[18,19,23,feature],col]=1;w[[16,17,24,25],col]=-1;t[col]=1
        if col in [129,130]:w[[20,31],col]=1
    w[20,448]=1
    for j,col in [(18,160),(19,160),(16,161),(17,161)]+[(39+4*i,168+i) for i in range(4)]+[(16+i,184+i) for i in range(4)]:w[j,col]=1;w[22,col]=1;t[col]=1
    layers.append(('native_source_gated_pixel_value',w,t,2))
    w=np.zeros((400,272),'i1');t=np.zeros(272,'i4')
    for b in range(96):
        w[384:388,b]=1;t[b]=1;w[384:388,96+b]=1
        for i in range(4):w[i*96+b,b]=1;w[i*96+b,96+b]=-1
    w[388,192]=1;w[384:388,192]=1;t[192]=1
    for i in range(4):w[384+i,193+i]=1;w[389+i,261+i]=1;w[393+i,265+i]=1
    for i,j in enumerate(OBSERVED_CELLS):w[j,197+i]=1
    w[397,269]=w[398,270]=1
    layers.append(('native_pixels_from_learned_value_connections',w,t,2))
    w=np.zeros((272,512),'i1');t=np.zeros(512,'i4')
    for b in range(96):
        w[:193,b]=1;w[96+b,b]=0;t[b]=96;w[:193,96+b]=1;w[b,96+b]=0;t[96+b]=96
    w[:193,400]=1;t[400]=96;w[192,448]=1
    for i in range(4):w[193+i,320+i]=1;w[261+i,272+i]=1;w[265+i,276+i]=1
    for i in range(64):w[197+i,200+i]=1
    for i,b in enumerate(MARKERS):w[:193,401+i]=1;w[b,401+i]=0;w[193+i,401+i]=1;t[401+i]=97
    layers.append(('native_complete_pixel_guard',w,t,1))
    for _,w,t,_ in layers:w.setflags(write=False);t.setflags(write=False)
    return tuple(layers)
def formatter():
    w=np.zeros((512,512),'i1')
    for b in range(96):w[b,b*3:b*3+3]=1;w[96+b,288+b]=1
    for i in range(64):w[200+i,384+i]=1
    for i in range(4):w[272+i,448+i]=1;w[320+i,468+i]=1
    for i in range(4):w[276+i,452+i]=1
    w[400,472]=w[401,473]=w[402,474]=1;w[448,Q]=w[448,AUX]=1
    return w
def fixed(x,e):return (np.asarray(x,'i4')@e[1].astype('i4')>e[2]).astype('u1')
class Circuit:
    def __init__(self,rams=None):self.banks=[Memory(k,None if rams is None else rams[k]) for k in range(2)]
    def step(self,x,mode=-1,label=0):
        p=np.array(self.banks[0].forward(x,label if mode==0 else -1),'i4');a=activate(p,0);code=fixed(fixed(a,matrices()[0]),matrices()[1]);q=np.array(self.banks[1].forward(code,label if mode==1 else -1),'i4');b=activate(q,1);v=fixed(fixed(b,matrices()[2]),matrices()[3]);y=v.astype('i4')@formatter().astype('i4');assert y.max()<=1
        return dict(output=y.astype('u1'),states=[np.asarray(m.ram,'<u4') for m in self.banks],potentials=[p,q],activations=[a,b],coded=code)
