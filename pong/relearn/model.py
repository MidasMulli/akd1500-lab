"""Stateful exact-rule CPU screen and a separate task-level online CPU baseline."""
from collections import deque
import math
import numpy as np


class Reference:
    def __init__(self,ram,n=1024,u=160,nw=64,classes=5):
        self.ram=list(map(int,ram));self.n=n;self.u=u;self.nw=nw;self.classes=classes
        self.words=n//32;self.stride=4+self.words;self.rot=self.cursor=0
        self.weights=[sum(self.ram[2+j*self.stride+4+k]<<(32*k) for k in range(self.words)) for j in range(u)]
    def forward(self,values,label):
        x=sum(1<<int(k) for k in np.flatnonzero(values));scores=[(w&x).bit_count() for w in self.weights]
        mix=lambda p:((p>>2)&7)|((p&3)<<3)
        self.rot=(self.rot+sum(map(mix,scores)))&31
        width=self.u//self.classes
        eligible=[j for j in range(label*width,(label+1)*width) if scores[j]>(self.ram[2+j*self.stride+3]>>16)]
        if not eligible:return scores
        win=max(eligible,key=lambda j:(scores[j],-j));base=2+win*self.stride;before=self.weights[win]
        remove=before&~x;insert=x&~before;count=min(self.ram[base+3]&65535,remove.bit_count(),insert.bit_count())
        if not count:return scores
        nr=ni=0;word=self.cursor;burst=scan=0;w=before
        while nr<count or ni<count:
            for mode in [0,1]:
                pool=remove if mode==0 else insert
                candidates=(pool>>(32*word))&0xffffffff
                if candidates and (nr if mode==0 else ni)<count:
                    bit=next(b for b in [(self.rot+k)&31 for k in range(32)] if candidates>>b&1)
                    address=word*32+bit
                    if mode==0:w&=~(1<<address);remove&=~(1<<address);nr+=1
                    else:w|=1<<address;insert&=~(1<<address);ni+=1
            self.rot=(self.rot+mix(scores[-1]))&31;scan+=1
            if nr==count and ni==count:break
            word=(word+1)%self.words;burst=0 if word==0 else (burst+1)%4
            assert scan<=self.words*(count+1)
        self.cursor=(word+1)%self.words;self.rot=(self.rot+min(3-burst,self.words-1-word)*mix(scores[-1]))&31
        initial=math.floor(self.nw*.5+.5)
        swap=max(math.ceil(self.nw*.25),(self.ram[base+3]&65535)-math.ceil(initial*.125))
        threshold=min(scores[win],math.floor(self.nw*.75+.5))
        self.ram[base+3]=(threshold<<16)|swap;self.weights[win]=w
        for k in range(self.words):self.ram[base+4+k]=(w>>(32*k))&0xffffffff
        return scores


class RecentNearest:
    """64 labeled observations per visible court; fixed newest tie resolution."""
    def __init__(self):self.memory={c:deque(maxlen=64) for c in [0,1]}
    def forward(self,x,label,court):
        bits=sum(1<<int(k) for k in np.flatnonzero(x));bank=self.memory[court]
        prediction=min(reversed(bank),key=lambda r:(bits^r[0]).bit_count())[1] if bank else 0
        if label!=4:bank.append((bits,int(label)))
        return prediction
