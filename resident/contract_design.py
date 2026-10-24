"""Frozen basis/sealed probes for the native FNP/CNP vector contract."""
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
S=P/'studies/20261023-handoff-contract'
KINDS=['source','all_fnp','direct_identity','mixed_identity','direct_compare','mixed_compare']


def data(split='development'):
    assert split in ['development','sealed']
    rng=np.random.default_rng(100101 if split=='development' else 100102)
    x=np.zeros((128,64),np.uint8)
    if split=='development':
        x[:64]=np.eye(64,dtype=np.uint8);x[65]=1;x[66,::2]=1;x[67,1::2]=1
        for i in range(68,128):x[i]=rng.random(64)<float(rng.choice([.125,.25,.5,.75,.875]))
    else:
        for block in range(8):
            levels=rng.permutation([2,6,10,14])
            for j in range(4):
                for group,count in enumerate(np.roll(levels,j)):x[block*4+j,group*16+rng.choice(16,int(count),replace=False)]=1
        for i in range(32,128):x[i]=rng.random(64)<float(rng.choice([.125,.25,.5,.75,.875]))
    return x


def comparison():
    pairs=[(i,j) for i in range(4) for j in range(4) if i!=j]
    w=np.zeros((64,32),np.int8);t=np.zeros(32,np.int32)
    for q,(i,j) in enumerate(pairs):w[i*16:(i+1)*16,q]=1;w[j*16:(j+1)*16,q]=-1;t[q]=-1 if i<j else 0
    return w,t


def expected(kind,x):
    if 'compare' not in kind:return x.astype(np.int32)
    w,t=comparison();return (x.astype(np.int32)@w>t).astype(np.int32)
