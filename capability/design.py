"""Frozen native primitive workload; no SDK or physical-device access."""
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
S=P/'studies/20261024-native-primitives'
DENSITIES=[0,8,64,256,512]
CELLS=[('fnp',128),('cnp',128),('fnp',256),('cnp',256),('fnp',512),('cnp',512),('cnp',1024),('cnp',2048)]


def weights():
    rng=np.random.default_rng(104101);w=np.zeros((512,2048),np.uint8)
    for j in range(2048):w[rng.choice(512,32,replace=False),j]=1
    return w


def inputs():
    rng=np.random.default_rng(104102);x=[];rows=[]
    for block in range(32):
        for density in rng.permutation(DENSITIES):
            a=np.zeros(512,np.uint8)
            if density:a[rng.choice(512,int(density),replace=False)]=1
            rows.append(dict(index=len(x),block=block,density=int(density)));x.append(a)
    return np.asarray(x),rows


def expected(x,w):
    a=x.astype(np.int32)@w.astype(np.int32)>15;u=w.shape[1]
    return a.reshape(len(x),2,u//2).sum(2).astype('<i4')


def smoke_indices(rows):return [i for i,r in enumerate(rows) if r['block']<8]
