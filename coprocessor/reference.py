"""Independent numerical specification, without SDK/device access."""
import numpy as np
from pong.relearn.model import Reference

def conv(x,w,threshold):
    n,h,width,c=x.shape;oh,ow=(h+1)//2,(width+1)//2
    ph,pw=max(0,(oh-1)*2+3-h),max(0,(ow-1)*2+3-width)
    z=np.pad(x.astype(np.int32),((0,0),(ph//2,ph-ph//2),(pw//2,pw-pw//2),(0,0)))
    y=np.zeros((n,oh,ow,w.shape[-1]),np.int32)
    for ky in range(3):
        for kx in range(3):y+=z[:,ky:ky+2*oh:2,kx:kx+2*ow:2]@w[ky,kx].astype(np.int32)
    return (y>threshold).astype(np.uint8)

def features(z):
    a=conv(z['inputs'],z['w0'],z['t0'])
    b=conv(a,z['w1'][::-1,::-1],z['t1'])
    return b.transpose(0,3,2,1).reshape(len(a),512).copy()

def trajectory(initial,x,learn):
    ref=Reference(initial,n=512,u=128,nw=32,classes=2)
    scores=[];states=[]
    for i,v in enumerate(x):
        if learn:y=ref.forward(v,i%2)
        else:
            bits=sum(1<<int(k) for k in np.flatnonzero(v));y=[(bits&w).bit_count() for w in ref.weights]
        scores.append(y);states.append(ref.ram[:])
    return np.asarray(scores,'<i4'),np.asarray(states,'<u4')
