"""Independent integer execution of exported fixed spatial layers."""
import numpy as np

def conv(x,w,threshold):
    # Portable arithmetic extracted unchanged in meaning from
    # coprocessor.prepare.conv; no vendor SDK dependency on the training Mac.
    n,h,width,c=x.shape;oh,ow=(h+1)//2,(width+1)//2
    ph,pw=max(0,(oh-1)*2+3-h),max(0,(ow-1)*2+3-width)
    z=np.pad(x.astype('i4'),((0,0),(ph//2,ph-ph//2),(pw//2,pw-pw//2),(0,0)))
    y=np.zeros((n,oh,ow,w.shape[-1]),'i4')
    for ky in range(3):
        for kx in range(3):y+=z[:,ky:ky+oh*2:2,kx:kx+ow*2:2]@w[ky,kx].astype('i4')
    return y if threshold is None else (y>threshold).astype('u1')

def pool(z):
    n,h,w,c=z.shape
    return z.reshape(n,h//2,2,w//2,2,c).max((2,4))

def forward(images,weights,return_layers=False):
    a=conv(images,weights['w0'],weights['t0'])
    b=(pool(conv(a,weights['w1'][::-1,::-1],None))>weights['t1']).astype('u1')
    c=(b.astype('i4')@weights['w2'][0,0].astype('i4')>weights['t2']).astype('u1')
    f=c.transpose(0,3,2,1).reshape(len(images),512).copy()
    return (a,b,c,f) if return_layers else f
