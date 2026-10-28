"""Integer four-bit spatial pipeline, independent of training autograd."""
import numpy as np
from visual_memory.encoder import conv,pool

def forward(images,w,return_layers=False):
    a=images;layers=[]
    for i in range(6):
        if i in [0,1,2]:z=conv(a,w[f'w{i}'][::-1,::-1] if i else w['w0'],None)
        else:z=a.astype('i4')@w[f'w{i}'][0,0].astype('i4')
        if i in [1,2,3,4]:z=pool(z)
        if i==5:a=(z>w[f't{i}']).astype('u1')
        else:a=np.clip((np.maximum(z-w[f't{i}'],0)+w[f's{i}']-1)//w[f's{i}'],0,15).astype('u1')
        layers.append(a)
    f=a.reshape(len(a),512)
    return (*layers,f) if return_layers else f
