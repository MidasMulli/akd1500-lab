"""Independent Pillow checks; never chooses the next action."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fixtures(root,seed=71011,n=4):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed);paths=[]
    for i in range(n):
        w,h=[(640,360),(960,540),(800,480),(1280,720)][i%4]
        yy,xx=np.mgrid[:h,:w];a=np.stack([(xx*255//max(1,w-1)),(yy*255//max(1,h-1)),((xx//40+yy//40)%2)*200+25],axis=-1).astype(np.uint8)
        im=Image.fromarray(a);d=ImageDraw.Draw(im)
        for k in range(6):
            x,y=int(rng.integers(0,w-100)),int(rng.integers(0,h-65));color=tuple(map(int,rng.integers(25,231,3)))
            d.rectangle((x,y,x+80,y+45),fill=color,outline='white',width=2)
        d.text((18,18),f'Image service / fixture {seed}-{i}',fill='white',stroke_width=1,stroke_fill='black')
        p=root/f'fixture_{seed}_{i}.jpg';im.save(p,quality=93,subsampling=0);paths.append(p)
    return paths

def verify(source,output,stage):
    source,output=Path(source),Path(output);got=np.load(output,allow_pickle=False)
    assert got.dtype==np.uint8 and got.ndim==3 and got.shape[-1]==4
    if stage=='decode':
        with Image.open(source) as im:reference=np.asarray(im.convert('RGBA'))
        bound,p99=5,24
    else:
        inp=np.load(source,allow_pickle=False);h,w,_=inp.shape
        # Affine bilinear sampling has the same pixel-center contract as the workers.
        # Pillow resize() additionally low-pass filters during downsampling.
        reference=np.asarray(Image.fromarray(inp).transform((320,180),Image.Transform.AFFINE,
            (w/320,0,0,0,h/180,0),resample=Image.Resampling.BILINEAR))
        bound,p99=8,48
        with Image.open(output.with_suffix('.png')) as im:assert np.array_equal(np.asarray(im.convert('RGBA')),got),'PNG payload differs'
    assert got.shape==reference.shape and np.all(got[:,:,3]==255)
    delta=np.abs(got[:,:,:3].astype(np.int16)-reference[:,:,:3].astype(np.int16))
    mae=float(delta.mean());quantile=float(np.quantile(delta,.99));assert mae<=bound and quantile<=p99,('fidelity',mae,quantile,bound,p99)
    return dict(status='PASS',stage=stage,shape=list(got.shape),mae=mae,p99_absolute_error=quantile,
                source_sha256=sha(source),output_sha256=sha(output),png_sha256=sha(output.with_suffix('.png')) if stage=='export' else None)
