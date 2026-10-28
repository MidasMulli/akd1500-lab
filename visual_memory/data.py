"""Fixed CORe50 extraction and disjoint manifests; no external pickle execution."""
from pathlib import Path
import argparse, hashlib, io, json, re, time, zipfile
import numpy as np
from PIL import Image

RESERVED = list(range(1, 47, 5))
FIT_SESSIONS = [1, 2, 4, 5, 6]
OFFICIAL_TEST = [3, 7, 10]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,v):
    Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def masks(d):
    s,o=d['session'],d['object']; held=np.isin(o,RESERVED)
    return dict(fit=~held & np.isin(s,FIT_SESSIONS), dev=~held & (s==8),
                reserved_dev=held & (s==9), sealed=np.isin(s,OFFICIAL_TEST),
                reserved=held)

def extract(archive,out):
    out.mkdir(parents=True,exist_ok=False); started=time.time()
    rx=re.compile(r'(?:^|/)C_(\d+)_(\d+)_(\d+)\.png$')
    entries=[]
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():
            m=rx.search(n)
            if m:
                s,o,f=map(int,m.groups())
                if f in range(10,201,10):entries.append((s,o,f,n))
        entries.sort()
        assert len(entries)==11000,(len(entries),'expected fixed 20 frames x50objects x11sessions')
        x=[]
        for i,(s,o,f,n) in enumerate(entries):
            # read() checks each selected ZIP entry CRC; exact box average.
            im=np.asarray(Image.open(io.BytesIO(z.read(n))).convert('RGB'))
            assert im.shape==(128,128,3)
            im=(im.astype(np.uint16).reshape(64,2,64,2,3).sum((1,3))//4).astype('u1')
            x.append(im)
            if i%1000==0:print(json.dumps(dict(extracted=i)),flush=True)
    s=np.array([v[0] for v in entries],np.uint8);o=np.array([v[1] for v in entries],np.uint8)
    d=dict(images=np.stack(x),session=s,object=o,frame=np.array([v[2] for v in entries],np.uint16),labels=(o-1)//5)
    mm=masks(d);assert sum(mm['fit'])==4000 and sum(mm['dev'])==800 and sum(mm['reserved_dev'])==200
    assert not (mm['fit']&mm['dev']).any() and not (mm['fit']&mm['reserved_dev']).any()
    # No official test image in the fitting/development subsets.
    assert not ((mm['fit']|mm['dev']|mm['reserved_dev'])&mm['sealed']).any()
    np.savez(out/'core50_selected.npz',**d)
    save(out/'DATA.json',dict(status='DATA_SPLIT_PASS',source_url='http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip',
        source_page='https://vlomonaco.github.io/core50/',license='CC BY 4.0',archive_bytes=archive.stat().st_size,
        archive_sha256=sha(archive),selected_npz_sha256=sha(out/'core50_selected.npz'),
        images=len(x),counts={k:int(v.sum()) for k,v in mm.items()},reserved_objects=RESERVED,
        fit_sessions=FIT_SESSIONS,encoder_dev_session=8,reserved_dev_session=9,
        sealed_sessions=OFFICIAL_TEST,selected_frames=list(range(10,201,10)),
        input='Recorded real-camera images, downsampled by exact2x2box averaging; no live camera',
        elapsed_s=time.time()-started,paths=[v[3] for v in entries]))
    print(json.dumps(dict(status='DATA_SPLIT_PASS',images=len(x))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('output',type=Path);a=p.parse_args();extract(a.archive,a.output)
