"""MNIST identities, sealed split and exact binary nearest-exemplar references."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import struct
import urllib.request
import numpy as np

P=Path(__file__).resolve().parents[1]
S=P/'studies/20261023-image-compaction'
URL='https://storage.googleapis.com/cvdf-datasets/mnist/'
FILES={
 'train-images-idx3-ubyte.gz':'f68b3c2dcbeaaa9fbdd348bbdeb94873',
 'train-labels-idx1-ubyte.gz':'d53e105ee54ea40749a09fcbcd1e9432',
 't10k-images-idx3-ubyte.gz':'9fb629c4189551a2d022fa330f9573f3',
 't10k-labels-idx1-ubyte.gz':'ec29112dd5afa0611ce80d1b7f02629c'}


def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def dataset(split):
    values=[];sources=[]
    for kind in ['images-idx3','labels-idx1']:
        name=f'{split}-{kind}-ubyte.gz';p=S/'sources'/name
        if not p.exists():
            partial=p.with_suffix('.partial')
            with urllib.request.urlopen(URL+name,timeout=40) as response:partial.write_bytes(response.read())
            assert hashlib.md5(partial.read_bytes()).hexdigest()==FILES[name], 'dataset identity'
            partial.rename(p)
        b=p.read_bytes();assert hashlib.md5(b).hexdigest()==FILES[name]
        raw=gzip.decompress(b)
        if kind.startswith('images'):
            magic,count,h,w=struct.unpack('>IIII',raw[:16]);assert magic==2051 and h==w==28
            a=np.frombuffer(raw[16:],np.uint8).reshape(count,28,28)
        else:
            magic,count=struct.unpack('>II',raw[:8]);assert magic==2049
            a=np.frombuffer(raw[8:],np.uint8);assert len(a)==count
        values.append(a);sources.append(dict(url=URL+name,md5=FILES[name],sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
    return *values,sources


def select(labels,start,end,per_class,seed):
    rng=np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.flatnonzero(labels[start:end]==c)+start,per_class,replace=False) for c in range(10)])


def encode(images):
    a=np.zeros((len(images),32,32),np.uint8);a[:,2:30,2:30]=images>=128
    return a.reshape(-1,1024)


def compact(x):return np.asarray(x).reshape(-1,16,2,16,2).max(axis=(2,4)).reshape(-1,256)


def overlap(x,templates):return x.astype(np.int32)@templates.astype(np.int32).T


def decode(scores,templates):return (templates.sum(axis=1,dtype=np.int32)-2*scores).argmin(axis=1)//48


def evaluate(x,labels,templates):
    scores=overlap(x,templates);pred=decode(scores,templates)
    return dict(correct=int((pred==labels).sum()),count=len(labels),predictions=pred.tolist(),
        per_digit=[dict(digit=c,correct=int(((pred==labels)&(labels==c)).sum()),count=int((labels==c).sum())) for c in range(10)])


def prepare():
    out=S/'fixtures-data01';out.mkdir(exist_ok=False)
    images,labels,sources=dataset('train')
    exemplar_ids=select(labels,0,10000,48,98601);dev_ids=select(labels,10000,60000,20,98602)
    t=encode(images[exemplar_ids]);x=encode(images[dev_ids]);y=labels[dev_ids]
    raw=evaluate(x,y,t);pooled=evaluate(compact(x),y,compact(t))
    checks=dict(raw_accuracy=raw['correct']>=160,compact_accuracy=pooled['correct']>=160,
                compact_noninferior=pooled['correct']>=raw['correct']-6)
    report=dict(status='DEVELOPMENT_PASS' if all(checks.values()) else 'STOP_DEVELOPMENT_GATE',
        sources=sources,exemplar_train_indices=exemplar_ids.tolist(),development_train_indices=dev_ids.tolist(),
        development=dict(raw=raw,compacted=pooled,checks=checks),sealed_test_opened=False)
    np.savez_compressed(out/'data.npz',templates=t,development_inputs=x,development_labels=y,exemplar_labels=labels[exemplar_ids])
    save(out/'DATA.json',report);print(json.dumps(dict(status=report['status'],development=report['development'])),flush=True)
    if not all(checks.values()):return
    images,labels,sources=dataset('t10k');ids=select(labels,0,10000,100,98603)
    x=encode(images[ids]);y=labels[ids]
    raw=evaluate(x,y,t);pooled=evaluate(compact(x),y,compact(t))
    checks=dict(raw_accuracy=raw['correct']>=850,compact_accuracy=pooled['correct']>=850,
                compact_noninferior=pooled['correct']>=raw['correct']-20)
    report.update(status='SEALED_QUALITY_PASS' if all(checks.values()) else 'STOP_SEALED_QUALITY_GATE',
        sources=report['sources']+sources,sealed_test_opened=True,test_indices=ids.tolist(),
        sealed=dict(raw=raw,compacted=pooled,checks=checks))
    np.savez_compressed(out/'data.npz',templates=t,development_inputs=encode(dataset('train')[0][dev_ids]),
        development_labels=dataset('train')[1][dev_ids],exemplar_labels=np.repeat(np.arange(10,dtype=np.uint8),48),
        test_inputs=x,test_labels=y)
    save(out/'DATA.json',report);print(json.dumps(dict(status=report['status'],sealed=report['sealed'])),flush=True)


if __name__=='__main__':prepare()
