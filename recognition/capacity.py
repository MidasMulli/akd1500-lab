"""One prospective larger-bank repair with an unexamined test subset."""
from pathlib import Path
import hashlib
import json
import numpy as np
from recognition import data

S=data.P/'studies/20261023-image-capacity'


def decode(scores,templates):return (templates.sum(axis=1,dtype=np.int32)-2*scores).argmin(axis=1)//320


def evaluate(x,labels,templates):
    scores=data.overlap(x,templates);pred=decode(scores,templates)
    return dict(correct=int((pred==labels).sum()),count=len(labels),predictions=pred.tolist(),
        per_digit=[dict(digit=c,correct=int(((pred==labels)&(labels==c)).sum()),count=int((labels==c).sum())) for c in range(10)])


def main():
    out=S/'fixtures-data01';out.mkdir(exist_ok=False)
    prior=json.loads((data.S/'fixtures-data01/DATA.json').read_text())
    images,labels,sources=data.dataset('train');old=np.asarray(prior['exemplar_train_indices']);rng=np.random.default_rng(98604)
    new=np.concatenate([np.r_[old[c*48:(c+1)*48],rng.choice(np.setdiff1d(np.flatnonzero(labels[:10000]==c),old),272,replace=False)] for c in range(10)])
    assert len(new)==len(set(new))==3200 and set(old)<=set(new)
    dev=np.asarray(prior['development_train_indices']);t=data.encode(images[new]);x=data.encode(images[dev]);y=labels[dev]
    raw=evaluate(x,y,t);compact=evaluate(data.compact(x),y,data.compact(t))
    checks=dict(raw_accuracy=raw['correct']>=160,compact_accuracy=compact['correct']>=160,compact_noninferior=compact['correct']>=raw['correct']-6)
    report=dict(status='DEVELOPMENT_PASS' if all(checks.values()) else 'STOP_DEVELOPMENT_GATE',sources=sources,
        parent_status=prior['status'],preserved_parent_indices=old.tolist(),exemplar_train_indices=new.tolist(),development_train_indices=dev.tolist(),
        development=dict(raw=raw,compacted=compact,checks=checks),sealed_test_opened=False)
    arrays=dict(templates=t,exemplar_labels=labels[new],development_inputs=x,development_labels=y)
    np.savez_compressed(out/'data.npz',**arrays);data.save(out/'DATA.json',report)
    print(json.dumps(dict(status=report['status'],correct=[raw['correct'],compact['correct']])),flush=True)
    if not all(checks.values()):return
    images,labels,sources=data.dataset('t10k');oldtest=np.asarray(prior['test_indices']);rng=np.random.default_rng(98605)
    ids=np.concatenate([rng.choice(np.setdiff1d(np.flatnonzero(labels==c),oldtest),100,replace=False) for c in range(10)])
    assert not set(ids)&set(oldtest)
    x=data.encode(images[ids]);y=labels[ids];raw=evaluate(x,y,t);compact=evaluate(data.compact(x),y,data.compact(t))
    checks=dict(raw_accuracy=raw['correct']>=850,compact_accuracy=compact['correct']>=850,compact_noninferior=compact['correct']>=raw['correct']-20)
    report.update(status='SEALED_QUALITY_PASS' if all(checks.values()) else 'STOP_SEALED_QUALITY_GATE',sources=report['sources']+sources,
        sealed_test_opened=True,test_indices=ids.tolist(),excluded_parent_test_indices=oldtest.tolist(),sealed=dict(raw=raw,compacted=compact,checks=checks))
    arrays.update(test_inputs=x,test_labels=y);np.savez_compressed(out/'data.npz',**arrays);data.save(out/'DATA.json',report)
    print(json.dumps(dict(status=report['status'],correct=[raw['correct'],compact['correct']])),flush=True)


if __name__=='__main__':main()
