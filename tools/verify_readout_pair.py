"""Recount matched outputs and learned state across output-format contracts."""
from pathlib import Path
import argparse,hashlib,json
from verify import verify,read_words

def check(root,potential_run,mask_run):
    audits=[verify(root,d) for d in [potential_run,mask_run]]
    a,b=[json.loads((d/'RECEIPT.json').read_text()) for d in [potential_run,mask_run]]
    assert not a['activated'] and b['activated']
    assert (a['samples'],a['learning'])==(b['samples'],b['learning'])
    f,g=[root/r['fixture'] for r in [a,b]]
    n,u,_,_=map(int,f.with_suffix('.meta').read_text().split())
    assert f.with_suffix('.meta').read_bytes()==g.with_suffix('.meta').read_bytes()
    for ext in ['inputs','labels']:assert f.with_suffix('.'+ext).read_bytes()==g.with_suffix('.'+ext).read_bytes()
    p=read_words(potential_run/'native.scores');m=read_words(mask_run/'native.scores')
    assert [int(v>b['threshold']) for v in p]==m,'threshold responses changed'
    stride=4+n//32;learned=[2+j*stride+k for j in range(u) for k in range(3,stride)]
    for ext in ['ram']:
        x,y=[read_words(d/('native.'+ext)) for d in [potential_run,mask_run]]
        assert all(x[i]==y[i] for i in learned),'learned state depends on readout'
    x,y=[read_words(z.with_suffix('.ram')) for z in [f,g]]
    assert all(x[i]==y[i] for i in learned),'initial learned fields differ'
    fixed_differences=[i for i,(v,w) in enumerate(zip(x,y)) if v!=w]
    assert all(i>=2 and (i-2)%stride in [0,1,2] for i in fixed_differences)
    return dict(status='PASS',runs=[d.name for d in [potential_run,mask_run]],samples=a['samples'],neurons=u,threshold=b['threshold'],exact_threshold_responses=len(m),true_responses=sum(m),false_responses=len(m)-sum(m),learned_words_equal=len(learned),independent_audits=audits,verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('potential_run',type=Path);p.add_argument('mask_run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=check(Path(__file__).resolve().parents[1],a.potential_run,a.mask_run);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='independent_audits'}))
