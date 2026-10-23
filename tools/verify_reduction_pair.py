"""Independent comparison of a mask stream and its two-count hardware reduction."""
from pathlib import Path
import argparse,hashlib,json
from verify import verify,read_words

def check(root,mask_run,reduced_run):
    audits=[verify(root,d) for d in [mask_run,reduced_run]]
    a,b=[json.loads((d/'RECEIPT.json').read_text()) for d in [mask_run,reduced_run]]
    assert a.get('output_contract',int(a['activated']))==1 and b['output_contract']==2
    assert (a['samples'],a['learning'],a['threshold'])==(b['samples'],b['learning'],b['threshold'])
    f,g=[root/r['fixture'] for r in [a,b]];n,u,_,classes=map(int,f.with_suffix('.meta').read_text().split())
    for ext in ['inputs','labels','meta','ram']:assert f.with_suffix('.'+ext).read_bytes()==g.with_suffix('.'+ext).read_bytes()
    mask=read_words(mask_run/'native.scores');got=read_words(reduced_run/'native.scores');expected=[]
    for i in range(a['samples']):
        for k in range(classes):expected.append(sum(mask[i*u+k*(u//classes):i*u+(k+1)*(u//classes)]))
    assert got==expected,'fixed downstream counts differ'
    assert (mask_run/'native.ram').read_bytes()==(reduced_run/'native.ram').read_bytes(),'complete learned RAM differs'
    return dict(status='PASS',runs=[d.name for d in [mask_run,reduced_run]],samples=a['samples'],neurons=u,outputs=classes,exact_group_counts=len(got),complete_RAM_words_equal=len(read_words(mask_run/'native.ram')),independent_audits=audits,verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mask_run',type=Path);p.add_argument('reduced_run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=check(Path(__file__).resolve().parents[1],a.mask_run,a.reduced_run);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='independent_audits'}))
