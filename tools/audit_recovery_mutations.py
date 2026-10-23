"""Check that new probe ingress and its fixed clock cannot be silently forged."""
from pathlib import Path
import argparse,importlib,json,sys
P=Path(__file__).resolve().parents[1]

def main(study):
    sys.path.insert(0,str(study));sys.path.insert(0,str(study/'service'))
    nv=importlib.import_module('verify_native');vr=importlib.import_module('verify_recovery')
    source=study/'runs/pilot_probe_71011';root=study/'evidence/MUTATION_CASES';root.mkdir(exist_ok=False)
    origin=Path(json.loads((source/'OWNER.json').read_text())['args'][1]).parent.parent
    def relocate(p):
        p=Path(p)
        try:return study/p.relative_to(origin)
        except ValueError:return p
    oldsha=vr.sha;oldartifact=vr.artifact
    vr.sha=lambda p:oldsha(relocate(p))
    vr.artifact=lambda a,b,stage:oldartifact(relocate(a),relocate(b),stage)
    assert vr.verify(source)['status']=='PASS','positive control before mutations'
    results=[]
    for name,filename in [('changed_probe_neighbor','stdout.jsonl'),('false_probe_clock','events.jsonl')]:
        d=root/name;d.mkdir()
        for f in source.iterdir():
            if f.name!=filename:(d/f.name).symlink_to(f,target_is_directory=f.is_dir())
        rows=[json.loads(l) for l in (source/filename).read_text().splitlines()]
        if filename=='stdout.jsonl':
            event=next(x for x in rows if x['kind']=='probe_request');event['bytes'][1]^=1
        else:
            event=next(x for x in rows if x['kind']=='operation' and x['probe']);event['probe']=False
        (d/filename).write_text(''.join(json.dumps(x)+'\n' for x in rows))
        try:
            # Clock mutation is checked before artifact-path reads on the Mac.
            nv.verify(d) if filename=='stdout.jsonl' else vr.verify(d)
        except AssertionError as exc:results.append(dict(case=name,status='REJECTED',reason=str(exc)))
        else:raise AssertionError('auditor accepted '+name)
    out=dict(status='PASS',mutations=results,source_unmodified=True)
    (study/'evidence/PROBE_AUDITOR_MUTATIONS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);a=p.parse_args();main(a.study.resolve())
