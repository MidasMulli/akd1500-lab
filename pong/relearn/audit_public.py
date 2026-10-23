"""Recount the new three-arm result directly from the public native archive."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pong.experiment import encode,sha
from pong.relearn.protocol import ARMS,N,U,schedule,gate
from pong.relearn.model import RecentNearest
from pong.relearn.audit import summarize,timing


def audit():
    report=json.loads((ROOT/'results/2026-10-23-pong-relearning.json').read_text());p=ROOT/'results/2026-10-23-pong-relearning.npz'
    assert sha(p)==report['archive_sha256'];z=np.load(p,allow_pickle=False);plan=report['protocol'];arms={};bound=0
    for arm in ARMS:
        key=f'development_nw64_s85301_{arm}';rows=schedule(plan,arm)
        xs=z[f'fixture__{key}__inputs'].reshape(-1,N);ys=z[f'fixture__{key}__labels'];scores=z[key+'__native_scores'].reshape(-1,U)
        stdout=z[key+'__native_stdout'].tobytes().decode();packets=[json.loads(line) for line in stdout.splitlines() if json.loads(line)['kind']=='forward']
        assert len(rows)==len(xs)==len(packets);cpu=RecentNearest();predictions=[];events=[]
        for i,(r,packet) in enumerate(zip(rows,packets)):
            assert np.array_equal(encode(r['observation']),xs[i]) and r['label']==ys[i]
            assert packet['scores']==scores[i].tolist()
            expected=cpu.forward(xs[i],int(ys[i]),r['observation']['court']);assert expected==packet['cpu_task_prediction'];predictions.append(expected)
            events.append(dict(**r,response=packet))
            if r['observation']['court']==1 and r['phase']!='prefix_B':assert r['label']==4
            bound+=1
        state_hash=hashlib.sha256(z[key+'__snapshots'][511].astype('<u4').tobytes()).hexdigest()
        score_hash=hashlib.sha256(scores[:512].astype('<i4').tobytes()).hexdigest()
        edits=sum(p['connection_bit_edits'] for p in packets[512:])
        summary=summarize(plan,rows,scores,predictions,state_hash,score_hash,edits)
        assert summary==report['triple']['arms'][arm]
        assert timing(events)==report['triple']['timing'][arm]
        arms[arm]=summary
    assert gate(plan,arms)==report['triple']['gate']
    return dict(status='PUBLIC_THREE_ARM_RECOUNT_PASS',bound_forwards=bound)


if __name__=='__main__':print(json.dumps(audit()))
