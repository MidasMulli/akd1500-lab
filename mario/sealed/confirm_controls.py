"""Prospectively specified CPU-model controls, kept separate from native arms."""
from pathlib import Path
import argparse
import json
import subprocess
import numpy as np
from bank import save,sha

ROOT=Path(__file__).resolve().parents[2]


def run(study):
    output=study/'cpu_runs/controls01';output.mkdir(parents=True,exist_ok=False);rows=[]
    for entry in json.loads((study/'fixtures-confirm02/BUILD.json').read_text())['rows']:
        if entry['arm'] not in ['action_only','shuffled']:continue
        f=study/'fixtures-confirm02'/entry['prefix'];out=output/entry['prefix'];meta=json.loads(f.with_suffix('.json').read_text())
        result=subprocess.run([str(ROOT/'build/learning_bench'),'cpu',str(f),str(out),str(meta['samples']),'1','1','0'],capture_output=True,text=True,timeout=60)
        out.with_suffix('.stdout').write_text(result.stdout);out.with_suffix('.stderr').write_text(result.stderr);assert result.returncode==0
        q=meta['evaluation_samples'];scores=np.fromfile(out.with_suffix('.scores'),'<i4').reshape(-1,192)[-q:]
        actions=(scores[:,64:128].max(1)-scores[:,:64].max(1)).reshape(-1,2).argmax(1);truth=np.array([r['success'] for r in meta['evaluation_rows']]).reshape(-1,2)
        rows.append(dict(seed=meta['seed'],arm=meta['arm'],successes=int(truth[np.arange(len(actions)),actions].sum()),contexts=len(actions),actions=actions.tolist(),raw_scores_sha256=sha(out.with_suffix('.scores'))))
    z=dict(status='CPU_MODEL_CONTROLS_ONLY',physical_device_opened=False,rows=rows,source_sha256=sha(__file__))
    save(output/'REPORT.json',z);print(json.dumps({**z,'rows':[{k:v for k,v in r.items() if k!='actions'} for r in rows]}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);a=p.parse_args();run(a.study.resolve())
