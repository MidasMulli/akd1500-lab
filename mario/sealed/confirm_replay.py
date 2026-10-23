"""Fresh emulator execution of the fixed native test choices; no teaching or search."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from collect import trial
from bank import encode,save,sha

ROOT=Path(__file__).resolve().parents[2]


def replay(study,output,seeds):
    report=json.loads((study/'evidence/RECOUNT.json').read_text());output.mkdir(parents=True,exist_ok=False)
    save(output/'NATIVE_SELECTIONS.json',report);rows=[]
    for run in report['runs']:
        if run.get('seed') not in seeds:continue
        prefix=ROOT/run['audit']['fixture'];meta=json.loads(prefix.with_suffix('.json').read_text())
        n=meta['samples'];scores=np.fromfile(study/'raw'/run['name']/'native.scores','<i4').reshape(n,192)[-256:]
        actions=(scores[:,64:128].max(1)-scores[:,:64].max(1)).reshape(128,2).argmax(1)
        assert actions.tolist()==run['actions']
        raw=np.frombuffer(prefix.with_suffix('.inputs').read_bytes(),np.uint8).reshape(n,1024)
        for i,(context,action,expected) in enumerate(zip(report['contexts'],actions,run['correct'])):
            actual,observation,_=trial(context,int(action));assert actual['valid']
            x=encode(**observation,action=int(action),encoder='projection',elapsed=context['wait']+context['start'])
            assert np.array_equal(x,raw[1120+2*i+action]),'fresh RGB/time observation differs from native test input'
            assert actual['success']==bool(expected),'fresh game outcome differs'
            rows.append(dict(seed=run['seed'],arm=run['arm'],context=context['id'],action=int(action),success=actual['success'],
                input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),frame_sha256=actual['frame_sha256'],advance_evaluation_only=actual['advance_evaluation_only']))
    save(output/'FRESH.json',dict(status='FRESH_NATIVE_SELECTIONS_MATCH',rows=rows,trials=len(rows),
        source_sha256=sha(__file__),native_selections_sha256=sha(output/'NATIVE_SELECTIONS.json'),
        note='Repeated policies on the same 128 contexts; not independent worlds or frame-by-frame native control.'))
    print(json.dumps(dict(status='FRESH_NATIVE_SELECTIONS_MATCH',trials=len(rows),seeds=seeds)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('output',type=Path);p.add_argument('--seeds',nargs='+',type=int,required=True);a=p.parse_args();replay(a.study.resolve(),a.output.resolve(),a.seeds)
