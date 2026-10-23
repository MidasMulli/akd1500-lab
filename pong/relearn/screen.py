"""Frozen one-factor exact-rule forecasts; these never open a physical device."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from verify import read_words
from pong.experiment import encode,save,sha
from pong.relearn.protocol import ARMS,U,schedule,gate
from pong.relearn.model import Reference,RecentNearest
from pong.relearn.audit import summarize


def simulate(study,seed,nw,arm):
    plan=json.loads((study/'PROTOCOL.json').read_text());fixture=study/'fixtures-relearn01'/f'nw{nw}_s{seed}'
    model=Reference(read_words(fixture.with_suffix('.ram')),nw=nw);cpu=RecentNearest()
    rows=schedule(plan,arm);scores=[];predictions=[];edits=0;trained=None
    for i,row in enumerate(rows):
        x=encode(row['observation']);before=model.weights[:]
        scores.append(model.forward(x,row['label']))
        predictions.append(cpu.forward(x,row['label'],row['observation']['court']))
        if i==511:trained=np.array(model.ram,'<u4').tobytes()
        if i>=512:edits+=sum((a^b).bit_count() for a,b in zip(before,model.weights))
    return summarize(plan,rows,scores,predictions,hashlib.sha256(trained).hexdigest(),
        hashlib.sha256(np.array(scores[:512],'<i4').tobytes()).hexdigest(),edits)


def validate_new_native(study):
    count=0
    for arm in ARMS:
        raw=study/'raw'/f'development_nw64_s85301_{arm}'
        model=Reference(read_words(raw/'fixture.ram'));xs=np.frombuffer((raw/'fixture.inputs').read_bytes(),np.uint8).reshape(-1,1024)
        ys=read_words(raw/'fixture.labels');scores=np.array(read_words(raw/'native.scores')).reshape(-1,U)
        for i,(x,y) in enumerate(zip(xs,ys)):
            assert model.forward(x,y)==scores[i].tolist()
            assert model.ram==read_words(raw/f'native.snapshot_{i}.ram')
            count+=1
    return count


def screen(study):
    plan=json.loads((study/'PROTOCOL.json').read_text())
    baseline=json.loads((study/'evidence/TRIPLE_development_nw64_s85301.json').read_text())
    assert baseline['gate']['status']=='STOP_DEVELOPMENT_GATE','screen not admitted after passing baseline'
    count=validate_new_native(study);results=[]
    for nw in plan['controlled_candidates']:
        rows=[]
        for seed in plan['development_seeds']:
            arms={arm:simulate(study,seed,nw,arm) for arm in ARMS}
            decision=gate(plan,arms);rows.append(dict(seed=seed,gate=decision,arms=arms))
            print(json.dumps(dict(status='CPU_FORECAST_ONLY',num_weights=nw,seed=seed,gate=decision)),flush=True)
        results.append(dict(num_weights=nw,rows=rows,qualifies=all(r['gate']['status']=='PASS' for r in rows),
            worst_seed_accuracy=min(r['arms']['relearn']['curves']['6']['new_A']['accuracy'] for r in rows)))
    passing=[r for r in results if r['qualifies']]
    selected=max(passing,key=lambda r:(r['worst_seed_accuracy'],-r['num_weights']))['num_weights'] if passing else None
    output=dict(status='CPU_ONE_FACTOR_SCREEN_COMPLETE',physical_device_opened=False,
        exact_reference_validated_against_new_native_forwards=count,source_sha256=sha(__file__),
        results=results,selected_num_weights=selected,
        disposition='ADMIT_ONE_CANDIDATE_FIRST_SEED_ONLY' if selected else 'STOP_NO_QUALIFYING_CANDIDATE',
        limitations='Forecasts from exact reconstructed rule on fixed development data; no hardware or sealed confirmation evidence')
    save(study/'evidence/CPU_SCREEN.json',output);print(json.dumps({k:v for k,v in output.items() if k!='results'}))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);screen(p.parse_args().study.resolve())
