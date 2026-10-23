"""Public replay binding: scores, edits and inputs must belong to physical raw data."""
from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pong.experiment import N,U,CLASSES,encode,decode,sha,PROTECTED
from pong.live import capture_cases


def audit(data=None):
    data=json.loads((ROOT/'visual/pong-physics-data.json').read_text()) if data is None else data
    p=ROOT/'results/2026-10-23-pong-physics.npz';assert sha(p)==data['archive_sha256']
    z=np.load(p,allow_pickle=False);cases=capture_cases(data['protocol']);bound=0
    for arm,capture in data['captures'].items():
        path=ROOT/capture['game_path'];assert sha(path)==capture['game_sha256']
        frames=np.load(path,allow_pickle=False)['frames'];key=capture['raw_key'];origin=capture['receipt']['origin_monotonic_ns']
        inputs=z[f'fixture__{key}__inputs'].reshape(-1,N);labels=z[f'fixture__{key}__labels']
        scores=z[f'{key}__native_scores'].reshape(-1,U);states=z[f'{key}__snapshots'];initial=z[f'fixture__{key}__ram']
        for event in capture['events']:
            if 'response' not in event:continue
            r=event['response'];i=r['index'];assert r['id']==i+1
            assert scores[i].tolist()==r['scores'],'substituted native scores'
            obs,changed,teach,phase=cases[event['case']]
            assert np.array_equal(inputs[i],encode(obs))
            assert labels[i]==r['label']
            before=initial if i==0 else states[i-1];after=states[i]
            records=[];bits=0
            for j in range(U):
                b=2+j*36
                if np.any(before[b:b+36]!=after[b:b+36]):records.append(j)
                bits+=sum((int(a)^int(b)).bit_count() for a,b in zip(before[b+4:b+36],after[b+4:b+36]))
            assert r['changed_records']==records and r['connection_bit_edits']==bits
            if event['kind']=='prediction':
                assert r['label']==PROTECTED and event['prediction']==decode(r['scores'])
                f=frames[frames[:,1]==event['case']]
                assert np.all(f[:,6]==event['prediction']),'frame prediction not from native response'
                assert r['received_monotonic_ns']<=origin+f[0,0],'reply shown before receipt'
            else:
                assert event['kind']=='feedback' and teach
                landing=next(e for e in capture['events'] if e['kind']=='landing' and e['case']==event['case'])
                assert r['sent_monotonic_ns']>=landing['mono_ns']
                assert r['label']==(PROTECTED if arm=='frozen' else landing['truth'])
            bound+=1
        if arm=='frozen':assert np.array_equal(states[511],states[-1])
    return dict(status='PUBLIC_PONG_BINDING_PASS',bound_replies=bound)


if __name__=='__main__':print(json.dumps(audit()))
