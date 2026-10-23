"""Hardware-free causal checks of the public paired capture and RGB/time inputs."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from mario.sealed.bank import encode


def audit(data=None):
    if data is None:data=json.loads((ROOT/'visual/mario-continuous-data.json').read_text())
    raw_path=ROOT/'results/2026-10-23-mario-continuous.npz';assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==data['archive_sha256']
    prefix=ROOT/'visual/mario-continuous-prefixes.npz';assert hashlib.sha256(prefix.read_bytes()).hexdigest()==data['prefix_sha256'];prefixes=np.load(prefix,allow_pickle=False)
    raw=np.load(raw_path,allow_pickle=False);pairs=responses=frame_count=0
    for arm,c in data['captures'].items():
        p=ROOT/c['game_path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==c['game_sha256'];game=np.load(p,allow_pickle=False);images=game['frames'];clock=game['mono_offsets_ns']
        events=c['events'];frames={e['frame_id']:e for e in events if e['kind']=='frame'};assert set(frames)==set(range(len(images)))
        origin=next(e['mono_ns'] for e in events if e['kind']=='game_start');assert np.all(np.diff(clock)>0)
        assert all(e['mono_ns']==origin+int(clock[i]) for i,e in frames.items());frame_count+=len(frames)
        name='dev_'+arm+'01';inputs=raw[f'fixture__{name}__inputs'].reshape(-1,1024);labels=raw[f'fixture__{name}__labels'];scores=raw[name+'__native_scores'].reshape(-1,192)
        forwards={e['response']['id']:e for e in events if e['kind']=='forward'}
        def check(reply):
            nonlocal responses
            i=reply['index'];assert reply['scores']==scores[i].tolist() and reply['label']==int(labels[i])
            assert reply['input_sha256']==hashlib.sha256(inputs[i].tobytes()).hexdigest();responses+=1
        for d in [e for e in events if e['kind']=='decision']:
            f=frames[d['input_frame']];assert d['controlled']==f['controlled'] and d['elapsed']==f['elapsed']
            history=prefixes[f'{arm}__{f["episode"]}__history'];sky=prefixes[f'{arm}__{f["episode"]}__sky']
            if f['controlled']==0:
                assert np.array_equal(images[f['frame_id']],history[-1]);previous=history[-9]
            else:
                old=next(e for e in frames.values() if e['episode']==f['episode'] and e['controlled']==f['controlled']-8);previous=images[old['frame_id']]
            replies=[forwards[i]['response'] for i in d['native_ids']]
            for a,r in enumerate(replies):
                check(r);expected=encode(images[f['frame_id']],previous,sky,a,'projection',f['elapsed'])
                assert np.array_equal(expected,inputs[r['index']]),'live RGB/time source mismatch'
                assert forwards[r['id']]['input_frame']==f['frame_id'] and r['label']==2
            margin=[max(r['scores'][64:128])-max(r['scores'][:64]) for r in replies]
            assert d['margins']==margin and d['action']==int(margin[1]>margin[0]),'action does not follow native replies'
            assert max(r['received_monotonic_ns'] for r in replies)<=d['mono_ns']
            outcome=next(e for e in events if e['kind']=='outcome' and e['input_frame']==f['frame_id'])
            feedback=forwards[outcome['response_id']]['response'];check(feedback)
            assert np.array_equal(inputs[feedback['index']],inputs[replies[d['action']]['index']])
            assert feedback['label']==(2 if arm=='frozen' else int(outcome['success']))
            end=frames[outcome['end_frame']];assert 0<end['controlled']-f['controlled']<=24
            assert feedback['sent_monotonic_ns']>=end['mono_ns']
            for step in range(1,end['controlled']-f['controlled']+1):
                e=frames[f['frame_id']+step];expected_button=2 if d['action']==0 and step<=8 else 1
                assert e['button']==expected_button and e['mono_ns']>=d['mono_ns']
            pairs+=1
        assert len(forwards)==3*sum(e['kind']=='decision' for e in events)
    return dict(status='PUBLIC_CONTINUOUS_BINDING_PASS',decisions=pairs,bound_responses=responses,frames=frame_count)

if __name__=='__main__':print(json.dumps(audit()))
