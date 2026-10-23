"""Recount every captured physical reply, landing and causal display timestamp."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
from verify import verify, read_words
from pong.experiment import N, U, encode, decode, schedule, save, sha, PROTECTED
from pong.live import capture_cases


def audit(study, arm):
    capture = study / 'runs' / f'live_85201_{arm}'; raw = study / 'raw' / f'live_85201_{arm}'
    receipt = json.loads((capture / 'CAPTURE.json').read_text())
    for name, key in [('events.jsonl','events_sha256'),('game.npz','game_sha256')]:
        assert sha(capture / name) == receipt[key]
    independent = verify(ROOT, raw)
    events = [json.loads(line) for line in (capture / 'events.jsonl').read_text().splitlines()]
    reply_events = [e for e in events if 'response' in e]
    scores = np.array(read_words(raw / 'native.scores')).reshape(-1, U)
    inputs = np.frombuffer((raw / 'fixture.inputs').read_bytes(), np.uint8).reshape(-1,N)
    labels = read_words(raw / 'fixture.labels')
    native = [json.loads(line) for line in (raw / 'native.stdout').read_text().splitlines()]
    replies = [e for e in native if e['kind'] == 'forward']
    assert len(replies) == len(reply_events)
    plan = json.loads((study / 'PROTOCOL.json').read_text())
    prefix = [r for r in schedule(plan) if r['kind'] == 'initial_training']
    cases = capture_cases(plan)
    initial_ram = read_words(raw / 'fixture.ram'); last_ram = initial_ram
    edits = []
    for i, (event, native_reply) in enumerate(zip(reply_events, replies)):
        for key in native_reply: assert event['response'][key] == native_reply[key]
        assert event['response']['scores'] == scores[i].tolist()
        if event['kind'] == 'preparation':
            expected = prefix[event['fixture_index']]; obs = expected['observation']; label = expected['label']
        else:
            obs, changed, teach, phase = cases[event['case']]
            label = event['truth'] if event['kind'] == 'feedback' and arm == 'full' else PROTECTED
        assert np.array_equal(encode(obs),inputs[i]) and labels[i] == label
        ram = read_words(raw / f'native.snapshot_{i}.ram')
        count = sum((a^b).bit_count() for j in range(U) for a,b in zip(last_ram[2+j*36+4:2+(j+1)*36],ram[2+j*36+4:2+(j+1)*36]))
        assert count == native_reply['connection_bit_edits']; edits.append(count)
        if label == PROTECTED: assert ram == last_ram
        last_ram = ram
    frames = np.load(capture / 'game.npz',allow_pickle=False)['frames']
    assert np.all(np.diff(frames[:,0]) > 0)
    origin = receipt['origin_monotonic_ns']
    for case, (obs, changed, teach, phase) in enumerate(cases):
        p = next(e for e in events if e['kind']=='prediction' and e['case']==case)
        landed = next(e for e in events if e['kind']=='landing' and e['case']==case)
        f = frames[frames[:,1] == case]
        assert p['response']['received_monotonic_ns'] <= origin + f[0,0]
        assert np.all(f[:,6] == decode(p['response']['scores']))
        # Separate scalar reconstruction, deliberately not experiment.flight().
        y, v = obs['y'], obs['vy']; trajectory=[(.08,y,v)]
        a = (-.8 if changed else .8) if obs['court']==0 else .3
        for k in range(144):
            v += a/120; y += v/120
            if y < 0: y,v=-y,-v
            if y > 1: y,v=2-y,-v
            trajectory.append((.08+.84*(k+1)/144,y,v))
        expected=np.array([trajectory[int(k)] for k in f[:,2]])
        assert np.allclose(expected,f[:,3:6],rtol=0,atol=1e-13)
        assert abs(landed['landing']-y)<1e-13 and landed['truth']==min(3,int(y*4))
        assert landed['mono_ns'] >= origin+f[72,0]
        feedback=[e for e in events if e['kind']=='feedback' and e['case']==case]
        assert len(feedback)==int(teach)
        for event in feedback: assert event['response']['sent_monotonic_ns'] >= landed['mono_ns']
        if receipt['controller'].startswith('fixed'): assert np.all(f[:,7]==.5)
    result=dict(status='INDEPENDENT_LIVE_AUDIT_PASS', raw=independent, frames=len(frames),
        trials=len(cases), preparation_forwards=len(prefix), post_prefix_connection_bit_edits=sum(edits[len(prefix):]),
        queries_before_launch=True, observed_labels_after_landing=True, all_saved_positions_reconstructed=True,
        source_game_fps=receipt['source_game_fps'], events_sha256=sha(capture/'events.jsonl'))
    save(capture/'AUDIT.json',result); print(json.dumps(result)); return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('arm',choices=['full','frozen']);a=p.parse_args();audit(a.study.resolve(),a.arm)
