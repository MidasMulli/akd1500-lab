"""Wall-paced physical Pong prediction capture; control is separately gated.

Runs the same serve-and-flight script sequentially in each arm. Native predictions
are received before a ball is launched; labels are sent only after it lands.
The two arms are NEVER described as concurrent hardware runs.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pong.experiment import encode, decode, flight, outcome, schedule, save, sha, PROTECTED
from pong.run import Native


def capture_cases(plan):
    a = [c for c in plan['development'] if c['court'] == 0]
    teaching = [c for c in plan['adapt'] if c['court'] == 0]
    return ([(c, False, False, 'original physics') for c in a[:2]] +
        [(c, True, False, 'physics changed / before feedback') for c in a[:4]] +
        [(c, True, True, 'observed landing / native feedback') for c in teaching[:6]] +
        [(c, True, False, 'changed physics / prediction check') for c in a[:4]])


def run(study, arm, output, controller=False):
    state = json.loads((study / 'STATE.json').read_text())
    assert not state['research_admissions_closed'] and state['admission'] == 'PONG_LIVE'
    if controller: assert state['live_control_admitted'], 'prediction gate not passed'
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((study / 'PROTOCOL.json').read_text())
    name = f'live_85201_{arm}'; fixture = f'studies/{study.name}/fixtures-pong01/s85201'
    native = Native(study, name, fixture, 1024)
    frames, events = [], []
    with (output / 'events.jsonl').open('w') as log:
        def record(row):
            events.append(row); log.write(json.dumps(row) + '\n'); log.flush()
        try:
            # Identical deterministic native prefix; visible RAM is never loaded
            # as a substitute for the selector's actual learning history.
            for row in schedule(plan):
                if row['kind'] != 'initial_training': break
                record(dict(kind='preparation', fixture_index=row['index'], response=native.forward(encode(row['observation']), row['label'])))
            origin = time.monotonic_ns(); record(dict(kind='capture_start', mono_ns=origin, controller=controller, arm=arm))
            for case, (obs, changed, teach, phase) in enumerate(capture_cases(plan)):
                x = encode(obs)
                reply = native.forward(x, PROTECTED)
                prediction = decode(reply['scores'])
                record(dict(kind='prediction', case=case, observation=obs, changed=changed,
                    phase=phase, response=reply, prediction=prediction))
                states = flight(obs, changed)
                target_time = time.monotonic()
                # One sample per 1/60 second; the integrator itself is 120 Hz.
                for k in range(0, len(states), 2):
                    if k: time.sleep(max(0, target_time + (k / 120) - time.monotonic()))
                    frames.append([time.monotonic_ns() - origin, case, k, *states[k], prediction,
                        (prediction + .5) / 4 if controller else .5])
                truth, landing = outcome(obs, changed)
                record(dict(kind='landing', case=case, mono_ns=time.monotonic_ns(), truth=truth,
                    landing=landing, prediction=prediction, hit=abs(frames[-1][-1] - landing) <= .13,
                    physical_meaning='simulated paddle hit; not an independent hardware endpoint'))
                if teach:
                    reply = native.forward(x, PROTECTED if arm == 'frozen' else truth)
                    record(dict(kind='feedback', case=case, mono_ns=reply['received_monotonic_ns'], truth=truth,
                        response=reply, frozen=arm == 'frozen'))
                for _ in range(9):
                    time.sleep(1 / 60)
                    frames.append([time.monotonic_ns() - origin, case, len(states) - 1, *states[-1], prediction,
                        (prediction + .5) / 4 if controller else .5])
            cleanup = native.close()
            np.savez_compressed(output / 'game.npz', frames=np.array(frames, np.float64))
            save(output / 'CAPTURE.json', dict(status='WALL_PACED_NATIVE_CAPTURE_PASS', arm=arm,
                controller='host follows native predicted band' if controller else 'fixed centre paddle; native predictions are shadow only',
                native_cleanup=cleanup, source_game_fps=(len(frames)-1) / ((frames[-1][0]-frames[0][0]) / 1e9),
                duration_seconds=(frames[-1][0]-frames[0][0]) / 1e9,
                origin_monotonic_ns=origin, events_sha256=sha(output / 'events.jsonl'), game_sha256=sha(output / 'game.npz'),
                timing='actual Mac monotonic timestamps, native query pauses preserved; no cross-host clock subtraction',
                physical_run=name, trials=len(capture_cases(plan))))
            print(json.dumps(json.loads((output / 'CAPTURE.json').read_text())))
        except BaseException as error:
            if frames: np.savez_compressed(output / 'partial-game.npz', frames=np.array(frames))
            save(output / 'FAILURE.json', dict(status='FAILED_PRESERVED', error=repr(error), frames=len(frames)))
            raise
        finally:
            if native.process.poll() is None:
                try: native.close()
                except Exception: native.process.terminate(); native.process.wait(timeout=10)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('study', type=Path); p.add_argument('arm', choices=['full','frozen'])
    p.add_argument('output', type=Path); p.add_argument('--controller', action='store_true'); a = p.parse_args()
    run(a.study.resolve(), a.arm, a.output.resolve(), a.controller)
