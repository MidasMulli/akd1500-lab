"""Wall-paced gameplay with actual resident AKD queries and causal response timestamps."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import select
import subprocess
import time
import gymnasium as gym
import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from bank import encode, save, sha


class Native:
    def __init__(self, study, name, fixture):
        remote = '/home/nicklomeli/akd1500-lab'
        command = f'cd {remote} && timeout --signal=INT --kill-after=5 178 /home/nicklomeli/akida-cc-venv/bin/python mario/sealed/serve.py studies/{study.name} {name} {fixture} --forwards 4096'
        self.process = subprocess.Popen(['ssh', '-T', 'jetson', command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.id = 0
        self.ready = self.read()
        assert self.ready['kind'] == 'ready'

    def read(self):
        assert select.select([self.process.stdout], [], [], 10)[0], 'native response timeout'
        line = self.process.stdout.readline()
        assert line, 'native stream ended: ' + self.process.stderr.read().decode()
        return json.loads(line)

    def forward(self, x, label):
        self.id += 1
        start = time.monotonic_ns()
        self.process.stdin.write(f'F {self.id} {label} {x.tobytes().hex()}\n'.encode()); self.process.stdin.flush()
        result = self.read()
        assert result['kind'] == 'forward' and result['id'] == self.id
        result.update(sent_monotonic_ns=start, received_monotonic_ns=time.monotonic_ns(), input_sha256=hashlib.sha256(x.tobytes()).hexdigest())
        return result

    def close(self):
        self.process.stdin.write(b'X\n'); self.process.stdin.flush()
        result = self.read()
        assert result['kind'] == 'closed'
        assert self.process.wait(timeout=10) == 0
        return result


def run(study, name, fixture, plan, output, frozen):
    assert 'fixtures-cal' in fixture, 'no held-out test admission'
    output.mkdir(parents=True, exist_ok=False)
    local = study / 'fixtures-cal01' / Path(fixture).name
    meta = json.loads(local.with_suffix('.json').read_text())
    inputs = np.frombuffer(local.with_suffix('.inputs').read_bytes(), np.uint8).reshape(meta['samples'], 1024)
    labels = np.frombuffer(local.with_suffix('.labels').read_bytes(), '<i4')
    tasks = json.loads(plan.read_text())['contexts']
    native = Native(study, name, fixture)
    events, frames, timings = [], [], []
    log = (output / 'events.jsonl').open('w')
    def record(row):
        events.append(row); log.write(json.dumps(row) + '\n'); log.flush()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending = None
    latest = None
    origin = None
    try:
        for i, (x, label) in enumerate(zip(inputs, labels)):
            reply = native.forward(x, int(label))
            # Retain every packet but keep offline training out of the game timeline.
            record(dict(kind='preparation', fixture_index=i, response=reply))
        record(dict(kind='game_start', mono_ns=time.monotonic_ns(), frozen=frozen))
        origin = time.monotonic_ns()
        def pair(observation, elapsed, frame_id, purpose):
            values = []
            for action in [0, 1]:
                x = encode(**observation, action=action, encoder='projection', elapsed=elapsed)
                values.append(native.forward(x, 2))
            return dict(kind='response_pair', purpose=purpose, input_frame=frame_id,
                elapsed=elapsed, received_monotonic_ns=values[-1]['received_monotonic_ns'], replies=values)
        def drain(wait=False):
            nonlocal pending, latest
            if pending is not None and (wait or pending.done()):
                latest = pending.result(timeout=10); record(latest); pending = None
        for case, context in enumerate(tasks):
            env = JoypadSpace(gym.make(f'SuperMarioBros-{context["level"]}-v0', render_mode='rgb_array'), SIMPLE_MOVEMENT)
            try:
                frame, info = env.reset(seed=82800)
                sky = frame[50, 0].astype(np.int16)
                history = [frame.copy()]
                elapsed = 0
                target = time.monotonic()
                def sample(phase):
                    nonlocal pending, target
                    drain()
                    frame_id = len(frames)
                    frames.append(frame.copy()); timings.append(time.monotonic_ns() - origin)
                    observation = dict(frame=frame.copy(), previous=history[-9].copy() if len(history) >= 9 else history[0].copy(), sky=sky)
                    if elapsed % 6 == 0 and pending is None:
                        pending = executor.submit(pair, observation, elapsed, frame_id, 'shadow_telemetry_only')
                    record(dict(kind='frame', frame_id=frame_id, case=case, context=context['id'], elapsed=elapsed,
                        phase=phase, mono_ns=time.monotonic_ns(), latest_native_id=None if latest is None else latest['replies'][-1]['id']))
                    target += 1 / 60
                    time.sleep(max(0, target - time.monotonic()))
                    return observation, frame_id
                for k in range(context['wait'] + context['start']):
                    sample('supplied_prefix')
                    frame, _, terminated, truncated, info = env.step(0 if k < context['wait'] else 1)
                    history.append(frame.copy()); elapsed += 1
                    assert not terminated and not truncated and not info.get('death')
                decision_observation, frame_id = sample('native_decision')
                drain(True)
                decision = pair(decision_observation, elapsed, frame_id, 'action_selection')
                record(decision)
                margins = [max(r['scores'][64:128]) - max(r['scores'][:64]) for r in decision['replies']]
                action = int(margins[1] > margins[0])
                record(dict(kind='decision', case=case, context=context['id'], input_frame=frame_id, action=action,
                    margins=margins, native_ids=[r['id'] for r in decision['replies']], mono_ns=time.monotonic_ns()))
                start_x = int(info['x_pos']); dead = False
                for k in range(90):
                    frame, _, terminated, truncated, info = env.step(2 if 16 * action <= k < 16 * action + 20 else 1)
                    history.append(frame.copy()); elapsed += 1
                    sample('committed_macro_shadow_queries_do_not_replan')
                    if info.get('death') or terminated or truncated:
                        dead = True; break
                success = bool(not dead and info['x_pos'] - start_x >= 70)
                drain(True)
                x = encode(**decision_observation, action=action, encoder='projection', elapsed=context['wait'] + context['start'])
                feedback = native.forward(x, 2 if frozen else int(success))
                record(dict(kind='feedback', case=case, context=context['id'], input_frame=frame_id, action=action,
                    success=success, advance_evaluation_only=int(info['x_pos']) - start_x, response=feedback,
                    mono_ns=time.monotonic_ns(), frozen=frozen))
                for _ in range(30):
                    frames.append(frame.copy()); timings.append(time.monotonic_ns() - origin)
                    record(dict(kind='frame', frame_id=len(frames) - 1, case=case, context=context['id'], elapsed=elapsed,
                        phase='observed_feedback_hold', mono_ns=time.monotonic_ns(), latest_native_id=feedback['id']))
                    time.sleep(1 / 60)
            finally:
                env.close()
        drain(True)
        cleanup = native.close()
        np.savez_compressed(output / 'game.npz', frames=np.array(frames, np.uint8), mono_offsets_ns=np.array(timings, np.int64))
        responses = [e for e in events if e['kind'] == 'response_pair']
        duration = (timings[-1] - timings[0]) / 1e9
        save(output / 'LIVE_RECEIPT.json', dict(status='LIVE_PHYSICAL_CAPTURE_PASS', native_cleanup=cleanup,
            frozen=frozen, frames=len(frames), duration_seconds=duration, source_game_frames_per_second=(len(frames) - 1) / duration,
            query_pairs=len(responses), achieved_query_pairs_per_second=len(responses) / duration,
            frame_timing='Mac monotonic timestamps, no cross-host clock subtraction',
            controller='90-frame supplied macro; protected shadow telemetry does not replan',
            game_sha256=sha(output / 'game.npz'), events_sha256=sha(output / 'events.jsonl'),
            source_sha256=sha(__file__)))
        print(json.dumps(json.loads((output / 'LIVE_RECEIPT.json').read_text())))
    except BaseException as error:
        if frames:
            np.savez_compressed(output / 'partial-game.npz', frames=np.array(frames, np.uint8), mono_offsets_ns=np.array(timings, np.int64))
        save(output / 'LIVE_FAILURE.json', dict(status='FAILED_PRESERVED', error=repr(error), captured_frames=len(frames),
            last_native_id=native.id, source_sha256=sha(__file__)))
        raise
    finally:
        executor.shutdown(wait=True)
        log.close()
        if native.process.poll() is None:
            try: native.close()
            except Exception: native.process.terminate(); native.process.wait(timeout=10)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('study', type=Path)
    p.add_argument('name')
    p.add_argument('fixture')
    p.add_argument('plan', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--frozen', action='store_true')
    a = p.parse_args()
    run(a.study.resolve(), a.name, a.fixture, a.plan.resolve(), a.output.resolve(), a.frozen)
