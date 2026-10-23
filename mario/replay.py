"""Fresh emulator evaluation of saved native choices, with optional recorded frames."""
from pathlib import Path
import argparse
import hashlib
import json

import gymnasium as gym
import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from collect import encode


def fresh(start, action, capture=False):
    env = JoypadSpace(gym.make('SuperMarioBros-1-1-v0', render_mode='rgb_array'), SIMPLE_MOVEMENT)
    frames = []
    try:
        frame, info = env.reset(seed=81701)
        sky = frame[50, 0].copy()
        history = [frame.copy()]
        for _ in range(start):
            frame, _, terminated, truncated, info = env.step(1)
            assert not terminated and not truncated
            history.append(frame.copy())
        x = encode(frame, history[-9], sky, action)
        if capture:
            frames.extend(history[-49:])
        delay = 0 if action == 0 else 16
        dead = False
        for k in range(90):
            frame, _, terminated, truncated, info = env.step(2 if delay <= k < delay + 20 else 1)
            if capture:
                frames.append(frame.copy())
            if info.get('death') or terminated or truncated:
                dead = True
                break
        return dict(success=bool(not dead and info['x_pos'] >= 350),
                    end_x_evaluation_only=int(info['x_pos']), dead=bool(dead),
                    actor_input_sha256=hashlib.sha256(x.tobytes()).hexdigest()), frames
    finally:
        env.close()


def main(recount, dataset, output):
    report = json.loads(recount.read_text())
    experience = json.loads((dataset / 'EXPERIENCE.json').read_text())
    source_rows = {(r['start'], r['action']): r for r in experience['rows'] if r['split'] == 'test'}
    output.mkdir(parents=True, exist_ok=False)
    results, clips = [], {}
    for run in report['runs']:
        if 'actions' not in run:
            continue
        for start, action, expected in zip(report['test_starts'], run['actions'], run['selected_outcomes']):
            capture = (run['seed'], run['arm'], start) in {
                (81701, 'full', 108), (81701, 'frozen', 108),
                (81702, 'full', 116), (81702, 'frozen', 116)}
            actual, frames = fresh(start, action, capture)
            original = source_rows[start, action]
            assert actual['actor_input_sha256'] == original['actor_input_sha256'], 'fresh observation drift'
            assert actual['success'] == bool(expected), 'fresh physical/game outcome differs'
            row = dict(run=run['run'], seed=run['seed'], arm=run['arm'], start=start, action=action, **actual)
            if capture:
                key = f"s{run['seed']}_{run['arm']}_{start}"
                clips[key] = np.array(frames, np.uint8)
                row['clip_key'] = key
                row['frames_sha256'] = hashlib.sha256(clips[key].tobytes()).hexdigest()
            results.append(row)
    np.savez_compressed(output / 'clips.npz', **clips)
    z = dict(status='FRESH_EPISODES_MATCH', trials=len(results), rows=results,
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             native_recount_sha256=hashlib.sha256(recount.read_bytes()).hexdigest(),
             note='Fresh resets and fixed prefixes; native choices come from saved protected queries. This is not frame-by-frame native control.')
    (output / 'FRESH_EPISODES.json').write_text(json.dumps(z, indent=2) + '\n')
    print(json.dumps({k: v for k, v in z.items() if k != 'rows'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('recount', type=Path)
    p.add_argument('dataset', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    main(a.recount.resolve(), a.dataset.resolve(), a.output.resolve())
