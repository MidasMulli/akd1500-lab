"""Collect only requested sealed-bank splits; retain invalid prefixes without replacement."""
from pathlib import Path
import argparse
import hashlib
import json
import gymnasium as gym
import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from bank import design, encode, save, sha


def trial(context, action, capture=False):
    env = JoypadSpace(gym.make(f'SuperMarioBros-{context["level"]}-v0', render_mode='rgb_array'), SIMPLE_MOVEMENT)
    frames = []
    try:
        frame, info = env.reset(seed=82800)
        sky = frame[50, 0].astype(np.int16)
        history = [frame.copy()]
        for k in range(context['wait'] + context['start']):
            frame, _, terminated, truncated, info = env.step(0 if k < context['wait'] else 1)
            history.append(frame.copy())
            if info.get('death') or terminated or truncated:
                return dict(**context, action=action, valid=False, reason='prefix_already_dead'), None, []
        observations = dict(frame=frame.copy(), previous=history[-9], sky=sky)
        start_x = int(info['x_pos'])
        if capture:
            frames.extend(history[-49:])
        dead = False
        delay = 16 * action
        for k in range(90):
            frame, _, terminated, truncated, info = env.step(2 if delay <= k < delay + 20 else 1)
            if capture:
                frames.append(frame.copy())
            if info.get('death') or terminated or truncated:
                dead = True
                break
        row = dict(**context, action=action, valid=True, success=bool(not dead and info['x_pos'] - start_x >= 70),
            dead=bool(dead), advance_evaluation_only=int(info['x_pos']) - start_x,
            frame_sha256=hashlib.sha256(observations['frame'].tobytes()).hexdigest(),
            previous_sha256=hashlib.sha256(observations['previous'].tobytes()).hexdigest())
        return row, observations, frames
    finally:
        env.close()


def collect(manifest, output, splits, lock=None, clock=False):
    z = json.loads(manifest.read_text())
    assert z == design(), 'task identity drift'
    if 'test' in splits:
        assert lock is not None and json.loads(lock.read_text())['status'] == 'PIPELINE_FROZEN_BEFORE_TEST'
    output.mkdir(parents=True, exist_ok=False)
    rows, arrays = [], {}
    for context in z['contexts']:
        if context['split'] not in splits:
            continue
        pair = []
        for action in [0, 1]:
            row, observation, _ = trial(context, action)
            if observation is not None:
                for encoder in ['legacy', 'projection']:
                    x = encode(**observation, action=action, encoder=encoder,
                        elapsed=context['wait'] + context['start'] if clock else None)
                    arrays[f'{context["id"]}_{action}_{encoder}'] = x
                    row[encoder + '_input_sha256'] = hashlib.sha256(x.tobytes()).hexdigest()
            rows.append(row)
            pair.append(row)
        assert pair[0]['valid'] == pair[1]['valid']
        if pair[0]['valid']:
            assert pair[0]['frame_sha256'] == pair[1]['frame_sha256']
    repeat_context = next(c for c in z['contexts'] if c['id'] == next(r['id'] for r in rows if r['valid']))
    repeat, _, _ = trial(repeat_context, 0)
    original = next(r for r in rows if r['id'] == repeat['id'] and r['action'] == 0)
    assert all(original[k] == v for k, v in repeat.items())
    np.savez_compressed(output / 'inputs.npz', **arrays)
    save(output / 'EXPERIENCE.json', dict(status='COLLECTED', splits=splits, manifest_sha256=sha(manifest),
        source_sha256=sha(__file__), repeatability=True, actor_game_RAM=False, observed_clock=clock, rows=rows))
    print(json.dumps(dict(status='COLLECTED', splits=splits, valid_contexts=sum(r['valid'] for r in rows)//2,
        invalid_contexts=sum(not r['valid'] for r in rows)//2)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--splits', nargs='+', choices=['train', 'dev', 'test'], required=True)
    p.add_argument('--lock', type=Path)
    p.add_argument('--clock', action='store_true')
    a = p.parse_args()
    collect(a.manifest.resolve(), a.output.resolve(), a.splits, a.lock, a.clock)
