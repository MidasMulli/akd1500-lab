"""Fresh prefix replay and short, revisable motor primitives; truth is not an acting input."""
import gymnasium as gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np


def prefix_button(style, k):
    if style == 'walk': return 1
    phase = 0 if style == 'jump0' else 24
    return 2 if (k-phase) % 48 < 8 else 1


def button(action, k):
    return 2 if action == 0 and k < 8 else 1


def opened(context):
    env = JoypadSpace(gym.make(f'SuperMarioBros-{context["level"]}-v0', render_mode='rgb_array'), SIMPLE_MOVEMENT)
    frame, info = env.reset(seed=84200); history = [frame.copy()]; sky = frame[50, 0].astype(np.int16)
    valid = True
    for k in range(context['wait'] + context['start']):
        frame, _, terminated, truncated, info = env.step(0 if k < context['wait'] else prefix_button(context.get('style', 'walk'), k-context['wait']))
        history.append(frame.copy())
        if terminated or truncated or info.get('death'): valid = False; break
    return env, history, sky, info, valid


def observation(history, sky):
    return dict(frame=history[-1], previous=history[-9] if len(history) >= 9 else history[0], sky=sky)


def contexts(split):
    levels = ['1-1', '2-1', '3-1', '4-1']
    if split == 'train':
        return [dict(id=f'train_{level}_{wait}_{start}_{style}', level=level, wait=wait, start=start, style=style)
                for level in levels for wait in [0, 16] for start in [56, 80, 104, 128] for style in ['walk', 'jump0', 'jump24']]
    return [dict(id=f'{split}_{level}', level=level, wait=4 if split == 'dev' else 8, start=76 if split == 'dev' else 84, style='walk') for level in levels]
