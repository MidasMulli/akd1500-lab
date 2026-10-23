"""Frozen task identities and two authored RGB-only representations."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

LEVELS = ['1-1', '2-1', '3-1', '4-1']
SEEDS = list(range(82901, 82906))
MATRIX = np.random.default_rng(82800).choice(np.array([-1, 1], np.int16), size=(384, 256))


def design():
    splits = dict(train=([0, 16, 32, 48], [72, 88, 104, 120, 136]),
                  dev=([4, 20, 36, 52], [80, 96, 112, 128]),
                  test=([8, 24, 40, 56], [76, 84, 92, 100, 108, 116, 124, 132]))
    return dict(schema=2, reset_seed=82800, contexts=[dict(id=f'{split}_{level}_{wait}_{start}',
        split=split, level=level, wait=wait, start=start)
        for split, (waits, starts) in splits.items() for level in LEVELS for wait in waits for start in starts])


def encode(frame, previous, sky, action, encoder, elapsed=None):
    if encoder == 'legacy':
        def occupied(a):
            mask = np.max(np.abs(a[96:224].astype(np.int16) - sky.astype(np.int16)), axis=2) > 25
            return (mask.reshape(8, 16, 16, 16).sum(axis=(1, 3)) >= 24).reshape(-1)
        bits = np.concatenate([occupied(frame), occupied(previous)]).astype(np.uint8)
    else:
        assert encoder == 'projection'
        views = [np.asarray(Image.fromarray(a[96:208]).resize((8, 8), Image.Resampling.BOX), np.int16) - sky for a in [frame, previous]]
        projections = np.concatenate([a.reshape(-1) for a in views]).astype(np.int32) @ MATRIX.astype(np.int32)
        bits = np.zeros(256, np.uint8)
        bits[np.argsort(projections, kind='stable')[-32:]] = 1
    if elapsed is not None:
        clock = np.zeros(64, np.uint8)
        clock[(int(elapsed) // 4 + np.arange(8)) % 64] = 1
        bits = np.concatenate([bits, np.tile(clock, 4)])
    x = np.zeros(len(bits) * 2, np.uint8)
    x[len(bits) * action:len(bits) * (action + 1)] = bits
    return x


def initial(seed, n=512):
    rng = np.random.default_rng(seed)
    w = np.zeros((n, 192), np.int8)
    for j in range(192):
        action = (j % 64) % 2
        w[(n//2) * action + rng.choice(n//2, 32, replace=False), j] = 1
    return w


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')
