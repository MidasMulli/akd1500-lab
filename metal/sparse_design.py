"""Frozen sparse-work inputs and selected-connection populations; no device API."""
import numpy as np

SEED = 98201
CELLS = [(n, u, nw) for u in [64, 256] for n in [128, 512, 1024] for nw in [8, 32, 64]]


def weights(n, u, nw):
    rng = np.random.default_rng(SEED)
    w = np.zeros((n, u), np.int8)
    for j in range(u): w[rng.permutation(n)[:nw], j] = 1
    return w


def stream(n):
    rng = np.random.default_rng(SEED + 17 + n)
    x = np.zeros((144, n), np.uint8)
    rows = []
    for i in range(len(x)):
        x[i, rng.choice(n, n // 2, replace=False)] = 1
        rows.append(dict(phase='initial_inference' if i < 96 else 'native_learning' if i < 128 else 'retained_inference',
                         learning=96 <= i < 128, label=i % 2, block=i // 32, density=n // 2))
    return x, rows
