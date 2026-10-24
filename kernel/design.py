"""Frozen redundant-input compaction equivalence domain and owned kernels."""
import numpy as np

SEED = 98501


def weights(u):
    rng = np.random.default_rng(SEED); w = np.zeros((128, u), np.int8)
    for j in range(u): w[rng.choice(128, 32, replace=False), j] = 1
    return w


def stream():
    rng = np.random.default_rng(98502); xs, rows = [], []
    for block in range(3):
        for density in rng.permutation([0, 64, 128]):
            for repeat in range(16):
                x = np.zeros(128, np.uint8)
                if density: x[rng.choice(128, int(density), replace=False)] = 1
                xs.append(np.tile(x, 4)); rows.append(dict(block=block, density=int(density), label=0, learning=False))
    return np.asarray(xs), rows


def expected(x, u):
    assert x.shape[1] == 512 and all(np.array_equal(x[:, :128], x[:, k*128:(k+1)*128]) for k in range(4)), 'four-copy input domain'
    activity = x[:, :128].astype(np.int32) @ weights(u).astype(np.int32) > 15
    return activity.reshape(len(x), 2, u // 2).sum(axis=2).astype(np.int32)
