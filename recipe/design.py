"""Frozen delayed-feedback task and equal-resource policies; no device access."""
from pathlib import Path
import argparse
import json
import numpy as np
from pong.relearn.model import Reference

N, U, NW, CLASSES = 512, 128, 32, 4
SEEDS = [98401, 98402, 98403]
ARMS = ['stock', 'error_gate', 'split_stock', 'stable_plastic']
CHECKPOINTS = [0, 16, 32, 64, 128, 256]


def initial_weights(seed, split):
    rng = np.random.default_rng(seed)
    w = np.zeros((4, 2, 16, N), np.uint8)
    for c in range(4):
        for bank in range(2):
            for j in range(16): w[c, bank, j, rng.choice(N, NW, replace=False)] = 1
    if split: w = w.transpose(1, 0, 2, 3)
    return w.reshape(U, N)


def initial_ram(seed, split):
    w = initial_weights(seed, split); ram = np.zeros(2 + U * 20, np.uint32)
    for j in range(U):
        base = 2 + 20 * j; ram[base + 3] = NW // 2
        for bit in np.flatnonzero(w[j]): ram[base + 4 + int(bit) // 32] |= np.uint32(1 << (int(bit) % 32))
    return ram


def corpus():
    rng = np.random.default_rng(98410)
    old = np.zeros((4, 4, N), np.uint8)
    for c in range(4):
        for k in range(4): old[c, k, rng.choice(N, N // 2, replace=False)] = 1
    new = np.zeros_like(old)
    for c in range(4):
        for k in range(4):
            a = old[(c + 1) % 4, k].copy()
            remove = rng.choice(np.flatnonzero(a), 24, replace=False)
            insert = rng.choice(np.flatnonzero(1 - a), 24, replace=False)
            a[remove] = 0; a[insert] = 1; new[c, k] = a
    return old, new


def noisy(a, rng):
    b = a.copy(); remove = rng.choice(np.flatnonzero(b), 8, replace=False)
    insert = rng.choice(np.flatnonzero(1 - b), 8, replace=False)
    b[remove] = 0; b[insert] = 1
    return b


def schedule():
    old, new = corpus(); xs, rows = [], []
    def add(phase, patterns, repetitions, seed, teach, checkpoint=-1):
        rng = np.random.default_rng(seed); order = np.tile(np.arange(16), repetitions); rng.shuffle(order)
        for item in order:
            c, k = int(item) // 4, int(item) % 4
            xs.append(noisy(patterns[c, k], rng)); rows.append(dict(phase=phase, label=c, feedback=teach,
                    checkpoint=checkpoint, mode=k, index=len(rows)))
    add('old_training', old, 32, 98411, True)
    # Every probe is withheld from training and has no feedback.
    for checkpoint in CHECKPOINTS:
        if checkpoint:
            start = CHECKPOINTS[CHECKPOINTS.index(checkpoint) - 1]
            rng = np.random.default_rng(98412); order = np.tile(np.arange(16), 16); rng.shuffle(order)
            noise_rng = np.random.default_rng(98413)
            values = [noisy(new[int(i) // 4, int(i) % 4], noise_rng) for i in order]
            for i in range(start, checkpoint):
                xs.append(values[i]); rows.append(dict(phase='adaptation', label=int(order[i]) // 4, feedback=True,
                    checkpoint=-1, mode=int(order[i]) % 4, index=len(rows)))
        add('old_probe', old, 8, 98500 + checkpoint, False, checkpoint)
        add('new_probe', new, 8, 98600 + checkpoint, False, checkpoint)
    x = np.asarray(xs)
    assert len(x) == 2304 and np.all(x.sum(axis=1) == 256)
    return x, rows


def decode(scores, split):
    v = np.asarray(scores)
    support = v.reshape(2, 4, 16).max(axis=(0, 2)) if split else v.reshape(4, 32).max(axis=1)
    order = np.argsort(-support, kind='stable')
    return int(order[0]), int(support[order[0]]), int(support[order[0]] - support[order[1]])


def decision(arm, row, prediction, match, margin):
    if not row['feedback']: return False, -1
    split = arm in ['split_stock', 'stable_plastic']
    label = row['label'] + (4 if split and row['phase'] == 'adaptation' else 0)
    teach = row['phase'] == 'old_training' or arm in ['stock', 'split_stock'] or prediction != row['label'] or margin < 2 or match < 29
    return bool(teach), label


def metrics(rows, predictions, decisions):
    result = {}
    for point in CHECKPOINTS:
        result[str(point)] = {}
        for phase in ['old_probe', 'new_probe']:
            ids = [i for i, r in enumerate(rows) if r['phase'] == phase and r['checkpoint'] == point]
            result[str(point)][phase] = dict(correct=sum(predictions[i] == rows[i]['label'] for i in ids), count=len(ids))
    updates = sum(decisions); adapting = sum(d for d, r in zip(decisions, rows) if r['phase'] == 'adaptation')
    return dict(checkpoints=result, teaching_forwards=updates, adaptation_teaching_forwards=adapting,
                query_forwards=len(rows), total_native_forwards=len(rows) + updates)


def replay(seed, arm, ram=None, keep=False, policy=None):
    split = arm in ['split_stock', 'stable_plastic']; classes = 8 if split else 4
    net = Reference(initial_ram(seed, split) if ram is None else ram, N, U, NW, classes)
    x, rows = schedule(); predictions, gates, outputs, states = [], [], [], []
    stable = None
    for a, row in zip(x, rows):
        if row['phase'] != 'old_training' and stable is None: stable = net.ram[2:2 + 64 * 20]
        bits = sum(1 << int(k) for k in np.flatnonzero(a)); scores = [(v & bits).bit_count() for v in net.weights]
        pred, match, margin = decode(scores, split)
        do, label = (decision if policy is None else policy)(arm, row, pred, match, margin)
        if do: assert net.forward(a, label) == scores
        predictions.append(pred); gates.append(int(do))
        if keep: outputs.append(scores); states.append(net.ram[:])
        if split and stable is not None: assert net.ram[2:2 + 64 * 20] == stable, 'protected population changed'
    return dict(seed=seed, arm=arm, status='CPU_FORECAST_ONLY', metrics=metrics(rows, predictions, gates),
                predictions=predictions, gates=gates, outputs=outputs if keep else [], states=states if keep else [])


def gate(candidate, baseline):
    c, b = candidate['metrics'], baseline['metrics']
    final, reference = c['checkpoints']['256'], b['checkpoints']['256']
    checks = dict(old_prefix_accuracy=c['checkpoints']['0']['old_probe']['correct'] >= 115,
        final_old_accuracy=final['old_probe']['correct'] >= 109,
        final_new_accuracy=final['new_probe']['correct'] >= 109,
        retention_noninferior=final['old_probe']['correct'] >= reference['old_probe']['correct'] - 2,
        adaptation_noninferior=final['new_probe']['correct'] >= reference['new_probe']['correct'] - 2,
        fewer_update_jobs=c['adaptation_teaching_forwards'] <= .75 * b['adaptation_teaching_forwards'])
    return dict(status='PASS' if all(checks.values()) else 'STOP_RECIPE_GATE', checks=checks)


def forecast(study):
    results, gates = [], []
    for seed in SEEDS:
        rr = {arm: replay(seed, arm) for arm in ARMS}; results.extend(rr.values())
        for arm, baseline in [('error_gate', 'stock'), ('stable_plastic', 'split_stock')]:
            gates.append(dict(seed=seed, arm=arm, baseline=baseline, **gate(rr[arm], rr[baseline])))
    report = dict(status='CPU_FORECAST_ONLY', results=results, gates=gates)
    (study / 'CPU_FORECAST.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(gates=gates, metrics=[dict(seed=r['seed'], arm=r['arm'], **r['metrics']) for r in results])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('study', type=Path); forecast(p.parse_args().study)
