"""Frozen, synthetic signal-routing task; no hardware access or game-state features."""
from pathlib import Path
import argparse
import json
import numpy as np

N, U, NW, CLASSES, THRESHOLD = 512, 64, 32, 4, 27
SEEDS = [97101, 97102, 97103]
ARMS = ['recognition_repair', 'frozen', 'routing_only']


def dataset():
    rng = np.random.default_rng(97201)
    signatures = [[rng.choice(N, N // 2, replace=False).tolist() for _ in range(4)] for _ in range(2)]
    # This environment mapping is a teacher/evaluator target, never an input feature.
    destinations = np.random.default_rng(97202).permutation(4).tolist()
    return dict(signatures=signatures, destinations=destinations)


def noisy(signature, rng):
    x = np.zeros(N, np.uint8); x[signature] = 1
    remove = rng.choice(np.flatnonzero(x), 8, replace=False)
    insert = rng.choice(np.flatnonzero(1 - x), 8, replace=False)
    x[remove] = 0; x[insert] = 1
    assert x.sum() == 256
    return x


def schedule(arm):
    assert arm in ARMS
    data = dataset(); xs, rows = [], []
    def add(phase, domain, per_class, seed, mask, teacher):
        rng = np.random.default_rng(seed)
        order = np.repeat(np.arange(4), per_class); rng.shuffle(order)
        for identity in order:
            identity = int(identity); target = data['destinations'][identity]
            xs.append(noisy(data['signatures'][domain][identity], rng))
            rows.append(dict(index=len(rows), phase=phase, domain=domain, identity=identity,
                             destination=target, label=identity if teacher == 'identity' else target,
                             learning_mask=mask))
    # All arms replay this exact native prefix, including all selector histories.
    add('recognition_training', 0, 64, 97301, 1, 'identity')
    add('routing_training', 0, 64, 97302, 2, 'destination')
    add('old_probe', 0, 16, 97303, 0, 'identity')
    add('changed_before', 1, 16, 97304, 0, 'identity')
    mask = {'recognition_repair': 1, 'frozen': 0, 'routing_only': 2}[arm]
    add('repair', 1, 64, 97305, mask, 'destination' if arm == 'routing_only' else 'identity')
    add('changed_after', 1, 16, 97306, 0, 'identity')
    add('old_retention', 0, 16, 97307, 0, 'identity')
    assert len(rows) == 1024
    return np.asarray(xs), rows


def initial(seed):
    rng = np.random.default_rng(seed); result = []
    for _ in range(2):
        ram = [0] * (2 + U * (4 + N // 32))
        for j in range(U):
            base = 2 + j * (4 + N // 32); ram[base + 3] = NW // 2
            for a in rng.choice(N, NW, replace=False):
                ram[base + 4 + int(a) // 32] |= 1 << (int(a) % 32)
        result.append(ram)
    return result


def metrics(rows, outputs):
    prediction = np.asarray(outputs).argmax(axis=1)
    return {phase: dict(n=len(ii), correct=int(sum(prediction[i] == rows[i]['destination'] for i in ii)),
                        accuracy=float(np.mean([prediction[i] == rows[i]['destination'] for i in ii])))
            for phase in ['old_probe', 'changed_before', 'changed_after', 'old_retention']
            for ii in [[i for i, r in enumerate(rows) if r['phase'] == phase]]}


def gate(full, frozen):
    a, b = full['metrics'], frozen['metrics']
    checks = dict(old_accuracy=a['old_probe']['accuracy'] >= .85,
                  changed_accuracy=a['changed_after']['accuracy'] >= .85,
                  changed_gain=a['changed_after']['accuracy'] - b['changed_after']['accuracy'] >= .30,
                  old_retention=a['old_retention']['accuracy'] >= .80,
                  downstream_unchanged=full['downstream_unchanged_during_repair'])
    return dict(status='PASS' if all(checks.values()) else 'STOP_TASK_GATE', checks=checks)


def write(study):
    protocol = dict(version=1, task='Noisy beacon identity to learned destination through a resident categorical message',
                    seeds=SEEDS, development_seed=SEEDS[0], arms=ARMS,
                    dimensions=dict(inputs=N, neurons_per_bank=U, connections=NW, classes=CLASSES,
                                    activation_threshold=THRESHOLD, plastic_banks=2),
                    dataset=dataset(), intended_native_layers=['recognition', 'pairwise_comparison',
                    'categorical_message', 'routing', 'destination_support'],
                    labels='Different identity and destination teachers, enabled on exactly one bank at a time',
                    input='512 binary synthetic sensor features, 256 active; no identity, domain, destination or teacher in input',
                    caps=dict(samples_per_arm=1024, first_smoke=64, initial_physical_admission='smoke only'),
                    thresholds=dict(old_accuracy=.85, changed_accuracy=.85, gain=.30, old_retention=.80),
                    limits=['Synthetic task and one chip', 'Host supplies sensors, teachers and mode transitions',
                            'Native graph is feed-forward, with fixed supplied message decoding',
                            'No electrical power, recurrent autonomy or whole-chip peak claim',
                            'Intermediate responses are reconstructed unless directly measured',
                            'No hidden-state restore; deterministic complete prefix replay'])
    (study / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (study / 'STATE.json').write_text(json.dumps(dict(stage='SOURCE_BUILD', research_admissions_closed=False,
                    admitted_runs=[], reviewed_runs=[]), indent=2) + '\n')
    for arm in ARMS:
        x, rows = schedule(arm)
        (study / f'{arm}.inputs').write_bytes(x.tobytes())
        (study / f'{arm}.labels').write_bytes(np.asarray([r['label'] for r in rows], '<i4').tobytes())
        (study / f'{arm}.masks').write_bytes(bytes(r['learning_mask'] for r in rows))
        (study / f'{arm}.rows.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('study', type=Path); write(p.parse_args().study)
