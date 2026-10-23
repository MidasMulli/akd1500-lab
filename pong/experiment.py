"""Deterministic physics, visible-only motion encoding and a frozen staged protocol.

The host supplies position, measured velocity and the visibly marked court. The
AKD predicts one of four future landing bands. Acceleration and future outcomes
never enter the acting encoder. This is a small supervised predictor, not a claim
of pixel learning, reinforcement learning or native simulation on the chip.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

N, U, NW, CLASSES, PROTECTED = 1024, 160, 64, 5, 4
PER_CLASS = U // CLASSES
SEEDS = [85201, 85202, 85203, 85204, 85205]
DT, FLIGHT_STEPS = 1 / 120, 144


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(observation):
    # Fine population codes for observed height and velocity: two 32-bit
    # windows in a court-specific 512-bit bank. No future state is encoded.
    y = float(observation['y']); v = float(observation['vy']); court = int(observation['court'])
    assert 0 <= y <= 1 and -1 <= v <= 1 and court in (0, 1)
    iy = min(224, max(0, round(y * 224)))
    iv = min(224, max(0, round((v + 1) * 112)))
    x = np.zeros(N, np.uint8)
    x[court * 512 + iy:court * 512 + iy + 32] = 1
    x[court * 512 + 256 + iv:court * 512 + 256 + iv + 32] = 1
    assert x.sum() == NW
    return x


def acceleration(court, changed):
    return (-.80 if changed else .80) if court == 0 else .30


def flight(observation, changed):
    y, v = float(observation['y']), float(observation['vy'])
    a = acceleration(observation['court'], changed)
    states = [[.08, y, v]]
    for k in range(FLIGHT_STEPS):
        v += a * DT; y += v * DT
        while y < 0 or y > 1:
            if y < 0: y, v = -y, -v
            elif y > 1: y, v = 2 - y, -v
        states.append([.08 + .84 * (k + 1) / FLIGHT_STEPS, y, v])
    return np.array(states, np.float64)


def outcome(observation, changed):
    y = float(flight(observation, changed)[-1, 1])
    return min(3, int(y * 4)), y


def decode(scores):
    values = np.asarray(scores).reshape(CLASSES, PER_CLASS)[:4].max(axis=1)
    # Fixed lowest-index tie rule; no target-dependent/recency tie breaking.
    return int(values.argmax())


def make_cases(seed, count_per_court):
    rng = np.random.default_rng(seed); rows = []
    for court in [0, 1]:
        for i in range(count_per_court):
            rows.append(dict(id=f'{seed}-{court}-{i}', court=court,
                y=float(rng.uniform(.07, .93)), vy=float(rng.uniform(-.85, .85))))
    return rows


def design():
    # Development and held-out banks use different draws. They are generated
    # prospectively; no output or trajectory is used to select the representation.
    return dict(version=1, seeds=SEEDS, development_seed=85201,
        train=make_cases(91001, 64), adapt=make_cases(91002, 64),
        development=make_cases(91003, 48), confirmation=make_cases(91004, 96),
        prefix_epochs=4, adaptation_epochs=6,
        stages=['initial_training', 'old_physics_probe', 'changed_before_feedback',
                'adaptation_with_matched_queries', 'changed_after_feedback', 'unchanged_court_retention'],
        gates=dict(old_accuracy_min=.65, changed_accuracy_min=.65,
            changed_gain_min=.10, changed_mae_ratio_max=.80,
            retention_loss_max=.10, minimum_paired_seed_wins=4),
        claims=dict(inputs='host-supplied observed height, velocity and visible court identity',
            prediction='one of four future landing bands', feedback='observed landing band after flight',
            plasticity='native binary connection updates; complete visible RAM recorded after every forward',
            frozen='identical initialization AND identical native training/query prefix, then protected labels only',
            controls='equal query count, order, examples and feedback opportunities; no hidden-state restore',
            controller='host paddle logic may follow a predicted band only after prediction gate passes',
            limitations=['One chip; repeated initialization seeds are not independent hardware',
                'Fixed discretized observed-motion encoder and four-band decoder are supplied',
                'No CPU/GPU energy advantage, electrical spike timing or full-chip use claim']))


def schedule(plan, split='development', frozen=False):
    rows = []
    def add(kind, cases, changed, teaching=False):
        for case in cases:
            target, landing = outcome(case, changed)
            rows.append(dict(index=len(rows), kind=kind, observation=case,
                changed=changed, truth=target, landing=landing,
                label=target if teaching and not frozen else PROTECTED))
    # Both arms teach the same prefix, including native selector history.
    for epoch in range(plan['prefix_epochs']):
        ordered = np.random.default_rng(92000 + epoch).permutation(len(plan['train']))
        for i in ordered:
            case = plan['train'][int(i)]; target, landing = outcome(case, False)
            rows.append(dict(index=len(rows), kind='initial_training', observation=case,
                changed=False, truth=target, landing=landing, label=target))
    add('old_physics_probe', plan[split], False)
    add('changed_before_feedback', [c for c in plan[split] if c['court'] == 0], True)
    for epoch in range(plan['adaptation_epochs']):
        ordered = np.random.default_rng(93000 + epoch).permutation(len(plan['adapt']))
        add('adaptation_with_matched_queries', [plan['adapt'][int(i)] for i in ordered], True, True)
    add('changed_after_feedback', [c for c in plan[split] if c['court'] == 0], True)
    add('unchanged_court_retention', [c for c in plan[split] if c['court'] == 1], True)
    return rows


def metrics(rows, predictions, kind):
    chosen = [(r, p) for r, p in zip(rows, predictions) if r['kind'] == kind]
    return dict(n=len(chosen), correct=sum(p == r['truth'] for r, p in chosen),
        accuracy=float(np.mean([p == r['truth'] for r, p in chosen])),
        mean_absolute_landing_error=float(np.mean([abs((p + .5) / 4 - r['landing']) for r, p in chosen])))


def pair_gate(plan, full, frozen):
    gates = plan['gates']; f, z = full['metrics'], frozen['metrics']
    after = f['changed_after_feedback']; control = z['changed_after_feedback']
    retention_before = full['old_court_b_accuracy']
    checks = dict(shared_training_state=full['trained_ram_sha256'] == frozen['trained_ram_sha256'],
        shared_prefix_scores=full['prefix_scores_sha256'] == frozen['prefix_scores_sha256'],
        frozen_ram_unchanged=frozen['post_prefix_connection_bit_edits'] == 0,
        old_accuracy=f['old_physics_probe']['accuracy'] >= gates['old_accuracy_min'],
        changed_accuracy=after['accuracy'] >= gates['changed_accuracy_min'],
        changed_gain=after['accuracy'] - control['accuracy'] >= gates['changed_gain_min'],
        changed_landing_error=after['mean_absolute_landing_error'] <= gates['changed_mae_ratio_max'] * control['mean_absolute_landing_error'],
        retention=f['unchanged_court_retention']['accuracy'] >= retention_before - gates['retention_loss_max'])
    return dict(status='PASS' if all(checks.values()) else 'STOP_PREDICTION_GATE', checks=checks,
        changed_gain=after['accuracy'] - control['accuracy'])


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('output', type=Path); a = p.parse_args()
    save(a.output, design())
