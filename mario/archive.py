"""Publish measured scores/RAM without licensed programs, runtime or game ROMs."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from verify import read_words


def export(study, fresh):
    report = json.loads((study / 'evidence/RECOUNT.json').read_text())
    actual = json.loads((fresh / 'FRESH_EPISODES.json').read_text())
    assert actual['status'] == 'FRESH_EPISODES_MATCH' and actual['trials'] == 72
    assert actual['native_recount_sha256'] == hashlib.sha256((study / 'evidence/RECOUNT.json').read_bytes()).hexdigest()
    arrays, fixtures, runs = {}, {}, []
    for row in report['runs']:
        run = study / 'raw' / row['run']
        receipt = json.loads((run / 'RECEIPT.json').read_text())
        prefix = ROOT / receipt['fixture']
        key = prefix.name
        if key not in fixtures:
            fixtures[key] = dict(inputs=512, neurons=96, num_weights=32, classes=3)
            for ext, dtype in [('inputs', 'u1'), ('labels', '<u4'), ('ram', '<u4')]:
                arrays[f'fixture__{key}__{ext}'] = np.frombuffer(prefix.with_suffix('.' + ext).read_bytes(), dtype=dtype).copy()
        for ext in ['scores', 'ram']:
            for mode in ['native', 'cpu']:
                arrays[f"{run.name}__{mode}_{ext}"] = np.array(read_words(run / f'{mode}.{ext}'), np.uint32)
        arrays[run.name + '__snapshots'] = np.array([read_words(run / f'native.snapshot_{i}.ram') for i in range(receipt['samples'])], np.uint32)
        runs.append(dict(name=run.name, fixture_key=key, samples=receipt['samples'], learning=receipt['learning'],
                         program_sha256=receipt['program_sha256'], binary_sha256=receipt['binary_sha256'], **row))
    archive = ROOT / 'results/2026-10-23-mario.npz'
    np.savez_compressed(archive, **arrays)
    fresh_public = ROOT / 'results/2026-10-23-mario-fresh.json'
    fresh_public.write_bytes((fresh / 'FRESH_EPISODES.json').read_bytes())
    z = dict(status='SMALL_SKILL_GATE_PASS_WITH_MIXED_FROZEN_CONTROL',
        physical_chips=1, native_sessions=10, native_forwards=752,
        seed_results=report['cohorts'], successes=dict(learned=21, frozen=20, action_only=15, best_constant=18, opportunities=24),
        fresh_emulator_trials=actual['trials'], fresh_outcomes_match=True,
        encoded_train_test_overlap=0, learned_visual_representation=False, actor_uses_game_RAM=False,
        description='Native consequence associations from recorded pixel/action experience; fixed visual encoder, action ranking and two motor primitives.',
        limits=['Eight nearby starts at the first encounter in one level',
                'Three initialization seeds on the same chip and same examples, not independent environments',
                'One frozen seed scores 8/8 and exceeds its trained counterpart; aggregate benefit is only one start',
                'Passing the predeclared two-of-three screen is not a robust generalization or statistical result',
                'Fresh episodes execute saved native-selected macros; no frame-by-frame native control or autonomous full-game RL',
                'Exact CPU reference reproduces every score and RAM word; no CPU, energy or power-loss persistence advantage'],
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        dataset_sha256=report['dataset_sha256'], independent_verifier_sha256=report['independent_verifier_sha256'],
        fresh_replay_sha256=actual['source_sha256'], fixtures=fixtures, runs=runs)
    z['fresh_receipt_sha256'] = hashlib.sha256(fresh_public.read_bytes()).hexdigest()
    (ROOT / 'results/2026-10-23-mario.json').write_text(json.dumps(z, indent=2) + '\n')
    print(json.dumps({k: v for k, v in z.items() if k not in ['fixtures', 'runs', 'limits']}))


def materialize(root, public=ROOT):
    """Restore only the public raw arrays into the existing independent auditor's format."""
    z = json.loads((public / 'results/2026-10-23-mario.json').read_text())
    archive = public / 'results/2026-10-23-mario.npz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == z['archive_sha256']
    data = np.load(archive, allow_pickle=False)
    paths = []
    for r in z['runs']:
        key = r['fixture_key']
        prefix = root / 'fixtures' / key
        prefix.parent.mkdir(exist_ok=True)
        for ext in ['inputs', 'labels', 'ram']:
            prefix.with_suffix('.' + ext).write_bytes(data[f'fixture__{key}__{ext}'].tobytes())
        prefix.with_suffix('.meta').write_text('512 96 32 3\n')
        run = root / 'raw' / r['name']
        run.mkdir(parents=True)
        for mode in ['native', 'cpu']:
            for ext in ['scores', 'ram']:
                (run / f'{mode}.{ext}').write_bytes(data[f"{r['name']}__{mode}_{ext}"].astype('<u4').tobytes())
        for i, ram in enumerate(data[r['name'] + '__snapshots']):
            (run / f'native.snapshot_{i}.ram').write_bytes(ram.astype('<u4').tobytes())
        receipt = dict(status='PASS', device_released=True, fixture=str(prefix.relative_to(root)),
            samples=r['samples'], learning=r['learning'],
            hashes={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in run.iterdir()})
        (run / 'RECEIPT.json').write_text(json.dumps(receipt))
        paths.append(run)
    return paths


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('study', type=Path)
    p.add_argument('fresh', type=Path)
    a = p.parse_args()
    export(a.study.resolve(), a.fresh.resolve())
