"""Publish the failed admission gate and verified physical development diagnostics."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from verify import verify, read_words


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(study, full, frozen):
    state = json.loads((study / 'STATE.json').read_text())
    assert state['research_admissions_closed'] and not state['test_outcomes_collected']
    calibration = json.loads((study / 'cpu_runs/calibration01/CALIBRATION.json').read_text())
    assert calibration['status'] == 'STOP_CALIBRATION_GATE'
    arrays, fixtures, runs, cohort = {}, {}, [], []
    for run in sorted((study / 'raw').iterdir()):
        rec = json.loads((run / 'RECEIPT.json').read_text())
        audit = verify(ROOT, run)
        prefix = ROOT / rec['fixture']
        key = run.name
        n, u, nw, classes = map(int, prefix.with_suffix('.meta').read_text().split())
        fixtures[key] = dict(inputs=n, neurons=u, num_weights=nw, classes=classes)
        for suffix, dtype in [('inputs', 'u1'), ('labels', '<u4'), ('ram', '<u4')]:
            arrays[f'fixture__{key}__{suffix}'] = np.frombuffer(prefix.with_suffix('.' + suffix).read_bytes(), dtype=dtype).copy()
        for mode in ['native', 'cpu']:
            for suffix in ['scores', 'ram']:
                arrays[f'{key}__{mode}_{suffix}'] = np.array(read_words(run / f'{mode}.{suffix}'), np.uint32)
        arrays[key + '__snapshots'] = np.array([read_words(run / f'native.snapshot_{i}.ram') for i in range(rec['samples'])], np.uint32)
        row = dict(name=key, fixture_key=key, samples=rec['samples'], learning=1, audit=audit,
                   binary_sha256=rec['binary_sha256'], program_sha256=rec['program_sha256'])
        if key.startswith('diag_'):
            meta = json.loads(prefix.with_suffix('.json').read_text())
            scores = arrays[key + '__native_scores'].astype(np.int32).reshape(-1, 192)[-128:]
            margins = (scores[:, 64:128].max(1) - scores[:, :64].max(1)).reshape(-1, 2)
            actions = margins.argmax(1)
            truth = np.array([r['success'] for r in meta['evaluation_rows']]).reshape(-1, 2)
            row.update(seed=meta['seed'], arm=meta['arm'], actions=actions.tolist(),
                successes=int(truth[np.arange(64), actions].sum()), best_constant=int(truth.sum(0).max()),
                evaluation_ids=[r['id'] for r in meta['evaluation_rows'][::2]],
                evaluation_outcomes=truth.astype(int).tolist())
        runs.append(row)
    for seed in [82811, 82812, 82813]:
        arms = {r['arm']: r for r in runs if r.get('seed') == seed}
        assert set(arms) == {'full', 'frozen'}
        cohort.append(dict(seed=seed, learned=arms['full']['successes'], frozen=arms['frozen']['successes'],
                           best_constant=arms['full']['best_constant'], contexts=64))
    archive = ROOT / 'results/2026-10-23-mario-sealed.npz'
    np.savez_compressed(archive, **arrays)
    live = {}
    for name, directory in [('learned', full), ('frozen', frozen)]:
        audit = json.loads((directory / 'INDEPENDENT_AUDIT.json').read_text())
        receipt = json.loads((directory / 'LIVE_RECEIPT.json').read_text())
        assert audit['status'] == 'INDEPENDENT_LIVE_AUDIT_PASS'
        live[name] = dict(receipt=receipt, audit=audit)
    result = dict(status='DEVELOPMENT_GATE_FAILED_SEALED_TEST_UNOPENED',
        parent_status='Both RGB encoders fail development gate; each aliases one contradictory input',
        revision='Observed elapsed-time code plus class-balanced training; causal effects not separately isolated',
        calibration=calibration['summaries'], native_development=cohort,
        development_totals=dict(learned=sum(r['learned'] for r in cohort), frozen=sum(r['frozen'] for r in cohort),
            best_constant=sum(r['best_constant'] for r in cohort), opportunities=192),
        sealed_test_contexts=128, test_outcomes_collected=False, physical_chips=1,
        physical_sessions=len(runs), verified_native_forwards=sum(r['samples'] for r in runs),
        archive_sha256=digest(archive), fixtures=fixtures, runs=runs, live=live,
        limitations=['Development data were used to select and repair the representation; these are not sealed-test results',
            'Three initializations, one chip, same 64 development contexts; not independent worlds',
            'Best constant exceeds the learned aggregate; no useful Mario advantage established',
            'Fixed RGB/time encoder, score ranking, prefixes and 90-frame motor primitives are supplied',
            'Shadow queries sample responses without changing the committed macro',
            'Colors show measured match values; no electrical spike timing, full-game RL, CPU/energy or power-loss claim'])
    (ROOT / 'results/2026-10-23-mario-sealed.json').write_text(json.dumps(result, indent=2) + '\n')
    events = [json.loads(line) for line in (full / 'events.jsonl').read_text().splitlines()]
    selected = [r for r in events if r['kind'] != 'preparation']
    game = np.load(full / 'game.npz', allow_pickle=False)
    frames = [r for r in selected if r['kind'] == 'frame']
    origin_lower = next(r['mono_ns'] for r in selected if r['kind'] == 'game_start')
    origin_upper = min(r['mono_ns'] - int(game['mono_offsets_ns'][r['frame_id']]) for r in frames)
    assert origin_lower <= origin_upper
    shutil.copy2(full / 'game.npz', ROOT / 'visual/mario-live-game.npz')
    replay = dict(status='RECORDED_LIVE_PHYSICAL_DATA', events=selected,
        origin_lower_ns=origin_lower, origin_upper_ns=origin_upper,
        alignment_uncertainty_ns=origin_upper - origin_lower, device_run='live_full01',
        learning_np=json.loads((study / 'fixtures-cal01/projection_s82811_full.json').read_text())['learning_np'],
        game_sha256=digest(ROOT / 'visual/mario-live-game.npz'),
        raw_archive_sha256=digest(archive), native_development=cohort,
        receipt=live['learned']['receipt'], audit=live['learned']['audit'],
        renderer_timing='Use conservative origin lower bound; never display a response received after the rendered time')
    (ROOT / 'visual/mario-live-data.json').write_text(json.dumps(replay, separators=(',', ':')) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['fixtures', 'runs', 'live', 'limitations', 'calibration']}))


def materialize(root, public=ROOT, stem='2026-10-23-mario-sealed'):
    report = json.loads((public / f'results/{stem}.json').read_text())
    source = public / f'results/{stem}.npz'
    assert digest(source) == report['archive_sha256']
    z = np.load(source, allow_pickle=False)
    paths = []
    for row in report['runs']:
        key = row['fixture_key']; f = root / 'fixtures' / key
        f.parent.mkdir(exist_ok=True)
        for suffix in ['inputs', 'labels', 'ram']:
            f.with_suffix('.' + suffix).write_bytes(z[f'fixture__{key}__{suffix}'].tobytes())
        m = report['fixtures'][key]
        f.with_suffix('.meta').write_text(f"{m['inputs']} {m['neurons']} {m['num_weights']} {m['classes']}\n")
        d = root / 'raw' / row['name']; d.mkdir(parents=True)
        for mode in ['native', 'cpu']:
            for suffix in ['scores', 'ram']:
                (d / f'{mode}.{suffix}').write_bytes(z[f'{key}__{mode}_{suffix}'].astype('<u4').tobytes())
        for i, state in enumerate(z[key + '__snapshots']):
            (d / f'native.snapshot_{i}.ram').write_bytes(state.astype('<u4').tobytes())
        rec = dict(status='PASS', device_released=True, fixture=str(f.relative_to(root)), samples=row['samples'], learning=1,
            hashes={p.name: digest(p) for p in d.iterdir()})
        (d / 'RECEIPT.json').write_text(json.dumps(rec))
        paths.append(d)
    return paths


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('study', type=Path)
    p.add_argument('full', type=Path)
    p.add_argument('frozen', type=Path)
    a = p.parse_args()
    export(a.study.resolve(), a.full.resolve(), a.frozen.resolve())
