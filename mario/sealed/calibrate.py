"""Read-only CPU calibration; never opens the physical device or the test bank."""
from pathlib import Path
import argparse
import json
import subprocess
import numpy as np
from bank import save, sha

ROOT = Path(__file__).resolve().parents[2]


def calibrate(fixtures, output):
    output.mkdir(parents=True, exist_ok=False)
    build = json.loads((fixtures / 'BUILD.json').read_text())
    assert build['phase'] == 'calibration'
    rows = []
    for row in build['rows']:
        prefix = fixtures / row['prefix']
        target = output / row['prefix']
        command = [str(ROOT / 'build/learning_bench'), 'cpu', str(prefix), str(target), str(row['samples']), '1', '1', '0']
        r = subprocess.run(command, capture_output=True, text=True, timeout=60)
        target.with_suffix('.stdout').write_text(r.stdout)
        target.with_suffix('.stderr').write_text(r.stderr)
        assert r.returncode == 0, r.stderr
        meta = json.loads(prefix.with_suffix('.json').read_text())
        scores = np.fromfile(str(target) + '.scores', '<i4').reshape(row['samples'], 192)
        count = row['evaluation_samples']
        margins = lambda a: (a[:, 64:128].max(axis=1) - a[:, :64].max(axis=1)).reshape(-1, 2)
        actions = margins(scores[-count:]).argmax(axis=1)
        truth = np.array([r['success'] for r in meta['evaluation_rows']]).reshape(-1, 2)
        rows.append(dict(**row, actions=actions.tolist(), successes=int(truth[np.arange(len(actions)), actions].sum()),
            contexts=len(actions), best_constant=int(truth.sum(axis=0).max()), raw_scores_sha256=sha(str(target) + '.scores')))
    summaries = []
    for encoder in ['legacy', 'projection']:
        full = [r for r in rows if r['encoder'] == encoder and r['arm'] == 'full']
        frozen = [r for r in rows if r['encoder'] == encoder and r['arm'] == 'frozen']
        wins = sum(a['successes'] > max(b['successes'], a['best_constant']) for a, b in zip(full, frozen))
        summaries.append(dict(encoder=encoder, mean_full=float(np.mean([r['successes'] for r in full])),
            mean_frozen=float(np.mean([r['successes'] for r in frozen])), seeds_beating_both=wins,
            aliases=max(r['opposite_label_aliases'] for r in full), full=[r['successes'] for r in full],
            frozen=[r['successes'] for r in frozen], best_constant=full[0]['best_constant'], contexts=full[0]['contexts']))
    selected = max(summaries, key=lambda r: (r['mean_full'], r['encoder'] == 'legacy'))
    admitted = selected['seeds_beating_both'] >= 2 and selected['mean_full'] > selected['mean_frozen'] and selected['aliases'] == 0
    save(output / 'CALIBRATION.json', dict(status='PASS_FOR_SEALED_TEST' if admitted else 'STOP_CALIBRATION_GATE',
        selected_encoder=selected['encoder'], summaries=summaries, runs=rows, physical_device_opened=False))
    print(json.dumps(dict(status='PASS_FOR_SEALED_TEST' if admitted else 'STOP_CALIBRATION_GATE', summaries=summaries)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('fixtures', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    calibrate(a.fixtures.resolve(), a.output.resolve())
