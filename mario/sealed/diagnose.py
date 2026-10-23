"""Post-hoc development-only CPU ablations; no new pipeline selection or hardware."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import numpy as np
from bank import save, sha

ROOT = Path(__file__).resolve().parents[2]


def diagnose(study):
    out = study / 'cpu_runs/posthoc01'
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'PLAN.json', dict(status='POSTHOC_DEVELOPMENT_ONLY', arms=['action_only', 'shuffled_feedback'],
        seeds=[82811, 82812, 82813], source_sha256=sha(__file__), physical_device_opened=False,
        test_outcomes_collected=False, note='Explain the failed development gate; do not select another pipeline.'))
    rows = []
    for seed in [82811, 82812, 82813]:
        original = study / f'fixtures-cal01/projection_s{seed}_full'
        meta = json.loads(original.with_suffix('.json').read_text())
        inputs = np.frombuffer(original.with_suffix('.inputs').read_bytes(), np.uint8).reshape(-1, 1024)
        labels = np.frombuffer(original.with_suffix('.labels').read_bytes(), '<i4')
        truth = np.array([r['success'] for r in meta['evaluation_rows']]).reshape(64, 2)
        for arm in ['action_only', 'shuffled_feedback']:
            prefix = out / f's{seed}_{arm}'
            x, y = inputs.copy(), labels.copy()
            if arm == 'action_only':
                actions = (x[:, :512].sum(1) == 0).astype(int); x[:] = 0
                for i, action in enumerate(actions): x[i, 512*action:512*action+64] = 1
            else:
                y[128:-128] = y[128:-128][np.random.default_rng(82999).permutation(len(y)-256)]
            for suffix in ['meta', 'ram']:shutil.copy2(original.with_suffix('.'+suffix), prefix.with_suffix('.'+suffix))
            prefix.with_suffix('.inputs').write_bytes(x.tobytes()); prefix.with_suffix('.labels').write_bytes(y.tobytes())
            result = subprocess.run([str(ROOT/'build/learning_bench'), 'cpu', str(prefix), str(prefix), str(len(y)), '1', '1', '0'], capture_output=True, text=True, timeout=60)
            prefix.with_suffix('.stdout').write_text(result.stdout); prefix.with_suffix('.stderr').write_text(result.stderr); assert result.returncode == 0
            scores = np.fromfile(prefix.with_suffix('.scores'), '<i4').reshape(-1, 192)[-128:]
            choice = (scores[:,64:128].max(1)-scores[:,:64].max(1)).reshape(64,2).argmax(1)
            rows.append(dict(seed=seed, arm=arm, contexts=64, successes=int(truth[np.arange(64),choice].sum()),
                unique_selected_actions=sorted(set(map(int,choice))), raw_scores_sha256=sha(prefix.with_suffix('.scores'))))
    report = dict(status='POSTHOC_CPU_MODEL_DIAGNOSTIC', rows=rows, physical_device_opened=False,
                  test_outcomes_collected=False, source_sha256=sha(__file__))
    save(out/'REPORT.json',report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);a=p.parse_args();diagnose(a.study.resolve())
