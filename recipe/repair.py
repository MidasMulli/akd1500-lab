"""A separately frozen, two-candidate development repair; no native launch."""
from pathlib import Path
import argparse
import json
from recipe.design import replay, gate

DEVELOPMENT_SEED = 98401
CONFIRMATION_SEEDS = [98404, 98405, 98406]
POLICIES = [('conservative31', 31, 4), ('conservative32', 32, 4)]


def policy(match_threshold, margin_threshold):
    def decide(arm, row, prediction, match, margin):
        if not row['feedback']: return False, -1
        return bool(row['phase'] == 'old_training' or prediction != row['label'] or margin < margin_threshold or match < match_threshold), row['label']
    return decide


def forecast(study):
    baseline = replay(DEVELOPMENT_SEED, 'stock'); results = []; gates = []
    for name, match, margin in POLICIES:
        r = replay(DEVELOPMENT_SEED, 'error_gate', policy=policy(match, margin)); r['arm'] = name
        results.append(r); gates.append(dict(name=name, **gate(r, baseline)))
    passed = [r for r in results if next(g for g in gates if g['name'] == r['arm'])['status'] == 'PASS']
    chosen = min(passed, key=lambda r: (r['metrics']['adaptation_teaching_forwards'], r['arm']))['arm'] if passed else None
    report = dict(status='CPU_FORECAST_ONLY', baseline=baseline, development_results=results, gates=gates, selected_policy=chosen, confirmation=[])
    if chosen:
        _, match, margin = next(r for r in POLICIES if r[0] == chosen)
        for seed in CONFIRMATION_SEEDS:
            b = replay(seed, 'stock'); r = replay(seed, 'error_gate', policy=policy(match, margin)); r['arm'] = chosen
            report['confirmation'].append(dict(seed=seed, baseline=b, candidate=r, gate=gate(r, b)))
    (study / 'CPU_FORECAST.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(selected=chosen, development=[dict(arm=r['arm'], metrics=r['metrics']) for r in results], gates=gates,
                          confirmation=[dict(seed=r['seed'], gate=r['gate'], candidate=r['candidate']['metrics']) for r in report['confirmation']])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('study', type=Path); forecast(p.parse_args().study)
