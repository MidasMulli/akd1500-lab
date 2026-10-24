"""Sealed out-of-sample validation of one exploratory conservative policy."""
from pathlib import Path
import argparse
import json
from recipe.design import replay
from recipe.repair import policy

SEEDS = [98411, 98412, 98413, 98414]


def quality_gate(c, b):
    cm, bm = c['metrics'], b['metrics']; cf, bf = cm['checkpoints']['256'], bm['checkpoints']['256']
    checks = dict(old_prefix=cm['checkpoints']['0']['old_probe']['correct'] >= 115,
        final_old=cf['old_probe']['correct'] >= 109, final_new=cf['new_probe']['correct'] >= 109,
        old_noninferior=cf['old_probe']['correct'] >= bf['old_probe']['correct'] - 2,
        new_noninferior=cf['new_probe']['correct'] >= bf['new_probe']['correct'] - 2,
        fewer_adaptation_jobs=(256 + cm['adaptation_teaching_forwards']) <= .95 * (256 + bm['adaptation_teaching_forwards']))
    return dict(status='PASS' if all(checks.values()) else 'STOP_VALIDATION_GATE', checks=checks)


def forecast(study):
    result = []
    for seed in SEEDS:
        b = replay(seed, 'stock'); c = replay(seed, 'error_gate', policy=policy(31, 4)); c['arm'] = 'conservative31'
        result.append(dict(seed=seed, baseline=b, candidate=c, gate=quality_gate(c, b)))
    report = dict(status='CPU_FORECAST_ONLY', policy=dict(match_threshold=31, margin_threshold=4), results=result,
        overall='PASS' if all(r['gate']['status'] == 'PASS' for r in result) else 'STOP_VALIDATION_GATE')
    (study / 'CPU_FORECAST.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(overall=report['overall'], results=[dict(seed=r['seed'], gate=r['gate'],
        teaching=r['candidate']['metrics']['adaptation_teaching_forwards'], final=r['candidate']['metrics']['checkpoints']['256']) for r in result])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('study', type=Path); forecast(p.parse_args().study)
