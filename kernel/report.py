"""Post-closeout analysis of exact outputs and matched native service costs.

This module was authored after execution. The physical instrument and runner
remain preserved by the pre-execution manifest and private source archive.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from kernel.experiment import BASE, recount, names

P = Path(__file__).resolve().parents[1]
OUT = P / 'results/2026-10-23-compaction-assessment.json'


def summarize(report, archive):
    runs = {r['name']: r for r in report['runs']}
    assert len(report['runs']) == 16 and set(runs) == names(), 'complete treatment set'
    assert len({r['source_freeze_sha256'] for r in runs.values()}) == 1, 'one physical source freeze'
    readbacks = 0
    for name, rec in runs.items():
        data = {k.split('__', 1)[1]: archive[k] for k in archive if k.startswith(name + '__')}
        assert recount(rec, data) == rec['audit'], 'raw-derived audit'
        for field in ['native.burst_cycles', 'native.burst_latency_ns']:
            assert data[field].shape == (3, 2) and np.all(data[field] > 0), 'positive paired burst measurements'
        readbacks += 2 * len(rec['fixture_info']['packets'])
    widths = []
    for units in [256, 512]:
        pairs = []
        for prefix in ['screen', 'repeat1', 'repeat2']:
            d, c = [runs[f'{prefix}_u{units}_{kind}'] for kind in ['direct', 'compacted']]
            serial = []
            for density in [0, 64, 128]:
                dg, cg = [next(g for g in r['audit']['groups'] if g['density'] == density) for r in [d, c]]
                serial.append(dict(latent_active_bits=density, direct_cycles=dg['median_cycles'],
                    compacted_cycles=cg['median_cycles'], reduction_percent=100 * (1 - cg['median_cycles'] / dg['median_cycles']),
                    direct_host_median_ns=dg['median_host_ns'], compacted_host_median_ns=cg['median_host_ns']))
            suffix = [r['rows'][-8:] for r in [d, c]]
            assert suffix[0] == suffix[1] and len({r['density'] for r in suffix[0]}) == 1, 'matched burst activity'
            db, cb = [np.asarray(archive[r['name'] + '__native.burst_cycles'])[:, 1] for r in [d, c]]
            dh, ch = [np.asarray(archive[r['name'] + '__native.burst_latency_ns'])[:, 1] for r in [d, c]]
            burst = dict(jobs_per_burst=8, latent_active_bits=suffix[0][0]['density'],
                direct_cycles=db.tolist(), compacted_cycles=cb.tolist(),
                reduction_percent=(100 * (1 - cb.astype(float) / db)).tolist(),
                direct_release_to_completion_ns=dh.tolist(), compacted_release_to_completion_ns=ch.tolist())
            passes = serial[1]['reduction_percent'] >= 5 and bool(np.all(100 * (1 - cb.astype(float) / db) >= 5))
            pairs.append(dict(stage=prefix, serial=serial, burst=burst, passes_frozen_cost_gate=passes))
        ram = {kind: sum(p['words'] - 2 for p in runs[f'screen_u{units}_{kind}']['fixture_info']['packets']) * 4
               for kind in ['direct', 'compacted']}
        widths.append(dict(detector_records=units, paired_sessions=pairs, captured_parameter_bytes=ram,
            captured_parameter_reduction_percent=100 * (1 - ram['compacted'] / ram['direct']),
            gate='PASS' if all(p['passes_frozen_cost_gate'] for p in pairs) else 'FAIL'))
    return dict(status='RAW_RECOUNTED_BOUNDED_NATIVE_GRAPH_IMPROVEMENT', physical_sessions=16,
        native_forwards=sum(r['forwards'] for r in runs.values()), physical_parameter_region_readbacks=readbacks,
        widths=widths, native_learning_enabled=False, external_input_bytes=512, external_output_integer_counts=2,
        limitations=[
            'One physical AKD1500; fixed synthetic four-copy input domain and one weight seed.',
            'Rewrite adds a third NPU. It is not an arbitrary instruction or learning-engine replacement.',
            'Serial query and queued burst have different timing boundaries and activity conditions.',
            'Burst suffix is eight zero-activity inputs; output equivalence there is not nonzero-work throughput proof.',
            'DMA monitor measures buffer service, not arithmetic-only cycles or energy.',
            'Burst host timing excludes input preload, descriptor construction and result drainage.',
            'Captured parameter payload excludes other SRAM, configuration and transport memory.',
            'No CPU or calibrated electrical efficiency advantage is established.'])


def load_summary():
    report = json.loads(BASE.with_suffix('.json').read_text())
    assert hashlib.sha256(BASE.with_suffix('.npz').read_bytes()).hexdigest() == report['archive_sha256']
    with np.load(BASE.with_suffix('.npz'), allow_pickle=False) as archive:
        return summarize(report, archive)


if __name__ == '__main__':
    result = load_summary()
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], forwards=result['native_forwards'],
        gates={w['detector_records']: w['gate'] for w in result['widths']})))
