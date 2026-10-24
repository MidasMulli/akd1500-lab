"""Hardware-free recount of all public native observations, including queue arms."""
from pathlib import Path
import hashlib
import json
import numpy as np
from pong.relearn.model import Reference

P = Path(__file__).resolve().parents[1]
BASE = P / 'results/2026-10-23-metal'


def audit_contents(report, data):
    checked, snapshots, values = 0, 0, 0
    queued = {}
    for run in report['runs']:
        key = run['name'] + '__'; info = report['fixtures'][run['fixture']]
        count = run['samples']; u = info['outputs']
        # Cache decompressed arrays once per run.
        names = ['inputs','labels','outputs','cycles','counter_bounds']
        if info['kind'] == 'learner': names += ['initial','snapshots','final']
        if not run['queued_teaching']: names += ['modes','idle_counter','config_bounds','serial_suffix','burst_suffix']
        a = {n:data[key+n] for n in names}
        assert a['inputs'].shape == (count, 512) and a['outputs'].shape == (count, u)
        assert np.all((a['inputs'] == 0) | (a['inputs'] == 1))
        if info['kind'] == 'learner':
            reference = Reference(a['initial'], 512, u, 32, 2)
            assert a['snapshots'].shape == ((count//8 if run['queued_teaching'] else count), len(a['initial']))
            for i in range(count):
                if run['queued_teaching'] or a['modes'][i]: want = reference.forward(a['inputs'][i], int(a['labels'][i]))
                else:
                    bits = sum(1 << int(j) for j in np.flatnonzero(a['inputs'][i]))
                    want = [(w & bits).bit_count() for w in reference.weights]
                assert np.array_equal(a['outputs'][i], want), ('native output', run['name'], i)
                if not run['queued_teaching'] or i % 8 == 7:
                    at = i//8 if run['queued_teaching'] else i
                    assert np.array_equal(a['snapshots'][at], reference.ram), ('complete RAM', run['name'], i)
                    snapshots += 1
            assert np.array_equal(a['final'], reference.ram)
        else:
            # Independently reconstruct owned random binary kernels; no vendor program needed.
            units = info['units']; seed = 98101 + (19 if info['kind'] == 'fixed_cnp' else 0)
            rng = np.random.default_rng(seed); w = np.zeros((512, units), np.int32)
            for j in range(units): w[rng.choice(512, 32, replace=False), j] = 1
            responses = a['inputs'].astype(np.int32) @ w > 15
            want = np.stack([responses[:, :units//2].sum(axis=1), responses[:, units//2:].sum(axis=1)], axis=1)
            assert np.array_equal(a['outputs'], want), ('fixed native graph', run['name'])
        bounds = a['counter_bounds'].astype(np.uint64)
        assert np.array_equal((bounds[:, 1]-bounds[:, 0]) % 2**32, a['cycles']), ('counter delta', run['name'])
        if run['queued_teaching']:
            native = run['native_result']; mode = int(native['pipeline_override'])
            assert (native['control_before'] >> 8) & 3 == 0
            assert (native['control_after'] >> 8) & 3 == mode
            assert native['control_before'] & ~(3<<8) == native['control_after'] & ~(3<<8)
            queued[mode] = a
        else:
            assert len(run['rows']) == count
            assert np.array_equal(a['inputs'].sum(axis=1), [r['density'] for r in run['rows']]), ('input density', run['name'])
            assert np.array_equal(a['modes'], [int(r['learning']) for r in run['rows']])
            idle = a['idle_counter']; cfg = a['config_bounds']
            assert len(idle) == 6 and idle[0] == idle[1] == idle[2] == idle[3] and idle[4] == idle[5], ('idle control', run['name'])
            assert np.all(cfg[:, 0] == cfg[:, 1]), ('configuration included', run['name'])
            assert np.array_equal(a['serial_suffix'], a['burst_suffix'])
            assert all(np.array_equal(row, a['outputs'][-8:]) for row in a['serial_suffix'])
            # Recompute all displayed density statistics from the raw counters.
            for g in run['audit']['groups']:
                ids = [i for i, row in enumerate(run['rows']) if row['phase'] == g['phase'] and row['density'] == g['density']]
                assert g['count'] == len(ids)
                assert g['median_cycles'] == float(np.median(a['cycles'][ids]))
                assert g['minimum_cycles'] == int(a['cycles'][ids].min()) and g['maximum_cycles'] == int(a['cycles'][ids].max())
            values += a['serial_suffix'].size + a['burst_suffix'].size
        values += a['outputs'].size; checked += run['forwards']
    assert set(queued) == {0,1}
    for name in ['inputs','labels','initial','outputs','snapshots','final']:
        assert np.array_equal(queued[0][name], queued[1][name]), ('matched teaching treatments', name)
    assert checked == report['native_forwards'] == 9568
    assert snapshots == report['captured_learning_snapshots']
    assert values == report['native_output_values']
    return dict(status='PASS', native_forwards=checked, full_learning_snapshots=snapshots,
                native_output_values=values, physical_sessions=len(report['runs']))


def audit():
    report = json.loads(BASE.with_suffix('.json').read_text())
    assert hashlib.sha256(BASE.with_suffix('.npz').read_bytes()).hexdigest() == report['archive_sha256']
    with np.load(BASE.with_suffix('.npz'), allow_pickle=False) as data: return audit_contents(report, data)


if __name__ == '__main__': print(json.dumps(audit()))
