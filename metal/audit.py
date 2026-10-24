"""Independent Python-int recount of native outputs, all RAM and counter bounds."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from pong.relearn.model import Reference

P = Path(__file__).resolve().parents[1]
S = P / 'studies/20261023-metal-characterization'


def audit(d):
    rec = json.loads((d / 'RECEIPT.json').read_text())
    assert rec['status'] == 'PASS' and rec['device_released']
    for name, expected in rec['hashes'].items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == expected, name
    f = P / rec['fixture']; info = rec['fixture_info']; count = rec['samples']; u = info['outputs']
    x = np.fromfile(d / 'stream.inputs', np.uint8).reshape(count, 512)
    labels = np.fromfile(d / 'stream.labels', '<i4')
    modes = np.fromfile(d / 'stream.modes', np.uint8)
    got = np.fromfile(d / 'native.scores', '<i4').reshape(count, u)
    edited, reference = [], None
    if info['kind'] == 'learner':
        initial = np.fromfile(f.with_suffix('.ram'), '<u4')
        assert np.array_equal(initial, np.fromfile(d / 'native.initial.ram', '<u4'))
        reference = Reference(initial, 512, u, 32, 2)
        snapshots = np.fromfile(d / 'native.snapshots', '<u4').reshape(count, len(initial))
        for i in range(count):
            before = list(reference.ram)
            if modes[i]: want = reference.forward(x[i], int(labels[i]))
            else:
                bits = sum(1 << int(k) for k in np.flatnonzero(x[i]))
                want = [(w & bits).bit_count() for w in reference.weights]
            assert np.array_equal(got[i], want), ('native output', i)
            assert np.array_equal(snapshots[i], reference.ram), ('native RAM', i)
            changes = np.flatnonzero((np.asarray(before[2:], np.uint32).reshape(u, 20)
                                     != snapshots[i, 2:].reshape(u, 20)).any(axis=1)).tolist()
            if not modes[i]: assert not changes, ('frozen RAM changed', i)
            edited.append(changes)
        assert np.array_equal(np.fromfile(d / 'native.final.ram', '<u4'), reference.ram)
    else:
        expected = np.fromfile(d / 'stream.expected', '<i4').reshape(count, u)
        assert np.array_equal(got, expected), 'fixed graph output'
    serial = np.fromfile(d / 'native.serial_suffix', '<i4').reshape(3, 8, u)
    burst = np.fromfile(d / 'native.burst_suffix', '<i4').reshape(3, 8, u)
    assert np.array_equal(serial, burst)
    assert all(np.array_equal(a, got[-8:]) for a in serial)
    bounds = np.fromfile(d / 'native.counter_bounds', '<u4').reshape(count, 2)
    cycles = np.fromfile(d / 'native.cycles', '<u4')
    assert np.array_equal((bounds[:, 1].astype(np.uint64) - bounds[:, 0].astype(np.uint64)) % 2**32, cycles)
    idle = np.fromfile(d / 'native.idle_counter', '<u4')
    assert len(idle) == 6
    cfg = np.fromfile(d / 'native.config_bounds', '<u4').reshape(count, 2)
    rows = json.loads((d / 'stream.rows.json').read_text())
    groups = []
    for phase in dict.fromkeys(r['phase'] for r in rows):
        for density in sorted({r['density'] for r in rows if r['phase'] == phase}):
            ids = [i for i, r in enumerate(rows) if r['phase'] == phase and r['density'] == density]
            groups.append(dict(phase=phase, density=density, count=len(ids),
                median_cycles=float(np.median(cycles[ids])), minimum_cycles=int(cycles[ids].min()), maximum_cycles=int(cycles[ids].max())))
    qualified = bool(idle[0] == idle[1] == idle[2] == idle[3] and idle[4] == idle[5]
                     and np.any(cycles > 0) and np.all(cfg[:, 0] == cfg[:, 1]))
    report = dict(status='INDEPENDENT_OUTPUT_RAM_PASS', name=d.name, fixture=info['name'],
        events=count, total_forwards=rec['forwards'], outputs_recounted=int(got.size),
        complete_RAM_snapshots=count if reference else 0,
        counter_passes_idle_and_config_controls=qualified, idle_reads=idle.tolist(),
        configuration_counter_changes=int(np.count_nonzero(cfg[:, 0] != cfg[:, 1])),
        maximum_edited_records_per_event=max(map(len, edited), default=0),
        teaching_events_with_edits=sum(bool(e) for e in edited),
        effective_queue=rec['result'], groups=groups,
        serialized_burst_device_cycles=np.fromfile(d / 'native.burst_cycles', '<u4').reshape(3, 2).tolist())
    (d / 'AUDIT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('run', type=Path); a = p.parse_args()
    audit(a.run if a.run.is_absolute() else S / 'raw' / a.run)
