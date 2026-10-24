"""Bounded sparse-work admissions, native execution and hardware-free recount."""
from pathlib import Path
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import subprocess
import numpy as np
from metal.run import P, digest
from metal.sparse_design import SEED, CELLS, stream, weights
from pong.relearn.model import Reference

S = P / 'studies/20261023-sparse-work'
BASE = P / 'results/2026-10-23-sparse-work'


def expected_runs():
    anchors = [f'n{n}_u64_w{nw}_{k}' for n, nw in [(128, 8), (1024, 8), (128, 64)] for k in ['learner', 'compact']]
    result = {'screen_' + f'n{n}_u{u}_w{nw}_{k}' for n, u, nw in CELLS for k in ['learner', 'compact']}
    result.update(prefix + f for prefix in ['smoke_', 'repeat1_', 'repeat2_'] for f in anchors)
    return result


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def freeze():
    files = list((P / 'metal').glob('*.py'))
    files += [P / p for p in ['src/metal_bench.cpp', 'src/readout_bench.cpp', 'src/mmio_driver.hpp',
        'tools/prepare.py', 'tools/prepare_network.py', 'tools/prepare_reduction.py', 'tools/program_codec.py',
        'pong/relearn/model.py', 'tools/build.sh', 'build/metal_bench']]
    files += [p for p in (S / 'fixtures-sparse01').iterdir() if p.is_file()]
    files += [S / 'PROTOCOL.md', S / 'AUTHORITY.json']
    target = S / 'evidence/FROZEN.json'; assert not target.exists()
    save(target, dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), status='FROZEN_BEFORE_PHYSICAL',
                      hashes={str(p.relative_to(P)): digest(p) for p in files}))
    print(json.dumps(dict(status='FROZEN_BEFORE_PHYSICAL', files=len(files))))


def run(name):
    with open('/tmp/akd1500-research.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((S / 'STATE.json').read_text())
        assert not state['research_admissions_closed'] and not state.get('halted', False)
        found = [r for r in state['admitted_runs'] if r['name'] == name]; assert len(found) == 1
        entry = found[0]; auth = json.loads((S / 'AUTHORITY.json').read_text())
        now = dt.datetime.now(dt.timezone.utc); cutoff = dt.datetime.fromisoformat(auth['scientific_cutoff_utc'])
        assert (cutoff - now).total_seconds() >= 5
        frozen = S / 'evidence/FROZEN.json'
        for p, h in json.loads(frozen.read_text())['hashes'].items(): assert digest(P / p) == h, ('source drift', p)
        assert subprocess.run(['fuser', '/dev/akd1500_0'], capture_output=True).returncode == 1
        f = S / 'fixtures-sparse01' / entry['fixture']; meta = json.loads(f.with_suffix('.json').read_text())
        x, rows = stream(meta['inputs'])
        indices = list(range(0, 96, 6)) + list(range(96, 104)) + list(range(128, 136)) if entry['smoke'] else list(range(144))
        selected = [dict(rows[i], original_index=i) for i in indices]; forwards = len(indices) + 48
        prior = [json.loads(p.read_text()) for p in (S / 'raw').glob('*/OWNER.json')]
        assert len(prior) < auth['maximum_physical_sessions']
        assert sum(r['forwards'] for r in prior) + forwards <= auth['maximum_forwards']
        d = S / 'raw' / name; d.mkdir(exist_ok=False); sf = d / 'stream'
        sf.with_suffix('.inputs').write_bytes(x[indices].tobytes())
        sf.with_suffix('.labels').write_bytes(np.asarray([r['label'] for r in selected], '<i4').tobytes())
        sf.with_suffix('.modes').write_bytes(bytes(int(r['learning']) for r in selected))
        save(d / 'stream.rows.json', selected)
        rec = dict(**entry, samples=len(indices), forwards=forwards, fixture_info=meta,
                   source_freeze_sha256=digest(frozen), start_utc=now.isoformat())
        save(d / 'OWNER.json', rec)
        with (d / 'native.stdout').open('w') as out, (d / 'native.stderr').open('w') as err:
            proc = subprocess.Popen(['taskset', '-c', '2', str(P / 'build/metal_bench'), str(f), str(sf),
                                     str(d / 'native'), str(len(indices)), meta['kind']], stdout=out, stderr=err)
            rec['native_pid'] = proc.pid; save(d / 'OWNER.json', rec)
            try: code = proc.wait(timeout=min(auth['maximum_individual_seconds'], int((cutoff - now).total_seconds())))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try: code = proc.wait(timeout=3)
                except subprocess.TimeoutExpired: proc.kill(); code = proc.wait()
        rec.update(status='PASS' if code == 0 else 'FAILED_PRESERVED', exit_code=code,
                   device_released=subprocess.run(['fuser', '/dev/akd1500_0'], capture_output=True).returncode == 1,
                   finish_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        if code == 0:
            rec['native_result'] = json.loads(next(line for line in (d / 'native.stdout').read_text().splitlines() if line.startswith('{"status"')))
        rec['hashes'] = {p.name: digest(p) for p in d.iterdir() if p.is_file() and p.name != 'RECEIPT.json'}
        save(d / 'RECEIPT.json', rec)
        if code or not rec['device_released']:
            state.update(halted=True, failure=name); save(S / 'STATE.json', state)
        assert code == 0 and rec['device_released'], (d / 'native.stderr').read_text()
        try: report = audit(d)
        except Exception:
            state.update(halted=True, independent_audit_failure=name); save(S / 'STATE.json', state)
            raise
        print(json.dumps(report), flush=True)


def arrays(d, rec):
    c, meta = rec['samples'], rec['fixture_info']; n, u, nw = meta['inputs'], meta['units'], meta['weights_per_neuron']
    size = 2 + u * (4 + n // 32); out = meta['outputs']
    shapes = {'stream.inputs': (np.uint8, (c, n)), 'stream.labels': ('<i4', (c,)),
        'stream.modes': (np.uint8, (c,)), 'native.scores': ('<i4', (c, out)),
        'native.initial.ram': ('<u4', (size,)), 'native.final.ram': ('<u4', (size,)),
        'native.snapshots': ('<u4', (c, size)), 'native.cycles': ('<u4', (c,)),
        'native.counter_bounds': ('<u4', (c, 2)), 'native.config_bounds': ('<u4', (c, 2)),
        'native.idle_counter': ('<u4', (6,)), 'native.latency_ns': ('<i8', (c,)),
        'native.caller_ns': ('<i8', (c,)), 'native.serial_suffix': ('<i4', (3, 8, out)),
        'native.burst_suffix': ('<i4', (3, 8, out)), 'native.burst_cycles': ('<u4', (3, 2)),
        'native.burst_latency_ns': ('<i8', (3, 2))}
    if meta['kind'] == 'compact':
        shapes.update({'native.fixed.initial': ('<u4', (-1,)), 'native.fixed.final': ('<u4', (-1,))})
    return {k: np.fromfile(d / k, dtype).reshape(shape) for k, (dtype, shape) in shapes.items()}


def recount(rec, data):
    meta = rec['fixture_info']; n, u, nw = meta['inputs'], meta['units'], meta['weights_per_neuron']
    assert (n, u, nw) in CELLS and meta['seed'] == SEED and meta['kind'] in ['learner', 'compact']
    compact = meta['kind'] == 'compact'; count = rec['samples']
    assert meta['outputs'] == (2 if compact else u) and meta['threshold'] == nw // 2 - 1
    placements = [p for layer in meta['mapping'] for p in layer['nps']]
    assert [p['np'] for p in placements] == ([[1, 3, 0], [3, 3, 0]] if compact else [[1, 3, 0]]) and all(p['type'] == 'Type.FNP3' for p in placements), 'declared FNP3 placement'
    x, rows = stream(n)
    indices = list(range(0, 96, 6)) + list(range(96, 104)) + list(range(128, 136)) if rec['smoke'] else list(range(144))
    assert count == len(indices) and rec['forwards'] == count + 48
    assert rec['rows'] == [dict(rows[i], original_index=i) for i in indices], 'frozen stream metadata'
    assert np.array_equal(data['stream.inputs'], x[indices]), 'frozen input stream'
    assert np.array_equal(data['stream.labels'], [rows[i]['label'] for i in indices]), 'teacher labels'
    assert np.array_equal(data['stream.modes'], [rows[i]['learning'] for i in indices]), 'learning intervals'
    initial = data['native.initial.ram']; ref = Reference(initial, n, u, nw, 2)
    w = weights(n, u, nw)
    expected_weights = [sum(int(w[k, j]) << k for k in range(n)) for j in range(u)]
    assert ref.weights == expected_weights, 'initial selected connections'
    assert hashlib.sha256(initial.astype('<u4').tobytes()).hexdigest() == meta['ram_sha256'], 'initial memory packet'
    edited = []
    for i, a in enumerate(data['stream.inputs']):
        before = ref.ram[:]
        if data['stream.modes'][i]: scores = ref.forward(a, int(data['stream.labels'][i]))
        else:
            bits = sum(1 << int(k) for k in np.flatnonzero(a)); scores = [(v & bits).bit_count() for v in ref.weights]
        want = np.asarray(scores)
        if compact: want = (want > meta['threshold']).reshape(2, u // 2).sum(axis=1)
        assert np.array_equal(data['native.scores'][i], want), ('native output', i)
        assert np.array_equal(data['native.snapshots'][i], ref.ram), ('complete RAM', i)
        changes = np.flatnonzero((np.asarray(before[2:], np.uint32).reshape(u, ref.stride) != data['native.snapshots'][i, 2:].reshape(u, ref.stride)).any(axis=1))
        if not data['stream.modes'][i]: assert not len(changes), ('query changed memory', i)
        edited.append(int(len(changes)))
    assert np.array_equal(data['native.final.ram'], ref.ram), 'final memory'
    if compact:
        assert np.array_equal(data['native.fixed.initial'], data['native.fixed.final']), 'fixed readout memory'
        assert hashlib.sha256(data['native.fixed.initial'].astype('<u4').tobytes()).hexdigest() == meta['fixed_ram_sha256'], 'fixed readout packet'
    idle = data['native.idle_counter']; bounds = data['native.counter_bounds']; cfg = data['native.config_bounds']
    assert idle[0] == idle[1] == idle[2] == idle[3] and idle[4] == idle[5], 'idle control'
    assert np.all(cfg[:, 0] == cfg[:, 1]), 'configuration counter control'
    assert np.array_equal((bounds[:, 1].astype(np.uint64) - bounds[:, 0].astype(np.uint64)) % 2**32, data['native.cycles']), 'counter delta'
    assert np.all(data['native.cycles'] > 0)
    assert np.all(data['native.latency_ns'] > 0) and np.all(data['native.caller_ns'] >= 0), 'host timing bounds'
    assert np.array_equal(data['native.serial_suffix'], data['native.burst_suffix']), 'resident burst outputs'
    assert all(np.array_equal(row, data['native.scores'][-8:]) for row in data['native.serial_suffix']), 'frozen suffix'
    groups = []
    for phase in ['initial_inference', 'native_learning', 'retained_inference']:
        ids = [i for i, row in enumerate(rec['rows']) if row['phase'] == phase]
        groups.append(dict(phase=phase, count=len(ids), median_cycles=float(np.median(data['native.cycles'][ids])),
                           minimum_cycles=int(data['native.cycles'][ids].min()), maximum_cycles=int(data['native.cycles'][ids].max()),
                           median_host_ns=float(np.median(data['native.latency_ns'][ids]))))
    return dict(status='INDEPENDENT_OUTPUT_RAM_COUNTER_PASS', name=rec['name'], fixture=meta['name'],
                native_forwards=rec['forwards'], outputs_recounted=int(data['native.scores'].size + data['native.serial_suffix'].size + data['native.burst_suffix'].size),
                complete_RAM_snapshots=count, maximum_edited_records=max(edited), groups=groups)


def audit(d):
    rec = json.loads((d / 'RECEIPT.json').read_text()); assert rec['status'] == 'PASS' and rec['device_released']
    for p, h in rec['hashes'].items(): assert digest(d / p) == h, p
    rec['rows'] = json.loads((d / 'stream.rows.json').read_text())
    report = recount(rec, arrays(d, rec)); save(d / 'AUDIT.json', report); return report


def export():
    assert json.loads((S / 'STATE.json').read_text())['research_admissions_closed'], 'close admissions before final export'
    assert {d.name for d in (S / 'raw').iterdir()} == expected_runs(), 'complete sparse treatment set'
    data, runs = {}, []
    for d in sorted((S / 'raw').iterdir()):
        rec = json.loads((d / 'RECEIPT.json').read_text()); report = audit(d)
        rec['rows'] = json.loads((d / 'stream.rows.json').read_text())
        for k, a in arrays(d, rec).items(): data[d.name + '__' + k] = a
        runs.append(dict(**{k: rec[k] for k in ['name', 'samples', 'forwards', 'smoke', 'fixture_info', 'rows', 'source_freeze_sha256']},
                         audit=report, receipt_sha256=digest(d / 'RECEIPT.json')))
    np.savez_compressed(BASE.with_suffix('.npz'), **data)
    report = dict(status='OBSERVED_NATIVE_SPARSE_WORK', physical_chips=1, runs=runs,
                  physical_sessions=len(runs), native_forwards=sum(r['forwards'] for r in runs),
                  archive_sha256=digest(BASE.with_suffix('.npz')))
    save(BASE.with_suffix('.json'), report); print(json.dumps(audit_public()))


def audit_public():
    report = json.loads(BASE.with_suffix('.json').read_text()); assert digest(BASE.with_suffix('.npz')) == report['archive_sha256']
    assert {r['name'] for r in report['runs']} == expected_runs() and len(report['runs']) == 54, 'complete sparse treatment set'
    assert report['physical_sessions'] == len(report['runs']) and report['native_forwards'] == sum(r['forwards'] for r in report['runs']), 'sparse accounting'
    with np.load(BASE.with_suffix('.npz'), allow_pickle=False) as z:
        for rec in report['runs']:
            result = recount(rec, {k.split('__', 1)[1]: z[k] for k in z.files if k.startswith(rec['name'] + '__')})
            assert result == rec['audit'], 'saved sparse assessment'
    return dict(status='PASS', physical_sessions=len(report['runs']), native_forwards=sum(r['forwards'] for r in report['runs']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command'); a = p.parse_args()
    if a.command == 'freeze': freeze()
    elif a.command == 'export': export()
    elif a.command == 'audit-public': print(json.dumps(audit_public()))
    elif a.command.startswith('audit:'): print(json.dumps(audit(S / 'raw' / a.command.split(':', 1)[1])))
    else: run(a.command)
