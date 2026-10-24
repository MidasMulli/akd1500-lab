"""Bounded native-compaction sessions and complete public numerical recount."""
from pathlib import Path
import argparse
import datetime as dt
import fcntl
import json
import subprocess
import hashlib
import numpy as np
from metal.run import P, digest
from metal.sparse import save
from kernel.design import stream, expected

S = P / 'studies/20261023-native-compaction'
BASE = P / 'results/2026-10-23-native-compaction'


def freeze():
    files = list((P / 'kernel').glob('*.py')) + [P / p for p in ['src/metal_bench.cpp', 'src/readout_bench.cpp', 'src/mmio_driver.hpp',
        'metal/run.py', 'metal/sparse.py', 'metal/prepare.py', 'metal/ablation_prepare.py', 'metal/prepare_cnp.py',
        'tools/build.sh', 'tools/prepare.py', 'tools/prepare_network.py', 'tools/program_codec.py', 'build/metal_bench']]
    files += list((S / 'fixtures-kernel01').iterdir()) + [S / 'AUTHORITY.json', S / 'PROTOCOL.md']
    f = S / 'evidence/FROZEN.json'; assert not f.exists()
    save(f, dict(status='FROZEN_BEFORE_PHYSICAL', utc=dt.datetime.now(dt.timezone.utc).isoformat(), hashes={str(p.relative_to(P)): digest(p) for p in files}))
    print('frozen', len(files), 'files')


def run(name):
    with open('/tmp/akd1500-research.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((S / 'STATE.json').read_text()); assert not state['research_admissions_closed'] and not state.get('halted', False)
        entry = [r for r in state['admitted_runs'] if r['name'] == name]; assert len(entry) == 1; entry = entry[0]
        auth = json.loads((S / 'AUTHORITY.json').read_text()); now = dt.datetime.now(dt.timezone.utc)
        cutoff = dt.datetime.fromisoformat(auth['scientific_cutoff_utc']); assert (cutoff - now).total_seconds() >= 5
        frozen = S / 'evidence/FROZEN.json'
        for p, h in json.loads(frozen.read_text())['hashes'].items(): assert digest(P / p) == h, ('source drift', p)
        assert subprocess.run(['fuser', '/dev/akd1500_0'], capture_output=True).returncode == 1
        f = S / 'fixtures-kernel01' / entry['fixture']; meta = json.loads(f.with_suffix('.json').read_text())
        x, rows = stream(); ids = list(range(0, 144, 5)) + [1, 2, 3] if entry['smoke'] else list(range(144))
        assert len(ids) == (32 if entry['smoke'] else 144)
        count, forwards = len(ids), len(ids) + 48
        prior = [json.loads(p.read_text()) for p in (S / 'raw').glob('*/OWNER.json')]
        assert len(prior) < auth['maximum_physical_sessions'] and sum(r['forwards'] for r in prior) + forwards <= auth['maximum_forwards']
        d = S / 'raw' / name; d.mkdir(exist_ok=False)
        (d / 'stream.inputs').write_bytes(x[ids].tobytes()); (d / 'stream.labels').write_bytes(np.zeros(count, '<i4').tobytes())
        (d / 'stream.modes').write_bytes(bytes(count)); (d / 'stream.expected').write_bytes(expected(x[ids], meta['units']).astype('<i4').tobytes())
        rows = [dict(rows[i], original_index=i) for i in ids]; save(d / 'stream.rows.json', rows)
        rec = dict(**entry, samples=count, forwards=forwards, fixture_info=meta, start_utc=now.isoformat(), source_freeze_sha256=digest(frozen))
        save(d / 'OWNER.json', rec)
        with (d / 'native.stdout').open('w') as out, (d / 'native.stderr').open('w') as err:
            proc = subprocess.Popen(['taskset', '-c', '2', str(P / 'build/metal_bench'), str(f), str(d / 'stream'), str(d / 'native'), str(count), 'fixed'], stdout=out, stderr=err)
            rec['native_pid'] = proc.pid; save(d / 'OWNER.json', rec)
            try: code = proc.wait(timeout=min(auth['maximum_individual_seconds'], int((cutoff - now).total_seconds())))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try: code = proc.wait(timeout=3)
                except subprocess.TimeoutExpired: proc.kill(); code = proc.wait()
        rec.update(status='PASS' if code == 0 else 'FAILED_PRESERVED', exit_code=code,
                   device_released=subprocess.run(['fuser', '/dev/akd1500_0'], capture_output=True).returncode == 1,
                   finish_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        rec['hashes'] = {p.name: digest(p) for p in d.iterdir() if p.is_file() and p.name != 'RECEIPT.json'}
        save(d / 'RECEIPT.json', rec)
        if code or not rec['device_released']: state.update(halted=True, failure=name); save(S / 'STATE.json', state)
        assert code == 0 and rec['device_released'], (d / 'native.stderr').read_text()
        try: report = audit(d)
        except Exception:
            state.update(halted=True, audit_failure=name); save(S / 'STATE.json', state); raise
        print(json.dumps(report), flush=True)


def arrays(d, rec):
    count = rec['samples']; shapes = {'stream.inputs': (np.uint8, (count, 512)),
        'native.scores': ('<i4', (count, 2)), 'native.cycles': ('<u4', (count,)),
        'native.counter_bounds': ('<u4', (count, 2)), 'native.config_bounds': ('<u4', (count, 2)),
        'native.latency_ns': ('<i8', (count,)), 'native.caller_ns': ('<i8', (count,)),
        'native.idle_counter': ('<u4', (6,)), 'native.serial_suffix': ('<i4', (3, 8, 2)),
        'native.burst_suffix': ('<i4', (3, 8, 2)), 'native.burst_cycles': ('<u4', (3, 2)), 'native.burst_latency_ns': ('<i8', (3, 2))}
    for i, packet in enumerate(rec['fixture_info']['packets']):
        for when in ['initial', 'final']: shapes[f'native.packet{i}.{when}'] = ('<u4', (packet['words'],))
    return {k: np.fromfile(d / k, dtype).reshape(shape) for k, (dtype, shape) in shapes.items()}


def recount(rec, data):
    meta = rec['fixture_info']; n = meta['units']; kind = meta['kind']; assert n in [256, 512] and kind in ['direct', 'compacted']
    x, rows = stream(); ids = list(range(0, 144, 5)) + [1, 2, 3] if rec['smoke'] else list(range(144))
    assert rec['samples'] == len(ids) and rec['forwards'] == len(ids) + 48
    assert rec['rows'] == [dict(rows[i], original_index=i) for i in ids], 'frozen kernel stream metadata'
    assert np.array_equal(data['stream.inputs'], x[ids]), 'frozen kernel inputs'
    assert np.array_equal(data['native.scores'], expected(x[ids], n)), 'native graph output equivalence'
    targets = ([[1, 3, 0]] if kind == 'compacted' else []) + [[2, 1, 0], [3, 3, 0]]
    placements = [p for r in meta['mapping'] for p in r['nps']]
    assert [p['np'] for p in placements] == targets and all(p['type'] == 'Type.FNP3' for p in placements), 'fixed native mapping'
    assert [p['np'] for p in meta['packets']] == targets
    for i, p in enumerate(meta['packets']):
        a, b = data[f'native.packet{i}.initial'], data[f'native.packet{i}.final']
        assert np.array_equal(a, b), 'native fixed RAM changed'
        assert hashlib.sha256(a.astype('<u4').tobytes()).hexdigest() == p['sha256'], 'native fixed RAM packet'
    idle = data['native.idle_counter']; cfg = data['native.config_bounds']; bounds = data['native.counter_bounds']
    assert idle[0] == idle[1] == idle[2] == idle[3] and idle[4] == idle[5], 'idle counter control'
    assert np.all(cfg[:, 0] == cfg[:, 1]), 'config counter control'
    assert np.array_equal((bounds[:, 1].astype(np.uint64) - bounds[:, 0].astype(np.uint64)) % 2**32, data['native.cycles']), 'counter deltas'
    assert np.all(data['native.cycles'] > 0) and np.all(data['native.latency_ns'] > 0)
    assert np.array_equal(data['native.serial_suffix'], data['native.burst_suffix']), 'frozen queue outputs'
    assert all(np.array_equal(a, data['native.scores'][-8:]) for a in data['native.serial_suffix']), 'frozen queue reference'
    groups = []
    for density in [0, 64, 128]:
        ii = [j for j, i in enumerate(ids) if rows[i]['density'] == density]
        groups.append(dict(density=density, count=len(ii), median_cycles=float(np.median(data['native.cycles'][ii])),
                           minimum_cycles=int(data['native.cycles'][ii].min()), maximum_cycles=int(data['native.cycles'][ii].max()),
                           median_host_ns=float(np.median(data['native.latency_ns'][ii]))))
    return dict(status='INDEPENDENT_NATIVE_GRAPH_MEMORY_COUNTER_PASS', name=rec['name'], forwards=rec['forwards'],
                groups=groups, physical_regions=len(targets), burst_cycles=data['native.burst_cycles'].tolist())


def audit(d):
    rec = json.loads((d / 'RECEIPT.json').read_text()); assert rec['status'] == 'PASS' and rec['device_released']
    for p, h in rec['hashes'].items(): assert digest(d / p) == h, p
    rec['rows'] = json.loads((d / 'stream.rows.json').read_text()); r = recount(rec, arrays(d, rec)); save(d / 'AUDIT.json', r); return r


def names(): return {f'{prefix}_u{u}_{kind}' for prefix in ['smoke', 'screen', 'repeat1', 'repeat2'] for u in [256, 512] for kind in ['direct', 'compacted']}


def export():
    assert json.loads((S / 'STATE.json').read_text())['research_admissions_closed']
    assert {d.name for d in (S / 'raw').iterdir()} == names()
    data, runs = {}, []
    for d in sorted((S / 'raw').iterdir()):
        a = audit(d); rec = json.loads((d / 'RECEIPT.json').read_text()); rec['rows'] = json.loads((d / 'stream.rows.json').read_text())
        for k, v in arrays(d, rec).items(): data[d.name + '__' + k] = v
        runs.append(dict(**{k: rec[k] for k in ['name', 'fixture', 'fixture_info', 'samples', 'forwards', 'smoke', 'rows', 'source_freeze_sha256']}, audit=a, receipt_sha256=digest(d / 'RECEIPT.json')))
    np.savez_compressed(BASE.with_suffix('.npz'), **data)
    save(BASE.with_suffix('.json'), dict(status='MEASURED_NATIVE_GRAPH_REWRITE', physical_chips=1, physical_sessions=len(runs),
        native_forwards=sum(r['forwards'] for r in runs), runs=runs, archive_sha256=digest(BASE.with_suffix('.npz'))))
    print(json.dumps(audit_public()))


def audit_public():
    r = json.loads(BASE.with_suffix('.json').read_text()); assert digest(BASE.with_suffix('.npz')) == r['archive_sha256']
    assert len(r['runs']) == 16 and {x['name'] for x in r['runs']} == names(), 'complete graph treatment set'
    assert r['physical_sessions'] == len(r['runs']) and r['native_forwards'] == sum(x['forwards'] for x in r['runs'])
    with np.load(BASE.with_suffix('.npz'), allow_pickle=False) as z:
        for rec in r['runs']:
            a = {k.split('__', 1)[1]: z[k] for k in z.files if k.startswith(rec['name'] + '__')}
            assert recount(rec, a) == rec['audit'], 'saved kernel assessment'
    return dict(status='PASS', physical_sessions=len(r['runs']), native_forwards=r['native_forwards'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command'); a = p.parse_args()
    if a.command == 'freeze': freeze()
    elif a.command == 'export': export()
    elif a.command == 'audit-public': print(json.dumps(audit_public()))
    else: run(a.command)
