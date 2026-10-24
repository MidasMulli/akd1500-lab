"""Four-mask learning-resource test, exact two-bank recount and public archive."""
from pathlib import Path
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import shutil
import subprocess
import numpy as np
from metal.run import P, digest
from metal.sparse import save
from relay.design import SEEDS, schedule, dataset, noisy
from relay.reference import Network

S = P / 'studies/20261023-learning-interference'
BASE = P / 'results/2026-10-23-learning-interference'


def expected_runs():
    return {f'smoke_s97101_m{m}' for m in range(4)} | {f'full_s{seed}_m{m}' for seed in SEEDS for m in range(4)}


def stream(mask, smoke=False):
    x, rows = schedule('recognition_repair'); x = list(x[:512])
    rows = [dict(r, phase='shared_prefix') for r in rows[:512]]
    data = dataset(); rng = np.random.default_rng(98301)
    for i in range(144):
        identity = i % 4
        x.append(noisy(data['signatures'][0][identity], rng))
        rows.append(dict(index=len(rows), phase='test' if i < 128 else 'retained_query',
                         label=identity, learning_mask=mask if i < 128 else 0, identity=identity))
    indices = list(range(528)) + list(range(640, 648)) if smoke else list(range(656))
    return np.asarray(x)[indices], [dict(rows[i], original_index=i) for i in indices]


def forecast(meta, mask, smoke):
    f = S / 'fixtures-coupled01' / f's{meta["seed"]}'
    ram = [np.fromfile(str(f) + f'.bank{k}.ram', '<u4') for k in range(2)]
    net = Network(ram); x, rows = stream(mask, smoke)
    messages, states, edits = [], [], []
    for a, row in zip(x, rows):
        before = [b.ram[:] for b in net.banks]
        y, detail = net.forward(a, row['label'], row['learning_mask'])
        messages.append(detail['identity']); states.append([b.ram[:] for b in net.banks])
        edits.append([int(before[k] != net.banks[k].ram) for k in range(2)])
    return dict(messages=np.asarray(messages), states=np.asarray(states, np.uint32), edits=np.asarray(edits))


def prepare():
    source = P / 'studies/20261023-signal-relay/fixtures-relay03'
    out = S / 'fixtures-coupled01'; assert not list(out.iterdir())
    cells = []
    for seed in SEEDS:
        for f in source.glob(f's{seed}.*'): shutil.copy2(f, out / f.name)
        meta = json.loads((out / f's{seed}.json').read_text())
        meta['bank_ram_sha256'] = [digest(out / f's{seed}.bank{k}.ram') for k in range(2)]
        meta['fixed_ram_sha256'] = [digest(out / f's{seed}.fixed{k}.ram') for k in range(3)]
        save(out / f's{seed}.json', meta)
        for smoke in [True, False]:
            arms = {mask: forecast(meta, mask, smoke) for mask in range(4)}
            assert all(np.array_equal(arms[m]['messages'], arms[0]['messages']) for m in range(4)), 'unequal native message forecast'
            assert np.array_equal(arms[1]['states'][:, 0], arms[3]['states'][:, 0]), 'A solo/mixed state forecast'
            assert np.array_equal(arms[2]['states'][:, 1], arms[3]['states'][:, 1]), 'B solo/mixed state forecast'
            cells.append(dict(seed=seed, smoke=smoke, status='CPU_MESSAGE_AND_STATE_EQUALITY_PASS',
                              edits={m: a['edits'][512:].sum(axis=0).tolist() for m, a in arms.items()}))
    save(out / 'BUILD.json', dict(physical_device_opened=False, status='CPU_GATE_PASSED', cells=cells))
    print(json.dumps(cells))


def freeze():
    files = [P / p for p in ['metal/interference.py', 'metal/sparse.py', 'metal/run.py', 'src/relay_bench.cpp',
        'src/readout_bench.cpp', 'src/mmio_driver.hpp', 'relay/design.py', 'relay/reference.py',
        'pong/relearn/model.py', 'build/relay_bench', 'tools/build.sh']]
    files += list((S / 'fixtures-coupled01').glob('*')) + [S / 'PROTOCOL.md', S / 'AUTHORITY.json']
    f = S / 'evidence/FROZEN.json'; assert not f.exists()
    save(f, dict(status='FROZEN_BEFORE_PHYSICAL', utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 hashes={str(p.relative_to(P)): digest(p) for p in files})); print('frozen', len(files), 'files')


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
        assert entry['seed'] in SEEDS and entry['mask'] in range(4)
        x, rows = stream(entry['mask'], entry['smoke']); count = len(x)
        prior = [json.loads(p.read_text()) for p in (S / 'raw').glob('*/OWNER.json')]
        assert len(prior) < auth['maximum_physical_sessions'] and sum(r['samples'] for r in prior) + count <= auth['maximum_forwards']
        d = S / 'raw' / name; d.mkdir(exist_ok=False); f = S / 'fixtures-coupled01' / f's{entry["seed"]}'
        (d / 'stream.inputs').write_bytes(x.tobytes()); (d / 'stream.labels').write_bytes(np.asarray([r['label'] for r in rows], '<i4').tobytes())
        (d / 'stream.masks').write_bytes(bytes(r['learning_mask'] for r in rows)); save(d / 'stream.rows.json', rows)
        rec = dict(**entry, samples=count, start_utc=now.isoformat(), source_freeze_sha256=digest(frozen),
                   fixture_info=json.loads(f.with_suffix('.json').read_text()))
        save(d / 'OWNER.json', rec)
        with (d / 'native.stdout').open('w') as out, (d / 'native.stderr').open('w') as err:
            proc = subprocess.Popen(['taskset', '-c', '2', str(P / 'build/relay_bench'), str(f), str(d / 'stream'), str(d / 'native'), str(count), '1'], stdout=out, stderr=err)
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
            state.update(halted=True, independent_audit_failure=name); save(S / 'STATE.json', state)
            raise
        print(json.dumps(report), flush=True)


def arrays(d, rec):
    c = rec['samples']; data = {}
    shapes = {'stream.inputs': (np.uint8, (c, 512)), 'stream.labels': ('<i4', (c,)), 'stream.masks': (np.uint8, (c,)),
        'native.scores': ('<i4', (c, 4)), 'native.cycles': ('<u4', (c,)), 'native.counter_bounds': ('<u4', (c, 2)),
        'native.config_bounds': ('<u4', (c, 2)), 'native.control_bounds': ('<u4', (c, 2)), 'native.idle_counter': ('<u4', (6,)),
        'native.latency_ns': ('<i8', (c,)), 'native.caller_ns': ('<i8', (c,))}
    for k, (dtype, shape) in shapes.items(): data[k] = np.fromfile(d / k, dtype).reshape(shape)
    for k in range(2):
        data[f'bank{k}.initial'] = np.fromfile(d / f'native.bank{k}.initial.ram', '<u4')
        data[f'bank{k}.final'] = np.fromfile(d / f'native.bank{k}.ram', '<u4')
        data[f'bank{k}.snapshots'] = np.stack([np.fromfile(d / f'native.snapshot_{i}_bank{k}.ram', '<u4') for i in range(c)])
    for k in range(3):
        for when in ['initial', 'final']: data[f'fixed{k}.{when}'] = np.fromfile(d / f'native.fixed{k}.{when}.ram', '<u4')
    return data


def recount(rec, data):
    assert rec['seed'] in SEEDS and rec['mask'] in range(4)
    meta = rec['fixture_info']; assert meta['seed'] == rec['seed'], 'coupled seed identity'
    x, rows = stream(rec['mask'], rec['smoke']); c = len(rows)
    assert rec['samples'] == c and rec['rows'] == rows, 'frozen coupled schedule'
    assert np.array_equal(data['stream.inputs'], x), 'coupled input'
    assert np.array_equal(data['stream.labels'], [r['label'] for r in rows]), 'coupled labels'
    assert np.array_equal(data['stream.masks'], [r['learning_mask'] for r in rows]), 'coupled masks'
    ram = [data[f'bank{k}.initial'] for k in range(2)]
    for k in range(2): assert hashlib.sha256(ram[k].astype('<u4').tobytes()).hexdigest() == meta['bank_ram_sha256'][k], 'initial coupled RAM'
    net = Network(ram); edits, messages = [], []
    for i, (a, row) in enumerate(zip(x, rows)):
        before = [b.ram[:] for b in net.banks]; want, detail = net.forward(a, row['label'], row['learning_mask'])
        assert np.array_equal(data['native.scores'][i], want), ('coupled native output', i)
        change = []
        for k in range(2):
            got = data[f'bank{k}.snapshots'][i]
            assert np.array_equal(got, net.banks[k].ram), ('coupled complete RAM', i, k)
            ids = np.flatnonzero((np.asarray(before[k][2:], np.uint32).reshape(64, 20) != got[2:].reshape(64, 20)).any(axis=1))
            if not row['learning_mask'] & (1 << k): assert not len(ids), ('frozen coupled bank changed', i, k)
            change.append(len(ids))
        edits.append(change); messages.append(detail['identity'])
    for k in range(2): assert np.array_equal(data[f'bank{k}.final'], net.banks[k].ram), 'final coupled RAM'
    for k in range(3):
        assert np.array_equal(data[f'fixed{k}.initial'], data[f'fixed{k}.final']), 'fixed communication RAM'
        assert hashlib.sha256(data[f'fixed{k}.initial'].astype('<u4').tobytes()).hexdigest() == meta['fixed_ram_sha256'][k], 'fixed communication packet'
    idle = data['native.idle_counter']; cfg = data['native.config_bounds']; bounds = data['native.counter_bounds']; ctl = data['native.control_bounds']
    assert idle[0] == idle[1] == idle[2] == idle[3] and idle[4] == idle[5], 'coupled idle control'
    assert np.all(cfg[:, 0] == cfg[:, 1]), 'coupled config counter'
    assert np.array_equal((bounds[:, 1].astype(np.uint64) - bounds[:, 0].astype(np.uint64)) % 2**32, data['native.cycles']), 'coupled counter delta'
    assert np.all(data['native.cycles'] > 0) and np.all(ctl == ctl[0, 0]), 'coupled buffer-end control'
    assert int(ctl[0, 0]) & 0x300 == 0, 'coupled outbound counter scope'
    ii = [i for i, r in enumerate(rows) if r['phase'] == 'test']; ee = np.asarray(edits)[ii]
    return dict(status='INDEPENDENT_OUTPUT_TWO_BANK_RAM_COUNTER_PASS', name=rec['name'], samples=c,
        test_events=len(ii), test_bank_edit_events=ee.sum(axis=0).tolist(), same_event_both_banks_edited=int(np.count_nonzero((ee > 0).all(axis=1))),
        test_median_cycles=float(np.median(data['native.cycles'][ii])), test_median_host_ns=float(np.median(data['native.latency_ns'][ii])),
        test_cycles=data['native.cycles'][ii].tolist(), reconstructed_messages=messages,
        buffer_end_control=int(ctl[0, 0]), maximum_edited_records_per_bank=ee.max(axis=0).tolist())


def audit(d):
    rec = json.loads((d / 'RECEIPT.json').read_text()); assert rec['status'] == 'PASS' and rec['device_released']
    for p, h in rec['hashes'].items(): assert digest(d / p) == h, p
    rec['rows'] = json.loads((d / 'stream.rows.json').read_text()); report = recount(rec, arrays(d, rec))
    save(d / 'AUDIT.json', report); return {k: v for k, v in report.items() if k not in ['test_cycles', 'reconstructed_messages']}


def paired(runs, data):
    for seed in SEEDS:
        for smoke in [True, False]:
            rr = [r for r in runs if r['seed'] == seed and r['smoke'] == smoke]
            if not rr: continue
            assert sorted(r['mask'] for r in rr) == [0, 1, 2, 3], 'complete four-mask comparison'
            by = {r['mask']: r for r in rr}
            for mask in range(4):
                assert by[mask]['audit']['reconstructed_messages'] == by[0]['audit']['reconstructed_messages'], 'matched native messages'
                assert by[mask]['audit']['buffer_end_control'] == by[0]['audit']['buffer_end_control'], 'matched counter scope'
            for k, solo in [(0, 1), (1, 2)]:
                assert np.array_equal(data[by[solo]['name'] + f'__bank{k}.snapshots'], data[by[3]['name'] + f'__bank{k}.snapshots']), 'native solo/mixed state equality'


def export():
    assert json.loads((S / 'STATE.json').read_text())['research_admissions_closed'], 'close admissions before final export'
    assert {d.name for d in (S / 'raw').iterdir()} == expected_runs(), 'complete coupled treatment set'
    data, runs = {}, []
    for d in sorted((S / 'raw').iterdir()):
        audit(d); rec = json.loads((d / 'RECEIPT.json').read_text()); rec['rows'] = json.loads((d / 'stream.rows.json').read_text())
        report = json.loads((d / 'AUDIT.json').read_text())
        runs.append(dict(**{k: rec[k] for k in ['name', 'seed', 'mask', 'smoke', 'samples', 'rows', 'fixture_info', 'source_freeze_sha256']}, audit=report, receipt_sha256=digest(d / 'RECEIPT.json')))
        for k, a in arrays(d, rec).items(): data[d.name + '__' + k] = a
    paired(runs, data); np.savez_compressed(BASE.with_suffix('.npz'), **data)
    save(BASE.with_suffix('.json'), dict(status='OBSERVED_NATIVE_COUPLED_LEARNING', physical_chips=1, physical_sessions=len(runs),
         native_forwards=sum(r['samples'] for r in runs), runs=runs, archive_sha256=digest(BASE.with_suffix('.npz'))))
    print(json.dumps(audit_public()))


def audit_public():
    report = json.loads(BASE.with_suffix('.json').read_text()); assert digest(BASE.with_suffix('.npz')) == report['archive_sha256']
    assert {r['name'] for r in report['runs']} == expected_runs() and len(report['runs']) == 16, 'complete coupled treatment set'
    assert report['physical_sessions'] == len(report['runs']) and report['native_forwards'] == sum(r['samples'] for r in report['runs']), 'coupled accounting'
    with np.load(BASE.with_suffix('.npz'), allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
        for rec in report['runs']:
            result = recount(rec, {k.split('__', 1)[1]: v for k, v in data.items() if k.startswith(rec['name'] + '__')})
            assert result == rec['audit'], 'saved coupled assessment'
        paired(report['runs'], data)
    return dict(status='PASS', physical_sessions=len(report['runs']), native_forwards=sum(r['samples'] for r in report['runs']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command'); a = p.parse_args()
    if a.command == 'prepare': prepare()
    elif a.command == 'freeze': freeze()
    elif a.command == 'export': export()
    elif a.command == 'audit-public': print(json.dumps(audit_public()))
    elif a.command.startswith('audit:'): print(json.dumps(audit(S / 'raw' / a.command.split(':', 1)[1])))
    else: run(a.command)
