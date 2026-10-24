"""CPU-only contextual comparator; matches saved outputs and complete final RAM."""
from pathlib import Path
import argparse
import datetime as dt
import json
import platform
import subprocess
import numpy as np
from metal.run import P, digest
from metal.sparse import save

S = P / 'studies/20261023-coupled-cpu'
NATIVE = P / 'studies/20261023-learning-interference'
BASE = P / 'results/2026-10-23-coupled-cpu'


def run(name, repeats):
    assert repeats in [1, 128]
    state = json.loads((S / 'STATE.json').read_text()); assert not state['research_admissions_closed']
    assert dict(name=name, repeats=repeats) in state['admitted_runs']
    native_name = name.removeprefix('pilot_'); d = NATIVE / 'raw' / native_name
    rec = json.loads((d / 'RECEIPT.json').read_text()); assert rec['status'] == 'PASS' and not rec['smoke']
    f = NATIVE / 'fixtures-coupled01' / f's{rec["seed"]}'
    out = S / 'raw' / name; out.mkdir(parents=True, exist_ok=False)
    sources = ['src/coupled_cpu_bench.cpp', 'src/readout_bench.cpp', 'src/mmio_driver.hpp', 'tools/build.sh', 'metal/cpu_compare.py', 'build/coupled_cpu_bench']
    frozen = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), hashes={p: digest(P / p) for p in sources},
                  native_receipt_sha256=digest(d / 'RECEIPT.json'), repeats=repeats, name=name)
    save(out / 'BEFORE.json', frozen)
    # Host observations are context, not a calibrated CPU/IRQ/energy comparison.
    save(out / 'HOST.json', dict(platform=platform.platform(), processor=platform.processor(),
        proc_stat_before=Path('/proc/stat').read_text(), cpuinfo=Path('/proc/cpuinfo').read_text()))
    r = subprocess.run(['taskset', '-c', '2', str(P / 'build/coupled_cpu_bench'), str(f), str(d / 'stream'),
                        str(out / 'cpu'), '656', str(repeats)], capture_output=True, timeout=30)
    (out / 'cpu.stdout').write_bytes(r.stdout); (out / 'cpu.stderr').write_bytes(r.stderr)
    assert r.returncode == 0, r.stderr.decode()
    scores = np.fromfile(out / 'cpu.scores', '<i4').reshape(656, 4)
    assert np.array_equal(scores, np.fromfile(d / 'native.scores', '<i4').reshape(656, 4)), 'CPU/native outputs'
    for k in range(2):
        assert np.array_equal(np.fromfile(out / f'cpu.bank{k}.ram', '<u4'), np.fromfile(d / f'native.bank{k}.ram', '<u4')), 'CPU/native final complete RAM'
    wall = np.fromfile(out / 'cpu.probe_wall_ns', '<i8'); cpu = np.fromfile(out / 'cpu.probe_cpu_ns', '<i8')
    assert len(wall) == len(cpu) == repeats and np.all(wall > 0) and np.all(cpu > 0)
    report = dict(status='EXACT_NATIVE_OUTPUTS_AND_FINAL_RAM_PASS', name=name, native_run=native_name,
        seed=rec['seed'], mask=rec['mask'], repeats=repeats, timed_events_per_repeat=128,
        median_cpu_wall_ns_per_test_event=float(np.median(wall) / 128),
        median_thread_cpu_ns_per_test_event=float(np.median(cpu) / 128),
        hashes={p.name: digest(p) for p in out.iterdir() if p.is_file()})
    save(out / 'RECEIPT.json', report); print(json.dumps(report), flush=True)


def export():
    assert json.loads((S / 'STATE.json').read_text())['research_admissions_closed']
    native_report = json.loads((P / 'results/2026-10-23-learning-interference.json').read_text())
    runs, arrays = [], {}
    for d in sorted((S / 'raw').iterdir()):
        r = json.loads((d / 'RECEIPT.json').read_text())
        for p, h in r['hashes'].items(): assert digest(d / p) == h
        runs.append(r)
        for ext, dtype in [('probe_wall_ns', '<i8'), ('probe_cpu_ns', '<i8'), ('scores', '<i4'), ('bank0.ram', '<u4'), ('bank1.ram', '<u4')]:
            arrays[d.name + '__' + ext] = np.fromfile(d / ('cpu.' + ext), dtype)
    assert len(runs) == 13
    np.savez_compressed(BASE.with_suffix('.npz'), **arrays)
    save(BASE.with_suffix('.json'), dict(status='CPU_CONTEXT_ONLY', physical_device_opened=False,
        native_report_sha256=digest(P / 'results/2026-10-23-learning-interference.json'),
        native_archive_sha256=native_report['archive_sha256'], archive_sha256=digest(BASE.with_suffix('.npz')), runs=runs,
        limitations=['Compiler-O3 binary-popcount implementation with direct categorical decoding; not a proven optimal CPU implementation',
                     'Same dense input, teacher stream, output contract and exact final learning state',
                     'Timed interval is 128 probe events after replaying the entire 512-event prefix each repetition',
                     'Native adapter uses transport and 50us fetch sleeps; CPU has no PCIe round trip',
                     'Late native confirmations overlapped artifact handling, so host-latency comparison is descriptive',
                     'No matched whole-host IRQ accounting, calibrated power or useful-application CPU differentiation established']))
    print(json.dumps(dict(status='EXPORTED_CPU_CONTEXT', runs=len(runs))))


def audit_public():
    report = json.loads(BASE.with_suffix('.json').read_text())
    assert report['native_report_sha256'] == digest(P / 'results/2026-10-23-learning-interference.json')
    assert report['archive_sha256'] == digest(BASE.with_suffix('.npz'))
    expected = {f'full_s{seed}_m{m}' for seed in [97101, 97102, 97103] for m in range(4)} | {'pilot_full_s97101_m3'}
    assert len(report['runs']) == 13 and {r['name'] for r in report['runs']} == expected
    with np.load(BASE.with_suffix('.npz'), allow_pickle=False) as z, np.load(P / 'results/2026-10-23-learning-interference.npz', allow_pickle=False) as native:
        for r in report['runs']:
            key = r['name'] + '__'; nk = r['native_run'] + '__'
            assert np.array_equal(z[key + 'scores'].reshape(656, 4), native[nk + 'native.scores']), 'public CPU/native outputs'
            for k in range(2): assert np.array_equal(z[key + f'bank{k}.ram'], native[nk + f'bank{k}.final']), 'public CPU/native RAM'
            for suffix, metric in [('probe_wall_ns', 'median_cpu_wall_ns_per_test_event'), ('probe_cpu_ns', 'median_thread_cpu_ns_per_test_event')]:
                a = z[key + suffix]; assert len(a) == r['repeats'] and np.all(a > 0)
                assert float(np.median(a) / 128) == r[metric], 'CPU timing summary'
    return dict(status='PASS', cpu_runs=13, physical_device_opened=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command'); p.add_argument('--repeats', type=int, default=128); a = p.parse_args()
    if a.command == 'export': export()
    elif a.command == 'audit-public': print(json.dumps(audit_public()))
    else: run(a.command, a.repeats)
