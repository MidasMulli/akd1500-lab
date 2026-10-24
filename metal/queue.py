"""Owned, explicitly admitted native teaching bursts and independent recount."""
from pathlib import Path
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import subprocess
import numpy as np
from metal.run import P, S, digest
from pong.relearn.model import Reference


def freeze():
    target = S / 'evidence/QUEUE_FREEZE_01.json'; assert not target.exists()
    files = [P / p for p in ['metal/queue.py', 'src/queue_bench.cpp', 'src/readout_bench.cpp',
        'src/mmio_driver.hpp', 'pong/relearn/model.py', 'tools/build.sh', 'build/queue_bench']]
    files += [S / p for p in ['AUTHORITY.json', 'PROTOCOL.md', 'QUEUE_ADDENDUM.md']]
    f = S / 'fixtures-metal01/width64'
    files += [f.with_suffix(ext) for ext in ['.ram', '.bin', '.meta', '.inputs']]
    z = dict(status='FROZEN_BEFORE_PHYSICAL', utc=dt.datetime.now(dt.timezone.utc).isoformat(),
             hashes={str(p.relative_to(P)): digest(p) for p in files})
    target.write_text(json.dumps(z, indent=2) + '\n'); print(json.dumps(dict(files=len(files), status=z['status'])))


def audit(d):
    rec = json.loads((d / 'RECEIPT.json').read_text()); assert rec['status'] == 'PASS'
    for name, expected in rec['hashes'].items(): assert digest(d / name) == expected
    initial = np.fromfile((P / rec['fixture']).with_suffix('.ram'), '<u4')
    assert np.array_equal(initial, np.fromfile(d / 'native.initial.ram', '<u4'))
    reference = Reference(initial, 512, 64, 32, 2)
    x = np.fromfile(d / 'stream.inputs', np.uint8).reshape(64, 512)
    labels = np.fromfile(d / 'stream.labels', '<i4')
    scores = np.fromfile(d / 'native.scores', '<i4').reshape(64, 64)
    snapshots = np.fromfile(d / 'native.snapshots', '<u4').reshape(8, len(initial))
    for i in range(64):
        assert np.array_equal(scores[i], reference.forward(x[i], int(labels[i]))), ('output', i)
        if i % 8 == 7: assert np.array_equal(snapshots[i // 8], reference.ram), ('RAM', i)
    assert np.array_equal(reference.ram, np.fromfile(d / 'native.final.ram', '<u4'))
    report = dict(status='INDEPENDENT_OUTPUT_RAM_PASS', samples=64, outputs_recounted=4096,
        complete_RAM_snapshots=8, pipeline_override=rec['pipeline_override'],
        control_before=rec['result']['control_before'], control_after=rec['result']['control_after'],
        limitations='Changing buffer-end mode changes counter scope; no cross-mode cycle speed ratio',
        cycles=np.fromfile(d / 'native.cycles', '<u4').tolist())
    (d / 'AUDIT.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report))


def run(name):
    state = json.loads((S / 'STATE.json').read_text()); assert not state['research_admissions_closed']
    entries = [r for r in state['admitted_runs'] if r['name'] == name and r.get('method') == 'teaching_queue']
    assert len(entries) == 1; entry = entries[0]
    frozen = S / 'evidence/QUEUE_FREEZE_01.json'
    for p, expected in json.loads(frozen.read_text())['hashes'].items(): assert digest(P / p) == expected
    auth = json.loads((S / 'AUTHORITY.json').read_text()); now = dt.datetime.now(dt.timezone.utc)
    cutoff = dt.datetime.fromisoformat(auth['scientific_cutoff_utc']); assert now < cutoff
    prior = [json.loads(p.read_text()) for p in (S / 'raw').glob('*/OWNER.json')]
    assert len(prior) < auth['maximum_physical_sessions'] and sum(p['forwards'] for p in prior) + 64 <= auth['maximum_forwards']
    d = S / 'raw' / name; d.mkdir(exist_ok=False); f = P / entry['fixture']
    x = np.fromfile(f.with_suffix('.inputs'), np.uint8).reshape(-1, 512)[336:400]
    (d / 'stream.inputs').write_bytes(x.tobytes())
    (d / 'stream.labels').write_bytes(np.asarray([i % 2 for i in range(64)], '<i4').tobytes())
    rec = dict(**entry, samples=64, forwards=64, start_utc=now.isoformat(), source_freeze_sha256=digest(frozen))
    (d / 'OWNER.json').write_text(json.dumps(rec, indent=2) + '\n')
    with open('/tmp/akd1500-research.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert subprocess.run(['fuser', '/dev/akd1500_0'], capture_output=True).returncode == 1
        args = ['taskset', '-c', '2', str(P / 'build/queue_bench'), str(f), str(d / 'stream'), str(d / 'native'), str(entry['pipeline_override'])]
        with (d / 'native.stdout').open('w') as out, (d / 'native.stderr').open('w') as err:
            proc = subprocess.Popen(args, stdout=out, stderr=err)
            rec['native_pid'] = proc.pid; (d / 'OWNER.json').write_text(json.dumps(rec, indent=2) + '\n')
            try: code = proc.wait(timeout=min(180, int((cutoff - now).total_seconds())))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try: code = proc.wait(timeout=3)
                except subprocess.TimeoutExpired: proc.kill(); code = proc.wait()
        rec.update(status='PASS' if code == 0 else 'FAILED_PRESERVED', exit_code=code,
            device_released=subprocess.run(['fuser', '/dev/akd1500_0'], capture_output=True).returncode == 1,
            finish_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        if code == 0: rec['result'] = json.loads(next(s for s in (d / 'native.stdout').read_text().splitlines() if s.startswith('{"status"')))
        rec['hashes'] = {p.name: digest(p) for p in d.iterdir() if p.is_file() and p.name != 'RECEIPT.json'}
        (d / 'RECEIPT.json').write_text(json.dumps(rec, indent=2) + '\n')
        print(json.dumps(rec.get('result', rec))); assert code == 0 and rec['device_released']


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('name'); a = p.parse_args()
    if a.name == 'freeze': freeze()
    elif a.name.startswith('audit:'): audit(S / 'raw' / a.name.split(':', 1)[1])
    else: run(a.name)
