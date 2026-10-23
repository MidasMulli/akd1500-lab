"""Explicitly admitted, source-bound owned hardware sessions; never auto-expands."""
from pathlib import Path
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import subprocess
import numpy as np
from relay.design import ARMS, SEEDS, schedule

P = Path(__file__).resolve().parents[1]
S = P / 'studies/20261023-signal-relay'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    files = list((P / 'relay').glob('*.py')) + [P / p for p in [
        'src/relay_bench.cpp', 'src/readout_bench.cpp', 'src/mmio_driver.hpp',
        'pong/relearn/model.py', 'tools/build.sh', 'tools/prepare_network.py',
        'tools/prepare_reduction.py', 'tools/program_codec.py', 'build/relay_bench']]
    files += list((S / 'fixtures-relay03').glob('*'))
    files += [S / 'PROTOCOL.md', S / 'PROTOCOL.json', S / 'AUTHORITY.json']
    z = dict(status='FROZEN_BEFORE_PHYSICAL', utc=dt.datetime.now(dt.timezone.utc).isoformat(),
             hashes={str(f.relative_to(P)): digest(f) for f in files if f.is_file()})
    target = S / 'evidence/PHYSICAL_SOURCE_FREEZE.json'
    assert not target.exists(), 'never overwrite a physical freeze'
    target.write_text(json.dumps(z, indent=2) + '\n'); print(json.dumps(dict(status=z['status'], files=len(z['hashes']))))


def run(name, seed, arm, smoke=False):
    now = dt.datetime.now(dt.timezone.utc); auth = json.loads((S / 'AUTHORITY.json').read_text())
    state = json.loads((S / 'STATE.json').read_text())
    assert not state['research_admissions_closed']
    entry = dict(name=name, seed=seed, arm=arm, smoke=smoke)
    assert entry in state['admitted_runs'], ('not explicitly admitted', entry)
    cutoff = dt.datetime.fromisoformat(auth['scientific_cutoff_utc'])
    assert now < cutoff and seed in SEEDS and arm in ARMS
    frozen = json.loads((S / 'evidence/PHYSICAL_SOURCE_FREEZE.json').read_text())
    for path, expected in frozen['hashes'].items(): assert digest(P / path) == expected, ('source drift', path)
    prior = [json.loads(f.read_text()) for f in (S / 'raw').glob('*/OWNER.json')]
    x, rows = schedule(arm)
    indices = list(range(16)) + list(range(256, 272)) + list(range(512, 520)) + list(range(640, 648)) + list(range(896, 904)) + list(range(960, 968)) if smoke else list(range(1024))
    assert len(prior) < auth['maximum_physical_sessions']
    assert sum(r['samples'] for r in prior) + len(indices) <= auth['maximum_forwards']
    d = S / 'raw' / name; d.mkdir(exist_ok=False)
    stream = d / 'stream'
    (d / 'stream.inputs').write_bytes(x[indices].tobytes())
    (d / 'stream.labels').write_bytes(np.asarray([rows[i]['label'] for i in indices], '<i4').tobytes())
    (d / 'stream.masks').write_bytes(bytes(rows[i]['learning_mask'] for i in indices))
    selected = [dict(rows[i], original_index=i) for i in indices]
    (d / 'stream.rows.json').write_text(json.dumps(selected, indent=2) + '\n')
    rec = dict(**entry, samples=len(indices), start_utc=now.isoformat(), fixture=f'studies/{S.name}/fixtures-relay03/s{seed}',
               stream_hashes={f.name: digest(f) for f in d.glob('stream.*')}, source_freeze_sha256=digest(S / 'evidence/PHYSICAL_SOURCE_FREEZE.json'))
    (d / 'OWNER.json').write_text(json.dumps(rec, indent=2) + '\n')
    with open('/tmp/akd1500-research.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert subprocess.run(['fuser', '/dev/akd1500_0'], capture_output=True).returncode == 1, 'device has another owner'
        args = ['taskset', '-c', '2', str(P / 'build/relay_bench'), str(P / rec['fixture']), str(stream), str(d / 'native'), str(len(indices)), '1']
        with (d / 'native.stdout').open('w') as out, (d / 'native.stderr').open('w') as err:
            proc = subprocess.Popen(args, stdout=out, stderr=err)
            rec['native_pid'] = proc.pid; (d / 'OWNER.json').write_text(json.dumps(rec, indent=2) + '\n')
            timeout = min(auth['maximum_individual_seconds'], int((cutoff - now).total_seconds()))
            try: code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try: code = proc.wait(timeout=3)
                except subprocess.TimeoutExpired: proc.kill(); code = proc.wait()
        rec.update(status='PASS' if code == 0 else 'FAILED_PRESERVED', exit_code=code,
                   device_released=subprocess.run(['fuser', '/dev/akd1500_0'], capture_output=True).returncode == 1,
                   finish_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        if code == 0:
            rec['results'] = json.loads(next(s for s in (d / 'native.stdout').read_text().splitlines() if s.startswith('{"status"')))
        rec['hashes'] = {f.name: digest(f) for f in d.iterdir() if f.is_file() and f.name != 'RECEIPT.json'}
        (d / 'RECEIPT.json').write_text(json.dumps(rec, indent=2) + '\n')
        print(json.dumps(rec.get('results', rec)), flush=True)
        assert code == 0 and rec['device_released'], (d / 'native.stderr').read_text()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('name'); p.add_argument('--seed', type=int, default=SEEDS[0])
    p.add_argument('--arm', choices=ARMS, default=ARMS[0]); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    if a.name == 'freeze': freeze()
    else: run(a.name, a.seed, a.arm, a.smoke)
