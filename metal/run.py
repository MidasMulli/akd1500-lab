"""Explicit admissions, immutable source freezes and an exclusive device owner."""
from pathlib import Path
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import subprocess
import numpy as np

P = Path(__file__).resolve().parents[1]
S = P / 'studies/20261023-metal-characterization'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(name):
    files = list((P / 'metal').glob('*.py'))
    files += [P / p for p in ['src/metal_bench.cpp', 'src/readout_bench.cpp',
              'src/mmio_driver.hpp', 'pong/relearn/model.py', 'tools/build.sh',
              'tools/prepare.py', 'tools/prepare_network.py', 'build/metal_bench']]
    files += [p for d in S.glob('fixtures-*') for p in d.iterdir() if p.is_file()]
    files += [S / 'PROTOCOL.md', S / 'AUTHORITY.json']
    target = S / 'evidence' / (name + '.json')
    assert not target.exists(), 'never overwrite a source freeze'
    target.write_text(json.dumps(dict(status='FROZEN_BEFORE_PHYSICAL',
        utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        hashes={str(p.relative_to(P)): digest(p) for p in files}), indent=2) + '\n')
    print(json.dumps(dict(status='FROZEN_BEFORE_PHYSICAL', files=len(files), path=str(target))))


def run(name):
    auth = json.loads((S / 'AUTHORITY.json').read_text())
    state = json.loads((S / 'STATE.json').read_text())
    assert not state['research_admissions_closed']
    found = [r for r in state['admitted_runs'] if r['name'] == name]
    assert len(found) == 1, ('not explicitly admitted', name)
    entry = found[0]
    frozen = S / 'evidence' / (entry['freeze'] + '.json')
    for p, expected in json.loads(frozen.read_text())['hashes'].items():
        assert digest(P / p) == expected, ('source drift', p)
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = dt.datetime.fromisoformat(auth['scientific_cutoff_utc'])
    assert now < cutoff
    f = P / entry['fixture']; info = json.loads(f.with_suffix('.json').read_text())
    rows = json.loads((f.parent / 'STREAM.json').read_text())
    x = np.frombuffer(f.with_suffix('.inputs').read_bytes(), np.uint8).reshape(-1, 512)
    if entry['smoke']:
        indices = [i * 16 for i in range(7)] + [i * 16 + 1 for i in range(7)] + [2, 3]
        if info['kind'] == 'learner': indices += list(range(336, 344)) + list(range(400, 408))
        else: indices += list(range(400, 416))
    else: indices = list(range(len(rows)))
    selected = [dict(rows[i], original_index=i) for i in indices]
    if info['kind'] != 'learner':
        for row in selected: row['learning'] = False
    prior = [json.loads(p.read_text()) for p in (S / 'raw').glob('*/OWNER.json')]
    assert len(prior) < auth['maximum_physical_sessions']
    forwards = len(indices) + 48
    assert sum(r['forwards'] for r in prior) + forwards <= auth['maximum_forwards']
    d = S / 'raw' / name; d.mkdir(exist_ok=False)
    stream = d / 'stream'
    stream.with_suffix('.inputs').write_bytes(x[indices].tobytes())
    stream.with_suffix('.labels').write_bytes(np.asarray([r['label'] for r in selected], '<i4').tobytes())
    stream.with_suffix('.modes').write_bytes(bytes(int(r['learning']) for r in selected))
    stream.with_suffix('.rows.json').write_text(json.dumps(selected, indent=2) + '\n')
    expected = np.frombuffer(f.with_suffix('.expected').read_bytes(), '<i4').reshape(-1, info['outputs'])
    stream.with_suffix('.expected').write_bytes(expected[indices].tobytes())
    rec = dict(**entry, samples=len(indices), forwards=forwards, start_utc=now.isoformat(),
               source_freeze_sha256=digest(frozen), fixture_info=info)
    (d / 'OWNER.json').write_text(json.dumps(rec, indent=2) + '\n')
    with open('/tmp/akd1500-research.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check = subprocess.run(['fuser', '/dev/akd1500_0'], capture_output=True)
        assert check.returncode == 1, ('device has another owner or check failed', check.returncode)
        args = ['taskset', '-c', '2', str(P / 'build/metal_bench'), str(f), str(stream),
                str(d / 'native'), str(len(indices)), 'learner' if info['kind'] == 'learner' else 'fixed']
        with (d / 'native.stdout').open('w') as out, (d / 'native.stderr').open('w') as err:
            proc = subprocess.Popen(args, stdout=out, stderr=err)
            rec['native_pid'] = proc.pid
            (d / 'OWNER.json').write_text(json.dumps(rec, indent=2) + '\n')
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
            rec['result'] = json.loads(next(s for s in (d / 'native.stdout').read_text().splitlines() if s.startswith('{"status"')))
        rec['hashes'] = {p.name: digest(p) for p in d.iterdir() if p.is_file() and p.name != 'RECEIPT.json'}
        (d / 'RECEIPT.json').write_text(json.dumps(rec, indent=2) + '\n')
        print(json.dumps(rec.get('result', rec)), flush=True)
        assert code == 0 and rec['device_released'], (d / 'native.stderr').read_text()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('name'); p.add_argument('--freeze', action='store_true')
    a = p.parse_args(); freeze(a.name) if a.freeze else run(a.name)
