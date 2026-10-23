"""Bounded owner for the interactive physical stream; stdout is actual device replies."""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(study, name, fixture, cap):
    assert re.fullmatch(r'[A-Za-z0-9_-]+', name)
    authority = json.loads((study / 'AUTHORITY.json').read_text())
    state = json.loads((study / 'STATE.json').read_text())
    assert not state['research_admissions_closed'] and state['admission'] == 'CALIBRATION_MECHANISM_LIVE_ONLY'
    assert 'fixtures-cal' in str(fixture), 'sealed test is not admitted'
    now = datetime.datetime.now(datetime.timezone.utc)
    assert now < datetime.datetime.fromisoformat(authority['scientific_cutoff_utc'])
    owners = list((study / 'raw').glob('*/OWNER.json'))
    assert len(owners) < authority['maximum_physical_sessions']
    assert sum(json.loads(p.read_text())['samples'] for p in owners) + cap <= authority['maximum_forwards']
    manifest = json.loads((study / 'evidence/PHYSICAL_SOURCE_FREEZE.json').read_text())
    for path, digest in manifest['hashes'].items():
        assert sha(ROOT / path) == digest, ('source drift', path)
    target = study / 'raw' / name
    target.mkdir(exist_ok=False)
    for ext in ['bin', 'ram', 'meta']:
        shutil.copy2(fixture.with_suffix('.' + ext), target / ('fixture.' + ext))
    record = dict(name=name, fixture=str((target / 'fixture').relative_to(ROOT)), samples=cap, learning=1,
        start_utc=now.isoformat(), binary_sha256=sha(ROOT / 'build/mario_stream'),
        program_sha256=sha(fixture.with_suffix('.bin')), mode='interactive_native_with_inline_CPU')
    owner = target / 'OWNER.json'
    owner.write_text(json.dumps(record, indent=2) + '\n')
    child = None
    count = 0
    start = time.monotonic()
    status = 'FAILED_PRESERVED'
    try:
        with open('/tmp/akd1500-research.lock', 'a') as lock, (target / 'native.stderr').open('w') as error, (target / 'native.stdout').open('w') as log, (target / 'commands.txt').open('w') as commands:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert subprocess.run(['fuser', '/dev/akd1500_0'], capture_output=True).returncode == 1
            child = subprocess.Popen(['taskset', '-c', '2', str(ROOT / 'build/mario_stream'), str(target / 'fixture'), str(target / 'native'), str(cap)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error)
            record['native_pid'] = child.pid
            owner.write_text(json.dumps(record, indent=2) + '\n')
            selector = selectors.DefaultSelector()
            selector.register(sys.stdin.buffer, selectors.EVENT_READ, 'input')
            selector.register(child.stdout, selectors.EVENT_READ, 'device')
            closed = False
            while not closed:
                assert time.monotonic() - start < 170, 'stream wall deadline'
                assert datetime.datetime.now(datetime.timezone.utc) < datetime.datetime.fromisoformat(authority['scientific_cutoff_utc'])
                for key, _ in selector.select(timeout=.1):
                    if key.data == 'input':
                        line = sys.stdin.buffer.readline()
                        if not line:
                            line = b'X\n'
                            selector.unregister(sys.stdin.buffer)
                        commands.write(line.decode()); commands.flush()
                        child.stdin.write(line); child.stdin.flush()
                    else:
                        line = child.stdout.readline()
                        if not line:
                            raise RuntimeError('device stream ended without cleanup receipt')
                        row = json.loads(line)
                        log.write(line.decode()); log.flush()
                        sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
                        count += row['kind'] == 'forward'
                        closed = row['kind'] == 'closed'
            assert child.wait(timeout=3) == 0
            assert count <= cap
            status = 'PASS'
    finally:
        if child is not None and child.poll() is None:
            try:
                child.stdin.write(b'X\n'); child.stdin.flush()
                child.wait(timeout=3)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                child.terminate()
                try: child.wait(timeout=3)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
        record.update(status=status, samples=count, finish_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            device_released=subprocess.run(['fuser', '/dev/akd1500_0'], capture_output=True).returncode == 1,
            native_exit_code=None if child is None else child.returncode)
        owner.write_text(json.dumps(record, indent=2) + '\n')
        record['hashes'] = {p.name: sha(p) for p in target.iterdir() if p.is_file()}
        (target / 'RECEIPT.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    def interrupt(signum, frame):
        raise TimeoutError('owned stream interrupted')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    p = argparse.ArgumentParser()
    p.add_argument('study', type=Path)
    p.add_argument('name')
    p.add_argument('fixture', type=Path)
    p.add_argument('--forwards', type=int, default=4096)
    a = p.parse_args()
    run(a.study.resolve(), a.name, a.fixture.resolve(), a.forwards)
