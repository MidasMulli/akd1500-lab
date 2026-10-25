"""Owned, serial admissions; no background research or automatic retries."""
from pathlib import Path
import argparse, datetime as dt, fcntl, json, os, subprocess
from admission import latency as model
from admission.latency import P, S, save, sha

def now(): return dt.datetime.now(dt.timezone.utc)

def environment():
    paths = ['/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor',
             '/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq',
             '/sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq',
             '/sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq',
             '/sys/class/thermal/thermal_zone0/temp', '/proc/loadavg']
    return dict(uname=list(os.uname()), values={p: Path(p).read_text().strip() for p in paths if Path(p).exists()})

def run(mode, seed, stage, verification, repetition):
    assert mode in ['dictionary', 'replica', 'native'] and seed in model.SEEDS
    assert stage in ['gate', 'repeat'] and verification in ['all', 'final']
    auth = json.loads((S / 'AUTHORITY.json').read_text())
    state = json.loads((S / 'STATE.json').read_text())
    assert not state['closed'] and not state.get('failure')
    assert now() < dt.datetime.fromisoformat(auth['scientific_cutoff_utc'])
    fixture = S / 'fixtures' / f's{seed}'
    meta = json.loads((fixture / 'BUILD.json').read_text())
    for name, h in meta['inherited_files'].items(): assert sha(fixture / name) == h
    def gate(mode, verification):
        return list((S / 'raw').glob(f'{mode}_s124101_gate_{verification}_r*/AUDIT.json'))
    if mode == 'native': assert gate('replica', 'all'), 'CPU full-state qualification first'
    if verification == 'final': assert gate('native', 'all'), 'native full-state qualification first'
    if stage == 'repeat':
        assert gate('native', 'final') and gate('dictionary', 'final')
        assert verification == 'final' and repetition in [0, 1, 2]
    physical = list((S / 'raw').glob('native_*/OWNER.json'))
    n = len(model.schedule(stage))
    assert len(physical) + int(mode == 'native') <= auth['maximum_physical_sessions']
    assert sum(json.loads(p.read_text())['events'] for p in physical) + (n if mode == 'native' else 0) <= auth['maximum_forwards']
    name = f'{mode}_s{seed}_{stage}_{verification}_r{repetition}'
    d = S / 'raw' / name
    assert not d.exists(), 'preserve earlier attempt; no automatic retry'
    d.mkdir(parents=True)
    files = [*fixture.iterdir(), *S.glob('*.md'), S / 'AUTHORITY.json', P / 'build/consensus_latency',
             *[P / p for p in ['admission/latency.py', 'admission/latency_run.py', 'admission/latency_bench.cpp',
               'admission/build_latency.sh', 'admission/consensus.py', 'admission/revision.py', 'admission/transfer.py',
               'admission/model.py', 'src/readout_bench.cpp', 'src/mmio_driver.hpp']]]
    save(d / 'FROZEN.json', dict(utc=now().isoformat(), hashes={str(p.relative_to(P)): sha(p) for p in files if p.is_file()}))
    r = dict(name=name, mode=mode, seed=seed, stage=stage, verification=verification, repetition=repetition,
             events=n, start_utc=now().isoformat(), environment_before=environment(), freeze_sha256=sha(d / 'FROZEN.json'))
    save(d / 'OWNER.json', r)
    def execute():
        with (d / 'stdout').open('w') as out, (d / 'stderr').open('w') as err:
            proc = subprocess.Popen(['taskset', '-c', '2', 'build/consensus_latency', mode,
                 str(fixture), stage, verification, str(d / 'result')], cwd=P, stdout=out, stderr=err)
            r['pid'] = proc.pid; save(d / 'OWNER.json', r)
            try: code = proc.wait(timeout=min(120, (dt.datetime.fromisoformat(auth['scientific_cutoff_utc']) - now()).total_seconds()))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try: code = proc.wait(timeout=2)
                except subprocess.TimeoutExpired: proc.kill(); code = proc.wait()
        r.update(exit_code=code, finish_utc=now().isoformat(), environment_after=environment(),
                 device_free=subprocess.run(['fuser', '/dev/akd1500_0'], capture_output=True).returncode == 1,
                 hashes={p.name: sha(p) for p in d.iterdir() if p.is_file()})
        save(d / 'RECEIPT.json', r)
        try:
            a = model.audit(d); save(d / 'AUDIT.json', a)
        except Exception:
            state['failure'] = name; save(S / 'STATE.json', state); raise
        state['completed'].append(name); save(S / 'STATE.json', state)
        return dict(name=name, status=a['status'], events=n, accepted_transfer=a['accepted_transfer_operational'],
                    convergence=[dict(cycle=c['cycle'], key=c['key'], attempts=c['teaching_attempts']) for c in a['convergence']])
    if mode == 'native':
        with open('/tmp/akd1500-research.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert subprocess.run(['fuser', '/dev/akd1500_0'], capture_output=True).returncode == 1
            result = execute()
    else: result = execute()
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['dictionary', 'replica', 'native'])
    p.add_argument('seed', type=int); p.add_argument('stage', choices=['gate', 'repeat'])
    p.add_argument('verification', choices=['all', 'final']); p.add_argument('repetition', type=int)
    a = p.parse_args(); run(a.mode, a.seed, a.stage, a.verification, a.repetition)
