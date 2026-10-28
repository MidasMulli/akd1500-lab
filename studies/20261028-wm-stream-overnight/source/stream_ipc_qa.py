"""Exercise the compiled interactive CPU path against every sealed witness byte."""
import argparse, json, os, struct, subprocess
from pathlib import Path
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('actor', type=Path); p.add_argument('fixture', type=Path); p.add_argument('out', type=Path)
a = p.parse_args(); f = a.fixture
trace = np.load(f / 'TRACE.npz'); count = len(trace['outputs'])
inputs = np.fromfile(f / 'inputs.bin', 'u1').reshape(count, 1024)
fields = [np.fromfile(f / (n + 's.bin'), '<i4') for n in ['net', 'mode', 'label', 'external']]
W = int((f / 'geometry.txt').read_text().split()[4])
with Path(str(a.out) + '.stderr').open('wb') as err:
    proc = subprocess.Popen([str(a.actor), 'stream-cpu', str(f), str(a.out), str(count)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err)
    try:
        for i in range(count):
            proc.stdin.write(struct.pack('<iiii', *(int(v[i]) for v in fields)) + inputs[i].tobytes()); proc.stdin.flush()
            data = bytearray()
            while len(data) < W * 4:
                part = os.read(proc.stdout.fileno(), W * 4 - len(data))
                assert part, ('truncated actor reply', i, proc.poll())
                data.extend(part)
            assert np.array_equal(np.frombuffer(data, '<u4'), trace['outputs'][i]), ('output mismatch', i)
        proc.stdin.close(); tail = proc.stdout.read(); assert proc.wait(timeout=15) == 0, tail
        assert json.loads(tail)['status'] == 'PASS'
    finally:
        if proc.poll() is None: proc.terminate(); proc.wait(timeout=5)
for key in ['outputs', 'states0', 'states1']:
    actual = np.fromfile(str(a.out) + '.' + key, '<u4').reshape(trace[key].shape)
    assert np.array_equal(actual, trace[key]), key
r = dict(status='PASS', physical=False, forwards=count, interactive_outputs_and_complete_RAM_exact=True)
Path(str(a.out) + '.QA.json').write_text(json.dumps(r, indent=2) + '\n'); print(json.dumps(r))
