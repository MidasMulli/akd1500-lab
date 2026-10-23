"""Matched native prediction stream and raw-derived per-arm summaries."""
from pathlib import Path
import argparse
import hashlib
import json
import select
import subprocess
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pong.experiment import N, U, NW, CLASSES, PROTECTED, decode, encode, metrics, schedule, save, sha


class Native:
    def __init__(self, study, name, fixture, cap=4096):
        command = (f'cd /home/nicklomeli/akd1500-lab && timeout --signal=INT --kill-after=5 178 '
            f'/home/nicklomeli/akida-cc-venv/bin/python pong/serve.py studies/{study.name} {name} {fixture} --forwards {cap}')
        self.process = subprocess.Popen(['ssh', '-T', 'jetson', command], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.id = 0; self.ready = self.read(); assert self.ready['kind'] == 'ready'

    def read(self):
        assert select.select([self.process.stdout], [], [], 10)[0], 'native response timeout'
        line = self.process.stdout.readline()
        assert line, 'native stream ended: ' + self.process.stderr.read().decode()
        return json.loads(line)

    def forward(self, x, label):
        self.id += 1; start = time.monotonic_ns()
        self.process.stdin.write(f'F {self.id} {label} {x.tobytes().hex()}\n'.encode()); self.process.stdin.flush()
        row = self.read(); assert row['kind'] == 'forward' and row['id'] == self.id
        row.update(sent_monotonic_ns=start, received_monotonic_ns=time.monotonic_ns(),
            input_sha256=hashlib.sha256(x.tobytes()).hexdigest())
        return row

    def close(self):
        self.process.stdin.write(b'X\n'); self.process.stdin.flush()
        row = self.read(); assert row['kind'] == 'closed' and self.process.wait(timeout=10) == 0
        return row


def summarize(rows, replies, trained_ram_hash):
    predicted = [decode(r['scores']) for r in replies]
    kinds = sorted({r['kind'] for r in rows if r['kind'] != 'initial_training'})
    prefix_count = sum(r['kind'] == 'initial_training' for r in rows)
    old_b = [(r, p) for r, p in zip(rows, predicted) if r['kind'] == 'old_physics_probe' and r['observation']['court'] == 1]
    return dict(metrics={k: metrics(rows, predicted, k) for k in kinds},
        trained_ram_sha256=trained_ram_hash,
        prefix_scores_sha256=hashlib.sha256(np.array([r['scores'] for r in replies[:prefix_count]], '<i4').tobytes()).hexdigest(),
        old_court_b_accuracy=float(np.mean([p == r['truth'] for r, p in old_b])),
        post_prefix_connection_bit_edits=sum(r['connection_bit_edits'] for r in replies[prefix_count:]),
        native_forwards=len(replies), prediction=predicted)


def run(study, seed, arm, split):
    name = f'{split}_{seed}_{arm}'; output = study / 'runs' / name
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((study / 'PROTOCOL.json').read_text()); rows = schedule(plan, split, arm == 'frozen')
    fixture = f'studies/{study.name}/fixtures-pong01/s{seed}'
    native = Native(study, name, fixture, len(rows)); replies = []
    with (output / 'events.jsonl').open('w') as log:
        try:
            for row in rows:
                reply = native.forward(encode(row['observation']), row['label']); replies.append(reply)
                log.write(json.dumps(dict(**row, response=reply)) + '\n'); log.flush()
            cleanup = native.close()
            save(output / 'CAPTURE.json', dict(status='MATCHED_NATIVE_COMPLETE', cleanup=cleanup,
                events_sha256=sha(output / 'events.jsonl'), seed=seed, arm=arm, split=split))
        except BaseException as error:
            save(output / 'FAILURE.json', dict(status='FAILED_PRESERVED', error=repr(error), completed=len(replies)))
            raise
        finally:
            if native.process.poll() is None:
                try: native.close()
                except Exception: native.process.terminate(); native.process.wait(timeout=10)
    print(json.dumps(dict(status='MATCHED_NATIVE_COMPLETE', run=name, forwards=len(replies))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('study', type=Path); p.add_argument('seed', type=int)
    p.add_argument('arm', choices=['full', 'frozen']); p.add_argument('--split', default='development', choices=['development', 'confirmation'])
    a = p.parse_args(); run(a.study.resolve(), a.seed, a.arm, a.split)
