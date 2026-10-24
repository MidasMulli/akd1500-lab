"""Virtual/compiler preflight only. Never discovers or opens physical devices."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import traceback
import numpy as np
import akida

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'tools'))
from prepare import PARAMS, parts
from prepare_network import TARGETS

N = 512
WIDTHS = [8, 32, 64, 128, 256, 512]
DENSITIES = [0, 1, 8, 32, 128, 256, 512]
SEED = 98101


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Select(akida.MapConstraints):
    def select_nps(self, source_nps, num_nps, type):
        found = [q for target in TARGETS for q in source_nps
                 if [q.col, q.row, q.id] == target]
        assert type == akida.NP.Type.FNP3 and len(found) >= num_nps, (str(type), num_nps)
        return found[:num_nps]


def mapping(model):
    return [dict(layer=l.name, nps=[dict(type=str(q.type),
                 np=[q.ident.col, q.ident.row, q.ident.id]) for q in l.mapping.nps])
            for l in model.sequences[0].passes[0].layers]


def stream():
    rng = np.random.default_rng(SEED + 7)
    x, rows = [], []
    # Widths see identical inputs, labels and interleaved density order.
    for block in range(3):
        for d in rng.permutation(DENSITIES):
            for repeat in range(16):
                a = np.zeros(N, np.uint8)
                if d: a[rng.choice(N, int(d), replace=False)] = 1
                x.append(a)
                rows.append(dict(phase='initial_inference', learning=False,
                                 density=int(d), block=block, label=repeat % 2))
    for i in range(64):
        a = np.zeros(N, np.uint8); a[rng.choice(N, N // 2, replace=False)] = 1
        x.append(a)
        rows.append(dict(phase='native_learning', learning=True,
                         density=N // 2, block=0, label=i % 2))
    for i in range(16):
        a = np.zeros(N, np.uint8); a[rng.choice(N, N // 2, replace=False)] = 1
        x.append(a)
        rows.append(dict(phase='retained_inference', learning=False,
                         density=N // 2, block=0, label=i % 2))
    # Same frozen suffix will be repeated as serialized and resident burst input.
    return np.asarray(x), rows


def add_fc(model, name, w, activation=False, inbounds=None):
    l = akida.FullyConnected(name=name, units=w.shape[1], weights_bits=1,
                            activation=activation, act_bits=1)
    model.add(l, [] if inbounds is None else inbounds)
    l.set_variable('weights', w.reshape(1, 1, *w.shape).astype(np.int8))
    if activation:
        l.set_variable('threshold', np.full(w.shape[1], 15, np.int32))
        l.set_variable('act_step', np.ones(w.shape[1], np.float32))
    return l


def save_common(out, name, model, x, expected, **metadata):
    d = akida.AKD1500()
    model.map(d, hw_only=True, mode=akida.MapMode.Minimal, constraints=Select(d))
    assert len(model.sequences) == 1 and len(model.sequences[0].passes) == 1
    b = bytes(model.sequences[0].program)
    f = out / name
    f.with_suffix('.bin').write_bytes(b)
    f.with_suffix('.inputs').write_bytes(x.tobytes())
    f.with_suffix('.expected').write_bytes(expected.astype('<i4').tobytes())
    metadata.update(name=name, mapping=mapping(model), inputs=N,
                    outputs=int(expected.shape[1]), program_sha256=hashlib.sha256(b).hexdigest())
    f.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return b, metadata


def main(study, attempt):
    out = study / attempt; out.mkdir(exist_ok=False)
    x, rows = stream()
    (out / 'STREAM.json').write_text(json.dumps(rows, indent=2) + '\n')
    results = []
    for u in WIDTHS:
        name = f'width{u}'
        try:
            rng = np.random.default_rng(SEED)
            w = np.zeros((N, u), np.int8)
            for j in range(u): w[rng.choice(N, 32, replace=False), j] = 1
            m = akida.Model(); m.add(akida.InputData(input_shape=(1, 1, N), input_bits=1))
            l = add_fc(m, 'learning_bank', w)
            m.compile(optimizer=akida.AkidaUnsupervised(**PARAMS))
            l.set_variable('weights', w.reshape(1, 1, N, u))
            l.set_variable('threshold_learning', np.zeros(u, np.int32))
            expected = x.astype(np.int32) @ w.astype(np.int32)
            got = m.forward(x.reshape(-1, 1, 1, N)).reshape(-1, u)
            assert np.array_equal(got, expected)
            b, meta = save_common(out, name, m, x, expected, kind='learner',
                                  units=u, weights_per_neuron=32, classes=2)
            ram, npid = parts(b); assert len(ram) == 2 + u * 20
            assert len({tuple(q['np']) for row in meta['mapping'] for q in row['nps']}) == 1
            (out / (name + '.ram')).write_bytes(np.asarray(ram, '<u4').tobytes())
            (out / (name + '.meta')).write_text(f'{N} {u} 32 2\n')
            results.append(dict(status='VIRTUAL_AND_COMPILER_PASS', **meta))
        except Exception as e:
            results.append(dict(name=name, status='FAILED_PRESERVED', error=str(e), traceback=traceback.format_exc()))
    # A compiler/virtual feasibility check for genuinely independent branches.
    for which in ['A', 'B', 'AB']:
        name = 'fork_' + which
        try:
            rng = np.random.default_rng(SEED + 11)
            m = akida.Model(); inp = akida.InputData(input_shape=(1, 1, N), input_bits=1); m.add(inp)
            branches, expected_parts = [], []
            for letter in 'AB':
                w = np.zeros((N, 64), np.int8)
                for j in range(64): w[rng.choice(N, 32, replace=False), j] = 1
                if letter not in which: continue
                branches.append(add_fc(m, 'branch_' + letter, w, True, [inp]))
                expected_parts.append((letter, (x.astype(np.int32) @ w.astype(np.int32) > 15).astype(np.int32)))
            rw = np.zeros((64 * len(branches), 2), np.int8)
            expected = np.zeros((len(x), 2), np.int32)
            for i, (letter, part) in enumerate(expected_parts):
                rw[i * 64:(i + 1) * 64, 'AB'.index(letter)] = 1
                expected[:, 'AB'.index(letter)] = part.sum(axis=1)
            add_fc(m, 'branch_supports', rw, False, branches)
            got = m.forward(x.reshape(-1, 1, 1, N)).reshape(-1, 2)
            assert np.array_equal(got, expected), 'independent branch virtual outputs'
            _, meta = save_common(out, name, m, x, expected, kind='fixed_fork', branches=which)
            results.append(dict(status='VIRTUAL_AND_COMPILER_PASS', **meta))
        except Exception as e:
            results.append(dict(name=name, status='FAILED_PRESERVED', error=str(e), traceback=traceback.format_exc()))
    report = dict(status='COMPILER_ONLY', physical_device_opened=False, sdk=akida.__version__,
                  cells=results, files={p.name: digest(p) for p in out.iterdir() if p.is_file()})
    (out / 'BUILD.json').write_text(json.dumps(report, indent=2) + '\n')
    for r in results: print(json.dumps({k: v for k, v in r.items() if k != 'traceback'}), flush=True)
    assert all(r['status'] == 'VIRTUAL_AND_COMPILER_PASS' for r in results[:len(WIDTHS)]), 'width preflight failed'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('study', type=Path)
    p.add_argument('--attempt', default='fixtures-metal01'); a = p.parse_args(); main(a.study, a.attempt)
