"""Compiler/virtual channel-partition probe, without physical device discovery."""
from pathlib import Path
import argparse
import json
import traceback
import numpy as np
import akida
from metal.prepare import N, SEED, stream, mapping


class Select(akida.MapConstraints):
    def select_nps(self, source_nps, num_nps, type):
        if type == akida.NP.Type.FNP3:
            found = [q for q in source_nps if [q.col, q.row, q.id] == [3, 3, 0]]
        else:
            assert type in [akida.NP.Type.CNP1, akida.NP.Type.CNP2], str(type)
            found = [q for q in source_nps if [q.col, q.row, q.id] != [3, 3, 0]]
        assert len(found) >= num_nps, (str(type), num_nps)
        return found[:num_nps]


def main(study, attempt):
    out = study / attempt; out.mkdir(exist_ok=False)
    x, rows = stream(); results = []
    (out / 'STREAM.json').write_text(json.dumps(rows, indent=2) + '\n')
    for u in [64, 128, 256, 512]:
        name = f'cnp{u}'
        try:
            rng = np.random.default_rng(SEED + 19)
            w = np.zeros((N, u), np.int8)
            for j in range(u): w[rng.choice(N, 32, replace=False), j] = 1
            m = akida.Model(); m.add(akida.InputData(input_shape=(1, 1, N), input_bits=1))
            m.add(akida.Convolutional(name='partitioned_resident_work', filters=u,
                  kernel_size=(1, 1), weights_bits=1, activation=True, act_bits=1))
            l = m.get_layer('partitioned_resident_work')
            l.set_variable('weights', w.reshape(1, 1, N, u))
            l.set_variable('threshold', np.full(u, 15, np.int32))
            l.set_variable('act_step', np.ones(u, np.float32))
            rw = np.zeros((u, 2), np.int8); rw[:u // 2, 0] = 1; rw[u // 2:, 1] = 1
            m.add(akida.FullyConnected(name='two_support_outputs', units=2, weights_bits=1, activation=False))
            m.get_layer('two_support_outputs').set_variable('weights', rw.reshape(1, 1, u, 2))
            expected = (x.astype(np.int32) @ w.astype(np.int32) > 15).astype(np.int32) @ rw.astype(np.int32)
            got = m.forward(x.reshape(-1, 1, 1, N)).reshape(-1, 2)
            assert np.array_equal(got, expected)
            d = akida.AKD1500(); m.map(d, hw_only=True, mode=akida.MapMode.Minimal, constraints=Select(d))
            assert len(m.sequences) == 1 and len(m.sequences[0].passes) == 1
            program = bytes(m.sequences[0].program)
            f = out / name
            for ext, data in [('bin', program), ('inputs', x.tobytes()), ('expected', expected.astype('<i4').tobytes())]:
                f.with_suffix('.' + ext).write_bytes(data)
            f.with_suffix('.meta').write_text(f'{N} 0 0 2\n')
            meta = dict(name=name, kind='fixed_cnp', inputs=N, outputs=2,
                        units=u, mapping=mapping(m), source='compiler-generated single-pass graph')
            f.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
            results.append(dict(status='VIRTUAL_AND_COMPILER_PASS', **meta))
        except Exception as e:
            results.append(dict(name=name, status='FAILED_PRESERVED', error=str(e), traceback=traceback.format_exc()))
    (out / 'BUILD.json').write_text(json.dumps(dict(status='COMPILER_ONLY', physical_device_opened=False,
                  cells=results), indent=2) + '\n')
    for row in results: print(json.dumps({k: v for k, v in row.items() if k != 'traceback'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('study', type=Path)
    p.add_argument('--attempt', default='fixtures-cnp01'); a = p.parse_args(); main(a.study, a.attempt)
