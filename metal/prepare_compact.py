"""Control for output-volume confounding: same fixed dot products, two counts."""
from pathlib import Path
import argparse
import json
import numpy as np
import akida
from metal.prepare import N, SEED, Select, stream, add_fc, save_common


def main(study):
    out = study / 'fixtures-compact01'; out.mkdir(exist_ok=False)
    x, rows = stream(); (out / 'STREAM.json').write_text(json.dumps(rows, indent=2) + '\n')
    results = []
    for u in [64, 512]:
        rng = np.random.default_rng(SEED); w = np.zeros((N, u), np.int8)
        for j in range(u): w[rng.choice(N, 32, replace=False), j] = 1
        m = akida.Model(); m.add(akida.InputData(input_shape=(1, 1, N), input_bits=1))
        add_fc(m, 'same_binary_dot_products', w, True)
        rw = np.zeros((u, 2), np.int8); rw[:u // 2, 0] = 1; rw[u // 2:, 1] = 1
        add_fc(m, 'two_support_outputs', rw)
        expected = (x.astype(np.int32) @ w.astype(np.int32) > 15).astype(np.int32) @ rw.astype(np.int32)
        got = m.forward(x.reshape(-1, 1, 1, N)).reshape(-1, 2)
        assert np.array_equal(got, expected)
        name = f'compact{u}'
        _, meta = save_common(out, name, m, x, expected, kind='fixed_compact', units=u,
                              control='same weights and input words; thresholded response then fixed two-count readout')
        (out / (name + '.meta')).write_text(f'{N} 0 0 2\n')
        results.append(dict(status='VIRTUAL_AND_COMPILER_PASS', **meta))
    (out / 'BUILD.json').write_text(json.dumps(dict(status='COMPILER_ONLY', physical_device_opened=False,
        cells=results), indent=2) + '\n')
    for row in results: print(json.dumps(row), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('study', type=Path); main(p.parse_args().study)
