"""Virtual fixtures for the fixed development selection and sealed test bank."""
from pathlib import Path
import argparse
import json
import struct
import sys
import akida
import numpy as np
from bank import initial, save, sha, SEEDS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare import parts


def dataset(directory, split, encoder):
    z = json.loads((directory / 'EXPERIENCE.json').read_text())
    arrays = np.load(directory / 'inputs.npz', allow_pickle=False)
    rows = [r for r in z['rows'] if r['split'] == split and r['valid']]
    return np.array([arrays[f'{r["id"]}_{r["action"]}_{encoder}'] for r in rows], np.uint8), np.array([r['success'] for r in rows], np.int32), rows


def prepare(train_dir, evaluation_dir, output, phase, selected=None):
    output.mkdir(parents=True, exist_ok=False)
    encoders = ['legacy', 'projection'] if phase == 'calibration' else [json.loads(selected.read_text())['encoder']]
    clock = json.loads((train_dir / 'EXPERIENCE.json').read_text()).get('observed_clock', False)
    seeds = ([82811,82812,82813] if clock else [82801,82802,82803]) if phase == 'calibration' else (list(range(82911,82916)) if clock else SEEDS)
    arms = ['full', 'frozen'] if phase == 'calibration' else ['full', 'frozen', 'action_only', 'shuffled']
    results = []
    for encoder in encoders:
        tx, ty, tr = dataset(train_dir, 'train', encoder)
        ex, ey, er = dataset(evaluation_dir, 'dev' if phase == 'calibration' else 'test', encoder)
        assert len(ex) > 0 and len(tx) > 0 and len(ex) % 2 == len(tx) % 2 == 0
        hashes = {}
        for x, y in zip(np.concatenate([tx, ex]), np.concatenate([ty, ey])):
            hashes.setdefault(x.tobytes(), set()).add(int(y))
        aliases = sum(len(v) > 1 for v in hashes.values())
        for seed in seeds:
            n = tx.shape[1]
            weights = initial(seed, n)
            model = akida.Model()
            model.add(akida.InputData(input_shape=(1, 1, n), input_bits=1))
            model.add(akida.FullyConnected(name='sealed_visual_association', units=192, weights_bits=1, activation=False))
            model.compile(optimizer=akida.AkidaUnsupervised(num_weights=32, num_classes=3,
                initial_plasticity=.5, min_plasticity=.25, plasticity_decay=.125, learning_competition=.25))
            layer = model.get_layer('sealed_visual_association')
            layer.set_variable('weights', weights.reshape(1, 1, n, 192))
            layer.set_variable('threshold_learning', np.array([0] * 128 + [32] * 64, np.int32))
            assert np.array_equal(model.forward(ex[:2].reshape(-1, 1, 1, n)).reshape(-1, 192), ex[:2].astype(np.int32) @ weights.astype(np.int32))
            model.save(str(output / f'{encoder}_{seed}.fbz'))
            model.map(akida.AKD1500(), hw_only=True)
            assert len(model.sequences) == len(model.sequences[0].passes) == 1
            program = bytes(model.sequences[0].program)
            ram, npid = parts(program)
            assert len(ram) == 2+192*(4+n//32)
            if clock:
                classes=[np.flatnonzero(ty==label) for label in [0,1]]
                assert all(len(c)>0 for c in classes)
                balanced=np.concatenate([np.resize(c,max(map(len,classes))) for c in classes])
                order=np.concatenate([np.random.default_rng(seed+e).permutation(balanced) for e in range(4)])
            else:order = np.concatenate([np.random.default_rng(seed + e).permutation(len(tx)) for e in range(4)])
            inputs = np.concatenate([ex, tx[order], ex])
            labels = np.concatenate([np.full(len(ex), 2), ty[order], np.full(len(ex), 2)]).astype('<i4')
            for arm in arms:
                prefix = output / f'{encoder}_s{seed}_{arm}'
                xx, ll = inputs.copy(), labels.copy()
                if arm == 'frozen':
                    ll[:] = 2
                if arm == 'action_only':
                    actions = (xx[:, :n//2].sum(axis=1) == 0).astype(int)
                    xx[:] = 0
                    for i, action in enumerate(actions):
                        xx[i, (n//2) * action:(n//2) * action + 64] = 1
                if arm == 'shuffled':
                    ll[len(ex):len(ex) + len(order)] = ty[order][np.random.default_rng(82999).permutation(len(order))]
                for suffix, content in [('bin', program), ('ram', struct.pack('<' + 'I' * len(ram), *ram)),
                    ('meta', f'{n} 192 32 3\n'.encode()), ('inputs', xx.tobytes()), ('labels', ll.tobytes())]:
                    prefix.with_suffix('.' + suffix).write_bytes(content)
                meta = dict(seed=seed, encoder=encoder, arm=arm, evaluation_samples=len(ex), training_presentations=len(order),
                    samples=len(inputs), evaluation_rows=er, train_rows=tr, order=order.tolist(),
                    opposite_label_aliases=aliases, learning_np=npid, program_sha256=sha(prefix.with_suffix('.bin')))
                save(prefix.with_suffix('.json'), meta)
                results.append(dict(prefix=prefix.name, **{k: v for k, v in meta.items() if k not in ['evaluation_rows', 'train_rows', 'order']}))
    save(output / 'BUILD.json', dict(status='VIRTUAL_ONLY_PASS', phase=phase, physical_device_opened=False,
        train_sha256=sha(train_dir / 'inputs.npz'), evaluation_sha256=sha(evaluation_dir / 'inputs.npz'), rows=results))
    print('VIRTUAL_ONLY_PASS', phase, len(results), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('train', type=Path)
    p.add_argument('evaluation', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--phase', choices=['calibration', 'test'], required=True)
    p.add_argument('--selected', type=Path)
    a = p.parse_args()
    prepare(a.train.resolve(), a.evaluation.resolve(), a.output.resolve(), a.phase, a.selected)
