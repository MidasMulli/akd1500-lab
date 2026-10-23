"""Compiler-only Pong fixture builder, never opens physical hardware."""
from pathlib import Path
import argparse
import importlib.util
import json
import struct
import sys
import akida
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pong.experiment import N, U, NW, CLASSES, PER_CLASS, SEEDS, save, sha
spec = importlib.util.spec_from_file_location('fixture_decoder', ROOT / 'tools/prepare.py')
decoder = importlib.util.module_from_spec(spec); spec.loader.exec_module(decoder)


def build(output):
    output.mkdir(parents=True, exist_ok=False); builds = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed); weights = np.zeros((N, U), np.int8)
        for j in range(U): weights[rng.choice(N, NW, replace=False), j] = 1
        model = akida.Model(); model.add(akida.InputData(input_shape=(1, 1, N), input_bits=1))
        model.add(akida.FullyConnected(name='landing_predictor', units=U, weights_bits=1, activation=False))
        model.compile(optimizer=akida.AkidaUnsupervised(num_weights=NW, num_classes=CLASSES,
            initial_plasticity=.5, min_plasticity=.25, plasticity_decay=.125, learning_competition=.25))
        layer = model.get_layer('landing_predictor'); layer.set_variable('weights', weights.reshape(1, 1, N, U))
        layer.set_variable('threshold_learning', np.array([0] * (U - PER_CLASS) + [NW] * PER_CLASS, np.int32))
        x = rng.integers(0, 2, (2, N), dtype=np.uint8)
        assert np.array_equal(model.forward(x.reshape(2, 1, 1, N)).reshape(2, U), x.astype(np.int32) @ weights.astype(np.int32))
        model.map(akida.AKD1500(), hw_only=True)
        assert len(model.sequences) == len(model.sequences[0].passes) == 1
        program = bytes(model.sequences[0].program); ram, npid = decoder.parts(program)
        assert len(ram) == 2 + U * (4 + N // 32)
        p = output / f's{seed}'
        p.with_suffix('.bin').write_bytes(program)
        p.with_suffix('.ram').write_bytes(struct.pack('<' + 'I' * len(ram), *ram))
        p.with_suffix('.meta').write_text(f'{N} {U} {NW} {CLASSES}\n')
        row = dict(seed=seed, prefix=p.name, learning_np=npid, inputs=N, neurons=U,
            num_weights=NW, classes=CLASSES, program_sha256=sha(p.with_suffix('.bin')))
        save(p.with_suffix('.json'), row); builds.append(row)
    save(output / 'BUILD.json', dict(status='COMPILER_ONLY_PASS', physical_device_opened=False, rows=builds))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('output', type=Path); build(p.parse_args().output.resolve())
