"""Compiler/virtual only: equal I/O, all compaction operations stay on chip."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import akida
from metal.prepare import add_fc, mapping
from metal.ablation_prepare import memory_packets
from kernel.design import weights, stream, expected


class Select(akida.MapConstraints):
    def __init__(self, device, targets): super().__init__(device); self.targets = targets
    def select_nps(self, source_nps, num_nps, type):
        assert type == akida.NP.Type.FNP3, ('unexpected kernel substitution', str(type))
        found = [p for target in self.targets for p in source_nps if [p.col, p.row, p.id] == target]
        assert len(found) >= num_nps; return found[:num_nps]


def main(study):
    out = study / 'fixtures-kernel01'; out.mkdir(exist_ok=False); x, rows = stream(); result = []
    (out / 'STREAM.json').write_text(json.dumps(rows, indent=2) + '\n')
    for u in [256, 512]:
        for kind in ['direct', 'compacted']:
            name = f'u{u}_{kind}'; m = akida.Model(); m.add(akida.InputData(input_shape=(1, 1, 512), input_bits=1))
            w = weights(u)
            if kind == 'compacted':
                cw = np.zeros((512, 128), np.int8)
                for k in range(4): cw[k*128:(k+1)*128] = np.eye(128, dtype=np.int8)
                l = add_fc(m, 'native_four_copy_consensus', cw, True)
                l.set_variable('threshold', np.full(128, 3, np.int32))
            else: w = np.vstack([w, np.zeros((384, u), np.int8)])
            add_fc(m, 'same_latent_detectors', w, True)
            rw = np.zeros((u, 2), np.int8); rw[:u // 2, 0] = 1; rw[u // 2:, 1] = 1
            add_fc(m, 'two_support_outputs', rw)
            want = expected(x, u); got = m.forward(x.reshape(-1, 1, 1, 512)).reshape(-1, 2)
            assert np.array_equal(got, want), 'virtual native-graph equivalence'
            targets = ([[1, 3, 0]] if kind == 'compacted' else []) + [[2, 1, 0], [3, 3, 0]]
            device = akida.AKD1500(); m.map(device, hw_only=True, mode=akida.MapMode.Minimal, constraints=Select(device, targets))
            assert len(m.sequences) == len(m.sequences[0].passes) == 1
            image = bytes(m.sequences[0].program); packets = memory_packets(image)
            assert len(packets) == len(targets) and [p['np'] for p in packets] == targets
            f = out / name
            for ext, data in [('bin', image), ('inputs', x.tobytes()), ('expected', want.astype('<i4').tobytes())]: f.with_suffix('.' + ext).write_bytes(data)
            f.with_suffix('.meta').write_text('512 0 0 2\n')
            f.with_suffix('.regions').write_text(str(len(packets)) + '\n' + '\n'.join(str(len(p['packet'])) for p in packets) + '\n')
            for j, p in enumerate(packets): (out / (name + f'.packet{j}.ram')).write_bytes(np.asarray(p['packet'], '<u4').tobytes())
            meta = dict(name=name, kind=kind, inputs=512, outputs=2, units=u, mapping=mapping(m),
                program_sha256=hashlib.sha256(image).hexdigest(), packets=[dict(np=p['np'], words=len(p['packet']),
                sha256=hashlib.sha256(np.asarray(p['packet'], '<u4').tobytes()).hexdigest()) for p in packets])
            f.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n'); result.append(meta)
    (out / 'BUILD.json').write_text(json.dumps(dict(status='COMPILER_VIRTUAL_EQUIVALENCE_PASS', physical_device_opened=False, cells=result), indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('study', type=Path); main(p.parse_args().study)
