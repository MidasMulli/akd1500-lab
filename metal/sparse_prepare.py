"""Compiler-only sparse-work cells; reuse the validated intermediate learner adapter."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import traceback
import numpy as np
import akida
from metal.prepare import add_fc, mapping
from metal.ablation_prepare import memory_packets
from metal.sparse_design import SEED, CELLS, weights, stream

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'tools'))
from prepare import PARAMS, parts
from prepare_reduction import assemble

class Select(akida.MapConstraints):
    def select_nps(self, source_nps, num_nps, type):
        assert type == akida.NP.Type.FNP3, ('reject kernel substitution', str(type))
        targets = [[1, 3, 0], [3, 3, 0]]
        found = [q for target in targets for q in source_nps if [q.col, q.row, q.id] == target]
        assert len(found) >= num_nps
        return found[:num_nps]


def main(study, attempt):
    out = study / attempt; out.mkdir(exist_ok=False)
    results = []
    for n, u, nw in CELLS:
        x, rows = stream(n)
        for kind in ['learner', 'compact']:
            name = f'n{n}_u{u}_w{nw}_{kind}'
            try:
                w = weights(n, u, nw)
                m = akida.Model(); m.add(akida.InputData(input_shape=(1, 1, n), input_bits=1))
                l = add_fc(m, 'sparse_learning_bank', w, kind == 'compact')
                m.compile(optimizer=akida.AkidaUnsupervised(**dict(PARAMS, num_weights=nw)))
                l.set_variable('weights', w.reshape(1, 1, n, u))
                l.set_variable('threshold_learning', np.zeros(u, np.int32))
                threshold = nw // 2 - 1
                if kind == 'compact':
                    l.set_variable('threshold', np.full(u, threshold, np.int32))
                    l.set_variable('act_step', np.ones(u, np.float32))
                d = akida.AKD1500(); m.map(d, hw_only=True, mode=akida.MapMode.Minimal, constraints=Select(d))
                assert len(m.sequences) == len(m.sequences[0].passes) == 1
                single = bytes(m.sequences[0].program)
                ram, npid = parts(single)
                assert npid == [1, 3, 0] and len(ram) == 2 + u * (4 + n // 32)
                scores = x.astype(np.int32) @ w.astype(np.int32)
                fixed, adapter = [], None
                m.map(None)
                if kind == 'compact':
                    rw = np.zeros((u, 2), np.int8); rw[:u // 2, 0] = 1; rw[u // 2:, 1] = 1
                    add_fc(m, 'fixed_two_count_readout', rw)
                    expected = (scores > threshold).astype(np.int32) @ rw.astype(np.int32)
                else: expected = scores
                got = m.forward(x.reshape(-1, 1, 1, n)).reshape(expected.shape)
                assert np.array_equal(got, expected), 'independent virtual dot/count output'
                if kind == 'compact':
                    d = akida.AKD1500(); m.map(d, hw_only=True, mode=akida.MapMode.Minimal, constraints=Select(d))
                    assert len(m.sequences) == len(m.sequences[0].passes) == 1
                    app = bytes(m.sequences[0].program)
                    fixed = [p for p in memory_packets(app) if p['np'] != npid]
                    assert len(fixed) == 1 and fixed[0]['np'] == [3, 3, 0]
                    program, adapter = assemble(single, app, n, u)
                    (out / (name + '.fixed.ram')).write_bytes(np.asarray(fixed[0]['packet'], '<u4').tobytes())
                    (out / (name + '.threshold')).write_text(str(threshold) + '\n')
                else:
                    program = single
                    d = akida.AKD1500(); m.map(d, hw_only=True, mode=akida.MapMode.Minimal, constraints=Select(d))
                for ext, data in [('bin', program), ('ram', np.asarray(ram, '<u4').tobytes()),
                                  ('inputs', x.tobytes()), ('expected', expected.astype('<i4').tobytes())]:
                    (out / (name + '.' + ext)).write_bytes(data)
                (out / (name + '.meta')).write_text(f'{n} {u} {nw} 2\n')
                meta = dict(name=name, kind=kind, inputs=n, units=u, outputs=int(expected.shape[1]),
                            weights_per_neuron=nw, classes=2, seed=SEED, threshold=threshold,
                            mapping=mapping(m), adapter=adapter, program_sha256=hashlib.sha256(program).hexdigest())
                meta['ram_sha256'] = hashlib.sha256(np.asarray(ram, '<u4').tobytes()).hexdigest()
                if fixed: meta['fixed_ram_sha256'] = hashlib.sha256(np.asarray(fixed[0]['packet'], '<u4').tobytes()).hexdigest()
                (out / (name + '.json')).write_text(json.dumps(meta, indent=2) + '\n')
                results.append(dict(status='VIRTUAL_AND_COMPILER_PASS', **meta))
            except Exception as e:
                results.append(dict(name=name, status='FAILED_PRESERVED', error=str(e), traceback=traceback.format_exc()))
            (out / 'BUILD.json').write_text(json.dumps(dict(physical_device_opened=False, cells=results), indent=2) + '\n')
            print(json.dumps({k: v for k, v in results[-1].items() if k not in ['traceback', 'adapter']}), flush=True)
    assert all(r['status'] == 'VIRTUAL_AND_COMPILER_PASS' for r in results)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('study', type=Path); p.add_argument('--attempt', default='fixtures-sparse01')
    a = p.parse_args(); main(a.study, a.attempt)
