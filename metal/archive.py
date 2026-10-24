"""Export owned numerical observations; exclude runtime and generated programs."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import numpy as np
from metal.run import P, S, digest
from metal.audit import audit
from metal.queue import audit as queue_audit

BASE = P / 'results/2026-10-23-metal'


def main():
    assert json.loads((S / 'CLOSEOUT.json').read_text())['status'] == 'CLOSED'
    arrays, runs, fixtures = {}, [], {}
    for d in sorted((S / 'raw').iterdir()):
        if not (d / 'RECEIPT.json').exists(): continue
        rec = json.loads((d / 'RECEIPT.json').read_text()); assert rec['status'] == 'PASS'
        queued = rec.get('method') == 'teaching_queue'
        with contextlib.redirect_stdout(io.StringIO()): report = queue_audit(d) if queued else audit(d)
        f = P / rec['fixture']; info = json.loads(f.with_suffix('.json').read_text())
        fixtures[info['name']] = info; key = d.name + '__'
        count, u = rec['samples'], info['outputs']
        shapes = [('inputs', np.uint8, (count, 512)), ('labels', '<i4', (count,))]
        if not queued: shapes.append(('modes', np.uint8, (count,)))
        for ext, dtype, shape in shapes:
            arrays[key + ext] = np.fromfile(d / ('stream.' + ext), dtype).reshape(shape)
        arrays[key + 'outputs'] = np.fromfile(d / 'native.scores', '<i4').reshape(count, u)
        arrays[key + 'cycles'] = np.fromfile(d / 'native.cycles', '<u4')
        arrays[key + 'counter_bounds'] = np.fromfile(d / 'native.counter_bounds', '<u4').reshape(-1, 2)
        arrays[key + 'latency_ns'] = np.fromfile(d / 'native.latency_ns', '<i8')
        learner = info['kind'] == 'learner'
        if learner:
            initial = np.fromfile(d / 'native.initial.ram', '<u4')
            arrays[key + 'initial'] = initial
            arrays[key + 'snapshots'] = np.fromfile(d / 'native.snapshots', '<u4').reshape(-1, len(initial))
            arrays[key + 'final'] = np.fromfile(d / 'native.final.ram', '<u4')
        if not queued:
            arrays[key + 'idle_counter'] = np.fromfile(d / 'native.idle_counter', '<u4')
            arrays[key + 'config_bounds'] = np.fromfile(d / 'native.config_bounds', '<u4').reshape(count, 2)
            arrays[key + 'serial_suffix'] = np.fromfile(d / 'native.serial_suffix', '<i4').reshape(3, 8, u)
            arrays[key + 'burst_suffix'] = np.fromfile(d / 'native.burst_suffix', '<i4').reshape(3, 8, u)
            arrays[key + 'burst_cycles'] = np.fromfile(d / 'native.burst_cycles', '<u4').reshape(3, 2)
            arrays[key + 'burst_latency_ns'] = np.fromfile(d / 'native.burst_latency_ns', '<i8').reshape(3, 2)
            rows = json.loads((d / 'stream.rows.json').read_text())
        else: rows = []
        runs.append(dict(name=d.name, queued_teaching=queued, fixture=info['name'],
            samples=count, forwards=rec['forwards'], rows=rows, audit=report,
            receipt_sha256=digest(d / 'RECEIPT.json'), native_result=rec.get('result'),
            source_freeze_sha256=rec['source_freeze_sha256']))
    assert len(runs) == 24 and sum(r['forwards'] for r in runs) == 9568
    np.savez_compressed(BASE.with_suffix('.npz'), **arrays)
    feasibility = json.loads((S / 'fixtures-metal01/BUILD.json').read_text())
    result = dict(status='NATIVE_CHARACTERIZATION_STAGE1_COMPLETE', physical_chips=1,
        physical_sessions=len(runs), native_forwards=sum(r['forwards'] for r in runs),
        captured_learning_snapshots=sum(len(arrays[r['name']+'__snapshots']) for r in runs if r['fixture'].startswith('width')),
        native_output_values=sum(arrays[r['name']+'__outputs'].size + (0 if r['queued_teaching'] else arrays[r['name']+'__serial_suffix'].size + arrays[r['name']+'__burst_suffix'].size) for r in runs),
        fixtures=fixtures, runs=runs,
        rejected_builds=[r for r in feasibility['cells'] if r['status'] != 'VIRTUAL_AND_COMPILER_PASS'],
        closeout=json.loads((S/'CLOSEOUT.json').read_text()),
        archive_sha256=digest(BASE.with_suffix('.npz')),
        limitations=[
            'One physical chip; temporal repetitions and controlled synthetic numerical workloads',
            'Counter is a DMA-path hardware monitor, not a per-neuron or arithmetic-only trace',
            'Buffer-end mode differs between inference and learning; no cross-mode cycle speed ratio',
            'Dense binary transport retained; activation sparsity is not sparse host ingress',
            'Channel-partition scaling supports usable parallel capacity, not per-neuron same-cycle execution',
            'Fixed CNP graphs have no native learning and no per-step learning-RAM snapshots',
            'Teaching pipeline override passes a bounded single-bank workload, not general concurrent teachers',
            'No CPU speed, electrical-energy, chip-peak or application-efficacy advantage established'])
    BASE.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ['status','physical_sessions','native_forwards','captured_learning_snapshots','native_output_values']}))


if __name__ == '__main__': main()
