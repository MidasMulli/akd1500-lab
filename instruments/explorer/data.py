"""Read-only, hash-checked export of measured native learned-policy streams.

No device imports, study writes, or generated electrical spike measurements.
Derived features and potentials are explicitly separate from captured RAM/output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from admission import learned_policy as model
from admission import learned_policy_run as run
from admission.run import masks

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/20261025-learned-policy'
LABELS = {'native': 'Learned policy', 'manager_frozen': 'Manager frozen',
          'open': 'Always admit', 'students_frozen': 'Students frozen',
          'cut': 'Admission route cut'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def records(ram, count):
    words = np.asarray(ram, dtype='<u4')[2:].reshape(count, 20)
    mm = masks(np.asarray(ram, dtype='<u4'), 512, count)
    return [dict(id=j, words=words[j, :4].astype('uint64').tolist(),
                 connections=np.flatnonzero(mm[j]).tolist()) for j in range(count)]


def export(base=BASE):
    from admission import learned_policy_archive as archive
    base = Path(base)
    meta_path, npz_path = base.with_suffix('.json'), base.with_suffix('.npz')
    meta = json.loads(meta_path.read_text())
    assert sha(npz_path) == meta['archive_sha256'], 'archive hash mismatch'
    result = dict(schema=1, title='AKD1500 / Network explorer',
                  source=dict(study='20261025-learned-policy',
                              archive_sha256=sha(npz_path), metadata_sha256=sha(meta_path)),
                  evidence=dict(ram='Captured native SRAM after every event',
                                output='Returned native output packet',
                                features='Reconstructed from measured inputs and verified state',
                                timing='Event order; playback spacing is editorial',
                                geometry='Logical layout, not a die floorplan'),
                  limits=['Externally supervised reliability task', 'Host-paced execution',
                          'Repeated designed cues on one board', 'Adaptive CPU controls also pass',
                          'No electrical spike or membrane-voltage measurement'],
                  banks=[], arms={})
    checks = []
    with np.load(npz_path, allow_pickle=False) as z:
        # Validate every byte that enters the explorer, then use the existing
        # independent numeric auditor for both chip and packed-CPU streams.
        for arm in model.ARMS:
            name = run.name('native', 151101, arm, 'pilot', 0)
            item = meta['runs'][name]
            fixture = meta['fixtures'][f's151101_{arm}']
            for v in [*item['arrays'].values(), *fixture['arrays'].values()]:
                assert digest(z[v['key']]) == v['sha256'], 'array hash mismatch'
            packets = z[item['packets']['key']]
            assert digest(packets) == item['packets']['sha256']
            for i, packet in enumerate(packets):
                assert digest(packet) == item['receipt']['hashes'][f'result.packet{i}']
            checks.append(archive.check_one(z, meta, name))
            cpu_name = run.name('cpu', 151101, arm, 'pilot', 0)
            cpu_item = meta['runs'][cpu_name]
            for v in cpu_item['arrays'].values():
                assert digest(z[v['key']]) == v['sha256']
            checks.append(archive.check_one(z, meta, cpu_name))
            initial = [z[fixture['arrays'][f'bank{k}.ram']['key']] for k in range(3)]
            derived = model.simulate(initial, 'pilot', arm, 0)
            n = len(derived['events'])
            get = lambda field: z[item['arrays'][field]['key']]
            outputs = get('scores').reshape(n, 128)
            raw_inputs = get('inputs').reshape(n, 128)
            states = [get(f'states{k}').reshape(n, -1) for k in range(3)]
            before = [records(initial[k], dim[0]) for k, dim in enumerate(model.DIMENSIONS)]
            initial_records = json.loads(json.dumps(before))
            events = []
            for i, e in enumerate(derived['events']):
                changes, counts = [], []
                for k, dim in enumerate(model.DIMENSIONS):
                    after = records(states[k][i], dim[0]); additions = removals = 0
                    for j, rec in enumerate(after):
                        old = before[k][j]
                        if rec != old:
                            a = sorted(set(rec['connections']) - set(old['connections']))
                            r = sorted(set(old['connections']) - set(rec['connections']))
                            additions += len(a); removals += len(r)
                            changes.append(dict(bank=k, record=j, before=old, after=rec,
                                                added=a, removed=r))
                    counts.append(dict(added=additions, removed=removals))
                    before[k] = after
                memory_flags = [int(outputs[i, 88]), int(outputs[i, model.PR].any())]
                ev = dict(**e, policy_flags=outputs[i, 84:86].tolist(),
                          memory_flags=memory_flags,
                          output_active=np.flatnonzero(outputs[i]).tolist(),
                          input_active=np.flatnonzero(raw_inputs[i]).tolist(),
                          changes=changes, counts=counts,
                          derived_features=[np.flatnonzero(derived[f'features{k}'][i]).tolist() for k in range(3)],
                          derived_potentials=[derived[f'potentials{k}'][i].tolist() for k in range(3)],
                          cpu_memory=derived['cpu'][i].tolist())
                events.append(ev)
            if not result['banks']:
                for k, (u, classes, plastic, helpers) in enumerate(model.DIMENSIONS):
                    result['banks'].append(dict(id=k, name=['Manager', 'Memory 0', 'Memory 1'][k],
                        units=u, plastic=plastic, helpers=helpers, classes=classes,
                        mapping=fixture['build']['banks'][k]['np'],
                        thresholds=model.thresholds(k)[1].tolist()))
                result['mapping'] = fixture['build']['mapping']
            result['arms'][arm] = dict(name=name, label=LABELS[arm],
                start_utc=item['receipt']['start_utc'], finish_utc=item['receipt']['finish_utc'],
                fixture_sha256={k:v['sha256'] for k,v in fixture['arrays'].items()},
                initial=initial_records, events=events,
                assessment=model.assess(dict(derived, outputs=outputs)),
                cpu_names=derived['cpu_names'], audit='SAVED_ARRAY_PASS')
        prefix = next(e['index'] for e in result['arms']['native']['events'] if e['block'])
        for arm in model.ARMS[1:]:
            for k in range(3):
                def s(a):
                    it=meta['runs'][run.name('native',151101,a,'pilot',0)]
                    return z[it['arrays'][f'states{k}']['key']].reshape(268,-1)[:prefix]
                assert np.array_equal(s(arm), s('native')), 'unmatched learned prefix'
    result['audit'] = dict(status='EXPLORER_EXPORT_PASS', independently_checked_streams=len(checks),
                          hardware_arms=5, events_per_arm=268, matched_prefix_events=prefix)
    return result


def write_export(out):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    data = export(); raw = json.dumps(data, separators=(',', ':')).encode()
    out.write_bytes(raw)
    receipt = dict(**data['audit'], data_sha256=hashlib.sha256(raw).hexdigest(),
                   bytes=len(raw), source=data['source'])
    out.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    print(json.dumps(write_export(p.parse_args().out)))
