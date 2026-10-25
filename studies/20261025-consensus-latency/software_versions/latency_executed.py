"""Sealed update/recall schedule and SDK-free independent timing analysis."""
from pathlib import Path
import argparse, datetime as dt, hashlib, itertools, json, shutil, types
import numpy as np
from admission import consensus as parent

P = Path(__file__).resolve().parents[1]
S = P / 'studies/20261025-consensus-latency'
SEEDS = parent.SEEDS
PROBES = 16
FIELDS = ['mode_ns', 'input_ns', 'submit_ns', 'release_ns', 'completion_observed_ns',
          'fetch_ns', 'compute_ns', 'audit_readback_ns', 'audit_check_ns',
          'start_ns', 'end_ns', 'read_bytes', 'write_bytes', 'polls']

def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + '\n')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def schedule(stage, arm='native'):
    assert stage in ['gate', 'repeat'] and arm == 'native'
    original = parent.schedule(stage, arm)
    rows = []
    for phase, group in itertools.groupby(original, lambda e: e['phase']):
        group = list(group)
        for event in group:
            rows.append(dict(event))
            if phase.startswith('accepted_transfer_') and event['role'] == 'transfer':
                for probe in range(PROBES):
                    rows.append(dict(event, role='exam', phase=phase.replace('accepted_transfer', 'settle_exam'),
                                     labels=[-1, -1, -1], wire=0, probe=probe))
    for i, row in enumerate(rows): row['index'] = i
    return rows

simulate = types.FunctionType(parent.simulate.__code__,
    dict(parent.simulate.__globals__, schedule=schedule), name='latency_reference')

def prepare(seed):
    source = P / 'studies/20261025-consensus-memory/fixtures02' / f's{seed}_native'
    target = S / 'fixtures' / f's{seed}'
    target.mkdir(parents=True, exist_ok=False)
    inherited = {}
    for q in source.iterdir():
        if q.suffix in ['.bin', '.ram', '.infer', '.learn', '.threshold', '.weights']:
            shutil.copyfile(q, target / q.name); inherited[q.name] = sha(q)
    initial = [np.fromfile(target / f'bank{k}.ram', '<u4') for k in range(3)]
    for stage in ['gate', 'repeat']:
        a = simulate(initial, stage, 'native'); rows = a['events']
        save(target / f'{stage}.events.json', rows)
        np.asarray([[e['key'], parent.ROLES.index(e['role']), *e['labels'], e['wire'], e['block']]
                    for e in rows], '<i4').tofile(target / f'{stage}.plan')
        a['inputs'].tofile(target / f'{stage}.inputs')
        a['outputs'].tofile(target / f'{stage}.scores')
        a['cpu'][:, 0].tofile(target / f'{stage}.dict')
        for k in range(3):
            a[f'states{k}'][-1].tofile(target / f'{stage}.final{k}')
            a[f'states{k}'].tofile(target / f'{stage}.states{k}')
        assert all(v['correct'] == v['queries'] for phase, v in parent.assess(a).items()
                   if phase.startswith(('protected_exam', 'updated_exam')))
    save(target / 'BUILD.json', dict(status='INHERITED_EXACT_PROGRAM', seed=seed,
         inherited_files=inherited, verification_probes=PROBES,
         events={stage: len(schedule(stage)) for stage in ['gate', 'repeat']}))

def operational(t): return t[:, :7].sum(1)

def decode(y):
    a, b = y[:, 88] != 0, y[:, :8].any(1)
    return np.where(a ^ b, b.astype('i4'), -1)

def stats(v):
    v = np.asarray(v, dtype='f8')
    return dict(n=len(v), median_ns=float(np.median(v)), p95_ns=float(np.percentile(v, 95)),
                min_ns=float(v.min()), max_ns=float(v.max())) if len(v) else None

def convergence(rows, answer, timing):
    """Commit and confirmation are distinct; diagnostic queries never teach."""
    out = []
    for cycle in range(1, 7):
        phase = f'accepted_transfer_{cycle}'
        candidates = [i for i, e in enumerate(rows) if e['phase'] == phase and e['role'] == 'transfer']
        if not candidates: continue
        ready = candidates[0] - 1
        assert rows[ready]['role'] == 'query'
        for key in [0, 1]:
            chosen = None
            for i in candidates:
                if rows[i]['key'] != key: continue
                probe_indices = list(range(i + 1, i + 1 + PROBES))
                assert all(rows[j]['phase'] == f'settle_exam_{cycle}' and rows[j]['key'] == key
                           and rows[j]['labels'] == [-1, -1, -1] for j in probe_indices)
                remaining = [j for j in range(i + 1, len(rows)) if rows[j]['key'] == key
                    and rows[j]['role'] == 'exam' and rows[j]['phase'] in [f'settle_exam_{cycle}', f'updated_exam_{cycle}']]
                if all(answer[j] == rows[j]['actual'] for j in remaining):
                    chosen = i, probe_indices; break
            assert chosen is not None, ('no stable updated recall', cycle, key)
            i, pp = chosen
            learning = [j for j in range(ready, i + 1) if rows[j]['phase'] != f'settle_exam_{cycle}']
            op = operational(timing)
            out.append(dict(cycle=cycle, key=key, answer=rows[i]['actual'], ready_event=ready,
                committed_event=i, confirmed_event=pp[-1],
                teaching_attempts=sum(rows[j]['role'] == 'transfer' and rows[j]['key'] == key
                                      for j in range(ready, i + 1)),
                correct_confirmation_queries=PROBES,
                learning_path_ns=int(op[learning].sum()),
                confirming_queries_ns=int(op[pp].sum()),
                operational_until_confirmed_ns=int(op[ready:pp[-1] + 1].sum()),
                audit_until_confirmed_ns=int(timing[ready:pp[-1] + 1, 7:9].sum()),
                observed_until_confirmed_ns=int(timing[pp[-1], 10] - timing[ready, 9])))
    return out

def audit(directory):
    d = Path(directory); r = json.loads((d / 'RECEIPT.json').read_text())
    assert r['exit_code'] == 0, (d / 'stderr').read_text()
    for name, h in r['hashes'].items(): assert sha(d / name) == h
    f = S / 'fixtures' / f"s{r['seed']}"; stage = r['stage']
    rows = json.loads((f / f'{stage}.events.json').read_text()); n = len(rows)
    t = np.fromfile(d / 'result.timing', '<i8').reshape(n, len(FIELDS))
    assert (t >= 0).all() and (t[:, 10] >= t[:, 9]).all()
    assert np.all(t[1:, 9] >= t[:-1, 10]), 'ordered event times'
    if r['mode'] == 'dictionary':
        answer = np.fromfile(d / 'result.answers', '<i4')
        assert np.array_equal(answer, np.fromfile(f / f'{stage}.dict', '<i4'))
    else:
        y = np.fromfile(d / 'result.scores', 'u1').reshape(n, parent.N)
        assert np.array_equal(y, np.fromfile(f / f'{stage}.scores', 'u1').reshape(n, parent.N))
        answer = decode(y)
        for k in range(3):
            got = np.fromfile(d / f'result.final{k}', '<u4')
            assert np.array_equal(got, np.fromfile(f / f'{stage}.final{k}', '<u4'))
            if r['verification'] == 'all':
                assert np.array_equal(np.fromfile(d / f'result.states{k}', '<u4'),
                                      np.fromfile(f / f'{stage}.states{k}', '<u4'))
    endpoints = {}
    for prefix in ['protected_exam', 'updated_exam']:
        ii = [i for i, e in enumerate(rows) if e['phase'].startswith(prefix)]
        correct = sum(answer[i] == rows[i]['actual'] for i in ii)
        assert correct == len(ii), (prefix, correct, len(ii))
        endpoints[prefix] = dict(correct=int(correct), queries=len(ii))
    report = json.loads((d / 'stdout').read_text())
    if r['mode'] == 'native':
        assert report['host_writes_during_execution'] == 0 and r['device_free']
        assert report['exact_scores_and_final_RAM'] and report['cleanup'] == 'unprogrammed'
    transfer = [i for i, e in enumerate(rows) if e['phase'].startswith('accepted_transfer') and e['role'] == 'transfer']
    return dict(status='TIMING_AND_SAVED_ARRAY_PASS', mode=r['mode'], verification=r['verification'],
        seed=r['seed'], stage=stage, events=n, endpoints=endpoints, timing_fields=FIELDS,
        accepted_transfer={name: stats(t[transfer, j]) for j, name in enumerate(FIELDS[:9])},
        accepted_transfer_operational=stats(operational(t)[transfer]),
        accepted_transfer_audit=stats(t[transfer, 7:9].sum(1)),
        convergence=convergence(rows, answer, t), report=report)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'audit']); p.add_argument('value')
    a = p.parse_args()
    if a.action == 'prepare': prepare(int(a.value))
    else:
        result = audit(a.value); save(Path(a.value) / 'AUDIT.json', result)
        print(json.dumps({k: v for k, v in result.items() if k not in ['convergence', 'report']}))
