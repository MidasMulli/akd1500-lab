"""Attribute saved native memory edits; causal CPU controls are labelled separately.

No SDK, camera, fitting or device access. Evaluation targets diagnose edits but
never enter the prefix-anchor admission policy.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path

import numpy as np

from visual_memory.admission_screen import Rule
from visual_memory.data import save, sha
from visual_memory.retention_v4 import gates, initial as initial512, native_quality

P = Path(__file__).resolve().parents[1]
DEFAULT = P / 'studies/20261027-visual-memory-attribution'
U, CLASSES, WIDTH = 220, 10, 20


def packed_input(v):
    return sum(1 << int(k) for k in np.flatnonzero(v))


def masks(ram, n):
    a = np.asarray(ram, dtype='<u4')[2:].reshape(U, 4 + n // 32)[:, 4:]
    return [int.from_bytes(row.tobytes(), 'little') for row in a]


def validate_physical_initial(actual, logical, n):
    # Compiler descriptors have nonzero global/record configuration words.
    # Match masks and learning parameters, then retain the entire native image.
    stride = 4 + n // 32
    np.testing.assert_array_equal(actual[2:].reshape(U, stride)[:, 3:],
                                  logical[2:].reshape(U, stride)[:, 3:])
    return actual


def score_grid(inputs, weights):
    return np.asarray([[(v & w).bit_count() for w in weights] for v in inputs], dtype='i4')


def predictions(scores, trained=None):
    values = np.array(scores[..., :200], copy=True)
    if trained is not None:
        values[..., ~np.asarray(trained[:200], dtype=bool)] = -1
    return values.reshape(-1, CLASSES, WIDTH).max(2).argmax(1)


def margin(scores, labels):
    cls = np.asarray(scores[:, :200]).reshape(-1, CLASSES, WIDTH).max(2)
    correct = cls[np.arange(len(labels)), labels].copy()
    cls[np.arange(len(labels)), labels] = -1
    return correct - cls.max(1)


def load_case(tag):
    v = P / 'studies/20261026-visual-memory-v4'
    files = []
    if tag == '160-development':
        src = P / 'studies/20261026-visual-admission/memory160-development'
        initial_path = P / 'studies/20261026-visual-admission/connection-budget01/w64/initial.ram'
        initial = np.fromfile(initial_path, '<u4')
        files.append(initial_path)
        n, role = 512, 'CPU_EXACT_RULE_FORECAST_CONSUMED_DATA'
    else:
        stage = 'development' if tag == '512-development' else 'fresh'
        src = v / (stage + '-screen')
        initial = initial512(True)
        n, role = 1024, 'SAVED_NATIVE_SCORES_AND_RAM_CONSUMED_DATA'
    row_path = src / 'ROWS.json'
    rows = json.loads(row_path.read_text())
    files.append(row_path)
    arms = {}
    for arm in ['adaptive', 'frozen']:
        if tag.startswith('160'):
            source = src / (arm + '.npz')
            z = np.load(source, allow_pickle=False)
            x, y, scores, states = (z[k] for k in ['inputs', 'labels', 'scores', 'ram'])
            files.append(source)
        else:
            stage = 'development' if tag == '512-development' else 'fresh'
            d = v / 'raw' / ('native-' + stage + '-' + arm)
            fixture = v / 'native-fixtures' / (stage + '-' + arm)
            paths = {k: d / ('result.' + k) for k in ['scores', 'snapshots']}
            paths.update(inputs=fixture / 'cell.cpu_inputs', labels=fixture / 'cell.labels', initial=fixture / 'cell.ram', audit=d / 'AUDIT.json')
            audit = json.loads(paths['audit'].read_text())
            assert audit['status'] == 'EXACT_PHYSICAL_PASS'
            actual_initial = validate_physical_initial(np.fromfile(paths['initial'], '<u4'), initial512(True), n)
            if arm == 'adaptive':
                initial = actual_initial
            else:
                np.testing.assert_array_equal(actual_initial, initial)
            x = np.fromfile(paths['inputs'], 'u1').reshape(len(rows), n)
            y = np.fromfile(paths['labels'], '<i4')
            scores = np.fromfile(paths['scores'], '<i4').reshape(len(rows), U)
            states = np.fromfile(paths['snapshots'], '<u4').reshape(len(rows), len(initial))
            files.extend(paths.values())
        arms[arm] = dict(inputs=x, labels=y, scores=scores, ram=states)
    np.testing.assert_array_equal(arms['adaptive']['inputs'], arms['frozen']['inputs'])
    return dict(tag=tag, n=n, role=role, initial=initial, rows=rows, arms=arms,
                hashes={str(p.relative_to(P)): sha(p) for p in files})


def verify(case):
    n = case['n']; stride = 4 + n // 32
    for tag, arm in case['arms'].items():
        before = case['initial'].copy()
        protected = before[2:].reshape(U, stride)[200:].copy()
        for i, (v, label, got, after) in enumerate(zip(arm['inputs'], arm['labels'], arm['scores'], arm['ram'])):
            assert np.isin(v, [0, 1]).all()
            want = score_grid([packed_input(v)], masks(before, n))[0]
            np.testing.assert_array_equal(want, got, err_msg=f'{tag} score event{i}')
            if int(label) == 10:
                np.testing.assert_array_equal(after, before, err_msg=f'{tag} readonly event{i}')
            a, b = (z[2:].reshape(U, stride) for z in [before, after])
            np.testing.assert_array_equal(b[200:], protected)
            changed = np.flatnonzero((a != b).any(1))
            assert len(changed) <= 1
            if len(changed):
                assert int(label) < 10 and changed[0] // WIDTH == int(label)
            assert all(w.bit_count() == 64 for w in masks(after, n))
            before = after
    return dict(scores_checked=2 * len(case['rows']) * U, all_readonly_and_protected_boundaries=True)


def attribution(case, arm):
    rows, n, initial = case['rows'], case['n'], case['initial']
    probe_ids = [i for i, r in enumerate(rows) if r['phase'] in ['prefix', 'old', 'after']]
    labels = np.asarray([rows[i]['label'] for i in probe_ids])
    phases = np.asarray([rows[i]['phase'] for i in probe_ids])
    packed = [packed_input(arm['inputs'][i]) for i in probe_ids]
    before = initial.copy(); w = masks(before, n); grid = score_grid(packed, w)
    trained = np.zeros(U, bool); updates = []; used = set()
    trained_scores = []
    prefix_state = None
    for i, after in enumerate(arm['ram']):
        # The decoder inspects training history only, before this event.
        trained_scores.append(int(predictions(arm['scores'][i:i+1], trained)[0]))
        after_w = masks(after, n)
        changed = [j for j in range(U) if w[j] != after_w[j]]
        if changed:
            j = changed[0]; old_pred = predictions(grid)
            old_margin = margin(grid, labels)
            grid[:, j] = [(v & after_w[j]).bit_count() for v in packed]
            new_pred = predictions(grid); new_margin = margin(grid, labels)
            item = dict(event=i + 1, phase=rows[i]['phase'], record=j,
                        label=int(arm['labels'][i]), session=rows[i]['session'], frame=rows[i]['frame'],
                        recruited=not bool(trained[j]), pre_update_score=int(arm['scores'][i, j]),
                        removed=(w[j] & ~after_w[j]).bit_count(), added=(after_w[j] & ~w[j]).bit_count(), probes={})
            for phase in ['prefix', 'old', 'after']:
                take = phases == phase
                good0, good1 = old_pred[take] == labels[take], new_pred[take] == labels[take]
                item['probes'][phase] = dict(correct_before=int(good0.sum()), correct_after=int(good1.sum()),
                    helped=int((~good0 & good1).sum()), harmed=int((good0 & ~good1).sum()),
                    mean_margin_change=float((new_margin[take] - old_margin[take]).mean()))
            updates.append(item); trained[j] = True; used.add(j)
        before, w = after, after_w
        if rows[i]['phase'] == 'prefix':
            prefix_state = (grid.copy(), trained.copy())
    assert prefix_state is not None
    # Prefix and end readouts on the same probes diagnose the inference pool.
    readout = {}
    for stage, (g, used_mask) in [('after_prefix', prefix_state), ('final', (grid, trained))]:
        readout[stage] = {}
        for policy, tr in [('all_records', None), ('trained_records_only', used_mask)]:
            pred = predictions(g, tr)
            readout[stage][policy] = {phase: dict(correct=int((pred[phases == phase] == labels[phases == phase]).sum()),
                total=int((phases == phase).sum())) for phase in ['prefix', 'old', 'after']}
    adapt = [u for u in updates if u['phase'] == 'adapt']
    return dict(updates=updates, changed_records=sorted(used), trained_records_per_class=np.bincount(np.asarray(sorted(used), int) // WIDTH, minlength=10).tolist(),
                adapt_updates=len(adapt), adapt_new_records=sum(u['recruited'] for u in adapt),
                adapt_immediate_flips={phase: {k: sum(u['probes'][phase][k] for u in adapt) for k in ['helped', 'harmed']} for phase in ['prefix', 'old', 'after']},
                readout_diagnosis=readout,
                chronological_trained_only_metrics=metrics(np.asarray(trained_scores), rows))


def metrics(pred, rows):
    result = {}
    for phase in ['old', 'before', 'after', 'retention']:
        ids = np.array([i for i, r in enumerate(rows) if r['phase'] == phase])
        correct = sum(int(pred[i]) == rows[i]['label'] for i in ids)
        result[phase] = dict(correct=correct, total=len(ids), accuracy=correct / len(ids))
    return result


def prefix_anchor_guard(inputs, teachers, initial, n, use_guard):
    """Only already-observed prefix inputs/labels can veto a proposed update."""
    if n == 512:
        live = Rule(initial, 64)
    else:
        from visual_memory.retention_v4 import NativeMemory
        live = NativeMemory(initial, n)
    anchor_x, anchor_y, decisions, outputs = [], [], [], []
    seen_teachers = 0
    for i, (v, y) in enumerate(zip(inputs, teachers)):
        y = int(y)
        outputs.append(live.forward(v, y, False))
        if y == 10:
            continue
        if seen_teachers < 80:
            live.forward(v, y, True)
            anchor_x.append(packed_input(v)); anchor_y.append(y)
        else:
            trial = copy.copy(live); trial.ram = live.ram.copy(); trial.weights = live.weights.copy()
            trial.forward(v, y, True)
            before = predictions(score_grid(anchor_x, live.weights))
            after = predictions(score_grid(anchor_x, trial.weights))
            losses = int(((before == anchor_y) & (after != anchor_y)).sum())
            admit = not use_guard or losses == 0
            decisions.append(dict(event=i+1, admit=admit, previously_correct_anchor_losses=losses))
            if admit:
                live = trial
        seen_teachers += 1
    assert seen_teachers == 160
    return np.asarray(outputs, 'i4'), decisions


def information(case):
    a = case['arms']['adaptive']; rows = case['rows']
    bank, labels, pred = [], [], []
    for v, y in zip(a['inputs'], a['labels']):
        x = packed_input(v)
        pred.append(labels[min(range(len(bank)), key=lambda k: (x ^ bank[k]).bit_count())] if bank else 0)
        if int(y) < 10:
            bank.append(x); labels.append(int(y))
    unique = {}
    for v, r in zip(a['inputs'], rows):
        unique.setdefault(packed_input(v), set()).add(r['label'])
    return dict(causal_nearest_hamming=metrics(np.asarray(pred), rows),
                unique_inputs=len(unique), cross_label_collisions=sum(len(v) > 1 for v in unique.values()))


def run(out):
    authority = json.loads((out / 'AUTHORITY.json').read_text())
    assert dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(authority['scientific_cutoff_utc'])
    frozen = json.loads((out / 'FROZEN.json').read_text())
    for path, digest in frozen['hashes'].items():
        assert sha(P / path) == digest, ('frozen source drift', path)
    destination = out / 'diagnostic01'; destination.mkdir(exist_ok=False)
    results = {}; source_hashes = {}
    for tag in ['160-development', '512-development', '512-consumed-confirmation']:
        c = load_case(tag); source_hashes.update(c['hashes'])
        for path, digest in c['hashes'].items():
            assert frozen['hashes'][path] == digest, ('input drift', path)
        audit = verify(c)
        arms = {a: attribution(c, c['arms'][a]) for a in ['adaptive', 'frozen']}
        baseline = {a: native_quality(c['arms'][a]['scores'], c['rows']) for a in ['adaptive', 'frozen']}
        # First confirm the callable used by the prospective guard is exact.
        original, _ = prefix_anchor_guard(c['arms']['adaptive']['inputs'], c['arms']['adaptive']['labels'], c['initial'], c['n'], False)
        np.testing.assert_array_equal(original, c['arms']['adaptive']['scores'])
        scores, decisions = prefix_anchor_guard(c['arms']['adaptive']['inputs'], c['arms']['adaptive']['labels'], c['initial'], c['n'], True)
        guard = native_quality(scores, c['rows'])
        trained = {a: arms[a]['chronological_trained_only_metrics'] for a in arms}
        np.savez_compressed(destination / (tag + '-guard.npz'), scores=scores)
        result = dict(evidence_role=c['role'], audit=audit, baseline=baseline, arms=arms, information=information(c),
                      causal_CPU_guard=dict(metrics=guard, gates=gates(guard, baseline['frozen']),
                        rejected=sum(not d['admit'] for d in decisions), decisions=decisions,
                        scope='CPU reference with external anchor storage and exact trial clone; NOT a native learned admission controller'),
                      trained_only_host_decoder=dict(metrics=trained, gates=gates(trained['adaptive'], trained['frozen']), scope='Host reads training history. Diagnostic; not native inhibition.'))
        save(destination / (tag + '.json'), result)
        results[tag] = {k: v for k, v in result.items() if k != 'arms'}
        print(tag, json.dumps(dict(baseline=baseline, guard=guard, rejected=result['causal_CPU_guard']['rejected'], trained=trained, information=result['information'], recruitment=arms['adaptive']['trained_records_per_class'], flips=arms['adaptive']['adapt_immediate_flips'])), flush=True)
    save(destination / 'RESULT.json', dict(status='ATTRIBUTION_COMPLETE', physical_sessions=0, encoder_fits=0,
         new_confirmation_records_opened=0, probe_scope='Consumed data only; repeated/correlated images, not independent trials',
         native_controller_admitted=False, results=results))
    save(destination / 'SOURCE_MANIFEST.json', dict(protocol_sha256=sha(out/'PROTOCOL.json'), source_sha256=sha(__file__), inputs=source_hashes))
    for path, digest in frozen['hashes'].items():
        assert sha(P / path) == digest, ('source changed during diagnosis', path)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=DEFAULT)
    run(p.parse_args().out)
