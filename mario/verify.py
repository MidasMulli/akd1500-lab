"""Recount native consequences from saved scores and every measured RAM snapshot."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from verify import read_words, verify


def recount(study, dataset, root=ROOT):
    experience = json.loads((dataset / 'EXPERIENCE.json').read_text())
    assert experience['status'] == 'OBSERVABILITY_PASS'
    assert experience['repeatability'] and not experience['actor_uses_game_RAM']
    source = np.load(dataset / 'experience.npz', allow_pickle=False)
    test_rows = [r for r in experience['rows'] if r['split'] == 'test']
    outcomes = np.array([r['success'] for r in test_rows]).reshape(8, 2)
    starts = [r['start'] for r in test_rows[::2]]
    build = json.loads((study / 'fixtures/BUILD.json').read_text())
    digest = hashlib.sha256((dataset / 'experience.npz').read_bytes()).hexdigest()
    assert digest == build['dataset_sha256']
    summaries = []
    for run in sorted((study / 'raw').iterdir()):
        receipt = json.loads((run / 'RECEIPT.json').read_text())
        assert receipt['status'] == 'PASS', ('preserved failure needs disposition', run)
        audit = verify(root, run)
        prefix = root / receipt['fixture']
        seed, arm = prefix.name.split('_', 1)
        seed = int(seed[1:])
        assert (prefix.with_suffix('.meta').read_text().split() == ['512', '96', '32', '3'])
        initial = read_words(prefix.with_suffix('.ram'))
        labels = read_words(prefix.with_suffix('.labels'))
        inputs = np.frombuffer(prefix.with_suffix('.inputs').read_bytes(), np.uint8).reshape(80, 512)
        assert len(initial) == 1922 and all(labels[i] == 2 for i in list(range(16)) + list(range(64, 80)))
        assert arm in {'full', 'frozen', 'action_only'}
        order = np.concatenate([np.random.default_rng(seed + e).permutation(12) for e in range(4)])
        expected_inputs = np.concatenate([source['test_inputs'], source['train_inputs'][order], source['test_inputs']])
        expected_labels = np.concatenate([np.full(16, 2), source['train_labels'][order], np.full(16, 2)])
        if arm == 'frozen':
            expected_labels[:] = 2
        if arm == 'action_only':
            actions = (expected_inputs[:, :256].sum(axis=1) == 0).astype(int)
            expected_inputs = np.zeros_like(expected_inputs)
            for i, action in enumerate(actions):
                expected_inputs[i, 256 * action:256 * action + 64] = 1
        assert np.array_equal(inputs, expected_inputs), 'fixture inputs differ from frozen experience/order'
        assert np.array_equal(labels, expected_labels), 'fixture labels differ from actual experience outcomes'
        previous = initial
        queries = 0
        for i in range(receipt['samples']):
            state = read_words(run / f'native.snapshot_{i}.ram')
            assert state[2 + 64 * 20:] == initial[2 + 64 * 20:], ('protected class changed', run.name, i)
            if labels[i] == 2:
                assert state == previous, ('protected query changed RAM', run.name, i)
                queries += 1
            previous = state
        row = dict(run=run.name, seed=seed, arm=arm, audit=audit,
                   protected_queries_unchanged=queries, protected_class_unchanged=True)
        if receipt['samples'] == 80:
            scores = np.array(read_words(run / 'native.scores')).reshape(80, 96)
            margins = lambda a: a[:, 32:64].max(axis=1) - a[:, :32].max(axis=1)
            before = margins(scores[:16]).reshape(8, 2)
            after = margins(scores[64:]).reshape(8, 2)
            before_actions = before.argmax(axis=1)
            actions = after.argmax(axis=1)
            row.update(before_margins=before.tolist(), after_margins=after.tolist(),
                       before_actions=before_actions.tolist(), actions=actions.tolist(),
                       before_successes=int(outcomes[np.arange(8), before_actions].sum()),
                       successes=int(outcomes[np.arange(8), actions].sum()),
                       selected_outcomes=outcomes[np.arange(8), actions].astype(int).tolist())
        summaries.append(row)
    cohorts = []
    for seed in sorted({r['seed'] for r in summaries}):
        arms = {r['arm']: r for r in summaries if r['seed'] == seed and 'successes' in r}
        if set(arms) == {'full', 'frozen', 'action_only'}:
            cohorts.append(dict(seed=seed, full=arms['full']['successes'],
                frozen=arms['frozen']['successes'], action_only=arms['action_only']['successes'],
                useful=arms['full']['successes'] > max(int(outcomes.sum(axis=0).max()),
                    arms['frozen']['successes'], arms['action_only']['successes'])))
    result = dict(status='PASS_EXACT_MECHANICS', dataset_sha256=digest, test_starts=starts,
                  constant_successes=outcomes.sum(axis=0).astype(int).tolist(),
                  best_constant_successes=int(outcomes.sum(axis=0).max()),
                  useful_seed_count=sum(c['useful'] for c in cohorts), cohorts=cohorts,
                  runs=summaries, independent_verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('study', type=Path)
    p.add_argument('dataset', type=Path)
    a = p.parse_args()
    result = recount(a.study.resolve(), a.dataset.resolve())
    (a.study / 'evidence/RECOUNT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}))
