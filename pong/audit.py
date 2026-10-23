"""Independent recount of inputs, class scores, complete RAM and matched controls."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
from verify import verify, read_words
from pong.experiment import N, U, CLASSES, PROTECTED, encode, schedule, pair_gate, save, sha
from pong.run import summarize


def audit(study, seed, split):
    plan = json.loads((study / 'PROTOCOL.json').read_text()); summaries = {}; raw_audits = []
    for arm in ['full', 'frozen']:
        name = f'{split}_{seed}_{arm}'; raw = study / 'raw' / name; capture = study / 'runs' / name
        raw_audits.append(verify(ROOT, raw))
        events = [json.loads(line) for line in (capture / 'events.jsonl').read_text().splitlines()]
        rows = schedule(plan, split, arm == 'frozen'); assert len(rows) == len(events)
        inputs = np.frombuffer((raw / 'fixture.inputs').read_bytes(), np.uint8).reshape(-1, N)
        labels = read_words(raw / 'fixture.labels'); scores = np.array(read_words(raw / 'native.scores')).reshape(-1, U)
        native = [json.loads(line) for line in (raw / 'native.stdout').read_text().splitlines()]
        replies = [e for e in native if e['kind'] == 'forward']; assert len(replies) == len(events)
        for i, (expected, actual) in enumerate(zip(rows, events)):
            assert {k: actual[k] for k in expected} == expected
            assert np.array_equal(inputs[i], encode(expected['observation'])) and labels[i] == expected['label']
            assert actual['response']['scores'] == scores[i].tolist() == replies[i]['scores']
            for key in replies[i]: assert actual['response'][key] == replies[i][key]
            assert actual['response']['sent_monotonic_ns'] <= actual['response']['received_monotonic_ns']
            if expected['label'] == PROTECTED:
                before = raw / ('fixture.ram' if i == 0 else f'native.snapshot_{i-1}.ram')
                assert before.read_bytes() == (raw / f'native.snapshot_{i}.ram').read_bytes()
        prefix = sum(r['kind'] == 'initial_training' for r in rows)
        result = summarize(rows, [e['response'] for e in events], sha(raw / f'native.snapshot_{prefix-1}.ram'))
        summaries[arm] = result
        save(capture / 'AUDIT.json', dict(status='INDEPENDENT_PREDICTION_AUDIT_PASS', raw=raw_audits[-1], **result))
    gate = pair_gate(plan, summaries['full'], summaries['frozen'])
    report = dict(seed=seed, split=split, gate=gate, arms=summaries, independent_raw=raw_audits)
    save(study / 'evidence' / f'PAIR_{split}_{seed}.json', report)
    print(json.dumps(dict(seed=seed, gate=gate, arms={a: {k: v for k, v in s.items() if k != 'prediction'} for a, s in summaries.items()})))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('study', type=Path); p.add_argument('seed', type=int)
    p.add_argument('--split', default='development'); a = p.parse_args(); audit(a.study.resolve(), a.seed, a.split)
