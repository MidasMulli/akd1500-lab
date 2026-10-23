"""Verify the public physical evidence and reject forged matching CPU/native data."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mario.archive import materialize
from verify import verify


def rehash(run):
    rec = json.loads((run / 'RECEIPT.json').read_text())
    rec['hashes'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in run.iterdir() if f.name != 'RECEIPT.json'}
    (run / 'RECEIPT.json').write_text(json.dumps(rec))


class MarioEvidence(unittest.TestCase):
    def test_reported_action_results_match_scores_and_fresh_episodes(self):
        report = json.loads((ROOT / 'results/2026-10-23-mario.json').read_text())
        fresh = ROOT / 'results/2026-10-23-mario-fresh.json'
        self.assertEqual(hashlib.sha256(fresh.read_bytes()).hexdigest(), report['fresh_receipt_sha256'])
        rows = json.loads(fresh.read_text())['rows']
        self.assertEqual(len(rows), 72)
        data = np.load(ROOT / 'results/2026-10-23-mario.npz', allow_pickle=False)
        for run in report['runs']:
            if run['samples'] != 80:
                continue
            scores = data[run['name'] + '__native_scores'].reshape(80, 96)[64:]
            margins = (scores[:, 32:64].max(axis=1).astype(int) - scores[:, :32].max(axis=1).astype(int)).reshape(8, 2)
            actions = margins.argmax(axis=1)
            actual = sorted([r for r in rows if r['run'] == run['name']], key=lambda r: r['start'])
            self.assertEqual(actions.tolist(), [r['action'] for r in actual])
            self.assertEqual(sum(r['success'] for r in actual), run['successes'])
            # Full/frozen receive the actual rendered observation, including its action half.
            if run['arm'] != 'action_only':
                inputs = data[f"fixture__{run['fixture_key']}__inputs"].reshape(80, 512)
                for i, row in enumerate(actual):
                    self.assertEqual(hashlib.sha256(inputs[64 + 2 * i + row['action']].tobytes()).hexdigest(), row['actor_input_sha256'])

    def test_all_public_scores_and_every_RAM_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = materialize(root)
            self.assertEqual(len(runs), 10)
            audits = [verify(root, run) for run in runs]
            self.assertEqual(sum(r['samples'] for r in audits), 752)

    def test_matching_corrupted_scores_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = next(r for r in materialize(root) if r.name == 's81701_full80')
            for name in ['native.scores', 'cpu.scores']:
                p = run / name
                raw = bytearray(p.read_bytes())
                raw[0] ^= 1
                p.write_bytes(raw)
            rehash(run)
            with self.assertRaisesRegex(AssertionError, 'independent output mismatch'):
                verify(root, run)

    def test_query_state_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = next(r for r in materialize(root) if r.name == 's81701_full80')
            p = run / 'native.snapshot_0.ram'
            raw = bytearray(p.read_bytes())
            raw[24] ^= 1
            p.write_bytes(raw)
            rehash(run)
            with self.assertRaisesRegex(AssertionError, 'per-step measured RAM mismatch'):
                verify(root, run)


if __name__ == '__main__':
    unittest.main()
