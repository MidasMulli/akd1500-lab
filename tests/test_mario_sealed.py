"""Independent raw learning recount and causal binding of the synchronized visual."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mario.sealed.archive import materialize
from mario.sealed.audit_public import audit
from verify import verify


class SealedMarioEvidence(unittest.TestCase):
    def test_confirmation_native_streams_and_fresh_selected_outcomes(self):
        report = json.loads((ROOT / 'results/2026-10-23-mario-confirmation.json').read_text())
        fresh = json.loads((ROOT / 'results/2026-10-23-mario-confirmation-fresh.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT / 'results/2026-10-23-mario-confirmation-fresh.json').read_bytes()).hexdigest(), report['fresh_receipt_sha256'])
        self.assertEqual(len(fresh['rows']), 1280)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = materialize(root, stem='2026-10-23-mario-confirmation')
            self.assertEqual(len(runs), 11)
            self.assertEqual(sum(verify(root, run)['samples'] for run in runs), 13792)
            context_ids = [r['id'] for r in report['contexts']]
            totals = {'full': 0, 'frozen': 0}
            for row in report['runs']:
                if row['name'] == 'smoke32':
                    continue
                run = root / 'raw' / row['name']
                fixture = root / 'fixtures' / row['fixture_key']
                labels = np.fromfile(fixture.with_suffix('.labels'), '<u4')
                self.assertEqual(row['training_presentations'], 1120)
                self.assertEqual(row['test_queries'], 256)
                self.assertTrue(np.all(labels[1120:] == 2))
                if row['arm'] == 'frozen':
                    self.assertTrue(np.all(labels == 2))
                    self.assertEqual((run / 'native.ram').read_bytes(), fixture.with_suffix('.ram').read_bytes())
                else:
                    self.assertTrue(np.all(labels[:1120] < 2))
                score = np.fromfile(run / 'native.scores', '<u4').astype(np.int32).reshape(-1, 192)[1120:]
                actions = (score[:, 64:128].max(1) - score[:, :64].max(1)).reshape(-1, 2).argmax(1)
                outcomes = [r for r in fresh['rows'] if r['seed'] == row['seed'] and r['arm'] == row['arm']]
                self.assertEqual(len(outcomes), 128)
                for outcome in outcomes:
                    i = context_ids.index(outcome['context'])
                    self.assertEqual(outcome['action'], int(actions[i]))
                    self.assertEqual(outcome['success'], bool(report['test_truth'][i][int(actions[i])]))
                    offset = (1120 + 2*i + int(actions[i])) * 1024
                    source = fixture.with_suffix('.inputs').read_bytes()[offset:offset+1024]
                    self.assertEqual(hashlib.sha256(source).hexdigest(), outcome['input_sha256'])
                totals[row['arm']] += sum(r['success'] for r in outcomes)
            self.assertEqual(totals, {'full': 551, 'frozen': 534})

    def test_confirmation_protected_test_snapshot_cannot_be_forged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = next(r for r in materialize(root, stem='2026-10-23-mario-confirmation') if r.name == 'seed82911_full')
            path = run / 'native.snapshot_1120.ram'
            raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
            receipt = json.loads((run / 'RECEIPT.json').read_text())
            receipt['hashes'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (run / 'RECEIPT.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(AssertionError, 'per-step measured RAM mismatch'):
                verify(root, run)

    def test_every_published_native_snapshot_and_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = materialize(root)
            self.assertEqual(len(runs), 10)
            results = [verify(root, run) for run in runs]
            self.assertEqual(sum(r['samples'] for r in results), 11614)

    def test_live_RGB_clock_and_physical_response_binding(self):
        result = audit()
        self.assertEqual(result['reconstructed_response_inputs'], 264)
        self.assertEqual(result['frames'], 880)

    def test_mutually_matching_forged_native_and_CPU_scores_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = next(r for r in materialize(root) if r.name == 'live_full01')
            for name in ['native.scores', 'cpu.scores']:
                p = run / name; raw = bytearray(p.read_bytes()); raw[0] ^= 1; p.write_bytes(raw)
            receipt = json.loads((run / 'RECEIPT.json').read_text())
            receipt['hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir() if p.name != 'RECEIPT.json'}
            (run / 'RECEIPT.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(AssertionError, 'independent output mismatch'):
                verify(root, run)

    def test_game_source_frame_cannot_be_reassigned(self):
        replay = json.loads((ROOT / 'visual/mario-live-data.json').read_text())
        forged = copy.deepcopy(replay)
        event = next(e for e in forged['events'] if e['kind'] == 'response_pair')
        event['input_frame'] = 100
        with self.assertRaisesRegex(AssertionError, 'source-frame clock mismatch'):
            audit(replay=forged)


if __name__ == '__main__':
    unittest.main()
