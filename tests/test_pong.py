"""No-hardware checks of physics, information boundaries and matched controls."""
import unittest
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
from pong.experiment import design, encode, flight, outcome, schedule, PROTECTED


class PongProtocol(unittest.TestCase):
    def test_encoder_has_no_future_physics_or_outcome_input(self):
        obs = dict(court=0, y=.5, vy=.2)
        self.assertTrue(np.array_equal(encode(obs), encode(dict(**obs, changed=True, truth=3, acceleration=-.8))))
        self.assertEqual(int(encode(obs).sum()), 64)

    def test_exact_native_training_prefix_and_frozen_suffix(self):
        full, frozen = schedule(design()), schedule(design(), frozen=True)
        prefix = [r for r in full if r['kind'] == 'initial_training']
        self.assertEqual(full[:len(prefix)], frozen[:len(prefix)])
        self.assertTrue(all(r['label'] == PROTECTED for r in frozen[len(prefix):]))
        for a, b in zip(full, frozen):
            self.assertEqual(a['observation'], b['observation'])
            self.assertEqual(a['truth'], b['truth'])

    def test_reflection_and_invariant_court(self):
        for y in [.05, .5, .95]:
            for v in [-.8, 0, .8]:
                a = dict(court=1, y=y, vy=v)
                old, new = flight(a, False), flight(a, True)
                self.assertTrue(np.array_equal(old, new))
                self.assertTrue(np.all((new[:, 1] >= 0) & (new[:, 1] <= 1)))
                self.assertAlmostEqual(new[-1, 0], .92)

    def test_fine_encoder_does_not_alias_contradictory_development_rows(self):
        cases = design()['development']
        for changed in [False, True]:
            buckets = {}
            for case in cases:
                buckets.setdefault(encode(case).tobytes(), set()).add(outcome(case, changed)[0])
            self.assertTrue(all(len(v) == 1 for v in buckets.values()))


class PongEvidence(unittest.TestCase):
    def test_all_physical_scores_and_complete_RAM_snapshots(self):
        from mario.sealed.archive import materialize
        from tools.verify import verify
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp);runs=materialize(target,stem='2026-10-23-pong-physics')
            checked=[verify(target,r) for r in runs]
            self.assertEqual(len(checked),5)
            self.assertEqual(sum(r['samples'] for r in checked),4140)
            full=target/'raw/development_85201_full';frozen=target/'raw/development_85201_frozen'
            self.assertEqual((full/'native.snapshot_511.ram').read_bytes(),(frozen/'native.snapshot_511.ram').read_bytes())
            self.assertEqual((frozen/'native.snapshot_511.ram').read_bytes(),(frozen/'native.ram').read_bytes())
            self.assertNotEqual((full/'native.snapshot_511.ram').read_bytes(),(full/'native.ram').read_bytes())

    def test_saved_live_packets_are_bound_to_physical_data(self):
        from pong.audit_public import audit
        self.assertEqual(audit(),dict(status='PUBLIC_PONG_BINDING_PASS',bound_replies=44))

    def test_substituted_match_score_is_rejected(self):
        from pong.audit_public import audit,ROOT
        data=json.loads((ROOT/'visual/pong-physics-data.json').read_text())
        event=next(e for e in data['captures']['full']['events'] if 'response' in e)
        event['response']['scores'][0]+=1
        with self.assertRaisesRegex(AssertionError,'substituted native scores'):audit(data)

    def test_reply_cannot_appear_before_it_arrives(self):
        from pong.audit_public import audit,ROOT
        from tools.visualize_pong import available_index
        self.assertEqual(available_index([10,20],9),-1)
        self.assertEqual(available_index([10,20],19),0)
        data=json.loads((ROOT/'visual/pong-physics-data.json').read_text())
        event=next(e for e in data['captures']['full']['events'] if e['kind']=='prediction')
        event['response']['received_monotonic_ns']+=100_000_000_000
        with self.assertRaisesRegex(AssertionError,'reply shown before receipt'):audit(data)


if __name__ == '__main__': unittest.main()
