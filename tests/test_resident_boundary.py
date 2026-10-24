"""A failed scale-up must remain failed, even when a compiler mapping succeeds."""
import copy
import hashlib
import json
import unittest
import numpy as np
from resident import archive


class ResidentBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=json.loads(archive.BASE.with_suffix('.json').read_text())
        with np.load(archive.BASE.with_suffix('.npz'),allow_pickle=False) as z:
            cls.data={k:z[k] for k in z.files}

    def test_all_saved_successes_and_first_failures_recount(self):
        r=archive.audit_public()
        self.assertEqual(r['observed_native_forwards'],663)
        self.assertEqual(r['full_bank_captures'],796)
        self.assertFalse(r['post_gate_passed'])

    def test_compiler_success_cannot_become_native_success(self):
        m=copy.deepcopy(self.meta);m['qualified_learning_banks']=6
        with self.assertRaisesRegex(AssertionError,'no scaled success claim'):archive.recount(m,self.data)

    def test_omitting_a_failed_attempt_is_rejected(self):
        m=copy.deepcopy(self.meta);m['runs']=[r for r in m['runs'] if r['status']=='PASS']
        with self.assertRaisesRegex(AssertionError,'complete successful and failed'):archive.recount(m,self.data)

    def test_an_earlier_connection_error_is_rejected(self):
        d=dict(self.data);key='smoke_b2_s99101_repair__states';d[key]=d[key].copy();d[key][0,6]^=1
        with self.assertRaisesRegex(AssertionError,'preserved first divergence'):archive.recount(self.meta,d)

    def test_relabeling_failed_output_as_success_is_rejected(self):
        m=copy.deepcopy(self.meta);next(r for r in m['runs'] if r['name']=='smoke_b2_s99101_repair_a03')['status']='PASS'
        with self.assertRaises(AssertionError):archive.recount(m,self.data)

    def test_forward_count_is_observed_not_requested(self):
        m=copy.deepcopy(self.meta);m['observed_native_forwards']=1440
        with self.assertRaisesRegex(AssertionError,'executed forward accounting'):archive.recount(m,self.data)

    def test_missing_partial_bank_snapshot_is_rejected(self):
        m=copy.deepcopy(self.meta);m['runs'][-1]['snapshot_coordinates'].pop()
        with self.assertRaisesRegex(AssertionError,'partial capture accounting'):archive.recount(m,self.data)

    def test_latest_native_source_is_the_executed_source(self):
        for name,h in self.meta['executed_latest_sources'].items():
            self.assertEqual(hashlib.sha256((archive.P/name).read_bytes()).hexdigest(),h,name)
        self.assertEqual(hashlib.sha256((archive.P/'resident/archive.py').read_bytes()).hexdigest(),self.meta['postclose_auditor_sha256'])


if __name__=='__main__':unittest.main()
