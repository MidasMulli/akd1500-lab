import copy
import hashlib
import json
import unittest
import numpy as np
from resident import bridge,bridge_audit


class FixedBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=json.loads(bridge.BASE.with_suffix('.json').read_text())
        with np.load(bridge.BASE.with_suffix('.npz'),allow_pickle=False) as z:cls.data={k:z[k] for k in z.files}

    def test_frozen_patterns_outputs_and_memory_recount(self):
        self.assertEqual(bridge_audit.audit_public()['controls_full_exact'],[128,128])

    def test_dropping_mixed_failure_is_rejected(self):
        m=copy.deepcopy(self.meta);m['runs'].pop()
        with self.assertRaisesRegex(AssertionError,'complete controls'):bridge_audit.recount(m,self.data)

    def test_promoting_inference_to_learning_is_rejected(self):
        m=copy.deepcopy(self.meta);m['learning_sessions']=1
        with self.assertRaisesRegex(AssertionError,'inference-only'):bridge_audit.recount(m,self.data)

    def test_changed_control_output_is_rejected(self):
        d=dict(self.data);k='all_fnp_full__outputs';d[k]=d[k].copy();d[k][12,0]+=1
        with self.assertRaises(AssertionError):bridge_audit.recount(self.meta,d)

    def test_changed_inputs_and_recomputed_truth_are_still_rejected(self):
        d=dict(self.data);k='mixed_smoke__inputs';d[k]=d[k].copy();d[k][0,0]=0
        with self.assertRaisesRegex(AssertionError,'frozen bridge patterns'):bridge_audit.recount(self.meta,d)

    def test_parameter_edit_is_rejected(self):
        d=dict(self.data);k='mixed_smoke__packet0.final';d[k]=d[k].copy();d[k][5]^=1
        with self.assertRaisesRegex(AssertionError,'memory changed'):bridge_audit.recount(self.meta,d)

    def test_fake_exact_mixed_output_is_rejected(self):
        d=dict(self.data);k='mixed_smoke__outputs';d[k]=d[k].copy();d[k][0]=[128,0,0,0]
        with self.assertRaises(AssertionError):bridge_audit.recount(self.meta,d)

    def test_native_sources_match_executed_freeze(self):
        for p,h in self.meta['executed_sources'].items():self.assertEqual(hashlib.sha256((bridge.P/p).read_bytes()).hexdigest(),h,p)


if __name__=='__main__':unittest.main()
