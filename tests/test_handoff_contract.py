import copy
import hashlib
import json
import unittest
import numpy as np
from resident import contract_archive


class HandoffContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=json.loads(contract_archive.BASE.with_suffix('.json').read_text())
        with np.load(contract_archive.BASE.with_suffix('.npz'),allow_pickle=False) as z:cls.z={k:z[k] for k in z.files}

    def test_all_controls_and_collapse_vectors_recount(self):
        self.assertEqual(contract_archive.audit_public()['collapsed_comparator_exact'],128)

    def test_failed_candidate_cannot_be_dropped(self):
        m=copy.deepcopy(self.meta);m['runs']=[r for r in m['runs'] if 'bits2' not in r['name']]
        with self.assertRaisesRegex(AssertionError,'retain all'):contract_archive.recount(m,self.z)

    def test_collapse_cannot_be_promoted_to_repair(self):
        m=copy.deepcopy(self.meta);m['repair_qualified']=True
        with self.assertRaises(AssertionError):contract_archive.recount(m,self.z)

    def test_a_forged_mixed_vector_is_rejected(self):
        z=dict(self.z);k='mixed_identity_development_full__outputs';z[k]=z[k].copy();z[k][0,0]=1
        with self.assertRaises(AssertionError):contract_archive.recount(self.meta,z)

    def test_executed_sources_remain_identical(self):
        for p,h in self.meta['executed_sources'].items():
            self.assertEqual(hashlib.sha256((contract_archive.P/p).read_bytes()).hexdigest(),h,p)

    def test_failed_compiler_tracebacks_preserve_error_without_private_paths(self):
        probe=self.meta['single_layer_compiler_probe'];text=json.dumps(probe)
        self.assertNotIn('/home/',text);self.assertNotIn('/Users/',text)
        self.assertEqual(sum(c['status']=='FAILED_PRESERVED' and c['error']=='Type.FNP2' for c in probe['cells']),4)


if __name__=='__main__':unittest.main()
