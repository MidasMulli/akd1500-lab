"""Preserved physical failures, full-state replay and wide snapshot addresses."""
import copy,json,unittest
import numpy as np
from coprocessor import fnp2_archive as a
class FNP2Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=json.loads(a.BASE.with_suffix('.json').read_text());cls.z=dict(np.load(a.BASE.with_suffix('.npz'),allow_pickle=False))
    def test_every_native_prefix_and_failed_state_recounts(self):
        r=a.audit_public();self.assertEqual(r['native_forwards'],246);self.assertEqual(r['failed_physical_sessions'],2)
    def test_failed_learning_cannot_be_promoted(self):
        m=copy.deepcopy(self.m);next(r for r in m['runs'] if r['name']=='boundary_native_above_l1')['exit_code']=0
        with self.assertRaisesRegex(AssertionError,'both failures'):a.recount(m,self.z)
    def test_no_slow_or_failed_control_can_be_omitted(self):
        m=copy.deepcopy(self.m);m['runs'].pop()
        with self.assertRaisesRegex(AssertionError,'all contract'):a.recount(m,self.z)
    def test_full_ram_snapshot_tamper_is_rejected(self):
        z=dict(self.z);k='boundary_native_above_l1__delta.values';z[k]=z[k].copy();z[k][-1]^=1
        with self.assertRaisesRegex(AssertionError,'checksum'):a.recount(self.m,z)
    def test_host_write_guard_is_not_optional(self):
        z=dict(self.z);k='boundary_native_above_l1__guard_counts';z[k]=z[k].copy();z[k][0]=1
        with self.assertRaisesRegex(AssertionError,'write observer'):a.recount(self.m,z)
    def test_wide_snapshot_indices_preserve_addresses_above65535(self):
        initial=np.zeros(225280,'<u4');states=np.stack([initial.copy(),initial.copy()]);states[0,176983]=9;states[1,132003]=7
        enc=a.compact32(initial,states);self.assertEqual(enc['indices'].dtype,np.dtype('<u4'));self.assertTrue(np.array_equal(a.expand(initial,enc,2),states))
if __name__=='__main__':unittest.main()
