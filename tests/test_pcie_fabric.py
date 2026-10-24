"""Falsify misleading evidence and publication claims without hardware access."""
import copy,json,unittest
import numpy as np
from coprocessor.archive import BASE,audit_public,recount
from coprocessor.reference import features

class PCIeFabric(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=json.loads(BASE.with_suffix('.json').read_text());cls.z=dict(np.load(BASE.with_suffix('.npz'),allow_pickle=False))
    def test_all_native_operations_and_complete_state_independently_replay(self):
        r=audit_public();self.assertEqual(r['physical_sessions'],45);self.assertEqual(r['native_forwards'],3872);self.assertEqual(r['full_learning_ram_snapshots'],1016)
    def test_missing_slow_dma_write_arm_is_rejected(self):
        m=copy.deepcopy(self.m);m['runs']=[r for r in m['runs'] if r['name']!='transport_p3_l1_n96_r0']
        with self.assertRaisesRegex(AssertionError,'all staged'):recount(m,self.z)
    def test_wrong_spatial_axon_order_is_rejected(self):
        z=dict(self.z);k='rgb32_c16__features';z[k]=z[k].reshape(96,8,8,8).transpose(0,3,1,2).reshape(96,512)
        with self.assertRaisesRegex(AssertionError,'RGB-to-FC'):recount(self.m,z)
    def test_tampered_native_potential_fails_replay(self):
        z=dict(self.z);k='transport_p1_l1_n96_r0__scores';z[k]=z[k].copy();z[k][27,65]^=1
        with self.assertRaisesRegex(AssertionError,'all-potential'):recount(self.m,z)
    def test_tampered_intermediate_ram_cannot_hide_behind_matching_final(self):
        z=dict(self.z);k='transport_p1_l1_n96_r0__snapshots';z[k]=z[k].copy();z[k][4,125]^=1
        with self.assertRaisesRegex(AssertionError,'RAM boundaries'):recount(self.m,z)
    def test_no_foreign_clock_source_or_vendor_binary_in_public_arrays(self):
        self.assertTrue(all(a.dtype!=object for a in self.z.values()));self.assertFalse(any('program' in k for k in self.z))
        self.assertEqual(self.m['inventory']['observed_pci']['current_link_width'],'2')
        self.assertEqual(self.m['inventory']['descriptor']['type_counts']['Type.FNP3'],7)
if __name__=='__main__':unittest.main()
