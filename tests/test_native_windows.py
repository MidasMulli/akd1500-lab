"""Positive native workaround, strict failure retention and no host weight repair."""
import copy,json,unittest
import numpy as np
from coprocessor import window_archive as a
class NativeWindows(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.m=json.loads(a.BASE.with_suffix('.json').read_text());cls.z=dict(np.load(a.BASE.with_suffix('.npz'),allow_pickle=False))
 def test_every_native_state_and_cpu_control_recounts(self):
  r=a.recount(self.m,self.z);self.assertEqual(r['main_sealed_updates'],24);self.assertEqual(r['window']['native_forwards'],101)
 def test_live_weight_host_write_is_rejected(self):
  z=dict(self.z);key='s107101_window_n8__guard';z[key]=z[key].copy();z[key][0]=1
  with self.assertRaisesRegex(AssertionError,'host writes'):a.recount(self.m,z)
 def test_original_direct_failure_cannot_be_promoted(self):
  m=copy.deepcopy(self.m);next(r for r in m['runs'] if r['mode']=='direct')['exit_code']=0
  with self.assertRaisesRegex(AssertionError,'direct failure'):a.recount(m,self.z)
 def test_missing_frozen_or_seed_control_is_rejected(self):
  m=copy.deepcopy(self.m);m['runs'].pop()
  with self.assertRaisesRegex(AssertionError,'all CPU'):a.recount(m,self.z)
 def test_complete_bank_tamper_is_rejected(self):
  z=dict(self.z);key='s107401_window_n8__delta.values';z[key]=z[key].copy();z[key][-1]^=1
  with self.assertRaisesRegex(AssertionError,'checksum'):a.recount(self.m,z)
 def test_same_record_connection_crossing_is_preserved(self):
  r=next(r for r in self.m['forensics'] if r['name']=='word_above_l1');self.assertEqual(r['exit_code'],0)
  a0=self.z[r['fixture']+'__initial'];states=a.decode(self.z,r['name'],a0,2)
  self.assertEqual(np.flatnonzero(states[-1]!=a0).tolist(),[131043,131072])
 def test_invalid_external_placement_cannot_be_omitted(self):
  m=copy.deepcopy(self.m);m['forensics']=[r for r in m['forensics'] if r['study']!='ddr']
  with self.assertRaisesRegex(AssertionError,'invalid-placement'):a.recount(m,self.z)
if __name__=='__main__':unittest.main()
