"""Real-image decisions, CPU controls and preserved compiler/quality stops."""
from pathlib import Path
import copy
import hashlib
import json
import unittest
import numpy as np
from recognition import archive,data

P=Path(__file__).resolve().parents[1]


class ImageCapacity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=json.loads(archive.BASE.with_suffix('.json').read_text())
        cls.z=np.load(archive.BASE.with_suffix('.npz'),allow_pickle=False)

    @classmethod
    def tearDownClass(cls):cls.z.close()

    def test_independent_all_image_and_cpu_recount(self):
        self.assertEqual(archive.audit_public()['cpu_timed_classifications'],18000)

    def test_failed_parent_stays_failed(self):
        m=self.meta['parent'];self.assertEqual(m['status'],'STOP_SEALED_QUALITY_GATE')
        self.assertEqual([m['sealed'][k]['correct'] for k in ['raw','compacted']],[814,817])

    def test_revision_preserves_original_exemplars_and_has_fresh_test(self):
        a,b=self.meta['parent'],self.meta['capacity']
        self.assertTrue(set(a['exemplar_train_indices'])<=set(b['exemplar_train_indices']))
        self.assertFalse(set(a['test_indices'])&set(b['test_indices']))
        self.assertEqual([b['sealed'][k]['correct'] for k in ['raw','compacted']],[910,900])

    def test_no_new_physical_measurement_claim(self):
        self.assertEqual(self.meta['physical_sessions'],0);self.assertEqual(self.meta['native_forwards'],0)
        self.assertTrue(all(r['status']=='FAILED_PRESERVED' for r in self.meta['compiler']['cells']))

    def test_forged_native_result_rejected(self):
        m=copy.deepcopy(self.meta);m['native_forwards']=1
        with self.assertRaisesRegex(AssertionError,'no physical result'):archive.recount(m,self.z)

    def test_reused_test_image_rejected(self):
        m=copy.deepcopy(self.meta);m['capacity']['test_indices'][0]=m['parent']['test_indices'][0]
        with self.assertRaisesRegex(AssertionError,'fresh sealed probes'):archive.recount(m,self.z)

    def test_actual_cpu_score_edit_rejected(self):
        z={k:self.z[k] for k in self.z.files};key='raw__cpu.overlap_scores';z[key]=z[key].copy();z[key][0,0]+=1
        with self.assertRaisesRegex(AssertionError,'actual CPU overlap'):archive.recount(self.meta,z)

    def test_actual_cpu_prediction_edit_rejected(self):
        z={k:self.z[k] for k in self.z.files};key='raw__cpu.m2.predictions';z[key]=z[key].copy();z[key][0,0]=(z[key][0,0]+1)%10
        with self.assertRaisesRegex(AssertionError,'actual CPU decisions'):archive.recount(self.meta,z)

    def test_bogus_timer_rejected(self):
        z={k:self.z[k] for k in self.z.files};key='raw__cpu.m2.wall_ns';z[key]=z[key].copy();z[key][0,0]=0
        with self.assertRaisesRegex(AssertionError,'CPU timer'):archive.recount(self.meta,z)

    def test_forged_compiler_success_rejected(self):
        m=copy.deepcopy(self.meta);m['compiler']['cells'][0]['status']='PASS'
        with self.assertRaisesRegex(AssertionError,'preserved compiler rejection'):archive.recount(m,self.z)

    def test_pooling_has_exact_spatial_meaning(self):
        x=np.zeros((2,1024),np.uint8);x[0,33]=1;x[1,1023]=1
        want=np.zeros((2,256),np.uint8);want[0,0]=1;want[1,255]=1
        np.testing.assert_array_equal(archive.pooled(x),want)
        np.testing.assert_array_equal(data.compact(x),want)

    def test_stronger_cpu_baseline_is_reported(self):
        r=[r for r in self.meta['cpu_timings'] if r['representation']=='compacted']
        self.assertEqual({a['method'] for a in r},{'exact_overlap','scalar_packed','NEON_packed'})
        self.assertEqual(next(a for a in r if a['method']=='NEON_packed')['median_wall_ns'],20544.5)

    def test_measured_cpu_source_identity(self):
        self.assertEqual(hashlib.sha256((P/'recognition/cpu_bench.cpp').read_bytes()).hexdigest(),self.meta['cpu_provenance']['source_sha256'])

    def test_image_source_binding(self):
        b=json.loads((P/'visual/IMAGE_CAPACITY_SOURCE_BINDING.json').read_text())
        for name,h in b['hashes'].items():self.assertEqual(hashlib.sha256((P/name).read_bytes()).hexdigest(),h,name)


if __name__=='__main__':unittest.main()
