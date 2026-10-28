from media_binding import binding_digest
import unittest
import copy,json,hashlib
from pathlib import Path
import numpy as np
from streaming.design import encode,locate,band,observation,episodes,commands,plan,decode
from streaming.archive import BASE,recount

def image(x=30,y=112,context=0):
    a=np.full((224,224,3),245,np.uint8);a[:80,108:116]=25 if context==0 else 200
    yy,xx=np.mgrid[:224,:224];a[(xx-x)**2+(yy-y)**2<49]=[220,50,50];return a

class StreamingProtocolTests(unittest.TestCase):
    def test_encoder_has_fixed_activity_and_visible_context_isolation(self):
        for c in [0,1]:
            o=observation(image(y=105,context=c),image(y=112,context=c),[.15,.5],c);x=encode(o)
            self.assertEqual(x.shape,(1024,));self.assertEqual(int(x.sum()),64)
            self.assertEqual(int(x[(1-c)*512:(2-c)*512].sum()),0)
    def test_future_outcome_does_not_enter_encoder(self):
        o=observation(image(y=105),image(y=112),[.15,.5],0)
        self.assertEqual(set(o),{'y','x','vy','action_y','context'})
        original=encode(o);band(image(y=200));self.assertTrue(np.array_equal(original,encode(o)))
    def test_context_must_be_visible(self):
        with self.assertRaises(AssertionError):observation(image(),image(),[0,0],1)
    def test_unresolved_pixels_are_not_silently_labelled(self):
        a=np.zeros((224,224,3),np.uint8)
        with self.assertRaises(ValueError):band(a)
        self.assertEqual(band(a,False),-1)
    def test_command_schedule_does_not_use_outcomes(self):
        self.assertEqual(commands(42),commands(42));self.assertEqual(len(commands(42)),8)
        self.assertTrue(all(a[0]==.15 for a in commands(42)))
    def test_protected_feedback_and_frozen_prefix_contract(self):
        p=plan();self.assertEqual(p['query_teacher'],4);self.assertFalse(p['skipping_initially_enabled'])
        for e in episodes():
            if e['phase']=='retention':self.assertEqual(e['context'],1);self.assertFalse(e['teach']);self.assertFalse(e['reversed'])
            if e['reversed']:self.assertEqual(e['context'],0)
        ids=[r['scene'] for r in episodes()];self.assertEqual(len(ids),len(set(ids)))
    def test_decoder_excludes_reserved_query_records(self):
        x=np.zeros(160,int);x[-1]=64;x[32]=31
        self.assertEqual(decode(x),(1,31,31))
    def test_target_bands_are_observed_pixels(self):
        self.assertEqual([band(image(y=y)) for y in [25,80,140,200]],[0,1,2,3])

class StreamingEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=json.loads(BASE.with_suffix('.json').read_text())
        with np.load(BASE.with_suffix('.npz')) as z:cls.z=dict(z)
    def test_complete_public_recount(self):
        r=recount(self.meta,self.z);self.assertEqual(r['native_forwards'],2148);self.assertEqual(r['skipped_inferences'],0)
    def test_skipping_cannot_be_promoted(self):
        m=copy.deepcopy(self.meta);m['skipped_inferences']=1
        with self.assertRaises(AssertionError):recount(m,self.z)
    def test_missing_frozen_control_rejected(self):
        m=copy.deepcopy(self.meta);m['runs']=[r for r in m['runs'] if r['arm']!='frozen']
        with self.assertRaises(AssertionError):recount(m,self.z)
    def test_future_label_cannot_be_in_query(self):
        z=dict(self.z);k='s105101_frozen__labels';z[k]=z[k].copy();z[k][0]=0
        with self.assertRaises(AssertionError):recount(self.meta,z)
    def test_feedback_before_prediction_rejected(self):
        z=dict(self.z);k='s105101_frozen__clock_ns';z[k]=z[k].copy();z[k][0,2]=z[k][0,0]-1
        with self.assertRaises(AssertionError):recount(self.meta,z)
    def test_native_output_corruption_rejected(self):
        z=dict(self.z);k='s105101_frozen__scores';z[k]=z[k].copy();z[k][0,0]^=1
        with self.assertRaises(AssertionError):recount(self.meta,z)
    def test_complete_snapshot_corruption_rejected(self):
        z=dict(self.z);k='s105101_frozen__delta.values';z[k]=z[k].copy();z[k][0]^=1
        with self.assertRaises(AssertionError):recount(self.meta,z)
    def test_future_pixel_truth_cannot_be_relabelled(self):
        m=copy.deepcopy(self.meta);m['runs'][0]['rows'][0]['truth']=(m['runs'][0]['rows'][0]['truth']+1)%4
        with self.assertRaises(AssertionError):recount(m,self.z)
    def test_quality_threshold_cannot_be_lowered(self):
        m=copy.deepcopy(self.meta);m['protocol']['gates']['minimum_changed_accuracy']=.70
        with self.assertRaises(AssertionError):recount(m,self.z)
    def test_source_bound_video_and_population_metrics(self):
        root=BASE.parents[1];p=root/'visual/STREAMING_PREDICTION_SOURCE_BINDING.json'
        d=json.loads(p.read_text());self.assertEqual((d['fps'],d['seconds']),(60,24));self.assertEqual(len(d['selected_samples']),12)
        for path,h in d['hashes'].items():self.assertEqual(binding_digest(root,path),h,path)
        m=json.loads((root/'visual/streaming-prediction-metrics.json').read_text());self.assertEqual(m['changed'],dict(native_correct=49,frozen_correct=27,cpu_correct=61,nano_correct=41,n=69))
        self.assertEqual(m['retention'],dict(native_correct=56,frozen_correct=56,n=74))

if __name__=='__main__':unittest.main()
