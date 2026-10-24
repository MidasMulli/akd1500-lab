"""Whole-archive reconstruction, failed policy screens and corruption controls."""
from pathlib import Path
import copy
import hashlib
import json
import unittest
import numpy as np
from kernel import design, experiment, report
from recipe import design as learning
from recipe import repair, validation

P = Path(__file__).resolve().parents[1]


class NativeCompaction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads((P / 'results/2026-10-23-native-compaction.json').read_text())
        cls.archive = np.load(P / 'results/2026-10-23-native-compaction.npz', allow_pickle=False)

    @classmethod
    def tearDownClass(cls): cls.archive.close()

    def cell(self):
        rec = copy.deepcopy(next(r for r in self.results['runs'] if r['name'] == 'screen_u512_compacted'))
        data = {k.split('__',1)[1]: self.archive[k].copy() for k in self.archive.files if k.startswith(rec['name']+'__')}
        return rec, data

    def test_complete_native_archive(self):
        self.assertEqual(experiment.audit_public(), dict(status='PASS', physical_sessions=16, native_forwards=2624))

    def test_derived_cost_gates_and_memory(self):
        a = report.load_summary()
        self.assertEqual(a, json.loads(report.OUT.read_text()))
        self.assertEqual(a['physical_parameter_region_readbacks'],80)
        self.assertEqual([w['gate'] for w in a['widths']],['FAIL','PASS'])
        self.assertEqual(a['widths'][1]['captured_parameter_bytes'],dict(direct=41120,compacted=26784))
        self.assertFalse(a['native_learning_enabled'])

    def test_exact_numerical_output_change_rejected(self):
        rec,data=self.cell();data['native.scores'][32,0]+=1
        with self.assertRaisesRegex(AssertionError,'output equivalence'):experiment.recount(rec,data)

    def test_fixed_native_ram_edit_rejected(self):
        rec,data=self.cell();data['native.packet0.final'][20]^=1
        with self.assertRaisesRegex(AssertionError,'fixed RAM changed'):experiment.recount(rec,data)

    def test_consistently_changed_before_after_ram_rejected(self):
        rec,data=self.cell()
        for when in ['initial','final']:data['native.packet1.'+when][20]^=1
        with self.assertRaisesRegex(AssertionError,'fixed RAM packet'):experiment.recount(rec,data)

    def test_monitor_delta_change_rejected(self):
        rec,data=self.cell();data['native.cycles'][0]+=1
        with self.assertRaisesRegex(AssertionError,'counter deltas'):experiment.recount(rec,data)

    def test_input_domain_change_rejected(self):
        x,_=design.stream();x[0,200]^=1
        with self.assertRaisesRegex(AssertionError,'four-copy input domain'):design.expected(x,512)

    def test_stream_substitution_rejected(self):
        rec,data=self.cell();data['stream.inputs'][0,200]^=1
        with self.assertRaisesRegex(AssertionError,'frozen kernel inputs'):experiment.recount(rec,data)

    def test_mapping_substitution_rejected(self):
        rec,data=self.cell();rec['fixture_info']['mapping'][0]['nps'][0]['type']='Type.FNP2'
        with self.assertRaisesRegex(AssertionError,'fixed native mapping'):experiment.recount(rec,data)

    def test_missing_physical_treatment_rejected(self):
        r=copy.deepcopy(self.results);r['runs'].pop()
        with self.assertRaisesRegex(AssertionError,'complete treatment set'):report.summarize(r,self.archive)

    def test_invalid_burst_cost_rejected(self):
        z={k:self.archive[k].copy() for k in self.archive.files}
        z['screen_u512_compacted__native.burst_cycles'][0,1]=0
        r=copy.deepcopy(self.results)
        next(v for v in r['runs'] if v['name']=='screen_u512_compacted')['audit']['burst_cycles'][0][1]=0
        with self.assertRaisesRegex(AssertionError,'positive paired burst'):report.summarize(r,z)

    def test_burst_activity_is_zero_and_not_implied_half_active(self):
        a=report.load_summary()
        self.assertTrue(all(p['burst']['latent_active_bits']==0 for w in a['widths'] for p in w['paired_sessions']))

    def test_execution_source_and_new_analysis_are_distinguished(self):
        p=json.loads((P/'results/2026-10-23-compaction-provenance.json').read_text())
        self.assertEqual(len(p['hashes']),55)
        self.assertNotIn('kernel/report.py',p['hashes'])
        for name in ['kernel/design.py','kernel/experiment.py','src/metal_bench.cpp']:
            self.assertEqual(hashlib.sha256((P/name).read_bytes()).hexdigest(),p['hashes'][name])

    def test_visual_source_binding(self):
        b=json.loads((P/'visual/NATIVE_COMPACTION_SOURCE_BINDING.json').read_text())
        for name,h in b['hashes'].items():self.assertEqual(hashlib.sha256((P/name).read_bytes()).hexdigest(),h,name)


class LearningPolicyScreens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=json.loads((P/'studies/20261023-native-recipe/CPU_FORECAST.json').read_text())
        cls.repair=json.loads((P/'studies/20261023-native-recipe-r1/CPU_FORECAST.json').read_text())
        cls.validation=json.loads((P/'studies/20261023-native-policy-validation/CPU_FORECAST.json').read_text())

    def test_all_original_forecast_predictions_and_gates_reproduce(self):
        rr={}
        for saved in self.original['results']:
            got=learning.replay(saved['seed'],saved['arm']);self.assertEqual(got,saved)
            rr[saved['seed'],saved['arm']]=got
        for g in self.original['gates']:
            got=learning.gate(rr[g['seed'],g['arm']],rr[g['seed'],g['baseline']])
            self.assertEqual(g,dict(seed=g['seed'],arm=g['arm'],baseline=g['baseline'],**got))

    def test_repair_preserves_failed_cost_gate(self):
        baseline=learning.replay(repair.DEVELOPMENT_SEED,'stock')
        self.assertEqual(baseline,self.repair['baseline'])
        for name,match,margin in repair.POLICIES:
            got=learning.replay(repair.DEVELOPMENT_SEED,'error_gate',policy=repair.policy(match,margin));got['arm']=name
            self.assertEqual(got,next(r for r in self.repair['development_results'] if r['arm']==name))
            self.assertEqual(learning.gate(got,baseline),{k:v for k,v in next(g for g in self.repair['gates'] if g['name']==name).items() if k!='name'})
        self.assertIsNone(self.repair['selected_policy']);self.assertEqual(self.repair['confirmation'],[])

    def test_all_sealed_validation_seeds_and_failures_reproduce(self):
        self.assertEqual([r['seed'] for r in self.validation['results']],validation.SEEDS)
        for saved in self.validation['results']:
            b=learning.replay(saved['seed'],'stock')
            c=learning.replay(saved['seed'],'error_gate',policy=repair.policy(31,4));c['arm']='conservative31'
            self.assertEqual(b,saved['baseline']);self.assertEqual(c,saved['candidate'])
            self.assertEqual(validation.quality_gate(c,b),saved['gate'])
        self.assertEqual(self.validation['overall'],'STOP_VALIDATION_GATE')
        self.assertEqual([r['gate']['status'] for r in self.validation['results']],['PASS','STOP_VALIDATION_GATE','STOP_VALIDATION_GATE','PASS'])

    def test_queries_have_no_teaching_or_phase_label_in_input(self):
        x,rows=learning.schedule()
        self.assertEqual(x.shape,(2304,512));self.assertTrue(np.all(x.sum(axis=1)==256))
        for saved in self.original['results']:
            self.assertTrue(all(not gate for gate,row in zip(saved['gates'],rows) if not row['feedback']))
        # The readout uses only scores and layout, with no domain or true label.
        scores=np.zeros(128,dtype=int);scores[88]=20
        self.assertEqual(learning.decode(scores,False),(2,20,20))
        self.assertEqual(learning.decode(scores,True),(1,20,20))

    def test_equal_initial_resources_are_only_reordered(self):
        a=learning.initial_weights(98401,False);b=learning.initial_weights(98401,True)
        self.assertEqual(a.shape,b.shape);self.assertTrue(np.all(a.sum(axis=1)==32))
        self.assertEqual(sorted(map(bytes,a)),sorted(map(bytes,b)))

    def test_frozen_logical_population_does_not_guarantee_final_retention(self):
        r=next(r for r in self.original['results'] if r['seed']==98401 and r['arm']=='stable_plastic')
        self.assertEqual(r['metrics']['checkpoints']['256']['old_probe']['correct'],102)
        self.assertLess(102,109)


if __name__=='__main__':unittest.main()
