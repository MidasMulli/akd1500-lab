"""Reject corrupted native measurements and unsupported changes to queue evidence."""
from pathlib import Path
import copy
import json
import unittest
import numpy as np
from metal.audit_public import audit, audit_contents
from metal.ablation import recount

P = Path(__file__).resolve().parents[1]


class NativeMetalEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((P / 'results/2026-10-23-metal.json').read_text())
        with np.load(P / 'results/2026-10-23-metal.npz', allow_pickle=False) as z:
            cls.data = {k:z[k] for k in z.files}
        cls.ablation = json.loads((P / 'results/2026-10-23-metal-ablation.json').read_text())
        with np.load(P / 'results/2026-10-23-metal-ablation.npz', allow_pickle=False) as z:
            cls.ablation_data = {k:z[k] for k in z.files}

    def altered(self, key):
        data = dict(self.data); data[key] = data[key].copy(); return data

    def test_all_measurements_recount(self):
        r = audit()
        self.assertEqual((r['native_forwards'], r['physical_sessions'], r['full_learning_snapshots']), (9568, 24, 4208))

    def test_a_changed_fixed_output_is_rejected(self):
        key = 'confirm1_cnp128__outputs'; data = self.altered(key); data[key][0,0] += 1
        with self.assertRaisesRegex(AssertionError, 'fixed native graph'):
            audit_contents(self.report, data)

    def test_a_changed_learning_snapshot_is_rejected(self):
        key = 'confirm1_width8__snapshots'; data = self.altered(key); data[key][0,6] ^= 1
        with self.assertRaisesRegex(AssertionError, 'complete RAM'):
            audit_contents(self.report, data)

    def test_counter_delta_tampering_is_rejected(self):
        key = 'confirm1_cnp128__cycles'; data = self.altered(key); data[key][0] += 1
        with self.assertRaisesRegex(AssertionError, 'counter delta'):
            audit_contents(self.report, data)

    def test_an_advancing_idle_counter_is_rejected(self):
        key = 'confirm1_cnp128__idle_counter'; data = self.altered(key); data[key][2] += 1
        with self.assertRaisesRegex(AssertionError, 'idle control'):
            audit_contents(self.report, data)

    def test_density_metadata_tampering_is_rejected(self):
        report = copy.deepcopy(self.report)
        report['runs'][0]['rows'][0]['density'] += 1
        with self.assertRaisesRegex(AssertionError, 'input density'):
            audit_contents(report, self.data)

    def test_queue_register_treatment_is_verified(self):
        report = copy.deepcopy(self.report)
        r = next(r for r in report['runs'] if r['name'] == 'teaching_queue_pipeline1')
        r['native_result']['control_after'] = r['native_result']['control_before']
        with self.assertRaises(AssertionError): audit_contents(report, self.data)

    def test_all_physical_ablation_observations_recount(self):
        result = recount(self.ablation, self.ablation_data)
        self.assertEqual(result['native_forwards'], 256)
        self.assertEqual(result['initial_and_final_region_readbacks'], 40)

    def test_ablation_output_corruption_is_rejected(self):
        data = dict(self.ablation_data); key = 'baseline__outputs'
        data[key] = data[key].copy(); data[key][0,0] += 1
        with self.assertRaisesRegex(AssertionError, 'causal native output'):
            recount(self.ablation, data)

    def test_ablation_physical_memory_corruption_is_rejected(self):
        data = dict(self.ablation_data); key = 'baseline__packet0_final'
        data[key] = data[key].copy(); data[key][2] ^= 1
        with self.assertRaisesRegex(AssertionError, 'physical memory changed'):
            recount(self.ablation, data)

    def test_ablation_physical_target_is_verified(self):
        report = copy.deepcopy(self.ablation)
        report['runs'][0]['fixture']['packets'][0]['np'] = [1,2,2]
        with self.assertRaisesRegex(AssertionError, 'physical target'):
            recount(report, self.ablation_data)


if __name__ == '__main__': unittest.main()
