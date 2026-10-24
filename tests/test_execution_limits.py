"""Recount the new hardware evidence and reject altered state/control metadata."""
from pathlib import Path
import copy
import hashlib
import json
import unittest
from unittest.mock import patch
import numpy as np
from metal import sparse, interference, cpu_compare

P = Path(__file__).resolve().parents[1]


class ExecutionLimits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sr = json.loads((P / 'results/2026-10-23-sparse-work.json').read_text())
        cls.ir = json.loads((P / 'results/2026-10-23-learning-interference.json').read_text())
        cls.sz = np.load(P / 'results/2026-10-23-sparse-work.npz', allow_pickle=False)
        cls.iz = np.load(P / 'results/2026-10-23-learning-interference.npz', allow_pickle=False)

    @classmethod
    def tearDownClass(cls): cls.sz.close(); cls.iz.close()

    def cell(self, coupled=False, name=None):
        r, z = (self.ir, self.iz) if coupled else (self.sr, self.sz)
        name = name or ('full_s97101_m3' if coupled else 'screen_n128_u64_w8_compact')
        rec = copy.deepcopy(next(x for x in r['runs'] if x['name'] == name))
        data = {k.split('__', 1)[1]: z[k].copy() for k in z.files if k.startswith(name + '__')}
        return rec, data

    def test_sparse_native_archive(self):
        self.assertEqual(sparse.audit_public()['native_forwards'], 9696)

    def test_coupled_native_archive(self):
        self.assertEqual(interference.audit_public()['native_forwards'], 10016)

    def test_cpu_matches_native_outputs_and_both_final_memories(self):
        self.assertEqual(cpu_compare.audit_public()['cpu_runs'], 13)

    def test_sparse_output_change_rejected(self):
        r, d = self.cell(); d['native.scores'][0, 0] += 1
        with self.assertRaisesRegex(AssertionError, 'native output'): sparse.recount(r, d)

    def test_sparse_query_memory_change_rejected(self):
        r, d = self.cell(); d['native.snapshots'][0, 8] ^= 1
        with self.assertRaisesRegex(AssertionError, 'complete RAM'): sparse.recount(r, d)

    def test_sparse_counter_change_rejected(self):
        r, d = self.cell(); d['native.cycles'][0] += 1
        with self.assertRaisesRegex(AssertionError, 'counter delta'): sparse.recount(r, d)

    def test_compact_readout_change_rejected(self):
        r, d = self.cell(); d['native.fixed.final'][3] ^= 1
        with self.assertRaisesRegex(AssertionError, 'fixed readout memory'): sparse.recount(r, d)

    def test_kernel_substitution_rejected(self):
        r, d = self.cell(); r['fixture_info']['mapping'][0]['nps'][0]['type'] = 'Type.FNP2'
        with self.assertRaisesRegex(AssertionError, 'declared FNP3 placement'): sparse.recount(r, d)

    def test_connection_budget_mislabel_rejected(self):
        r, d = self.cell(); r['fixture_info']['weights_per_neuron'] = 32; r['fixture_info']['threshold'] = 15
        with self.assertRaisesRegex(AssertionError, 'initial selected connections'): sparse.recount(r, d)

    def test_coupled_output_change_rejected(self):
        r, d = self.cell(True); d['native.scores'][512, 0] += 1
        with self.assertRaisesRegex(AssertionError, 'coupled native output'): interference.recount(r, d)

    def test_coupled_memory_change_rejected(self):
        r, d = self.cell(True); d['bank1.snapshots'][512, 4] ^= 1
        with self.assertRaisesRegex(AssertionError, 'coupled complete RAM'): interference.recount(r, d)

    def test_coupled_teacher_change_rejected(self):
        r, d = self.cell(True); d['stream.masks'][512] = 1
        with self.assertRaisesRegex(AssertionError, 'coupled masks'): interference.recount(r, d)

    def test_coupled_counter_scope_change_rejected(self):
        r, d = self.cell(True); d['native.control_bounds'][:] |= 256
        with self.assertRaisesRegex(AssertionError, 'outbound counter scope'): interference.recount(r, d)

    def test_missing_solo_arm_rejected(self):
        rr = [copy.deepcopy(r) for r in self.ir['runs'] if r['seed'] == 97101 and not r['smoke'] and r['mask'] != 2]
        with self.assertRaisesRegex(AssertionError, 'complete four-mask comparison'): interference.paired(rr, {})

    def test_missing_entire_sparse_stage_rejected(self):
        report = copy.deepcopy(self.sr); report['runs'].pop()
        original = json.loads
        with patch.object(sparse.json, 'loads', side_effect=lambda text: report if 'OBSERVED_NATIVE_SPARSE_WORK' in text else original(text)):
            with self.assertRaisesRegex(AssertionError, 'complete sparse treatment set'): sparse.audit_public()

    def test_missing_entire_coupled_seed_rejected(self):
        report = copy.deepcopy(self.ir); report['runs'] = [r for r in report['runs'] if r['seed'] != 97103]
        original = json.loads
        with patch.object(interference.json, 'loads', side_effect=lambda text: report if 'OBSERVED_NATIVE_COUPLED_LEARNING' in text else original(text)):
            with self.assertRaisesRegex(AssertionError, 'complete coupled treatment set'): interference.audit_public()

    def test_chart_source_binding(self):
        b = json.loads((P / 'visual/EXECUTION_LIMITS_SOURCE_BINDING.json').read_text())
        for p, h in b['hashes'].items(): self.assertEqual(hashlib.sha256((P / p).read_bytes()).hexdigest(), h, p)


if __name__ == '__main__': unittest.main()
