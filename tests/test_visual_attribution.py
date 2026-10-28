import unittest
from unittest.mock import patch

import numpy as np

from visual_memory.attribution import masks, packed_input, score_grid, predictions, prefix_anchor_guard, verify, validate_physical_initial
from visual_memory.admission_screen import initial


class AttributionTests(unittest.TestCase):
    def test_native_descriptor_metadata_is_preserved(self):
        logical = initial(160, 64); actual = logical.copy()
        actual[:2] = [345, 678]; actual[2:5] = [9, 8, 64]
        np.testing.assert_array_equal(validate_physical_initial(actual, logical, 512), actual)

    def test_native_initial_mask_mismatch_rejected(self):
        logical = initial(160, 64); actual = logical.copy(); actual[6] ^= 1
        with self.assertRaises(AssertionError): validate_physical_initial(actual, logical, 512)

    def test_little_endian_masks_agree_independent_dense(self):
        ram = initial(160, 64); w = masks(ram, 512)
        dense = np.unpackbits(ram[2:].reshape(220, 20)[:, 4:].copy().view('u1'), axis=1, bitorder='little')
        rng = np.random.default_rng(7)
        x = rng.integers(0, 2, (11, 512), dtype='u1')
        np.testing.assert_array_equal(score_grid([packed_input(v) for v in x], w), x.astype('i4') @ dense.T.astype('i4'))

    def test_readout_ignores_helpers_and_uses_first_tie(self):
        x = np.zeros((2, 220), 'i4'); x[0, 200:] = 64; x[1, 20] = 40; x[1, 0] = 40
        np.testing.assert_array_equal(predictions(x), [0, 0])
        used = np.zeros(220, bool); used[20] = True
        np.testing.assert_array_equal(predictions(x, used), [1, 1])

    def test_guard_only_uses_given_teachers_no_probe_targets(self):
        # This controller cannot accept evaluation labels: they are absent from
        # its signature. It must behave identically for any evaluator metadata.
        import inspect
        self.assertEqual(list(inspect.signature(prefix_anchor_guard).parameters), ['inputs', 'teachers', 'initial', 'n', 'use_guard'])

    def test_saved_score_corruption_is_rejected(self):
        ram = initial(160, 64); x = np.zeros((1, 512), 'u1')
        arm = dict(inputs=x, labels=np.array([10]), scores=np.zeros((1, 220), 'i4'), ram=ram[None, :].copy())
        case = dict(n=512, initial=ram, rows=[{}], arms={'adaptive':arm, 'frozen':arm})
        verify(case)
        arm['scores'][0, 0] = 1
        with self.assertRaises(AssertionError): verify(case)

    def test_readonly_metadata_write_is_rejected(self):
        ram = initial(160, 64); x = np.zeros((1, 512), 'u1')
        arm = dict(inputs=x, labels=np.array([10]), scores=np.zeros((1, 220), 'i4'), ram=ram[None, :].copy())
        case = dict(n=512, initial=ram, rows=[{}], arms={'adaptive':arm, 'frozen':arm})
        arm['ram'][0, 5] ^= 1
        with self.assertRaises(AssertionError): verify(case)

    def test_rejected_trial_preserves_live_hidden_state(self):
        class FakeRule:
            def __init__(self, ram, nw):
                self.ram = [0]; self.weights = [0] * 220; self.counter = 0
            def forward(self, v, y, learning=True):
                answer = [self.counter] * 220
                if learning:
                    self.counter += 1; self.ram[0] += 1; self.weights[0] += 1
                return answer
        x = np.zeros((160, 512), 'u1'); y = np.zeros(160, 'i4')
        calls = 0
        def fake_pred(values):
            nonlocal calls
            calls += 1
            return np.zeros(80, 'i4') if calls % 2 else np.ones(80, 'i4')
        with patch('visual_memory.attribution.Rule', FakeRule), patch('visual_memory.attribution.predictions', fake_pred):
            scores, decisions = prefix_anchor_guard(x, y, [], 512, True)
        self.assertTrue(all(not d['admit'] for d in decisions))
        np.testing.assert_array_equal(scores[80:, 0], np.full(80, 80))


if __name__ == '__main__':
    unittest.main()
