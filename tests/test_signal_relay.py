"""Claims depend on numerical traces, isolation and inference without teaching."""
from pathlib import Path
import json
import unittest
import numpy as np
from relay.audit_public import audit,audit_contents
from relay.design import schedule,initial
from relay.reference import Network
from relay.cpu_baseline import evaluate

P=Path(__file__).resolve().parents[1]


class SignalRelayEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((P/'results/2026-10-23-signal-relay.json').read_text())
        with np.load(P/'results/2026-10-23-signal-relay.npz',allow_pickle=False) as data:
            cls.data={k:data[k] for k in data.files}

    def test_all_native_outputs_and_bank_snapshots_recount(self):
        r=audit();self.assertEqual(r['native_forwards'],9280);self.assertEqual(r['full_bank_snapshots'],18560)

    def test_a_changed_native_destination_is_rejected(self):
        key='s97101_recognition__outputs';data=dict(self.data);data[key]=data[key].copy();data[key][0,0]+=1
        with self.assertRaisesRegex(AssertionError,'native output recount'):audit_contents(self.report,data)

    def test_a_frozen_downstream_connection_edit_is_rejected(self):
        key='s97101_recognition__bank1_snapshots';data=dict(self.data);data[key]=data[key].copy()
        data[key][640,6]^=1
        with self.assertRaisesRegex(AssertionError,'native full RAM recount'):audit_contents(self.report,data)

    def test_inference_label_cannot_change_output_or_memory(self):
        x,rows=schedule('recognition_repair');a=Network(initial(97101));b=Network(initial(97101))
        for values,row in zip(x[:512],rows[:512]):
            a.forward(values,row['label'],row['learning_mask']);b.forward(values,row['label'],row['learning_mask'])
        for values,row in zip(x[512:576],rows[512:576]):
            ya,_=a.forward(values,0,0);yb,_=b.forward(values,3,0);self.assertEqual(ya,yb)
        self.assertEqual([r.ram for r in a.banks],[r.ram for r in b.banks])

    def test_control_prefixes_are_identical_before_repair(self):
        ref_x,ref_rows=schedule('recognition_repair')
        for arm in ['frozen','routing_only']:
            x,rows=schedule(arm);self.assertTrue(np.array_equal(ref_x,x));self.assertEqual(ref_rows[:640],rows[:640])

    def test_diagnostic_CPU_result_is_reproducible(self):
        published=json.loads((P/'results/2026-10-23-signal-relay-cpu.json').read_text())
        self.assertEqual(evaluate(),published)


if __name__=='__main__':unittest.main()
