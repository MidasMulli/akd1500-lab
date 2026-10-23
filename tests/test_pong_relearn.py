"""Information, history, retention and exact-reference checks before new runs."""
from pathlib import Path
import unittest
import json
import tempfile
import numpy as np
from pong.relearn.protocol import design,schedule
from pong.relearn.model import Reference,RecentNearest
ROOT=Path(__file__).resolve().parents[1]


class RelearningProtocol(unittest.TestCase):
    def test_no_retention_refresh_and_matched_new_teaching(self):
        p=design();streams={a:schedule(p,a) for a in ['frozen','fresh','relearn']}
        self.assertEqual(streams['frozen'][:512],streams['relearn'][:512])
        for arm,rows in streams.items():
            self.assertTrue(all(r['label']==4 for r in rows if r['observation']['court']==1 and r['phase']!='prefix_B'))
            self.assertTrue(all(r['label']==4 for r in rows if r['phase'] in ['fit_A','new_A','retention_B','old_A']))
        self.assertEqual([r for r in streams['fresh'] if r['phase']=='teach_A'],[r for r in streams['relearn'] if r['phase']=='teach_A'])
        self.assertTrue(all(r['label']==4 for r in streams['fresh'] if r['phase']=='prefix_A'))
        self.assertTrue(all(r['label']==4 for r in streams['frozen'][512:]))

    def test_generic_reference_replays_existing_native_scores_and_RAM(self):
        z=np.load(ROOT/'results/2026-10-23-pong-physics.npz',allow_pickle=False);name='development_85201_full'
        model=Reference(z[f'fixture__{name}__ram']);x=z[f'fixture__{name}__inputs'].reshape(-1,1024)
        y=z[f'fixture__{name}__labels'];scores=z[name+'__native_scores'].reshape(-1,160);states=z[name+'__snapshots']
        for i,(a,b) in enumerate(zip(x,y)):
            self.assertEqual(model.forward(a,int(b)),scores[i].tolist());self.assertEqual(model.ram,states[i].tolist())

    def test_CPU_has_no_future_labels_and_queries_never_teach(self):
        model=RecentNearest();x=np.zeros(1024,np.uint8);x[:64]=1
        self.assertEqual(model.forward(x,3,0),0)
        self.assertEqual(model.forward(x,4,0),3)
        before=list(model.memory[0]);self.assertEqual(model.forward(x,4,0),3)
        self.assertEqual(list(model.memory[0]),before)
        self.assertEqual(model.forward(x,1,0),3)
        self.assertEqual(model.forward(x,4,0),1)


class RelearningEvidence(unittest.TestCase):
    def test_all_new_native_scores_and_per_step_RAM(self):
        from mario.sealed.archive import materialize
        from tools.verify import verify
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs=materialize(root,stem='2026-10-23-pong-relearning')
            self.assertEqual(len(runs),4)
            self.assertEqual(sum(verify(root,r)['samples'] for r in runs),6752)

    def test_public_three_arm_metrics_CPU_predictions_and_timing_recount(self):
        from pong.relearn.audit_public import audit
        self.assertEqual(audit(),dict(status='PUBLIC_THREE_ARM_RECOUNT_PASS',bound_forwards=6720))

    def test_forecast_is_reproducible_and_distinct_from_hardware(self):
        from pong.relearn.screen import simulate
        report=json.loads((ROOT/'results/2026-10-23-pong-relearning.json').read_text())
        self.assertFalse(report['cpu_forecasts']['physical_device_opened'])
        self.assertIsNone(report['cpu_forecasts']['selected_num_weights'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'fixtures-relearn01').mkdir()
            (root/'PROTOCOL.json').write_text(json.dumps(report['protocol']))
            z=np.load(ROOT/'results/2026-10-23-pong-relearning.npz',allow_pickle=False)
            (root/'fixtures-relearn01/nw48_s85301.ram').write_bytes(z['forecast_initial__nw48_s85301'].astype('<u4').tobytes())
            result=simulate(root,85301,48,'relearn')
            expected=next(r for r in report['cpu_forecasts']['results'] if r['num_weights']==48)['rows'][0]['arms']['relearn']
            self.assertEqual(result,expected)


class SubstrateEvidence(unittest.TestCase):
    def test_every_transport_mode_replays_original_Pong_learning(self):
        from mario.sealed.archive import materialize
        from tools.verify import verify
        source=np.load(ROOT/'results/2026-10-23-pong-relearning.npz',allow_pickle=False)
        expected=source['development_nw64_s85301_relearn__native_scores'].tobytes()
        final=source['development_nw64_s85301_relearn__native_ram'].tobytes()
        total=0
        for stem,count in [('2026-10-23-pong-substrate',12),('2026-10-23-pong-read-dma',6)]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);runs=materialize(root,stem=stem);self.assertEqual(len(runs),count)
                for r in runs:
                    total+=verify(root,r)['samples']
                    self.assertEqual((r/'native.scores').read_bytes(),expected)
                    self.assertEqual((r/'native.ram').read_bytes(),final)
        self.assertEqual(total,40320)

    def test_transport_cost_arithmetic_and_no_silent_DMA_fallback(self):
        from pong.substrate_archive import summary
        for stem in ['2026-10-23-pong-substrate','2026-10-23-pong-read-dma']:
            r=json.loads((ROOT/f'results/{stem}.json').read_text())
            self.assertEqual(summary(r['rows'],list(r['protocol']['configurations'])),r['summary'])
            for row in r['rows']:
                a,b=row['results']['native'],row['results']['cpu'];c=r['protocol']['configurations'][row['configuration']]
                self.assertAlmostEqual(row['native_elapsed_us_per_input'],a['wall_ns']/2240/1000)
                self.assertAlmostEqual(row['exact_CPU_elapsed_us_per_input'],b['wall_ns']/2240/32/1000)
                if c['dma_read_min']:self.assertGreater(a['transaction_DMA_reads'],0)
                if c['dma_write_min']:self.assertGreater(a['transaction_DMA_writes'],0)
                else:self.assertEqual(a['transaction_DMA_writes'],0)
                if c['resident']:
                    self.assertEqual(a['queue_slots'],9);self.assertEqual(a['batch'],8)
                    self.assertEqual(a['host_writes_during_execution'],0)


if __name__=='__main__':unittest.main()
