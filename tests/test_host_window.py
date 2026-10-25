"""Semantic timing/memory guard failures, independent of checksum enforcement."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from coprocessor import host_window_run as W

P=Path(__file__).resolve().parents[1]
class HostWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b=P/'results/20261025-reference-scale';meta=json.loads(b.with_suffix('.json').read_text());f=meta['fixtures']['s64_s107501'];cls.meta=dict(f['metadata'],output_contract='potentials')
        with np.load(b.with_suffix('.npz'),allow_pickle=False) as z:
            cls.fixture={k:z[v['key']].copy() for k,v in f['arrays'].items()}
            r=meta['runs']['cpu-blas4_s64_s107501_a1_n64_r0'];cls.trace={k:z[v['key']].copy() for k,v in r['arrays'].items() if k in ['scores','snapshots','ram']}
    def make(self,root,busy=False):
        f=root/'fixtures/s64_s107501_potentials';f.mkdir(parents=True);(f/'BUILD.json').write_text(json.dumps(self.meta))
        for k,v in self.fixture.items():v.tofile(f/('cell.'+k))
        ref=root/'reference';ref.mkdir()
        for k in ['scores','snapshots']:self.trace[k].tofile(ref/('s64_s107501_a1.'+k))
        d=root/'cell';d.mkdir();start=1_000_000_000;period=50_000_000
        arrays=dict(scores=self.trace['scores'].copy(),ram=self.trace['ram'].copy(),snapshots=self.trace['ram'].copy(),boundary_index=np.array([64],'u4'),
                    started_ns=start+np.arange(64,dtype='i8')*period,finished_ns=start+np.arange(64,dtype='i8')*period+1_000_000,latency_ns=np.ones(64,'i8')*1_000_000,caller_ns=np.ones(64,'i8')*500_000)
        for k,v in arrays.items():v.tofile(d/('result.'+k))
        receipt=dict(status='EXECUTION_PASS',hashes={},device_free=True,mode='cpu-blas4',busy=busy,seed=107501,arm=1,count=64,start_ns=start,end_ns=start+64*period,period_ns=period,audit_released_ns=start+64*period+2_000_000)
        if busy:
            timing=np.array([[start,start+10_000_000,5_000_000],[start+10_000_000,start+20_000_000,5_000_000]],'i8');np.save(d/'foreground-timing.npy',timing)
            (d/'MODEL_RECEIPT.json').write_text(json.dumps(dict(weights_unchanged=True,start_ns=receipt['start_ns'],end_ns=receipt['end_ns'],process_cpu_ns=10_000_000)))
        (d/'RECEIPT.json').write_text(json.dumps(receipt));return d,arrays,receipt
    def test_exact_output_final_ram_and_window_pass(self):
        for busy in [False,True]:
            with tempfile.TemporaryDirectory() as tmp,patch.object(W,'S',Path(tmp)):
                d,_,_=self.make(Path(tmp),busy);self.assertEqual(W.audit(d)['status'],'EXACT_WINDOW_PASS')
    def test_wrong_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(W,'S',Path(tmp)):
            d,a,_=self.make(Path(tmp));a['scores'][3]^=1;a['scores'].tofile(d/'result.scores')
            with self.assertRaisesRegex(AssertionError,'outputs'):W.audit(d)
    def test_overlapping_actor_timestamps_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(W,'S',Path(tmp)):
            d,a,_=self.make(Path(tmp));a['finished_ns'][0]=a['started_ns'][1]+1;a['finished_ns'].tofile(d/'result.finished_ns')
            with self.assertRaises(AssertionError):W.audit(d)
    def test_audit_release_cannot_overlap_foreground(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(W,'S',Path(tmp)):
            d,_,r=self.make(Path(tmp),True);r['audit_released_ns']=r['start_ns'];(d/'RECEIPT.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(AssertionError,'oracle overlapped'):W.audit(d)
    def test_wrong_complete_memory_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(W,'S',Path(tmp)):
            d,a,_=self.make(Path(tmp));a['ram'][19]^=1;a['ram'].tofile(d/'result.ram')
            with self.assertRaisesRegex(AssertionError,'full RAM'):W.audit(d)

if __name__=='__main__':unittest.main()
