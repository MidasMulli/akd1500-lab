"""Reject semantic corruption even when result digests are bypassed."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from coprocessor import service_audit as A

P=Path(__file__).resolve().parents[1]
class HostServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b=P/'results/20261025-reference-scale'
        meta=json.loads(b.with_suffix('.json').read_text());f=meta['fixtures']['s32_s107501'];cls.meta=f['metadata']
        with np.load(b.with_suffix('.npz'),allow_pickle=False) as z:
            cls.fixture={k:z[v['key']].copy() for k,v in f['arrays'].items()}
            r=meta['runs']['cpu-blas4_s32_s107501_a1_n64_r0']
            cls.trace={k:z[v['key']].copy() for k,v in r['arrays'].items() if k in ['scores','snapshots','ram']}
    def cell(self,root,full=True,mask=False):
        f=root/'fixtures/test';f.mkdir(parents=True);m=dict(self.meta,output_contract='mask' if mask else 'potentials')
        (f/'BUILD.json').write_text(json.dumps(m))
        for k,v in self.fixture.items():v.tofile(f/('cell.'+k))
        # These fixed inference words may change; learned fields must not.
        initial=self.fixture['ram'].copy()
        if mask:initial[2]=initial[2]^1
        initial.tofile(f/'cell.ram');n=m['n'];count=64;batch=8
        ref=root/'reference';ref.mkdir();self.trace['scores'].tofile(ref/'s32_s107501_a1.scores');self.trace['snapshots'].tofile(ref/'s32_s107501_a1.snapshots')
        expected=A.expected_states(initial,self.trace['snapshots'].reshape(64,-1),n)
        idx=np.arange(8,65,8,dtype='u4') if full else np.array([64],dtype='u4')
        d=root/'raw/test';d.mkdir(parents=True)
        arrays={'scores':(self.trace['scores']>24).astype('i4') if mask else self.trace['scores'],
                'snapshots':expected[idx-1], 'ram':expected[-1], 'boundary_index':idx,
                'latency_ns':np.ones(8,dtype='i8')*1000000,'caller_ns':np.ones(8,dtype='i8')*500000}
        for k,v in arrays.items():v.tofile(d/('result.'+k))
        r=dict(exit_code=0,hashes={},mode='cpu-blas4',fixture='test',arm=1,count=64,batch=8,full_audit=full,repeats=1,sleep_us=0,profile=1)
        (d/'RECEIPT.json').write_text(json.dumps(r));return d,arrays
    def test_exact_reference_and_compact_contract_pass(self):
        for mask in [False,True]:
            with tempfile.TemporaryDirectory() as tmp,patch.object(A,'S',Path(tmp)):
                d,_=self.cell(Path(tmp),mask=mask);self.assertEqual(A.audit(d)['status'],'EXACT_SERVICE_PASS')
    def test_wrong_output_rejected_without_digest_gate(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(A,'S',Path(tmp)):
            d,a=self.cell(Path(tmp));a['scores'][5]^=1;a['scores'].tofile(d/'result.scores')
            with self.assertRaisesRegex(AssertionError,'exact outputs'):A.audit(d)
    def test_missing_native_boundary_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(A,'S',Path(tmp)):
            d,a=self.cell(Path(tmp));a['boundary_index'][-1]-=1;a['boundary_index'].tofile(d/'result.boundary_index')
            with self.assertRaisesRegex(AssertionError,'boundary'):A.audit(d)
    def test_corrupt_state_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(A,'S',Path(tmp)):
            d,a=self.cell(Path(tmp));a['snapshots'][-1,12]^=1;a['snapshots'].tofile(d/'result.snapshots');a['snapshots'][-1].tofile(d/'result.ram')
            with self.assertRaisesRegex(AssertionError,'RAM boundary'):A.audit(d)

if __name__=='__main__':unittest.main()
