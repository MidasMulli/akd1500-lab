"""Semantic corruption tests for the new parameterized spatial audit."""
import unittest
from pathlib import Path
import numpy as np
from coprocessor.extent_audit import masks,validate,load_fixture,BASE,audit
from coprocessor.deployment import describe

P=Path(__file__).resolve().parents[1]
class ReferenceScaleTests(unittest.TestCase):
    def test_mask_record_stride_at_all_extents(self):
        for n in [128,512,1152,2048]:
            r=np.zeros(2+128*(4+n//32),dtype='<u4')
            records=r[2:].reshape(128,4+n//32)
            records[:,4]=0xffffffff;records[73,-1]=0x80000000
            m=masks(r,n)
            self.assertEqual(m.shape,(128,n));self.assertEqual(m[73,-1],1)
            self.assertEqual(m[0].sum(),32);self.assertEqual(m[73].sum(),33)
    def test_mapping_survives_json_roundtrip(self):
        import json
        class Ident:col,row,id=1,2,3
        class NP:ident,type=Ident(),'CNP'
        class Mapping:nps=[NP()]
        class Layer:name,mapping='a',Mapping()
        class Pass:layers=[Layer()]
        class Sequence:passes=[Pass()]
        class Model:sequences,external_memory_size=[Sequence()],0
        m=describe(Model());self.assertEqual(m,json.loads(json.dumps(m)))
    def fixture(self):
        d=P/'studies/20261025-reference-scale/extent_fixtures/s32_s107501'
        if d.exists():meta,a=load_fixture(d)
        else:
            import json
            meta=json.loads(BASE.with_suffix('.json').read_text())['fixtures']['s32_s107501']
            with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:a={k:z[v['key']].copy() for k,v in meta['arrays'].items()}
            meta=meta['metadata']
        n=meta['n'];initial=a['ram'];features=a['features'].reshape(64,n)
        # Frozen read-only smoke uses actual compiled masks and integer scores.
        arrays=dict(snapshots=np.tile(initial,8),ram=initial.copy(),scores=a['expected'][:8*128].copy(),latency_ns=np.ones(8,dtype='i8')*1000,caller_ns=np.ones(8,dtype='i8')*900)
        r=dict(cell='test',mode='native',arm=0,count=8,repeats=1)
        f=dict(n=n,u=128,features=features,expected=a['expected'].reshape(64,128),labels=a['labels'])
        return r,f,arrays,initial
    def test_frozen_score_corruption_rejected(self):
        r,f,a,i=self.fixture();validate(r,f,a.__getitem__,i);a['scores'][0]+=1
        with self.assertRaises(AssertionError):validate(r,f,a.__getitem__,i)
    def test_query_metadata_mutation_rejected(self):
        r,f,a,i=self.fixture();a['snapshots'][2]^=1
        with self.assertRaises(AssertionError):validate(r,f,a.__getitem__,i)
    def test_feature_corruption_rejected(self):
        r,f,a,i=self.fixture();f['features']=1-f['features']
        with self.assertRaises(AssertionError):validate(r,f,a.__getitem__,i)
    def test_portable_archive_reconstructs_all_physical_streams(self):
        r=audit();self.assertEqual(r['status'],'INDEPENDENT_NUMERICAL_ARCHIVE_PASS')
        self.assertEqual(r['physical_scale_forwards']+r['reference_forwards'],1824)
        self.assertEqual(r['compiler_failures_preserved'],2)
if __name__=='__main__':unittest.main()
