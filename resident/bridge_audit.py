"""Post-closeout frozen-input and matched-control checks, without an SDK."""
import json
import numpy as np
from resident.bridge import BASE,ORDER,inspect
from resident.run import digest


def recount(meta,data):
    assert meta['status']=='FIXED_BRIDGE_DIAGNOSTIC' and meta['learning_sessions']==0, 'inference-only boundary'
    assert meta['physical_sessions']==5 and meta['observed_native_forwards']==321
    assert [r['name'] for r in meta['runs']]==ORDER[:5], 'complete controls and failed mixed attempt'
    counts=np.random.default_rng(99901).integers(0,17,(128,4));counts[:4]=np.eye(4,dtype=int)*16
    counts[4:8]=[[0,0,0,0],[16,16,16,16],[0,8,8,0],[4,0,4,0]]
    want=(np.arange(16)[None,None,:]<counts[:,:,None]).reshape(128,64).astype(np.uint8)
    checks=[]
    for i,r in enumerate(meta['runs']):
        z={k.split('__',1)[1]:v for k,v in data.items() if k.startswith(r['name']+'__')}
        assert np.array_equal(z['inputs'][:,:64],want), 'frozen bridge patterns'
        assert r['fixture']['kind']==r['name'].rsplit('_',1)[0]
        result=inspect(r,z);assert result==r['check'];checks.append(result)
        assert result['status']==('EXACT_FIXED_BRIDGE_PASS' if i<4 else 'FIXED_BRIDGE_OUTPUT_DIVERGENCE')
        assert r['device_released']
    assert np.array_equal(data['mixed_smoke__expected'][0],[128,0,0,0])
    assert np.array_equal(data['mixed_smoke__outputs'][0],[0,0,0,128]), 'preserved mixed handoff failure'
    return dict(status='MATCHED_FIXED_BRIDGE_RECOUNT_PASS',physical_sessions=5,observed_native_forwards=321,
        controls_full_exact=[128,128],mixed_exact=0,mixed_observed=1,learning_sessions=0)


def audit_public():
    meta=json.loads(BASE.with_suffix('.json').read_text());assert digest(BASE.with_suffix('.npz'))==meta['archive_sha256']
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:return recount(meta,{k:z[k] for k in z.files})


if __name__=='__main__':print(json.dumps(audit_public()))
