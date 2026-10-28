"""Data leakage boundaries and exported arithmetic, not gate-mirroring tests."""
import numpy as np
import torch
from visual_memory.data import masks
from visual_memory.train import Encoder
from visual_memory.encoder import forward

def test_object_and_session_boundaries():
    s,o=np.meshgrid(np.arange(1,12),np.arange(1,51));d=dict(session=s.ravel(),object=o.ravel());m=masks(d)
    assert m['fit'].sum()==200 and m['dev'].sum()==40 and m['reserved_dev'].sum()==10
    for k in ['dev','reserved_dev','sealed']:assert not (m['fit']&m[k]).any()
    assert not set(d['object'][m['fit']])&set(d['object'][m['reserved_dev']])

def test_exported_integer_spatial_orientation_and_axon_order():
    torch.manual_seed(2401);torch.set_num_threads(2);m=Encoder()
    rng=np.random.default_rng(108);x=rng.integers(0,256,(2,64,64,3),np.uint8)
    # Nonzero fitted BN state avoids untrained exact-zero floating thresholds.
    m.train();m(torch.from_numpy(x).permute(0,3,1,2).float()/255,True);m.eval()
    want=m(torch.from_numpy(x).permute(0,3,1,2).float()/255,True,True).detach().numpy().astype('u1')
    weights=m.exported();got=forward(x,weights)
    assert np.array_equal(got,want)
    changed={k:v.copy() for k,v in weights.items()};changed['w1']=changed['w1'][::-1,::-1].copy()
    assert not np.array_equal(forward(x,changed),want),'orientation mutant must be detected'
    assert not np.array_equal(got.reshape(2,8,8,8).transpose(0,1,3,2).reshape(2,512),want),'axon order mutant must be detected'

def test_binary_ste_can_train_projection():
    torch.manual_seed(3);m=Encoder();x=torch.rand(4,3,64,64)
    before=m.blocks[2].w.detach().clone();optim=torch.optim.AdamW(m.parameters(),lr=.001)
    z=m(x,True);torch.nn.functional.cross_entropy(z,torch.arange(4)).backward();optim.step()
    assert torch.isfinite(m.blocks[2].w.grad).all() and not torch.equal(before,m.blocks[2].w)
