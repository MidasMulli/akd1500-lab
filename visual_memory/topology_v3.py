"""Virtual/mapping-only richer student qualification, no live-device discovery."""
from pathlib import Path
import argparse,json,traceback
import numpy as np
import akida
from visual_memory.encoder_v3 import forward
from visual_memory.data import save,sha
from coprocessor.prepare import Select
from metal.prepare import mapping
from tools.prepare import parts

def model(w,x):
    m=akida.Model();expected=forward(x,w,True)
    sizes=[32,128,256,256,256,512]
    for i,filters in enumerate(sizes):
        name=f'own_resident_{i}'
        if i==0:layer=akida.InputConvolutional(name=name,input_shape=(128,128,3),filters=filters,kernel_size=(3,3),kernel_stride=(2,2),weights_bits=8,act_bits=4)
        else:
            opts=dict(name=name,filters=filters,kernel_size=(3,3) if i in [1,2] else (1,1),kernel_stride=(2,2) if i in [1,2] else (1,1),weights_bits=2 if i==5 else 4,act_bits=1 if i==5 else 4)
            if i in [3,4]:opts.update(pool_size=(2,2),pool_stride=(2,2),pool_type=akida.PoolType.Max)
            layer=akida.Convolutional(**opts)
        m.add(layer);l=m.get_layer(name);l.set_variable('weights',w[f'w{i}']);l.set_variable('threshold',w[f't{i}']);l.set_variable('act_step',np.ones(filters,np.float32) if i==5 else w[f's{i}'].astype('f4'))
        if i in [1,2]:
            # Exact monotone-activation/pooling commute. Splitting avoids the
            # SDK's strided-convolution+pool spatial-multi-NP restriction.
            pn=f'identity_pool_{i}';m.add(akida.Convolutional(name=pn,filters=filters,kernel_size=(1,1),weights_bits=2,pool_size=(2,2),pool_stride=(2,2),pool_type=akida.PoolType.Max,act_bits=4))
            pl=m.get_layer(pn);pl.set_variable('weights',np.eye(filters,dtype='i1').reshape(1,1,filters,filters));pl.set_variable('threshold',np.zeros(filters,'i4'));pl.set_variable('act_step',np.ones(filters,'f4'))
        actual=m.forward(x);assert np.array_equal(actual,expected[i]),('SDK virtual layer mismatch',i,int((actual!=expected[i]).sum()))
    n,u,nw,classes=512,352,64,11;m.add(akida.FullyConnected(name='own_native_memory',units=u,weights_bits=1,activation=False));m.compile(optimizer=akida.AkidaUnsupervised(num_weights=nw,num_classes=classes,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25))
    rng=np.random.default_rng(20261026);lw=np.zeros((n,u),'i1')
    for j in range(u):lw[rng.choice(n,nw,False),j]=1
    l=m.get_layer('own_native_memory');l.set_variable('weights',lw.reshape(1,1,n,u));l.set_variable('threshold_learning',np.array([0]*320+[nw]*32,'i4'))
    assert np.array_equal(m.forward(x).reshape(-1,u),expected[-1].astype('i4')@lw.astype('i4'))
    return m

def check(study):
    r=dict(physical_device_opened=False)
    try:
        w=np.load(study/'dummy_weights.npz');x=np.load(study/'dummy_witness.npz')['images'];m=model(w,x);d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d))
        assert len(m.sequences)==len(m.sequences[0].passes)==1,'one resident pass required'
        b=bytes(m.sequences[0].program);ram,npid=parts(b);assert npid==[3,3,0] and len(ram)==7042
        (study/'dummy_program.bin').write_bytes(b);r.update(status='VIRTUAL_MAP_PASS',mapping=mapping(m),program_bytes=len(b),program_sha256=sha(study/'dummy_program.bin'),learning_np=npid,single_resident_pass=True,scope='Random-weight topology/virtual arithmetic only, not trained quality or physical qualification')
    except BaseException as e:r.update(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc())
    save(study/'TOPOLOGY.json',r);print(json.dumps(r),flush=True)
    assert r['status']=='VIRTUAL_MAP_PASS'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);a=p.parse_args();check(a.study)
