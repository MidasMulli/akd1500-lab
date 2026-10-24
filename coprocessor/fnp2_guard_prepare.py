"""Compile/decode one FNP2 contract fixture without discovering hardware."""
import json,struct
import numpy as np
import akida
from tools.prepare import P,Table,PARAMS
from metal.prepare import mapping

S=P/'studies/20261024-fnp2-guard'
def main(units):
    out=S/'fixtures01'/f'u{units}';out.mkdir(parents=True,exist_ok=False)
    n,u=512,units;rng=np.random.default_rng(107101);w=np.zeros((n,u),np.int8)
    for j in range(u):w[rng.choice(n,32,False),j]=1
    m=akida.Model();m.add(akida.InputData(input_shape=(1,1,n),input_bits=1));m.add(akida.FullyConnected(name='external_record_learner',units=u,weights_bits=1,activation=False));m.compile(optimizer=akida.AkidaUnsupervised(**PARAMS));l=m.get_layer('external_record_learner');l.set_variable('weights',w.reshape(1,1,n,u));l.set_variable('threshold_learning',np.zeros(u,np.int32))
    xrng=np.random.default_rng(107102);x=np.zeros((64,n),np.uint8)
    for i in range(64):x[i,xrng.choice(n,[0,512,64,256,64,256,128,256][i%8],False)]=1
    y=x.astype(np.int32)@w.astype(np.int32);assert np.array_equal(m.forward(x.reshape(-1,1,1,n)).reshape(64,u),y)
    m.map(akida.AKD1500(),hw_only=True,mode=akida.MapMode.Minimal);assert len(m.sequences)==len(m.sequences[0].passes)==1
    b=bytes(m.sequences[0].program);root=Table(b,4+struct.unpack_from('<I',b,4)[0]);learning=root.table(15);ram=learning.table(2);at=ram.field(1);assert at
    off,words=struct.unpack_from('<II',b,at);ident=list(b[at+8:at+11]);assert ident==[1,2,0]
    base=struct.unpack_from('<I',b)[0]+4;assert off%4==0 and words==u*20 and base+off+4*words<=len(b)
    a=np.frombuffer(b,'<u4',words,base+off).copy().reshape(u,20)
    packed=np.packbits(w.T,axis=1,bitorder='little').copy().view('<u4').reshape(u,16)
    assert np.array_equal(a[:,4:],packed),'FNP2 numeric connection layout'
    assert np.all(a[:,2]==32) and np.all(a[:,3]==16),'swap budget and threshold word'
    # Two synthetic zero words adapt to the existing independently tested
    # CPU selector class. They are not part of native FNP2 storage.
    cpu=np.concatenate([np.zeros(2,'<u4'),a.reshape(-1)])
    f=out/'learner'
    for ext,data in [('bin',b),('raw',a.tobytes()),('cpu_ram',cpu.tobytes()),('inputs',x.tobytes()),('expected',y.astype('<i4').tobytes()),('labels',(np.arange(64,dtype='<i4')%2).tobytes())]:f.with_suffix('.'+ext).write_bytes(data)
    meta=dict(status='COMPILER_VIRTUAL_AND_RECORD_DECODE_PASS',physical_device_opened=False,input_features=n,records=u,record_words=20,connections_per_record=32,classes=2,weights_seed=107101,input_seed=107102,parameters=PARAMS,mapping=mapping(m),learning_np=ident,external_record=dict(program_data_offset=off,word_count=words,byte_count=words*4),synthetic_cpu_header_words=2)
    f.with_suffix('.units').write_text(str(u)+'\n');f.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
if __name__=='__main__':
    import sys
    main(int(sys.argv[1]))
