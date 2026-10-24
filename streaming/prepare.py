"""Compiler-only single-FNP native fixtures, with a protected query class."""
import argparse,struct
from pathlib import Path
import numpy as np
import akida
from streaming.design import N,U,NW,CLASSES,SEEDS,save,sha
from tools.prepare import parts

def build(study):
    out=study/'fixtures-stream01';out.mkdir(exist_ok=False);rows=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed);w=np.zeros((N,U),np.int8)
        for j in range(U):w[rng.choice(N,NW,False),j]=1
        m=akida.Model();m.add(akida.InputData(input_shape=(1,1,N),input_bits=1))
        m.add(akida.FullyConnected(name='stream_predictor',units=U,weights_bits=1,activation=False))
        m.compile(optimizer=akida.AkidaUnsupervised(num_weights=NW,num_classes=CLASSES,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25))
        layer=m.get_layer('stream_predictor');layer.set_variable('weights',w.reshape(1,1,N,U))
        layer.set_variable('threshold_learning',np.array([0]*128+[NW]*32,np.int32))
        x=rng.integers(0,2,(4,N),dtype=np.uint8)
        assert np.array_equal(m.forward(x.reshape(4,1,1,N)).reshape(4,U),x.astype(np.int32)@w.astype(np.int32))
        m.map(akida.AKD1500(),hw_only=True);assert len(m.sequences)==len(m.sequences[0].passes)==1
        program=bytes(m.sequences[0].program);ram,npid=parts(program);assert len(ram)==2+U*(4+N//32)
        prefix=out/f's{seed}';prefix.with_suffix('.bin').write_bytes(program)
        prefix.with_suffix('.ram').write_bytes(struct.pack('<'+'I'*len(ram),*ram));prefix.with_suffix('.meta').write_text(f'{N} {U} {NW} {CLASSES}\n')
        row=dict(seed=seed,learning_np=npid,program_sha256=sha(prefix.with_suffix('.bin')),initial_ram_sha256=sha(prefix.with_suffix('.ram')))
        save(prefix.with_suffix('.json'),row);rows.append(row)
    save(out/'BUILD.json',dict(status='COMPILER_EXACT_FIXED_OUTPUT_PASS',physical_device_opened=False,rows=rows))
    print('compiler fixtures PASS',len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);build(p.parse_args().study.resolve())
