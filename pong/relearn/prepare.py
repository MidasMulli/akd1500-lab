"""Prospective compiler-only one-factor fixtures; old experiments stay frozen."""
from pathlib import Path
import argparse,importlib.util,struct,sys
import akida,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pong.relearn.protocol import N,U,CLASSES,design
from pong.experiment import save,sha
spec=importlib.util.spec_from_file_location('fixture_decoder',ROOT/'tools/prepare.py')
decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)


def build(output):
    output.mkdir(parents=True,exist_ok=False);plan=design();rows=[]
    for seed in plan['development_seeds']+plan['confirmation_seeds']:
        # Nested initial address sets within each seed across connection counts.
        rng=np.random.default_rng(seed);choices=[rng.choice(N,64,replace=False) for j in range(U)]
        for nw in [64,32,48]:
            weights=np.zeros((N,U),np.int8)
            for j in range(U):weights[choices[j][:nw],j]=1
            model=akida.Model();model.add(akida.InputData(input_shape=(1,1,N),input_bits=1))
            model.add(akida.FullyConnected(name='relearning_landing',units=U,weights_bits=1,activation=False))
            model.compile(optimizer=akida.AkidaUnsupervised(num_weights=nw,num_classes=CLASSES,**plan['fixed_parameters']))
            layer=model.get_layer('relearning_landing');layer.set_variable('weights',weights.reshape(1,1,N,U))
            layer.set_variable('threshold_learning',np.array([0]*128+[nw]*32,np.int32))
            x=rng.integers(0,2,(2,N),dtype=np.uint8)
            assert np.array_equal(model.forward(x.reshape(2,1,1,N)).reshape(2,U),x.astype(np.int32)@weights.astype(np.int32))
            model.map(akida.AKD1500(),hw_only=True);assert len(model.sequences)==len(model.sequences[0].passes)==1
            program=bytes(model.sequences[0].program);ram,npid=decoder.parts(program)
            assert len(ram)==2+U*(4+N//32)
            prefix=output/f'nw{nw}_s{seed}'
            prefix.with_suffix('.bin').write_bytes(program);prefix.with_suffix('.ram').write_bytes(struct.pack('<'+'I'*len(ram),*ram))
            prefix.with_suffix('.meta').write_text(f'{N} {U} {nw} {CLASSES}\n')
            row=dict(prefix=prefix.name,seed=seed,num_weights=nw,learning_np=npid,program_sha256=sha(prefix.with_suffix('.bin')))
            save(prefix.with_suffix('.json'),row);rows.append(row)
    save(output/'BUILD.json',dict(status='COMPILER_ONLY_PASS',physical_device_opened=False,rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);build(p.parse_args().output.resolve())
