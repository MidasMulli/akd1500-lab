"""Compile the existing native learning shape; train first, never prequery episode inputs."""
from pathlib import Path
import argparse,json,struct,sys
import akida,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT))
from prepare import parts
from mario.sealed.bank import initial,save,sha


def build(data,output):
    output.mkdir(parents=True,exist_ok=False); z=np.load(data/'inputs.npz',allow_pickle=False)
    rows=[r for r in json.loads((data/'EXPERIENCE.json').read_text())['rows'] if r['valid']]
    x=np.array([z[f'{r["id"]}_{r["action"]}'] for r in rows]); y=np.array([r['success'] for r in rows],np.int32)
    assert y.min()==0 and y.max()==1
    builds=[]
    for seed in [84101,84102,84103]:
        weights=initial(seed,1024);model=akida.Model();model.add(akida.InputData(input_shape=(1,1,1024),input_bits=1))
        model.add(akida.FullyConnected(name='continuous_consequence',units=192,weights_bits=1,activation=False))
        model.compile(optimizer=akida.AkidaUnsupervised(num_weights=32,num_classes=3,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25))
        layer=model.get_layer('continuous_consequence');layer.set_variable('weights',weights.reshape(1,1,1024,192))
        layer.set_variable('threshold_learning',np.array([0]*128+[32]*64,np.int32))
        assert np.array_equal(model.forward(x[:2].reshape(-1,1,1,1024)).reshape(-1,192),x[:2].astype(np.int32)@weights.astype(np.int32))
        model.map(akida.AKD1500(),hw_only=True);assert len(model.sequences)==len(model.sequences[0].passes)==1
        program=bytes(model.sequences[0].program);ram,npid=parts(program)
        classes=[np.flatnonzero(y==k) for k in [0,1]];balanced=np.concatenate([np.resize(c,max(map(len,classes))) for c in classes])
        order=np.concatenate([np.random.default_rng(seed+e).permutation(balanced) for e in range(4)])
        for arm in ['full','frozen']:
            p=output/f's{seed}_{arm}'
            for suffix,b in [('bin',program),('ram',struct.pack('<'+'I'*len(ram),*ram)),('meta',b'1024 192 32 3\n'),('inputs',x[order].tobytes()),('labels',(y[order] if arm=='full' else np.full(len(order),2)).astype('<i4').tobytes())]:p.with_suffix('.'+suffix).write_bytes(b)
            meta=dict(seed=seed,arm=arm,training_presentations=len(order),samples=len(order),learning_np=npid,order=order.tolist(),horizon=24)
            save(p.with_suffix('.json'),meta);builds.append(dict(prefix=p.name,**meta))
    save(output/'BUILD.json',dict(status='VIRTUAL_PASS',physical_device_opened=False,rows=builds,source_sha256=sha(__file__)))
    print('VIRTUAL_PASS',len(builds),'training_forwards',len(order))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('output',type=Path);a=p.parse_args();build(a.data.resolve(),a.output.resolve())
