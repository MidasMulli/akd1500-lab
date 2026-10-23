"""Virtual-only compiler for a protected-query Mario association bank."""
from pathlib import Path
import argparse,hashlib,json,struct,sys
import akida,numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'tools'))
from prepare import parts

def main(dataset,out):
    report=json.loads((dataset/'EXPERIENCE.json').read_text());assert report['status']=='OBSERVABILITY_PASS'
    raw=np.load(dataset/'experience.npz',allow_pickle=False);out.mkdir(parents=True,exist_ok=False)
    params=dict(num_weights=32,num_classes=3,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25)
    rows=[]
    for seed in [81701,81702,81703]:
        rng=np.random.default_rng(seed);weights=np.zeros((512,96),np.int8)
        for j in range(96):weights[rng.choice(512,32,replace=False),j]=1
        m=akida.Model();m.add(akida.InputData(input_shape=(1,1,512),input_bits=1));m.add(akida.FullyConnected(name='visual_consequence_association',units=96,weights_bits=1,activation=False))
        m.compile(optimizer=akida.AkidaUnsupervised(**params));layer=m.get_layer('visual_consequence_association')
        layer.set_variable('weights',weights.reshape(1,1,512,96));layer.set_variable('threshold_learning',np.array([0]*64+[32]*32,np.int32))
        assert np.array_equal(m.forward(raw['test_inputs'][:2].reshape(-1,1,1,512)).reshape(-1,96),raw['test_inputs'][:2].astype(np.int32)@weights.astype(np.int32))
        m.save(str(out/f'seed{seed}.fbz'));m.map(akida.AKD1500(),hw_only=True);assert len(m.sequences)==len(m.sequences[0].passes)==1
        program=bytes(m.sequences[0].program);ram,npid=parts(program)
        order=np.concatenate([np.random.default_rng(seed+e).permutation(12) for e in range(4)])
        inputs=np.concatenate([raw['test_inputs'],raw['train_inputs'][order],raw['test_inputs']])
        labels=np.concatenate([np.full(16,2,np.int32),raw['train_labels'][order],np.full(16,2,np.int32)])
        assert inputs.shape==(80,512) and labels.shape==(80,)
        for arm in ['full','frozen','action_only']:
            f=out/f's{seed}_{arm}';xx=inputs.copy();ll=labels.copy()
            if arm=='frozen':ll[:]=2
            if arm=='action_only':
                xx[:]=0
                for i,x in enumerate(inputs):a=0 if x[:256].any() else 1;xx[i,a*256:a*256+64]=1
            f.with_suffix('.bin').write_bytes(program);f.with_suffix('.ram').write_bytes(struct.pack('<'+'I'*len(ram),*ram))
            f.with_suffix('.meta').write_text('512 96 32 3\n');f.with_suffix('.inputs').write_bytes(xx.tobytes());f.with_suffix('.labels').write_bytes(ll.astype('<i4').tobytes())
            rows.append(dict(seed=seed,arm=arm,prefix=f.name,learning_np=npid,program_sha256=hashlib.sha256(program).hexdigest()))
    (out/'BUILD.json').write_text(json.dumps(dict(status='VIRTUAL_ONLY_PASS',rows=rows,parameters=params,dataset_sha256=hashlib.sha256((dataset/'experience.npz').read_bytes()).hexdigest(),physical_device_opened=False),indent=2)+'\n')
    print('VIRTUAL_ONLY_PASS',len(rows),'fixtures')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.dataset.resolve(),a.output.resolve())
