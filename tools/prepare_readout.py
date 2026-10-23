"""Compiler-only matched potential/activated fixtures; never opens hardware."""
from pathlib import Path
import argparse, hashlib, json, struct, traceback
import akida, numpy as np
from prepare import PARAMS, parts

def generate(out):
    out.mkdir(parents=True, exist_ok=False)
    rows=[]
    for n,u in [(128,8),(512,64),(512,512)]:
        rng=np.random.default_rng(72301+n+u)
        w=np.zeros((n,u),np.int8)
        for j in range(u):w[rng.choice(n,32,replace=False),j]=1
        x=np.zeros((1024,n),np.uint8)
        for i in range(1024):x[i,rng.choice(n,max(32,n//4),replace=False)]=1
        expected=x[:16].astype(np.int32)@w.astype(np.int32)
        pair=[]
        for activated in [False,True]:
            name=f'n{n}_u{u}_'+('mask' if activated else 'potentials')
            try:
                m=akida.Model();m.add(akida.InputData(input_shape=(1,1,n),input_bits=1))
                m.add(akida.FullyConnected(name='association',units=u,weights_bits=1,activation=activated,act_bits=1))
                m.compile(optimizer=akida.AkidaUnsupervised(**PARAMS));l=m.get_layer('association')
                l.set_variable('weights',w.reshape(1,1,n,u))
                l.set_variable('threshold_learning',np.zeros(u,np.int32))
                if activated:
                    l.set_variable('threshold',np.full(u,7,np.int32))
                    l.set_variable('act_step',np.ones(u,np.float32))
                y=m.forward(x[:16].reshape(16,1,1,n)).reshape(16,u)
                assert np.array_equal(y,(expected>7).astype(np.uint8) if activated else expected),'virtual output contract'
                m.save(str(out/(name+'.fbz')));m.map(akida.AKD1500(),hw_only=True)
                assert len(m.sequences)==1 and len(m.sequences[0].passes)==1
                mapping=[]
                for layer in m.sequences[0].passes[0].layers:
                    for p in layer.mapping.nps:
                        q=p.ident;mapping.append(dict(type=str(p.type),np=[q.col,q.row,q.id]))
                image=bytes(m.sequences[0].program);ram,npid=parts(image)
                assert len(ram)==2+u*(4+n//32)
                for ext,b in [('bin',image),('ram',struct.pack('<'+'I'*len(ram),*ram)),('inputs',x.tobytes()),('labels',np.array([i%2 for i in range(1024)],dtype='<i4').tobytes())]:(out/(name+'.'+ext)).write_bytes(b)
                (out/(name+'.meta')).write_text(f'{n} {u} 32 2\n')
                (out/(name+'.output')).write_text(f'{int(activated)} 7\n')
                row=dict(name=name,status='MAPPED',inputs=n,neurons=u,activated=activated,threshold=7,learning_np=npid,mapping=mapping,program_sha256=hashlib.sha256(image).hexdigest())
                pair.append(ram)
            except Exception as e:
                row=dict(name=name,status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
            rows.append(row)
            (out/'READOUT.json').write_text(json.dumps(dict(status='COMPILER_ONLY',physical_device_opened=False,cells=rows),indent=2)+'\n')
            print(json.dumps(row),flush=True)
        if len(pair)==2:
            stride=4+n//32
            differences=[i for i,(a,b) in enumerate(zip(*pair)) if a!=b]
            assert all(i>=2 and (i-2)%stride in [0,1,2] for i in differences),('learning fields differ',differences)
            (out/f'n{n}_u{u}_PAIR.json').write_text(json.dumps(dict(status='PASS',fixed_RAM_differences=differences,learned_fields_identical=True),indent=2)+'\n')
    assert all(r['status']=='MAPPED' for r in rows),'compiler failure preserved; no admission'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);generate(p.parse_args().output)
