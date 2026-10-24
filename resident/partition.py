"""One compiler-only attempt: two1600-filter banks plus explicit Concatenate."""
from pathlib import Path
import hashlib
import json
import traceback
import numpy as np
import akida
from recognition.data import P,compact,overlap,save
from recognition.prepare import Select
from metal.prepare import add_fc,mapping
from metal.ablation_prepare import memory_packets

S=P/'studies/20261023-resident-push'


def main():
    out=S/'fixtures-partition01';out.mkdir(exist_ok=False)
    with np.load(P/'results/2026-10-23-image-capacity.npz',allow_pickle=False) as z:
        templates=z['capacity__templates'];x=np.vstack([z['capacity__development_inputs'][:8],z['capacity__test_inputs'][:8]])
    banks=[np.concatenate([np.arange(c*320+b*160,c*320+(b+1)*160) for c in range(10)]) for b in range(2)]
    order=np.concatenate(banks);assert set(order)==set(range(3200));result=[]
    for kind in ['raw','compacted']:
        stage='build';virtual=False
        try:
            m=akida.Model();root=akida.InputData(input_shape=(1,1,1024),input_bits=1);m.add(root);features=x;t=templates
            if kind=='compacted':
                w=np.zeros((1024,256),np.int8)
                for r in range(32):
                    for c in range(32):w[r*32+c,(r//2)*16+c//2]=1
                root=add_fc(m,'native_spatial_OR',w,True);root.set_variable('threshold',np.zeros(256,np.int32))
                features=compact(x);t=compact(templates)
            branches=[]
            for b,ids in enumerate(banks):
                layer=akida.Convolutional(name=f'exemplar_bank{b}',filters=1600,kernel_size=(1,1),weights_bits=1,activation=False)
                m.add(layer,[root]);layer.set_variable('weights',t[ids].T.reshape(1,1,t.shape[1],1600).astype(np.int8));branches.append(layer)
            m.add(akida.Concatenate(name='explicit_bank_collection'),branches)
            stage='virtual';actual=m.forward(x.reshape(-1,1,1,1024)).reshape(-1,3200);want=overlap(features,t)[:,order]
            assert np.array_equal(actual,want),'partitioned virtual output and bank order';virtual=True;stage='mapping'
            d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d))
            assert len(m.sequences)==len(m.sequences[0].passes)==1,'single native pass'
            image=bytes(m.sequences[0].program);packets=memory_packets(image)
            f=out/kind;f.with_suffix('.bin').write_bytes(image);f.with_suffix('.order').write_bytes(order.astype('<i4').tobytes())
            f.with_suffix('.meta').write_text('1024 3200 320\n');f.with_suffix('.bias').write_bytes(t[order].sum(axis=1,dtype=np.int32).astype('<i4').tobytes())
            f.with_suffix('.regions').write_text(str(len(packets))+'\n'+'\n'.join(str(len(p['packet'])) for p in packets)+'\n')
            for j,p in enumerate(packets):(out/f'{kind}.packet{j}.ram').write_bytes(np.asarray(p['packet'],'<u4').tobytes())
            meta=dict(kind=kind,status='COMPILER_VIRTUAL_SINGLE_PASS',virtual_checked_images=len(x),mapping=mapping(m),
                program_sha256=hashlib.sha256(image).hexdigest(),output_order=order.tolist(),packets=[dict(np=p['np'],words=len(p['packet']),sha256=hashlib.sha256(np.asarray(p['packet'],'<u4').tobytes()).hexdigest()) for p in packets])
            save(f.with_suffix('.json'),meta);result.append(meta)
            print(json.dumps({k:v for k,v in meta.items() if k not in ['packets','output_order']}),flush=True)
        except Exception as e:
            row=dict(kind=kind,status='FAILED_PRESERVED',stage=stage,virtual_passed=virtual,error=str(e),traceback=traceback.format_exc());result.append(row);print(json.dumps(row),flush=True)
        save(out/'BUILD.json',dict(status='BOUNDED_COMPILER_PARTITION_ATTEMPT',physical_device_opened=False,cells=result))


if __name__=='__main__':main()
