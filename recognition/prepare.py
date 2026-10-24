"""Compiler/virtual gate for the frozen3200-exemplar CNP comparison."""
from pathlib import Path
import hashlib
import json
import traceback
import numpy as np
import akida
from recognition.data import compact,overlap,save
from recognition.capacity import S,decode
from metal.prepare import add_fc,mapping
from metal.ablation_prepare import memory_packets


class Select(akida.MapConstraints):
    def select_nps(self,source_nps,num_nps,type):
        encoder=[1,3,0]
        if type==akida.NP.Type.FNP3:
            found=[p for p in source_nps if [p.col,p.row,p.id]==encoder]
        else:
            assert type in [akida.NP.Type.CNP1,akida.NP.Type.CNP2],('unexpected kernel type',str(type))
            found=[p for p in source_nps if [p.col,p.row,p.id]!=encoder]
        assert len(found)>=num_nps,('insufficient native resources',str(type),num_nps,len(found))
        return found[:num_nps]


def main():
    source=S/'fixtures-data01';quality=json.loads((source/'DATA.json').read_text());assert quality['status']=='SEALED_QUALITY_PASS'
    out=S/'fixtures-native01';out.mkdir(exist_ok=False);results=[]
    with np.load(source/'data.npz',allow_pickle=False) as z:
        t=z['templates'];x=np.vstack([z['development_inputs'][:8],z['test_inputs'][:8]])
    for kind in ['raw','compacted']:
        try:
            m=akida.Model();m.add(akida.InputData(input_shape=(1,1,1024),input_bits=1))
            features=x;templates=t
            if kind=='compacted':
                cw=np.zeros((1024,256),np.int8)
                for r in range(32):
                    for c in range(32):cw[r*32+c,(r//2)*16+c//2]=1
                l=add_fc(m,'native_spatial_OR',cw,True);l.set_variable('threshold',np.zeros(256,np.int32))
                features=compact(x);templates=compact(t)
            m.add(akida.Convolutional(name='resident_exemplar_overlap',filters=3200,kernel_size=(1,1),weights_bits=1,activation=False))
            m.get_layer('resident_exemplar_overlap').set_variable('weights',templates.T.reshape(1,1,templates.shape[1],3200).astype(np.int8))
            want=overlap(features,templates)
            actual=m.forward(x.reshape(-1,1,1,1024)).reshape(-1,3200)
            assert np.array_equal(actual,want),'compiler virtual exact overlaps'
            assert np.array_equal(decode(actual,templates),decode(want,templates))
            device=akida.AKD1500();m.map(device,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(device))
            assert len(m.sequences)==len(m.sequences[0].passes)==1,'single native pass'
            mapping_info=mapping(m)
            cnn=next(r for r in mapping_info if r['layer']=='resident_exemplar_overlap')
            assert all(p['type'] in ['Type.CNP1','Type.CNP2'] for p in cnn['nps'])
            assert all(p['np']!=[1,3,0] for p in cnn['nps'])
            image=bytes(m.sequences[0].program);packets=memory_packets(image);assert 1<=len(packets)<=128
            f=out/kind;f.with_suffix('.bin').write_bytes(image);f.with_suffix('.meta').write_text('1024 3200 320\n')
            f.with_suffix('.bias').write_bytes(templates.sum(axis=1,dtype=np.int32).astype('<i4').tobytes())
            f.with_suffix('.regions').write_text(str(len(packets))+'\n'+'\n'.join(str(len(p['packet'])) for p in packets)+'\n')
            for j,p in enumerate(packets):(out/f'{kind}.packet{j}.ram').write_bytes(np.asarray(p['packet'],'<u4').tobytes())
            meta=dict(kind=kind,inputs=1024,outputs=3200,per_class=320,mapping=mapping_info,program_sha256=hashlib.sha256(image).hexdigest(),
                packets=[dict(np=p['np'],words=len(p['packet']),sha256=hashlib.sha256(np.asarray(p['packet'],'<u4').tobytes()).hexdigest()) for p in packets],
                virtual_checked_images=len(x),parameter_payload_bytes=sum(len(p['packet'])-2 for p in packets)*4)
            save(f.with_suffix('.json'),meta);results.append(dict(status='COMPILER_VIRTUAL_PASS',**meta))
            print(json.dumps(dict(kind=kind,status=results[-1]['status'],mapping=mapping_info,parameter_bytes=meta['parameter_payload_bytes'])),flush=True)
        except Exception as e:
            results.append(dict(kind=kind,status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc()))
            print(json.dumps(results[-1]),flush=True)
        save(out/'BUILD.json',dict(status='COMPILER_ONLY',physical_device_opened=False,cells=results))
    assert all(r['status']=='COMPILER_VIRTUAL_PASS' for r in results),'compiler gate failed; no physical admission'


if __name__=='__main__':main()
