"""Ordinary compiler programs for identical FNP/CNP mathematical primitives."""
import json
import traceback
import numpy as np
import akida
from capability.design import P,S,CELLS,weights,inputs,expected
from metal.prepare import mapping,Select as FNPSelect
from metal.prepare_cnp import Select as CNPSelect
from metal.ablation_prepare import memory_packets
from tools.prepare_network import fc
from metal.run import digest


def main():
    out=S/'fixtures01';out.mkdir(exist_ok=False);w=weights();x,rows=inputs();results=[]
    (out/'STREAM.json').write_text(json.dumps(rows,indent=2)+'\n')
    for kind,u in CELLS:
        name=f'{kind}{u}';f=out/name
        try:
            ww=w[:,:u].copy();m=akida.Model();m.add(akida.InputData(input_shape=(1,1,512),input_bits=1))
            if kind=='fnp':fc(m,'binary_matches',ww,np.full(u,15,np.int32),bits=1)
            else:
                m.add(akida.Convolutional(name='binary_matches',filters=u,kernel_size=(1,1),weights_bits=1,activation=True,act_bits=1))
                l=m.get_layer('binary_matches');l.set_variable('weights',ww.reshape(1,1,512,u).astype(np.int8));l.set_variable('threshold',np.full(u,15,np.int32));l.set_variable('act_step',np.ones(u,np.float32))
            rw=np.zeros((u,2),np.int8);rw[:u//2,0]=1;rw[u//2:,1]=1;fc(m,'two_counts',rw,None,activation=False,bits=1)
            y=expected(x,ww);actual=m.forward(x.reshape(-1,1,1,512)).reshape(-1,2);assert np.array_equal(actual,y),'virtual exact primitive'
            d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=(FNPSelect if kind=='fnp' else CNPSelect)(d))
            assert len(m.sequences)==len(m.sequences[0].passes)==1,'single resident pass'
            mp=mapping(m);assert all(p['type']!='Type.FNP2' for l in mp for p in l['nps']),'no external FNP2'
            program=bytes(m.sequences[0].program);packets=memory_packets(program);assert 1<=len(packets)<=32
            regions=[str(len(packets))]
            for j,pkt in enumerate(packets):
                a=np.asarray(pkt['packet'],'<u4');assert 2<len(a)<=20000
                (out/f'{name}.packet{j}.ram').write_bytes(a.tobytes());regions.append(str(len(a)))
            f.with_suffix('.regions').write_text('\n'.join(regions)+'\n')
            f.with_suffix('.bin').write_bytes(program);f.with_suffix('.inputs').write_bytes(x.tobytes());f.with_suffix('.expected').write_bytes(y.tobytes());f.with_suffix('.weights').write_bytes(ww.T.tobytes());f.with_suffix('.meta').write_text('512 0 0 2\n')
            r=dict(name=name,kind=kind,units=u,inputs=512,outputs=2,status='COMPILER_VIRTUAL_PASS',mapping=mp,weight_sha256=digest(f.with_suffix('.weights')),program_sha256=digest(f.with_suffix('.bin')),
                packets=[dict(np=p['np'],words=len(p['packet']),sha256=digest(out/f'{name}.packet{j}.ram')) for j,p in enumerate(packets)])
            f.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n')
        except Exception as e:r=dict(name=name,kind=kind,units=u,status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
        results.append(r);(out/'BUILD.json').write_text(json.dumps(dict(status='COMPILER_ONLY',physical_device_opened=False,sdk=akida.__version__,cells=results),indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='traceback'}),flush=True)


if __name__=='__main__':main()
