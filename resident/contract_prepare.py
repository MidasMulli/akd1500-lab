"""Compiler-only endpoint probes, with ordinary compiler programs."""
import json
import numpy as np
import akida
from resident.contract_design import P,S,KINDS,data,comparison,expected
from resident.prepare import Select,conv
from tools.prepare_network import fc
from metal.prepare import mapping
from metal.ablation_prepare import memory_packets
from resident.run import save,digest


def main():
    out=S/'fixtures01';out.mkdir(exist_ok=False);results=[]
    for kind in KINDS:
        n=64 if kind.startswith('direct') else 512;m=akida.Model();m.add(akida.InputData(input_shape=(1,1,n),input_bits=1))
        if not kind.startswith('direct'):
            w=np.zeros((512,64),np.int8);w[:64]=np.eye(64,dtype=np.int8);fc(m,'source',w,np.zeros(64,np.int32),bits=1)
        width=64
        if kind not in ['source','all_fnp']:
            w,t=comparison() if 'compare' in kind else (np.eye(64,dtype=np.int8),np.zeros(64,np.int32));width=w.shape[1]
            conv(m,'bridge',w,t)
        if kind!='source':fc(m,'readout',np.eye(width,dtype=np.int8),None,activation=False,bits=1)
        for split in ['development','sealed']:
            features=data(split);x=np.zeros((128,n),np.uint8);x[:,:64]=features;y=expected(kind,features)
            if m.device is not None:m.map(None)
            got=m.forward(x.reshape(-1,1,1,n)).reshape(-1,width)
            assert np.array_equal(got,y),('virtual vector contract',kind,split)
            (out/f'{kind}.{split}.inputs').write_bytes(x.tobytes());(out/f'{kind}.{split}.expected').write_bytes(y.astype('<i4').tobytes())
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d))
        assert len(m.sequences)==len(m.sequences[0].passes)==1
        mp=mapping(m);assert all(p['type']!='Type.FNP2' for l in mp for p in l['nps'])
        program=bytes(m.sequences[0].program);packets=memory_packets(program);f=out/kind
        f.with_suffix('.bin').write_bytes(program);layout=[f'{n} {width} 128 {len(packets)}'];meta=[]
        # The benchmark selects a complete frozen split by giving it a copied fixture prefix.
        for j,pkt in enumerate(packets):
            p=out/f'{kind}.packet{j}.ram';p.write_bytes(np.asarray(pkt['packet'],'<u4').tobytes());layout.append(' '.join(map(str,pkt['np']+[len(pkt['packet'])])))
            meta.append(dict(np=pkt['np'],words=len(pkt['packet']),sha256=digest(p)))
        f.with_suffix('.layout').write_text('\n'.join(layout)+'\n')
        r=dict(kind=kind,status='COMPILER_VIRTUAL_PASS',input_features=n,outputs=width,mapping=mp,packets=meta,program_sha256=digest(f.with_suffix('.bin')),
            data_sha256={split:{ext:digest(out/f'{kind}.{split}.{ext}') for ext in ['inputs','expected']} for split in ['development','sealed']})
        save(f.with_suffix('.json'),r);results.append(r);save(out/'BUILD.json',dict(status='COMPILER_ONLY',physical_device_opened=False,results=results))
        print(json.dumps({k:r[k] for k in ['kind','status','outputs']}),flush=True)


if __name__=='__main__':main()
