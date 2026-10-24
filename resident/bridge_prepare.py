"""Compiler-only fixed communication truth tables; no learning adapter."""
import json
import numpy as np
import akida
from resident.prepare import Select, communication
from tools.prepare_network import fc
from metal.prepare import mapping
from metal.ablation_prepare import memory_packets
from resident.run import P,save,digest

S=P/'studies/20261023-fixed-bridge'
KINDS=['all_fnp','direct_cnp','mixed']


def patterns():
    counts=np.random.default_rng(99901).integers(0,17,(128,4));counts[:4]=np.eye(4,dtype=int)*16
    counts[4:8]=np.asarray([[0,0,0,0],[16,16,16,16],[0,8,8,0],[4,0,4,0]])
    x=np.zeros((128,64),np.uint8)
    for i,cs in enumerate(counts):
        for j,c in enumerate(cs):x[i,j*16:j*16+c]=1
    y=np.zeros((128,4),np.int32);y[np.arange(128),counts.argmax(1)]=128
    return counts,x,y


def main():
    out=S/'fixtures01';out.mkdir(exist_ok=False);counts,features,y=patterns();results=[]
    for kind in KINDS:
        n=64 if kind=='direct_cnp' else 512;m=akida.Model();m.add(akida.InputData(input_shape=(1,1,n),input_bits=1))
        if kind!='direct_cnp':
            w=np.zeros((512,64),np.int8);w[:64]=np.eye(64,dtype=np.int8);fc(m,'known_source',w,np.zeros(64,np.int32),bits=1)
        if kind=='all_fnp':
            pairs=[(i,j) for i in range(4) for j in range(4) if i!=j];w=np.zeros((64,32),np.int8);t=np.zeros(32,np.int32)
            for q,(i,j) in enumerate(pairs):w[i*16:(i+1)*16,q]=1;w[j*16:(j+1)*16,q]=-1;t[q]=-1 if i<j else 0
            fc(m,'comparison',w,t);w=np.zeros((32,512),np.int8)
            for j in range(4):w[j*3:(j+1)*3,j*128:(j+1)*128]=1
            fc(m,'categorical',w,np.full(512,2,np.int32))
        else:communication(m,0)
        w=np.zeros((512,4),np.int8)
        for j in range(4):w[j*128:(j+1)*128,j]=1
        fc(m,'readout',w,None,activation=False,bits=1)
        x=np.zeros((128,n),np.uint8);x[:,:64]=features
        actual=m.forward(x.reshape(-1,1,1,n)).reshape(-1,4);assert np.array_equal(actual,y),'virtual communication truth table'
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d))
        assert len(m.sequences)==len(m.sequences[0].passes)==1
        mp=mapping(m);assert all(p['type']!='Type.FNP2' for l in mp for p in l['nps'])
        program=bytes(m.sequences[0].program);packets=memory_packets(program);f=out/kind
        f.with_suffix('.bin').write_bytes(program);f.with_suffix('.inputs').write_bytes(x.tobytes());f.with_suffix('.expected').write_bytes(y.astype('<i4').tobytes())
        layout=[f'{n} 4 128 {len(packets)}'];meta=[]
        for j,pkt in enumerate(packets):
            p=out/f'{kind}.packet{j}.ram';p.write_bytes(np.asarray(pkt['packet'],'<u4').tobytes());layout.append(' '.join(map(str,pkt['np']+[len(pkt['packet'])])))
            meta.append(dict(np=pkt['np'],words=len(pkt['packet']),sha256=digest(p)))
        f.with_suffix('.layout').write_text('\n'.join(layout)+'\n');r=dict(kind=kind,status='COMPILER_VIRTUAL_PASS',input_features=n,mapping=mp,packets=meta,program_sha256=digest(f.with_suffix('.bin')))
        save(f.with_suffix('.json'),r);results.append(r);print(json.dumps(dict(kind=kind,status=r['status'],nps=len({tuple(p['np']) for l in mp for p in l['nps']}))),flush=True)
        save(out/'BUILD.json',dict(status='COMPILER_ONLY',physical_device_opened=False,results=results))


if __name__=='__main__':main()
