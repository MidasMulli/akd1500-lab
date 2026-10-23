"""Measured RAM differences and scores for the 32-NPU evidence run."""
from pathlib import Path
import argparse,hashlib,json,struct
import numpy as np
def words(p):
    b=p.read_bytes();return list(struct.unpack('<'+'I'*(len(b)//4),b))
def main():
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=a.study/'raw/mesh32_evidence';f=a.study/'mesh_v6/mesh_32';initial=words(f.with_suffix('.ram'));scores=words(d/'native.scores');inputs=f.with_suffix('.inputs').read_bytes();before=initial;samples=[];n=128;u=64;stride=8
    for i in range(64):
        file=d/f'native.snapshot_{i}.ram';after=words(file);edits=[]
        for j in range(u):
            old=[b for b in range(n) if before[2+j*stride+4+b//32]>>(b%32)&1];new=[b for b in range(n) if after[2+j*stride+4+b//32]>>(b%32)&1]
            if old!=new:edits.append(dict(neuron=j,removed=sorted(set(old)-set(new)),added=sorted(set(new)-set(old))))
        samples.append(dict(id=i,input=list(inputs[i*n:(i+1)*n]),label=i%2,scores=scores[i*u:(i+1)*u],connection_edits=edits,ram_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        before=after
    m=json.loads((a.study/'mesh_v6/MESH.json').read_text());cell=next(c for c in m['cells'] if c['name']=='mesh_32')
    z=dict(schema=1,status='MEASURED_NATIVE_REPLAY',samples=samples,mapping=cell,source_hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in [d/'native.scores',d/'native.ram',d/'RECEIPT.json']},
           visual_boundary='Logical layer order and schematic layout; motion is illustrative replay. Match scores and connection differences come from actual output and per-step RAM readback. No electrical spike/current or physical neuron-position trace.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(z,indent=2)+'\n');print('PASS: 64 measured samples, 32 logical placements')
    np.savez_compressed(a.output.parent.parent/'results/mesh32-evidence.npz',
        initial_ram=np.array(initial,dtype=np.uint32),
        measured_ram_after=np.array([words(d/f'native.snapshot_{i}.ram') for i in range(64)],dtype=np.uint32),
        measured_scores=np.array(scores,dtype=np.int32).reshape(64,64),
        inputs=np.frombuffer(inputs[:64*128],np.uint8).reshape(64,128),
        labels=np.arange(64,dtype=np.int32)%2)
if __name__=='__main__':main()
