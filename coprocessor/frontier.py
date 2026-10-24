"""Bounded compiler/resource enumeration. No physical discovery or execution."""
import json
import traceback
import numpy as np
import akida
from coprocessor.prepare import P,S,Select,SEED
from tools.prepare import PARAMS,parts
from metal.prepare import mapping

def main():
    rows=[]
    # Deliberately straddle the descriptor's weight-storage budget at n=512.
    for n,u in [(128,512),(256,512),(512,512),(512,544),(512,560),(512,576),(512,1024),(1024,256),(1024,288),(1024,320),(2048,128)]:
        try:
            m=akida.Model();m.add(akida.InputData(input_shape=(1,1,n),input_bits=1));m.add(akida.FullyConnected(name='capacity_bank',units=u,weights_bits=1,activation=False));m.compile(optimizer=akida.AkidaUnsupervised(**PARAMS))
            rng=np.random.default_rng(SEED);w=np.zeros((n,u),np.int8)
            for j in range(u):w[rng.choice(n,32,False),j]=1
            l=m.get_layer('capacity_bank');l.set_variable('weights',w.reshape(1,1,n,u));l.set_variable('threshold_learning',np.zeros(u,np.int32))
            d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d))
            assert len(m.sequences)==len(m.sequences[0].passes)==1
            b=bytes(m.sequences[0].program)
            ram,ident=parts(b)
            row=dict(inputs=n,records=u,status='COMPILER_SINGLE_FNP3',mapping=mapping(m),learning_np=ident,packet_bytes=len(ram)*4,physical_execution=False)
        except Exception as e:row=dict(inputs=n,records=u,status='COMPILER_OR_DECODER_REJECTED',error=str(e),traceback=traceback.format_exc(),physical_execution=False)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='traceback'}),flush=True)
    (S/'evidence/COMPILER_FRONTIER.json').write_text(json.dumps(dict(status='COMPILER_ONLY_NOT_SILICON_CAPACITY',cells=rows),indent=2)+'\n')
if __name__=='__main__':main()
