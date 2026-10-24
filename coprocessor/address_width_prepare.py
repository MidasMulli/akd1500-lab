"""Offline compiler, virtual CPU and record-layout check for one frozen stride."""
from pathlib import Path
import struct,json
import numpy as np
import akida
from tools.prepare import P,Table,PARAMS
from pong.relearn.model import Reference
S=P/'studies/20261024-fnp2-width'
def main():
 n,u,nw,seed=1024,4096,32,107201;stride=4+n//32
 root=S/'fixtures01';root.mkdir(exist_ok=False)
 rng=np.random.default_rng(seed);w=np.zeros((n,u),np.int8)
 for j in range(u):w[rng.choice(n,nw,False),j]=1
 m=akida.Model();m.add(akida.InputData(input_shape=(1,1,n),input_bits=1));m.add(akida.FullyConnected(name='stride_learner',units=u,weights_bits=1,activation=False));m.compile(optimizer=akida.AkidaUnsupervised(**PARAMS));l=m.get_layer('stride_learner');l.set_variable('weights',w.reshape(1,1,n,u));l.set_variable('threshold_learning',np.zeros(u,np.int32))
 xs={}
 for case,t in [('below',3600),('above',3700)]:
  x=np.zeros((2,n),np.uint8);x[1]=w[:,t];active=np.flatnonzero(x[1]);inactive=np.flatnonzero(1-x[1]);x[1,active[0]]=0;x[1,inactive[0]]=1
  y=x.astype(np.int32)@w.astype(np.int32);assert y[1,t]==31 and y[1,u//2:].max()==31 and np.count_nonzero(y[1,u//2:]==31)==1
  assert np.array_equal(m.forward(x.reshape(2,1,1,n)).reshape(2,u),y)
  xs[case]=(t,x,y)
 m.map(akida.AKD1500(),hw_only=True,mode=akida.MapMode.Minimal);assert len(m.sequences)==len(m.sequences[0].passes)==1
 b=bytes(m.sequences[0].program);table=Table(b,4+struct.unpack_from('<I',b,4)[0]);track=table.table(15).table(2);off,words=struct.unpack_from('<II',b,track.field(1));assert words==u*stride
 base=4+struct.unpack_from('<I',b)[0];a=np.frombuffer(b,'<u4',words,base+off).copy().reshape(u,stride)
 packed=np.packbits(w.T,axis=1,bitorder='little').copy().view('<u4').reshape(u,n//32);assert np.array_equal(a[:,4:],packed)
 for case,(t,x,y) in xs.items():
  out=root/case;out.mkdir();f=out/'learner'
  ref=Reference([0,0]+a.reshape(-1).tolist(),n=n,u=u,nw=nw,classes=2)
  states=[]
  for i,z in enumerate(x):assert np.array_equal(ref.forward(z,i%2),y[i]);states.append(ref.ram[2:].copy())
  changed=np.flatnonzero(np.any(np.asarray(states[-1]).reshape(u,stride)!=a,axis=1));assert changed.tolist()==[t]
  f.with_suffix('.bin').write_bytes(b);a.tofile(f.with_suffix('.raw'));x.tofile(f.with_suffix('.inputs'));y.astype('<i4').tofile(f.with_suffix('.expected'))
  meta=dict(input_features=n,records=u,record_words=stride,record_bytes=stride*4,target=t,target_byte_offset=t*stride*4,weight_seed=seed,parameters=PARAMS,virtual_potentials_exact=True,record_layout_exact=True,physical_device_opened=False)
  f.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
  print(case,json.dumps(meta))
if __name__=='__main__':main()
