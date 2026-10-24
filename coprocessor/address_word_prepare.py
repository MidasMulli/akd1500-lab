"""Seal two single-word edits for the same native winner; compiler image unchanged."""
from pathlib import Path
import json,shutil
import numpy as np
from tools.prepare import P
from pong.relearn.model import Reference
S=P/'studies/20261024-fnp2-word'
def main():
 old=P/'studies/20261024-fnp2-width/fixtures01/below/learner'
 n,u,t,stride=1024,4096,3640,36
 a=np.fromfile(old.with_suffix('.raw'),'<u4').reshape(u,stride)
 w=np.unpackbits(a[:,4:].copy().view('u1'),axis=1,bitorder='little')
 active=np.flatnonzero(w[t]);lastword=int(active[-1]//32);assert lastword==28
 root=S/'fixtures01';root.mkdir(exist_ok=False)
 for case,word in [('below',0),('above',lastword)]:
  bits=w[t].copy();lo,hi=word*32,(word+1)*32
  remove=int(np.flatnonzero(bits[lo:hi])[0])+lo;add=int(np.flatnonzero(1-bits[lo:hi])[-1])+lo
  x=np.zeros((2,n),np.uint8);x[1]=bits;x[1,remove]=0;x[1,add]=1
  y=x.astype(np.int32)@w.T.astype(np.int32);assert y[1,t]==31 and y[1,u//2:].max()==31 and np.count_nonzero(y[1,u//2:]==31)==1
  ref=Reference([0,0]+a.reshape(-1).tolist(),n=n,u=u,nw=32,classes=2)
  for i,v in enumerate(x):assert np.array_equal(ref.forward(v,i%2),y[i])
  changed=np.flatnonzero(np.asarray(ref.ram[2:])!=a.reshape(-1));assert changed.tolist()==[t*stride+3,t*stride+4+word],changed
  out=root/case;out.mkdir();f=out/'learner'
  for ext in ['bin','raw']:shutil.copyfile(old.with_suffix('.'+ext),f.with_suffix('.'+ext))
  x.tofile(f.with_suffix('.inputs'));y.astype('<i4').tofile(f.with_suffix('.expected'))
  meta=dict(input_features=n,records=u,record_words=stride,target=t,condition=case,removed_bit=remove,added_bit=add,connection_word=word,expected_changed_word_indices=changed.tolist(),weight_seed=107201,physical_device_opened=False)
  f.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
if __name__=='__main__':main()
