import copy,struct
import akida,numpy as np
from program_codec import read,build,words
def fc(m,name,w,th,bits=4):
 m.add(akida.FullyConnected(name=name,units=w.shape[1],weights_bits=bits,activation=True,act_bits=1));l=m.get_layer(name);l.set_variable('weights',w.reshape(1,1,*w.shape).astype(np.int8));l.set_variable('threshold',np.asarray(th,np.int32));l.set_variable('act_step',np.ones(w.shape[1],np.float32))
def assemble(single,app,expected_fixed=4):
 si,sd=read(single);h,data=read(app);lr=si['learning'];assert lr and h['learning'] is None and len(h['passes'])==1;inf=words(sd,lr['inference']);lea=words(sd,lr['learning']);ram=words(sd,lr['ram'][0]);assert len(inf)==len(lea)==52 and len(ram)==1282
 mode=[i for i,(a,b) in enumerate(zip(inf,lea)) if a!=b];assert mode==[38,39,40],mode
 candidates=[]
 for ri,rec in enumerate(h['passes'][0]):
  for sp in rec:
   t=words(data,sp)
   if len(t)==1334 and t[0]==inf[0] and t[52:54]==ram[:2]:candidates.append((ri,t))
 assert len(candidates)==1,candidates;ri,t=candidates[0];routes=[i for i,(a,b) in enumerate(zip(inf,t[:52])) if a!=b];assert routes==[46,47,49],routes
 diffs=[i for i,(a,b) in enumerate(zip(ram,t[52:])) if a!=b];assert diffs==[2+20*j+k for j in range(64) for k in [1,2,3]],diffs
 assert all(ram[2+20*j+1]==16+((j//32+1)<<20) and t[54+20*j+1]==16 and t[54+20*j+2]==t[54+20*j+3]==0 for j in range(64))
 ni=t[:52];nl=ni[:]
 for k in mode:assert ni[k]==inf[k];nl[k]=lea[k]
 d=bytearray(data)
 def add(v):sp=[len(d),len(v)];d.extend(struct.pack('<'+'I'*len(v),*v));return sp
 ls=add(nl);is_=add(ni);rs=add(ram);struct.pack_into('<I',d,0,len(d)-4);a=copy.deepcopy(h);a['learning']=dict(learning=ls,inference=is_,ram=[rs],np=lr['np']);out=build(a,bytes(d));aa,dd=read(out);assert aa==a and aa['passes']==h['passes'] and all(words(dd,s)==words(data,s) for ps in h['passes'] for r in ps for s in r)
 fixed=[]
 for j,rec in enumerate(h['passes'][0]):
  if j==ri:continue
  for sp in rec:
   tr=words(data,sp)
   if len(tr)>54 and tr[52]&255==6:
    hd=tr[52:54];mask=(hd[0]>>8)&15;assert mask and not(mask&(mask-1));assert len(tr)==54+(hd[1]>>16);fixed.append(dict(np=[hd[0]>>24,(hd[0]>>16)&255,mask.bit_length()-1],words=tr[52:],record=j))
 assert len(fixed)==expected_fixed,len(fixed)
 return out,dict(np=lr['np'],initial_words=ram,fixed=fixed,mode_word_indices=mode,route_word_indices=routes,restored_auxiliary_indices=diffs,inference_tracks_unchanged=True)
