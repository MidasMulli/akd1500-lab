"""Compile class-window controls; never discover or open physical hardware."""
import json,struct
from pathlib import Path
import numpy as np
import akida
from tools.prepare import P,Table,PARAMS
from pong.relearn.model import Reference
S=P/'studies/20261024-fnp2-window'
N,U,TILE,STRIDE=512,11264,5632,20
DDR=0xC0000000;TILE_DATA=DDR+0x10000

def compile_model(units,seed):
 rng=np.random.default_rng(seed);w=np.zeros((N,units),np.int8)
 for j in range(units):w[rng.choice(N,32,False),j]=1
 m=akida.Model();m.add(akida.InputData(input_shape=(1,1,N),input_bits=1));m.add(akida.FullyConnected(name='window_learner',units=units,weights_bits=1,activation=False));m.compile(optimizer=akida.AkidaUnsupervised(**PARAMS));l=m.get_layer('window_learner');l.set_variable('weights',w.reshape(1,1,N,units));l.set_variable('threshold_learning',np.zeros(units,np.int32))
 m.map(akida.AKD1500(),hw_only=True,mode=akida.MapMode.Minimal);assert len(m.sequences)==len(m.sequences[0].passes)==1
 b=bytes(m.sequences[0].program);base=4+struct.unpack_from('<I',b)[0];tab=Table(b,4+struct.unpack_from('<I',b,4)[0]);track=tab.table(15).table(2);at=track.field(1);off,words=struct.unpack_from('<II',b,at);assert words==units*STRIDE
 a=np.frombuffer(b,'<u4',words,base+off).copy().reshape(units,STRIDE)
 packed=np.packbits(w.T,axis=1,bitorder='little').copy().view('<u4');assert np.array_equal(a[:,4:],packed)
 return b,base,at,off,a

def main(seed):
 out=S/'fixtures01'/f's{seed}';out.mkdir(parents=True,exist_ok=False)
 full,fb,fa,fo,a=compile_model(U,seed);tile,tb,ta,to,_=compile_model(TILE,seed)
 full_data=DDR+0x100000+((-fo)%64);record=full_data+fo
 assert record%64==0 and TILE_DATA+len(tile)-tb<full_data and full_data+len(full)-fb<DDR+0x400000
 assert np.all(a[:TILE,1]==0x100000) and np.all(a[TILE:,1]==0x200000),'native class tags'
 (out/'full.info').write_bytes(full[:fb]);(out/'full.data').write_bytes(full[fb:]);(out/'tile.data').write_bytes(tile[tb:])
 for label in [0,1]:
  info=bytearray(tile[:tb]);offset=record+label*TILE*STRIDE*4-TILE_DATA;assert offset>=0
  struct.pack_into('<I',info,ta,offset)
  check=Table(info,4+struct.unpack_from('<I',info,4)[0]).table(15).table(2);v,words=struct.unpack_from('<II',info,check.field(1));assert v==offset and words==TILE*STRIDE
  (out/f'tile{label}.info').write_bytes(info)
 a.tofile(out/'initial.raw');np.concatenate([np.zeros(2,'<u4'),a.reshape(-1)]).tofile(out/'cpu.raw')
 # The first target is in the previously failing class-1 region. All probes
 # are prospective and unambiguous; no native outputs select inputs or labels.
 targets=[8849,0,11263,5631,6600,256,10000,5000]
 ref=Reference([0,0]+a.reshape(-1).tolist(),n=N,u=U,nw=32,classes=2);xs=[];labels=[];rows=[]
 for t in targets:
  w=ref.weights[t];bits=np.array([(w>>i)&1 for i in range(N)],np.uint8);on=np.flatnonzero(bits);off=np.flatnonzero(1-bits);x=bits.copy();x[on[0]]=0;x[off[0]]=1;label=t//TILE
  scores=np.array([(v&sum(1<<int(k) for k in np.flatnonzero(x))).bit_count() for v in ref.weights]);assert scores[t]==31 and scores[label*TILE:(label+1)*TILE].max()==31 and np.count_nonzero(scores[label*TILE:(label+1)*TILE]==31)==1
  old=np.array(ref.ram[2:],'<u4');got=ref.forward(x,label);new=np.array(ref.ram[2:],'<u4');changed=np.flatnonzero(old!=new)
  w0=sum(int((old[t*STRIDE+4+k]^new[t*STRIDE+4+k])).bit_count() for k in range(N//32));assert w0==2 and np.unique(changed//STRIDE).tolist()==[t]
  xs.append(x);labels.append(label);rows.append(dict(target=t,label=label,removed_bit=int(on[0]),added_bit=int(off[0]),potential_before=31,potential_after=32,global_record_start_word=t*STRIDE,window_record_start_word=(t%TILE)*STRIDE))
 np.asarray(xs,np.uint8).tofile(out/'inputs.raw');np.asarray(labels,'<i4').tofile(out/'labels.raw')
 meta=dict(seed=seed,input_features=N,records=U,tile_records=TILE,record_words=STRIDE,record_address=record,full_data_address=full_data,tile_data_address=TILE_DATA,host_DDR_base=DDR,host_DDR_extent=0x400000,full_data_bytes=len(full)-fb,tile_data_bytes=len(tile)-tb,track_patch_byte_offset=ta,parameters=PARAMS,rows=rows,physical_device_opened=False,claim='Host chooses class view from teacher label; chip selects winner and changes connections; single-swap probes only')
 (out/'dims').write_text(f'{record} {full_data} {TILE_DATA}\n')
 (out/'meta.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
if __name__=='__main__':
 import sys
 main(int(sys.argv[1]))
