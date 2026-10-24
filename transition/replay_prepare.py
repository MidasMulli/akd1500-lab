"""Compiler-only finite multi-pass replay of captured learned weights; no device."""
from pathlib import Path
import argparse, json, traceback
import akida,numpy as np
from tools.program_codec import read
from metal.prepare import mapping
from tools.prepare_network import Select
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-learned-transition'
class Two(akida.MapConstraints):
    def select_nps(self,source_nps,num_nps,type):
        chosen=[p for p in source_nps if [p.col,p.row,p.id] in [[1,3,0],[2,1,0]]]
        return chosen[:num_nps] if len(chosen)>=num_nps else []
def main(out):
    out.mkdir(parents=True,exist_ok=False);f=S/'fixtures01/s110101';d=S/'raw/native_s110101_native_h32'
    ram=np.fromfile(d/'result.ram','<u4');assert len(ram)==98
    w=np.unpackbits(ram[2:].reshape(8,12)[:,4:].copy().view('u1'),axis=-1,bitorder='little').T.astype(np.int8)
    codes=np.fromfile(f/'codes','u1').reshape(8,256);partial=np.fromfile(f/'partial','u1').reshape(8,256);new=np.fromfile(f/'new','<i4');rows=[]
    for restricted in [True,False]:
      for loop in [False,True]:
       for h in [1,2,4,8]:
        name=f'{"two" if restricted else "seven"}_loop{int(loop)}_h{h}';row=dict(name=name,horizon=h,loopitself=loop,restricted_two_nps=restricted,physical_device_opened=False)
        try:
            m=akida.Model();m.add(akida.InputData(input_shape=(1,1,256),input_bits=1))
            for i in range(h):
                for tag,a,threshold in [('transition',w,24),('decode',codes.astype(np.int8),0)]:
                    name2=f'{tag}_{i}';m.add(akida.FullyConnected(name=name2,units=a.shape[1],weights_bits=1,activation=True,act_bits=1));l=m.get_layer(name2);l.set_variable('weights',a.reshape(1,1,*a.shape));l.set_variable('threshold',np.full(a.shape[1],threshold,np.int32));l.set_variable('act_step',np.ones(a.shape[1],np.float32))
            want=np.arange(8)
            for _ in range(h):want=new[want]
            assert np.array_equal(m.forward(partial.reshape(8,1,1,256)).reshape(8,256),codes[want]),'virtual captured-weight composition'
            dev=akida.AKD1500();c=(Two if restricted else Select)(dev);c.hwpr_loopitself=loop;m.map(dev,hw_only=True,mode=akida.MapMode.Minimal,constraints=c)
            assert len(m.sequences)==1;seq=m.sequences[0];program=bytes(seq.program);hdr,_=read(program);(out/f'{name}.bin').write_bytes(program);codes[want].tofile(out/f'{name}.expected');partial.tofile(out/f'{name}.inputs')
            row.update(status='COMPILER_PASS',passes=len(seq.passes),program_passes=len(hdr['passes']),input_dense=hdr['s4']==0,output_dense=hdr['s5']==0,learning=bool(hdr.get('learning')),mapping=mapping(m))
        except Exception as e:row.update(status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
        rows.append(row);(out/'BUILD.json').write_text(json.dumps(dict(status='COMPILER_ONLY',captured_weights_source='native_s110101_native_h32/result.ram',cells=rows),indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k not in ['traceback','mapping']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();main(a.out)
