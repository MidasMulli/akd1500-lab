"""Finite replay adapter: repeat one complete, same-shape compiler pass."""
from pathlib import Path
import copy,hashlib,json,struct
import numpy as np
from tools.program_codec import read,build
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-learned-transition'
def main():
    out=S/'replay_fixtures03';out.mkdir(exist_ok=False)
    src=S/'replay_fixtures01/two_loop0_h1.bin';image=src.read_bytes();header,data=read(image)
    assert len(header['passes'])==1 and header['learning'] is None and header['s4']==header['s5']==0 and header['input_dims']==header['output_dims']==[1,1,256]
    assert header['s12']>=sum(len(r) for r in header['passes'][0])
    f=S/'fixtures01/s110101';inputs=np.fromfile(f/'partial','u1').reshape(8,256);codes=np.fromfile(f/'codes','u1').reshape(8,256);new=np.fromfile(f/'new','<i4');cells=[]
    for h in [1,2,4,8]:
        hh=copy.deepcopy(header);hh['passes']=[];payload=bytearray(data)
        for i in range(h):
            p=copy.deepcopy(header['passes'][0])
            if i:
                for record in p:
                    for span in record:
                        offset,size=span;raw=data[offset:offset+size*4];span[0]=len(payload);payload.extend(raw);assert bytes(payload[span[0]:span[0]+size*4])==raw
            hh['passes'].append(p)
        struct.pack_into('<I',payload,0,len(payload)-4);b=build(hh,bytes(payload));rr,dd=read(b);assert rr==hh and dd==bytes(payload)
        (out/f'h{h}.bin').write_bytes(b);inputs.tofile(out/f'h{h}.inputs');want=np.arange(8)
        for _ in range(h):want=new[want]
        codes[want].tofile(out/f'h{h}.expected');cells.append(dict(horizon=h,passes=len(hh['passes']),sha256=hashlib.sha256(b).hexdigest(),expected=want.tolist()))
    (out/'BUILD.json').write_text(json.dumps(dict(status='FINITE_REPLAY_ADAPTER_PASS',source_program_sha256=hashlib.sha256(image).hexdigest(),unchanged_pass_track_bytes=True,adapter_changes='repeat full compiler-generated pass with unique backing track spans per pass; common input/output shape and encoding',learned_weights_source='raw/native_s110101_native_h32/result.ram',learning_enabled=False,cells=cells),indent=2)+'\n')
if __name__=='__main__':main()
