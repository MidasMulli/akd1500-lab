"""CPU-only targeted record inputs in the already tested 8192-record bank."""
import json
import numpy as np
from coprocessor.run import P,save
from coprocessor.fnp2_capacity_run import trajectory
S=P/'studies/20261024-fnp2-boundary'
def main():
    original=P/'studies/20261024-fnp2-capacity/fixtures01/u8192/learner'
    initial=np.fromfile(original.with_suffix('.raw'),'<u4');records=initial.reshape(8192,20)
    root=S/'fixtures01';root.mkdir(exist_ok=False)
    for case,target in [('below',6500),('above',6600)]:
        out=root/case;out.mkdir();f=out/'learner'
        for suffix in ['bin','raw','cpu_ram','units']:f.with_suffix('.'+suffix).write_bytes(original.with_suffix('.'+suffix).read_bytes())
        mask=np.unpackbits(records[target,4:].copy().view('u1'),bitorder='little')
        x=np.zeros((64,512),np.uint8);x[1]=mask;active=np.flatnonzero(mask);inactive=np.flatnonzero(1-mask);x[1,active[0]]=0;x[1,inactive[0]]=1
        ys,_=trajectory(initial,x[:8],8192,False);assert ys[1,target]==31 and np.sum(ys[1,4096:]==31)==1,'unique class-1 winner'
        ys,states=trajectory(initial,x[:8],8192,True);changed=np.flatnonzero((states[1].reshape(8192,20)!=records).any(1));assert changed.tolist()==[target]
        f.with_suffix('.inputs').write_bytes(x.tobytes());f.with_suffix('.labels').write_bytes((np.arange(64,dtype='<i4')%2).tobytes())
        save(f.with_suffix('.json'),dict(case=case,target_record=target,records=8192,record_offset_bytes=target*80,expected_potential=31,expected_updated_records=[target],weight_seed=107101,one_bit_removed=int(active[0]),one_bit_added=int(inactive[0]),physical_device_opened=False))
        print(case,target,'unique target, CPU-only PASS',flush=True)
if __name__=='__main__':main()
