"""Compile on a virtual device; no physical discovery or opening."""
from pathlib import Path
import argparse,json,sys,traceback
import numpy as np
import akida
from temporal.model import *
from tools.prepare import parts
from metal.prepare import mapping
from coprocessor.resident_prepare import packets
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'tools'))
from prepare_reduction import assemble
from prepare_network import Select

def main(out,seed):
    out.mkdir(parents=True,exist_ok=False);r=dict(seed=seed,physical_device_opened=False)
    try:
        codes,cues,w,old,new,changed=design(seed)
        m=akida.Model();m.add(akida.InputData(input_shape=(1,1,N),input_bits=1))
        m.add(akida.FullyConnected(name='context_conditioned_memory',units=U,weights_bits=1,activation=True,act_bits=1))
        m.compile(optimizer=akida.AkidaUnsupervised(num_weights=32,num_classes=8,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.75))
        l=m.get_layer('context_conditioned_memory');l.set_variable('weights',w.reshape(1,1,N,U))
        l.set_variable('threshold_learning',np.full(U,15,np.int32));l.set_variable('threshold',np.full(U,24,np.int32));l.set_variable('act_step',np.ones(U,np.float32))
        test=np.asarray([codes[c]+cues[4] for c in range(4)])
        assert np.array_equal(m.forward(test.reshape(4,1,1,N)).reshape(4,U),np.eye(U,dtype='u1')[old])
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));single=bytes(m.sequences[0].program);ram,npid=parts(single)
        m.map(None);m.add(akida.FullyConnected(name='fixed_resident_context_code',units=N,weights_bits=1,activation=True,act_bits=1))
        l=m.get_layer('fixed_resident_context_code');l.set_variable('weights',codes.reshape(1,1,U,N).astype('i1'));l.set_variable('threshold',np.zeros(N,np.int32));l.set_variable('act_step',np.ones(N,np.float32))
        assert np.array_equal(m.forward(test.reshape(4,1,1,N)).reshape(4,N),codes[old])
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d))
        assert len(m.sequences)==len(m.sequences[0].passes)==1
        program,adapter=assemble(single,bytes(m.sequences[0].program),N,U);pm=packets(program,npid);assert len(pm)==1
        (out/'model.bin').write_bytes(program);np.asarray(ram,'<u4').tofile(out/'initial.ram');np.asarray(pm[0][1],'<u4').tofile(out/'fixed.ram')
        for key,v in [('codes',codes),('cues',cues),('weights',w),('old',old),('new',new),('changed',changed)]:v.tofile(out/key)
        for arm in ARMS:
            for epochs in STAGES:
                a=simulate(ram,seed,arm,epochs);prefix=out/f'{arm}_e{epochs}'
                plan=np.asarray([[e['source'],e['cue'],e['label'] if arm in ['native','severed'] else -1,
                    int(arm=='severed' and e['phase']=='probe' and e['cue']>=4),e['delay_ms']] for e in a['events']],'<i4')
                plan.tofile(prefix.with_suffix('.plan'))
                for key in ['inputs','outputs','states','task_outputs','stateless_outputs']:a[key].tofile(prefix.with_suffix('.'+key))
                for e,y in zip(a['events'],a['outputs']):
                    if e['phase']=='probe' and arm=='native':assert np.array_equal(y,codes[e['target']]),(arm,epochs,e)
        r.update(status='COMPILER_REFERENCE_PASS',mapping=mapping(m),adapter=adapter,changed=changed.tolist(),old=old.tolist(),new=new.tolist(),initial_ram_words=len(ram))
    except Exception as e:r.update(status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
    (out/'BUILD.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
    if r['status']=='FAILED_PRESERVED':raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('seed',type=int);a=p.parse_args();main(a.out,a.seed)
