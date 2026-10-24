"""Compiler-only fixtures for repeated native adaptation, never opens hardware."""
from pathlib import Path
import argparse,json,sys,traceback
import numpy as np
import akida
from plasticity.model import *
from tools.prepare import parts
from tools.program_codec import read
from metal.prepare import mapping
from coprocessor.resident_prepare import packets
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'tools'))
from prepare_reduction import assemble
from prepare_network import Select
def main(out,seed,extra_queries=0,full_queries=False):
    out.mkdir(parents=True,exist_ok=False);report=dict(seed=seed,physical_device_opened=False)
    try:
        codes,partial,old,new,changed=design(seed);w=np.zeros((N,U),np.int8)
        for s in range(8):w[:,old[s]]=codes[s]
        mappings={};initials=[]
        for arm in (['native'] if extra_queries else ARMS):
            comp=.25 if arm=='baseline' else .75;m=akida.Model();m.add(akida.InputData(input_shape=(1,1,N),input_bits=1));m.add(akida.FullyConnected(name='resident_transition',units=U,weights_bits=1,activation=True,act_bits=1));m.compile(optimizer=akida.AkidaUnsupervised(num_weights=32,num_classes=8,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=comp));l=m.get_layer('resident_transition');l.set_variable('weights',w.reshape(1,1,N,U));l.set_variable('threshold_learning',np.full(U,15,np.int32));l.set_variable('threshold',np.full(U,24,np.int32));l.set_variable('act_step',np.ones(U,np.float32));assert np.array_equal(m.forward(partial.reshape(8,1,1,N)).reshape(8,U),np.eye(8,dtype='u1')[old])
            d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));single=bytes(m.sequences[0].program);ram,npid=parts(single);initials.append(ram)
            m.map(None);m.add(akida.FullyConnected(name='fixed_state_decode',units=N,weights_bits=1,activation=True,act_bits=1));l=m.get_layer('fixed_state_decode');l.set_variable('weights',codes.astype(np.int8).reshape(1,1,U,N));l.set_variable('threshold',np.zeros(N,np.int32));l.set_variable('act_step',np.ones(N,np.float32));assert np.array_equal(m.forward(partial.reshape(8,1,1,N)).reshape(8,N),codes[old])
            d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));assert len(m.sequences)==len(m.sequences[0].passes)==1;program,adapter=assemble(single,bytes(m.sequences[0].program),N,U);pm=packets(program,npid);assert len(pm)==1;h,_=read(program);assert h['s4']==h['s5']==0 and h['input_dims']==h['output_dims']==[1,1,N]
            (out/f'{arm}.bin').write_bytes(program);np.asarray(ram,'<u4').tofile(out/f'{arm}.ram');np.asarray(pm[0][1],'<u4').tofile(out/f'{arm}.fixed');mappings[arm]=dict(mapping=mapping(m),competition=comp,adapter=adapter)
            for epochs in ([2] if extra_queries else EPOCHS):
                a=simulate(ram,seed,arm,epochs,extra_queries,full_queries);prefix=out/f'{arm}_e{epochs}';(prefix.with_suffix('.events.json')).write_text(json.dumps(a['events'],indent=2)+'\n')
                plan=np.asarray([[{'old_example':0,'old_query':1,'pre_prediction':2,'fresh_example':3,'query':4,'feedback':5,'selector_query':6}[e['kind']],e['epoch'],e['label'],e['source'],e['context']] for e in a['events']],'<i4');plan.tofile(prefix.with_suffix('.plan'))
                for key in ['inputs','outputs','states','potentials','table_outputs']:a[key].tofile(prefix.with_suffix('.'+key))
                for epoch in range(epochs):
                    ids=[e['index'] for e in a['events'] if e['epoch']==epoch and e['kind']=='query'];truth=new if epoch%2==0 else old;pred=np.asarray(a['outputs'])[ids];expected=codes[truth if arm=='native' or arm=='baseline' and epoch==0 else new if arm=='baseline' else old];assert np.array_equal(pred,expected),(arm,epoch,'CPU query gate')
        assert all(a==initials[0] for a in initials)
        for key,a in [('codes',codes),('partial',partial),('old',old),('new',new),('changed',changed)]:a.tofile(out/key)
        report.update(full_queries=full_queries,extra_queries=extra_queries,status='COMPILER_REFERENCE_PASS',mappings=mappings,changed=changed.tolist(),old=old.tolist(),new=new.tolist(),events_per_epoch=13+H,initial_events=16,seed=seed,threshold_caps=dict(native=8,baseline=24))
    except Exception as e:report.update(full_queries=full_queries,extra_queries=extra_queries,status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
    (out/'BUILD.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='mappings'}),flush=True)
    if report['status']=='FAILED_PRESERVED':raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('seed',type=int);p.add_argument('--full-queries',action='store_true');p.add_argument('--extra-queries',type=int,default=0,choices=[0,1,8]);a=p.parse_args();main(a.out,a.seed,a.extra_queries,a.full_queries)
