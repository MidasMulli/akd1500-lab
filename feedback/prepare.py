"""Compiler-only native self-gating fixture; preserves raw and reduced paths."""
from pathlib import Path
import argparse,json,sys,traceback
import numpy as np
import akida
from feedback.design import N,U,initial,schedule,simulate,metrics
from tools.prepare import parts
from metal.prepare import mapping
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'tools'))
from prepare_reduction import assemble
from prepare_network import Select
from coprocessor.resident_prepare import packets

def main(out,seed):
    out.mkdir(parents=True,exist_ok=False);report=dict(seed=seed,physical_device_opened=False)
    try:
        w=initial(seed);m=akida.Model();m.add(akida.InputData(input_shape=(1,1,N),input_bits=1));m.add(akida.FullyConnected(name='signal_gated_memory',units=U,weights_bits=1,activation=True,act_bits=1))
        m.compile(optimizer=akida.AkidaUnsupervised(num_weights=32,num_classes=1,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25))
        l=m.get_layer('signal_gated_memory');l.set_variable('weights',w.reshape(1,1,N,U));l.set_variable('threshold_learning',np.full(U,24,np.int32));l.set_variable('threshold',np.full(U,13,np.int32));l.set_variable('act_step',np.ones(U,np.float32))
        x,rows,truth=schedule(seed,'native');expected=(x.astype('i4')@w.astype('i4')>13).astype('i4');assert np.array_equal(m.forward(x[:16].reshape(16,1,1,N)).reshape(16,U),expected[:16])
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));single=bytes(m.sequences[0].program);ram,npid=parts(single)
        # A raw-potential twin uses the same geometry, weights and learning rule.
        mr=akida.Model();mr.add(akida.InputData(input_shape=(1,1,N),input_bits=1));mr.add(akida.FullyConnected(name='raw_memory',units=U,weights_bits=1,activation=False));mr.compile(optimizer=akida.AkidaUnsupervised(num_weights=32,num_classes=1,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25))
        lr=mr.get_layer('raw_memory');lr.set_variable('weights',w.reshape(1,1,N,U));lr.set_variable('threshold_learning',np.full(U,24,np.int32));dr=akida.AKD1500();mr.map(dr,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(dr));raw=bytes(mr.sequences[0].program);rr,rnp=parts(raw)
        diffs=[i for i,(a,b) in enumerate(zip(rr,ram)) if a!=b];report['raw_reduced_aux_diff']=diffs
        assert rnp==npid and len(rr)==len(ram) and all(i>=2 and (i-2)%20 in [0,1,2] for i in diffs),diffs
        m.map(None);m.add(akida.FullyConnected(name='native_prediction_support',units=1,weights_bits=1,activation=False));m.get_layer('native_prediction_support').set_variable('weights',np.ones((1,1,U,1),np.int8))
        assert np.array_equal(m.forward(x[:16].reshape(16,1,1,N)).reshape(16),expected[:16].sum(1))
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));assert len(m.sequences)==len(m.sequences[0].passes)==1
        reduced,adapter=assemble(single,bytes(m.sequences[0].program),N,U);r2,p2=parts(reduced);assert r2==ram and p2==npid
        for name,b in [('raw',raw),('reduced',reduced)]:
            f=out/name;f.with_suffix('.bin').write_bytes(b);f.with_suffix('.ram').write_bytes(np.asarray(rr if name=='raw' else ram,'<u4').tobytes());pm=packets(b,npid)
            f.with_suffix('.layout').write_text(str(len(pm))+'\n')
            for j,(_,a) in enumerate(pm):f.with_suffix(f'.packet{j}.ram').write_bytes(np.asarray(a,'<u4').tobytes())
        screens={}
        for arm in ['native','frozen','disconnected']:
            x,rows,truth=schedule(seed,arm);y,states=simulate(ram,x,rows,arm);_,raw_states=simulate(rr,x,rows,arm);f=out/arm
            f.with_suffix('.inputs').write_bytes(x.tobytes());f.with_suffix('.potentials').write_bytes(y.tobytes());f.with_suffix('.states').write_bytes(states.tobytes());f.with_suffix('.rows.json').write_text(json.dumps(rows,indent=2)+'\n')
            f.with_suffix('.rawstates').write_bytes(raw_states.tobytes())
            # Queries are incapable of reaching threshold24, independent of mode.
            before=np.vstack([np.asarray(ram,'<u4'),states[:-1]])
            for i,row in enumerate(rows):
                if not row['feedback'] or not row['gate_present']:assert np.array_equal(before[i],states[i]),('query/gateless edit',i)
            z=metrics(y,rows,truth);screens[arm]={k:v for k,v in z.items() if k!='predictions'}
            key='post_adaptation_changed';assert z['groups'][key]['correct']==(2 if arm=='native' else 0),z['groups']
        report.update(status='COMPILER_AND_CPU_SCREEN_PASS',mapping=mapping(m),learning_np=npid,adapter=adapter,truth=truth,cpu_screens=screens,ram_words=len(ram),constant_teacher_class=0,learning_mode='always enabled in native arm; input match threshold gates writes',event_count=768)
    except Exception as e:report.update(status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
    (out/'BUILD.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if report['status']=='FAILED_PRESERVED':sys.exit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('seed',type=int);a=p.parse_args();main(a.out,a.seed)
