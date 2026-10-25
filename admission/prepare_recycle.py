"""Virtual/compiler gate only; never selects a physical device."""
from pathlib import Path
import argparse,json,traceback
import akida,numpy as np
from admission.recycle import *
from admission.prepare import bind,Select,mapping,parts

def fixed(m,name,w,t,bits=1):
    m.add(akida.FullyConnected(name=name,units=w.shape[1],weights_bits=bits,activation=True,act_bits=1))
    l=m.layers[-1];l.set_variable('weights',w.reshape(1,1,*w.shape));l.set_variable('threshold',t);l.set_variable('act_step',np.ones(w.shape[1],np.float32))

def main(out,seed,arm):
    out.mkdir(parents=True,exist_ok=False);r=dict(seed=seed,arm=arm,physical_device_opened=False)
    try:
        mats=matrices(arm);m=akida.Model();m.add(akida.InputData(input_shape=(1,1,N),input_bits=1))
        for k in range(3): fixed(m,['saved_prediction_actual_comparator','native_query_or_error_gate','binary_key_expansion'][k],*mats[k],2 if k==0 else 1)
        m.add(akida.FullyConnected(name='plastic_consequence_memory',units=U,weights_bits=1,activation=True,act_bits=1))
        m.compile(optimizer=akida.AkidaUnsupervised(num_weights=32,num_classes=2,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25))
        w=weights(seed);lt,t=thresholds();l=m.layers[-1]
        l.set_variable('weights',w.reshape(1,1,D,U));l.set_variable('threshold_learning',lt);l.set_variable('threshold',t);l.set_variable('act_step',np.ones(U,np.float32))
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d))
        assert len(m.sequences)==len(m.sequences[0].passes)==1
        prefix=bytes(m.sequences[0].program);ram,npid=parts(prefix);stride=4+D//32
        assert all(a==b for j,(a,b) in enumerate(zip(ram,packed(w))) if j>=2 and (j-2)%stride>=3)
        m.map(None);fixed(m,'prediction_and_raw_key_relay',*mats[3])
        test=simulate(ram,'truth',arm)['inputs'];v=test.astype('i4')
        for ww,tt in mats[:3]:v=(v@ww.astype('i4')>tt).astype('i4')
        v=(v@w.astype('i4')>t).astype('i4');v=(v@mats[3][0]>mats[3][1]).astype('i4')
        assert np.array_equal(m.forward(test.reshape(-1,1,1,N)).reshape(-1,N),v),'virtual signed comparator truth table'
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d))
        assert len(m.sequences)==len(m.sequences[0].passes)==1
        program,banks,pm,adapter=bind([prefix],bytes(m.sequences[0].program),[(D,U)])
        (out/'model.bin').write_bytes(program)
        for ext in ['ram','infer','learn']:np.asarray(banks[0][ext],'<u4').tofile(out/f'bank.{ext}')
        t.tofile(out/'bank.threshold')
        for k,b in enumerate(pm):np.asarray(b['packet'],'<u4').tofile(out/f'fixed{k}.ram')
        for k,(ww,tt) in enumerate(mats):ww.tofile(out/f'fixed{k}.weights');tt.tofile(out/f'fixed{k}.threshold')
        cells={}
        for stage in STAGES:
            a=simulate(ram,stage,arm);cells[stage]=assess(a)
            np.asarray([[e['key'],int(e['teacher']),e['label'],e['actual'],int(e['override']),e['p0'],e['p1'],int(e['cut']),int(e['erase'])] for e in a['events']],'<i4').tofile(out/f'{stage}.plan')
            for key,val in a.items():
                if key!='events':val.tofile(out/f'{stage}.{key}')
        r.update(status='VIRTUAL_COMPILER_PASS',mapping=mapping(m),banks=[dict(np=b['np'],n=b['n'],u=b['u'],words=len(b['ram'])) for b in banks],fixed_packets=[dict(np=b['np'],words=len(b['packet'])) for b in pm],adapter=adapter,cells=cells)
    except Exception as e:r.update(status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
    (out/'BUILD.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['cells','traceback']}),flush=True)
    if r['status']=='FAILED_PRESERVED':raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('seed',type=int,choices=SEEDS);p.add_argument('arm',choices=ARMS);a=p.parse_args();main(a.out,a.seed,a.arm)
