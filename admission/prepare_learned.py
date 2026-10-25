"""Compiler-only seven-group program with three resident plastic banks."""
from pathlib import Path
import argparse,json,traceback
import akida,numpy as np
from admission.learned import *
from admission.prepare import bind,fixed,Select,mapping,parts

def main(out,seed,arm):
    out.mkdir(parents=True,exist_ok=False);r=dict(seed=seed,arm=arm,physical_device_opened=False)
    try:
        mats=matrices(arm);m=akida.Model();m.add(akida.InputData(input_shape=(1,1,N),input_bits=1));prefixes=[]
        fixed(m,'raw_observation_encoding',*mats[0])
        for k,(n,u,classes) in enumerate(DIMENSIONS):
            if k:fixed(m,'synaptic_permission_policy' if k==1 else 'native_bank_bridge',*mats[k])
            m.add(akida.FullyConnected(name=['native_synaptic_recency_manager','always_update_consequence_memory','protected_consequence_memory'][k],units=u,weights_bits=1,activation=True,act_bits=1))
            m.compile(optimizer=akida.AkidaUnsupervised(num_weights=32,num_classes=classes,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25))
            w=weights(seed,k);lt,t=thresholds(k);l=m.layers[-1];l.set_variable('weights',w.reshape(1,1,n,u));l.set_variable('threshold_learning',lt);l.set_variable('threshold',t);l.set_variable('act_step',np.ones(u,np.float32))
            d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));assert len(m.sequences)==len(m.sequences[0].passes)==1
            prefix=bytes(m.sequences[0].program);prefixes.append(prefix);ram,npid=parts(prefix);stride=4+n//32
            assert len(ram)==2+u*stride
            assert all(a==b for j,(a,b) in enumerate(zip(ram,packed(w,k))) if j>=2 and (j-2)%stride>=3)
            m.map(None)
        fixed(m,'native_prediction_permission_and_observation_relay',*mats[3])
        rams=[parts(p)[0] for p in prefixes];test=simulate(rams,'gate',arm)['inputs'];v=test.astype('i4')
        v=(v@mats[0][0].astype('i4')>mats[0][1]).astype('i4')
        for k in range(3):
            v=(v@weights(seed,k).astype('i4')>thresholds(k)[1]).astype('i4')
            w,t=mats[k+1];v=(v@w.astype('i4')>t).astype('i4')
        assert np.array_equal(m.forward(test.reshape(-1,1,1,N)).reshape(-1,N),v),'full virtual fixed-state circuit'
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));assert len(m.sequences)==len(m.sequences[0].passes)==1
        program,banks,pm,adapter=bind(prefixes,bytes(m.sequences[0].program),[(n,u) for n,u,c in DIMENSIONS]);(out/'model.bin').write_bytes(program)
        for k,b in enumerate(banks):
            for ext in ['ram','infer','learn']:np.asarray(b[ext],'<u4').tofile(out/f'bank{k}.{ext}')
            thresholds(k)[1].tofile(out/f'bank{k}.threshold')
        for k,b in enumerate(pm):np.asarray(b['packet'],'<u4').tofile(out/f'fixed{k}.ram')
        for k,(w,t) in enumerate(mats):w.tofile(out/f'fixed{k}.weights');t.tofile(out/f'fixed{k}.threshold')
        cells={}
        for stage in STAGES:
            a=simulate(rams,stage,arm);cells[stage]=assess(a)
            np.asarray([[e['key'],int(e['teacher']),*e['labels'],OUTCOME[e['key']] if e['teacher'] else 0,int(arm=='severed' and e['teacher'] and e['phase'] not in ['bootstrap','gate'])] for e in a['events']],'<i4').tofile(out/f'{stage}.plan')
            for k,v in a.items():
                if k!='events':v.tofile(out/f'{stage}.{k}')
        r.update(status='VIRTUAL_COMPILER_PASS',mapping=mapping(m),banks=[dict(np=b['np'],n=b['n'],u=b['u'],words=len(b['ram'])) for b in banks],fixed_packets=[dict(np=b['np'],words=len(b['packet'])) for b in pm],adapter=adapter,cells=cells)
    except Exception as e:r.update(status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
    (out/'BUILD.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['cells','traceback']}),flush=True)
    if r['status']=='FAILED_PRESERVED':raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('seed',type=int,choices=SEEDS);p.add_argument('arm',choices=ARMS);a=p.parse_args();main(a.out,a.seed,a.arm)
