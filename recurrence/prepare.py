"""Compiler-only feedback fixtures; no physical discovery or device opening."""
from pathlib import Path
import argparse, json, sys, traceback
import numpy as np
import akida
from tools.prepare import parts
from tools.program_codec import read
from metal.prepare import mapping
from coprocessor.resident_prepare import packets
from pong.relearn.model import Reference
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'tools'))
from prepare_reduction import assemble
from prepare_network import Select
N=U=128
HORIZONS=[1,2,3,4,5,8,16,32]
SEEDS=[109101,109201,109301]

def weights(seed):
    rng=np.random.default_rng(seed);w=np.zeros((N,U),np.int8)
    for j in range(U):
        group=((j//32)-1)%4; a=np.arange(group*32,(group+1)*32)
        b=np.setdiff1d(np.arange(N),a)
        w[rng.choice(a,16,False),j]=1;w[rng.choice(b,16,False),j]=1
    return w

def simulate(ram,seed_input,arm,horizon,probes):
    ref=Reference(ram,n=N,u=U,nw=32,classes=1)
    x=seed_input.copy();ys=[];states=[];inputs=[];potentials=[]
    for i in range(horizon):
        inputs.append(x.copy())
        if arm=='frozen':
            bits=sum(1<<int(k) for k in np.flatnonzero(x));pot=np.asarray([(b&bits).bit_count() for b in ref.weights])
        else:pot=np.asarray(ref.forward(x,0))
        y=(pot>13).astype(np.uint8);ys.append(y);potentials.append(pot);states.append(ref.ram[:])
        x=seed_input.copy() if arm=='repeated_seed' else np.zeros(N,np.uint8) if arm=='severed' else y.copy()
    probe_y=[];probe_states=[]
    for a in probes:
        if arm=='frozen':
            bits=sum(1<<int(k) for k in np.flatnonzero(a));pot=np.asarray([(b&bits).bit_count() for b in ref.weights])
        else:pot=np.asarray(ref.forward(a,0))
        probe_y.append((pot>13).astype(np.uint8));probe_states.append(ref.ram[:])
    return {k:np.asarray(v,dtype=t) for k,v,t in [('inputs',inputs,'u1'),('outputs',ys,'u1'),('potentials',potentials,'<i4'),('states',states,'<u4'),('probe_outputs',probe_y,'u1'),('probe_states',probe_states,'<u4')]}

def main(out,seed):
    out.mkdir(parents=True,exist_ok=False);report=dict(seed=seed,physical_device_opened=False)
    try:
        w=weights(seed);m=akida.Model();m.add(akida.InputData(input_shape=(1,1,N),input_bits=1));m.add(akida.FullyConnected(name='recurrent_association',units=U,weights_bits=1,activation=True,act_bits=1))
        m.compile(optimizer=akida.AkidaUnsupervised(num_weights=32,num_classes=1,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25))
        l=m.get_layer('recurrent_association');l.set_variable('weights',w.reshape(1,1,N,U));l.set_variable('threshold_learning',np.full(U,15,np.int32));l.set_variable('threshold',np.full(U,13,np.int32));l.set_variable('act_step',np.ones(U,np.float32))
        cues=np.zeros((4,N),np.uint8)
        for g in range(4):cues[g,g*32:(g+1)*32]=1
        expected=(cues.astype('i4')@w.astype('i4')>13).astype('u1')
        assert np.array_equal(expected,np.roll(cues,-1,axis=0)), 'unambiguous four-state transition'
        assert np.array_equal(m.forward(cues.reshape(4,1,1,N)).reshape(4,U),expected)
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));single=bytes(m.sequences[0].program);ram,npid=parts(single)
        m.map(None);m.add(akida.FullyConnected(name='fixed_feedback_relay',units=U,weights_bits=1,activation=True,act_bits=1));l=m.get_layer('fixed_feedback_relay');l.set_variable('weights',np.eye(U,dtype=np.int8).reshape(1,1,U,U));l.set_variable('threshold',np.zeros(U,np.int32));l.set_variable('act_step',np.ones(U,np.float32))
        assert np.array_equal(m.forward(cues.reshape(4,1,1,N)).reshape(4,U),expected)
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));assert len(m.sequences)==len(m.sequences[0].passes)==1
        program,adapter=assemble(single,bytes(m.sequences[0].program),N,U);rr,nn=parts(program);assert rr==ram and nn==npid
        h,_=read(program);assert h['input_dims']==h['output_dims']==[1,1,N] and h['s4']==h['s5']==0 and h['s6']==1 and h['s7']==1,h
        pm=packets(program,npid);assert len(pm)==1
        probes=[]
        for g in range(4):
            target=((g+1)%4)*32;probe=np.zeros(N,np.uint8)
            missing=np.flatnonzero((cues[g]>0)&(w[:,target]==0));assert len(missing)==16
            probe[missing]=1;assert not np.any(probe.astype('i4')@w.astype('i4')>13),'partial cue frozen floor'
            probes.append(probe)
        probes=np.asarray(probes,np.uint8);seed_input=cues[(seed//100)%4]
        (out/'program.bin').write_bytes(program);(out/'initial.ram').write_bytes(np.asarray(ram,'<u4').tobytes());(out/'seed.input').write_bytes(seed_input.tobytes());(out/'probes.inputs').write_bytes(probes.tobytes());(out/'fixed.ram').write_bytes(np.asarray(pm[0][1],'<u4').tobytes())
        cells={}
        for arm in ['native','frozen','repeated_seed','severed']:
            for horizon in HORIZONS:
                a=simulate(ram,seed_input,arm,horizon,probes);f=out/f'{arm}_{horizon}'
                for k,v in a.items():f.with_suffix('.'+k).write_bytes(v.tobytes())
                if horizon>=4:
                    hits=a['probe_outputs'].sum(1).tolist();assert hits==([1]*4 if arm=='native' else [0]*4 if arm=='frozen' else [int(g==int(np.flatnonzero(seed_input)[0])//32) for g in range(4)]), (arm,hits)
                cells[f'{arm}_{horizon}']=dict(output_active=a['outputs'].sum(1).tolist(),probe_active=a['probe_outputs'].sum(1).tolist())
        report.update(status='COMPILER_AND_REFERENCE_PASS',mapping=mapping(m),learning_np=npid,ram_words=len(ram),adapter=adapter,cells=cells,seed_state=int(np.flatnonzero(seed_input)[0])//32,io='dense binary input and output, native output payload is next input')
    except Exception as e:report.update(status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
    (out/'BUILD.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if report['status']=='FAILED_PRESERVED':raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('seed',type=int);a=p.parse_args();main(a.out,a.seed)
