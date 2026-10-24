"""Compiler-only changed-transition memory, fixed neural decoder and scalar replay."""
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
N=256;U=C=8;SEEDS=[110101,110201,110301,110401,110501];HORIZONS=[0,1,2,4,32]

def simulate(ram,codes,partial,old,new,changed,arm,h):
    ref=Reference(ram,n=N,u=U,nw=32,classes=C)
    xs=[];labels=[];ys=[];states=[];potentials=[]
    # Outputs precede each update. Queries are explicitly unlabelled.
    events=[(codes[s],int(old[s]),True) for s in range(C)]
    events += [(partial[s],-1,False) for s in range(C)]
    events += [(codes[s],int(new[s]),arm=='native') for s in changed]
    events += [(partial[s],-1,False) for s in range(C)]
    for x,label,learn in events:
        bits=sum(1<<int(k) for k in np.flatnonzero(x))
        pot=ref.forward(x,label) if learn else [(b&bits).bit_count() for b in ref.weights]
        active=np.flatnonzero(np.asarray(pot)>24);y=np.maximum.reduce(codes[active],axis=0) if len(active) else np.zeros(N,'u1')
        xs.append(x);labels.append(label if learn else -1);ys.append(y);potentials.append(pot);states.append(ref.ram[:])
    diagnostic_ram=ref.ram[:];roll=[];roll_inputs=[]
    for start in range(C):
        x=partial[start].copy()
        for t in range(h):
            roll_inputs.append(x.copy());bits=sum(1<<int(k) for k in np.flatnonzero(x));pot=[(b&bits).bit_count() for b in ref.weights]
            active=np.flatnonzero(np.asarray(pot)>24);x=np.maximum.reduce(codes[active],axis=0) if len(active) else np.zeros(N,'u1');roll.append(x.copy())
    return {k:np.asarray(v,dtype=d) for k,v,d in [('inputs',xs,'u1'),('labels',labels,'<i4'),('outputs',ys,'u1'),('states',states,'<u4'),('potentials',potentials,'<i4'),('rollout_inputs',roll_inputs,'u1'),('rollout_outputs',roll,'u1'),('ram',diagnostic_ram,'<u4')]}

def main(out,seed):
    out.mkdir(parents=True,exist_ok=False);report=dict(seed=seed,physical_device_opened=False)
    try:
        rng=np.random.default_rng(seed);features=rng.permutation(N);anchor=features[:16];unique=features[16:144].reshape(C,16)
        codes=np.zeros((C,N),'u1');partial=codes.copy()
        for s in range(C):
            codes[s,anchor]=1;codes[s,unique[s]]=1;partial[s,anchor]=1;partial[s,rng.choice(unique[s],12,False)]=1
        ring=rng.permutation(C);old=np.empty(C,'<i4');old[ring]=np.roll(ring,-1);changed=np.sort(rng.choice(C,2,False));new=old.copy();new[changed]=new[changed[::-1]]
        w=np.zeros((N,U),np.int8)
        for s in range(C):w[:,old[s]]=codes[s]
        m=akida.Model();m.add(akida.InputData(input_shape=(1,1,N),input_bits=1));m.add(akida.FullyConnected(name='transition_memory',units=U,weights_bits=1,activation=True,act_bits=1))
        m.compile(optimizer=akida.AkidaUnsupervised(num_weights=32,num_classes=C,initial_plasticity=.5,min_plasticity=.25,plasticity_decay=.125,learning_competition=.25))
        l=m.get_layer('transition_memory');l.set_variable('weights',w.reshape(1,1,N,U));l.set_variable('threshold_learning',np.full(U,15,np.int32));l.set_variable('threshold',np.full(U,24,np.int32));l.set_variable('act_step',np.ones(U,np.float32))
        expected=np.eye(C,dtype='u1')[old];assert np.array_equal(m.forward(partial.reshape(C,1,1,N)).reshape(C,U),expected)
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));single=bytes(m.sequences[0].program);ram,npid=parts(single)
        m.map(None);m.add(akida.FullyConnected(name='fixed_state_decoder',units=N,weights_bits=1,activation=True,act_bits=1));l=m.get_layer('fixed_state_decoder');l.set_variable('weights',codes.astype(np.int8).reshape(1,1,U,N));l.set_variable('threshold',np.zeros(N,np.int32));l.set_variable('act_step',np.ones(N,np.float32))
        assert np.array_equal(m.forward(partial.reshape(C,1,1,N)).reshape(C,N),codes[old])
        d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d));assert len(m.sequences)==len(m.sequences[0].passes)==1
        program,adapter=assemble(single,bytes(m.sequences[0].program),N,U);rr,nn=parts(program);assert rr==ram and nn==npid
        header,_=read(program);assert header['input_dims']==header['output_dims']==[1,1,N] and header['s4']==header['s5']==0 and header['s6']==header['s7']==1
        pm=packets(program,npid);assert len(pm)==1
        (out/'program.bin').write_bytes(program);np.asarray(ram,'<u4').tofile(out/'initial.ram');np.asarray(pm[0][1],'<u4').tofile(out/'fixed.ram')
        for key,a in [('codes',codes),('partial',partial),('old',old),('new',new),('changed',changed.astype('<i4'))]:a.tofile(out/key)
        for arm in ['native','frozen','disconnected']:
            for h in HORIZONS:
                a=simulate(ram,codes,partial,old,new,changed,arm,h)
                for key,v in a.items():v.tofile(out/f'{arm}_{h}.{key}')
                assert np.array_equal(a['outputs'][8:16],codes[old]);assert np.array_equal(a['outputs'][18:26],codes[new if arm=='native' else old])
                if arm=='native':
                    delta=np.asarray(a['ram'][2:],'<u4').reshape(U,12)[:,4:]^np.asarray(ram[2:],'<u4').reshape(U,12)[:,4:];assert sum(int(v).bit_count() for v in delta.flat)==64
                else:assert np.array_equal(a['ram'],ram)
        report.update(status='COMPILER_AND_REFERENCE_PASS',mapping=mapping(m),learning_np=npid,ram_words=len(ram),adapter=adapter,old=old.tolist(),new=new.tolist(),changed=changed.tolist(),classes=C,prototypes_per_class=1,feature_encoding='16 common anchors + 16 state bits; partial cue removes four state bits',query_label=None)
    except Exception as e:report.update(status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
    (out/'BUILD.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if report['status']=='FAILED_PRESERVED':raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('seed',type=int);a=p.parse_args();main(a.out,a.seed)
