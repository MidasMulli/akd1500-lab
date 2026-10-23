"""Virtual compiler: native lookahead learns only externally observed worker outcomes."""
from pathlib import Path
import itertools,json,struct
import numpy as np
import akida
from compiler_helpers import fc
from assembly import two_memory_assembly
from layers import prediction_weights,message_encoder,history_weights,policy

P=Path(__file__).resolve().parent.parent;F=P/'frozen'
PARAMS=dict(num_weights=32,num_classes=2,initial_plasticity=.125,min_plasticity=.125,plasticity_decay=.125,learning_competition=.25)
HP=dict(num_weights=32,num_classes=2,initial_plasticity=.25,min_plasticity=.25,plasticity_decay=.125,learning_competition=.5)

def front():
    w=np.zeros((32,512),np.int8);t=np.zeros(512,np.int32);t[224:240]=1
    for s in range(4):
        w[s,:224]=1;w[s,224:228]=2
        w[4+s,56*s:56*s+56]=1;w[4+s,224:240]=1
        w[s,256+s]=w[4+s,256+s]=1
        w[s,260]=w[4+s,260]=1;w[s,261]=1
        for a in range(2):
            j=8+2*s+a;w[j,28*(2*s+a):28*(2*s+a)+28]=1
            # Actual outcome one-hot injected at input bytes24..27 only after execution.
            w[j,224:240]=1
        w[24+s,224+4*s:228+4*s]=1
    w[:8,262]=1;w[16,262]=1;t[262]=1
    for k in range(2):w[4:8,263+k]=1;w[28+k,263+k]=1;t[263+k]=1
    # Probe bit is meaningful on protected queries only, never a teacher bit.
    w[:8,265]=1;w[20,265]=1;t[265]=1
    return w,t

def expected(x,mapping,hist,twohop):
    idx=next(i for i in range(16) if x[i]);y=[0]*32
    if idx<4:
        s=idx;r1=[int(any(mapping[2*k+a]==0 for a in range(2))) for k in range(4)]
        direct=[int(mapping[2*s+a]==0) for a in range(2)]
        y[4+s]=1;y[16]=x[16];y[17]=1;y[24:28]=r1;y[28:30]=direct
        y[30]=int(hist[s] is not None and hist[s][0]==1);y[31]=int(any(direct));y[20]=x[20]
    elif idx<8:
        s=idx-4;direct=x[28:30];hop=[mapping[2*s+a] is not None and bool(x[24+mapping[2*s+a]]) for a in range(2)]
        last1=hist[s] is not None and hist[s][0]==1
        if x[20]:act=0 if last1 else 1;depth=0
        elif any(direct):act=direct.index(1);depth=1
        elif twohop and any(hop):act=hop.index(True);depth=2
        else:act=0 if last1 else 1;depth=0
        y[8+2*s+act]=1;y[16]=x[16];y[18]=1;y[20 if depth==1 else 21 if depth==2 else 22]=1;y[30]=int(last1)
    else:
        ctx=idx-8;act=ctx%2;out=x[24:28].index(1)
        y[3 if out==0 else out]=1;y[16]=act;y[19]=1;y[23]=int(out==0)
    return y

def main():
    F.mkdir(exist_ok=False);ew,et=front();mw=prediction_weights([None]*8);hw,ht=message_encoder()
    for ctx,out in itertools.product(range(8),range(4)):
        x=np.zeros(32,int);x[8+ctx]=1;x[24+out]=1;active=(x@ew>et).astype(int)
        assert active.sum()==32
        x[20]=1
        assert np.array_equal(active,(x@ew>et).astype(int)),'probe must not alter teaching features'
        mid=(active@mw>np.array([31]*32+[0]*32)).astype(int);active2=(mid@hw>ht).astype(int)
        assert active2.sum()==32
    rng=np.random.default_rng(71011);trials=[([None]*8,[None]*4)]+[
        ([None if k==4 else int(k) for k in rng.integers(0,5,8)],
         [[int(rng.integers(0,2)),int(rng.integers(0,4))] if rng.integers(0,2) else None for _ in range(4)]) for z in range(15)]
    xs=[]
    for s,prev in itertools.product(range(4),range(2)):
        x=[0]*32;x[s]=1;x[16]=prev;xs.append(x)
        for r1,direct in itertools.product(range(16),range(4)):
            x=[0]*32;x[4+s]=1;x[16]=prev;x[24:28]=[(r1>>k)&1 for k in range(4)];x[28:30]=[direct&1,direct>>1];xs.append(x)
    for ctx,out,prev in itertools.product(range(8),range(4),range(2)):
        x=[0]*32;x[8+ctx]=1;x[24+out]=1;x[16]=prev;xs.append(x)
    query_xs=[x[:] for x in xs if next(i for i in range(16) if x[i])<8]
    for x in query_xs:x[20]=1
    xs+=query_xs
    for twohop in [True,False]:
        name='twohop' if twohop else 'onestep';m=akida.Model();m.add(akida.InputData(input_shape=(1,1,32),input_bits=1))
        fc(m,'external_outcome_and_protected_query_encoder',ew,et)
        m.add(akida.FullyConnected(name='native_transition_memory',units=64,weights_bits=1,activation=True,act_bits=1));m.compile(optimizer=akida.AkidaUnsupervised(**PARAMS))
        l=m.get_layer('native_transition_memory');l.set_variable('weights',prediction_weights([None]*8).reshape(1,1,512,64));l.set_variable('threshold',np.array([31]*32+[0]*32,np.int32));l.set_variable('threshold_learning',np.array([0]*8+[32]*56,np.int32))
        m.save(str(F/(name+'_p1.fbz')));m.map(akida.AKD1500(),hw_only=True);p1=bytes(m.sequences[0].program)
        mm=akida.Model(str(F/(name+'_p1.fbz')));fc(mm,'native_message_encoder',hw,ht)
        mm.add(akida.FullyConnected(name='native_history_memory',units=64,weights_bits=1,activation=True,act_bits=1));mm.compile(optimizer=akida.AkidaUnsupervised(**HP))
        l=mm.get_layer('native_history_memory');l.set_variable('weights',history_weights([None]*4).reshape(1,1,512,64));l.set_variable('threshold',np.array([31]*32+[0]*32,np.int32));l.set_variable('threshold_learning',np.array([0]*4+[32]*60,np.int32))
        mm.save(str(F/(name+'_p2.fbz')));mm.map(akida.AKD1500(),hw_only=True);p2=bytes(mm.sequences[0].program)
        for lname,w,t in policy(0,3,twohop):fc(mm,lname,w,t)
        mm.map(None)
        for mapping,hist in trials:
            mm.get_layer('native_transition_memory').set_variable('weights',prediction_weights(mapping).reshape(1,1,512,64))
            mm.get_layer('native_history_memory').set_variable('weights',history_weights(hist).reshape(1,1,512,64))
            got=mm.forward(np.array(xs,np.uint8).reshape(-1,1,1,32)).reshape(-1,32).tolist()
            want=[expected(x,mapping,hist,twohop) for x in xs];assert got==want,(name,[(i,a,b) for i,(a,b) in enumerate(zip(got,want)) if a!=b][:1])
        mm.get_layer('native_transition_memory').set_variable('weights',prediction_weights([None]*8).reshape(1,1,512,64));mm.get_layer('native_history_memory').set_variable('weights',history_weights([None]*4).reshape(1,1,512,64))
        mm.map(akida.AKD1500(),hw_only=True);assert len(mm.sequences)==len(mm.sequences[0].passes)==1
        image,audit=two_memory_assembly(p1,p2,bytes(mm.sequences[0].program));(F/(name+'.bin')).write_bytes(image);(F/(name+'_AUDIT.json')).write_text(json.dumps(audit,indent=2)+'\n')
        (F/(name+'_PLAN.txt')).write_text(''.join(' '.join(map(str,v['np']+[len(v['initial_words'] if 'initial_words' in v else v['words'])]))+'\n' for v in audit['memories']+audit['fixed']))
        for k,mem in enumerate(audit['memories']):
            for mode in ['inference','learning']:(F/f'{name}_m{k}_{mode}.regs').write_bytes(struct.pack('<'+'I'*len(mem[mode]),*mem[mode]))
        print(name,'OFFLINE PASS',len(xs)*len(trials),flush=True)
    (F/'OFFLINE.json').write_text(json.dumps(dict(status='PASS',programs=2,cases_per_program=len(xs)*len(trials),external_observation_only=True,both_teacher_features=32,physical_device_opened=False),indent=2)+'\n')
if __name__=='__main__':main()
