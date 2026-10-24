"""Native carried context plus a current-only external symbol; no host history input."""
import numpy as np
from plasticity.model import Reference

N=256;U=8;SEEDS=[112101,112201,112301,112401,112501]
ARMS=['native','frozen','unlabelled','severed'];STAGES=[0,1,2,8]

def design(seed):
    rng=np.random.default_rng(seed)
    history=rng.permutation(128);current=128+rng.permutation(128)
    codes=np.zeros((U,N),'u1')
    for j in range(U):codes[j,history[16*j:16*(j+1)]]=1
    cues=np.zeros((6,N),'u1')
    for j in range(4):cues[j,current[:8]]=1;cues[j,current[8+24*j:32+24*j]]=1
    cues[4,current[104:120]]=1;cues[5]=cues[4]
    cues[5,rng.choice(current[104:120],4,False)]=0
    old=np.arange(4,8,dtype='<i4');changed=np.sort(rng.choice(4,2,False)).astype('<i4')
    new=old.copy();new[changed]=new[changed[::-1]]
    w=np.zeros((N,U),'i1')
    for j in range(4):w[:,j]=cues[j];w[:,old[j]]=codes[j]+cues[4]
    assert np.all(w.sum(0)==32) and np.all(codes[:,128:]==0)
    return codes,cues,w,old,new,changed

def schedule(seed,epochs):
    _,_,_,old,new,changed=design(seed);rng=np.random.default_rng(seed+31);events=[]
    def pair(context,phase,epoch,variant='full',teacher=False):
        truth=new if epoch>=0 and epoch%2==0 else old
        for cue in [int(context),5 if variant!='full' else 4]:
            is_query=cue>=4
            events.append(dict(index=len(events),source=len(events)-1,cue=cue,context=int(context),
                phase=phase,epoch=epoch,variant=variant,label=int(truth[context]) if teacher and is_query else -1,
                target=int(truth[context]) if is_query else int(context),
                delay_ms=(1 if variant=='short' else 20 if variant=='long' else 0) if is_query else 0))
    for c in rng.permutation(4):pair(c,'old_query',-1)
    for epoch in range(epochs):
        for repeat in range(2):
            for c in rng.permutation(changed):pair(c,'teacher',epoch,teacher=True)
        for variant in ['full','short','long']:
            for c in rng.permutation(4):pair(c,'probe',epoch,variant)
    return events

def simulate(initial,seed,arm,epochs):
    codes,cues,w,old,new,changed=design(seed);ref=Reference(initial,.75)
    events=schedule(seed,epochs);xs=[];ys=[];states=[];potentials=[];last=np.zeros(N,'u1')
    task=[];stateless=[];history=-1;table=old.copy()
    for e in events:
        x=last.copy();assert np.all(x[128:]==0)
        x[128:]=cues[e['cue'],128:]
        sever=arm=='severed' and e['phase']=='probe' and e['cue']>=4
        if sever:x[:128]=0
        label=e['label'] if arm in ['native','severed'] else -1
        p=ref.forward(x,label) if label>=0 else ref.infer(x)
        active=np.flatnonzero(np.asarray(p)>24)
        last=np.maximum.reduce(codes[active],axis=0) if len(active) else np.zeros(N,'u1')
        xs.append(x);ys.append(last);states.append(ref.ram[:]);potentials.append(p)
        if e['cue']<4:history=e['cue'];ty=history;sy=history
        else:
            assert history>=0;ty=int(table[history]);sy=4
            if label>=0:table[history]=label
        task.append(codes[ty]);stateless.append(codes[sy])
    return dict(events=events,inputs=np.asarray(xs,'u1'),outputs=np.asarray(ys,'u1'),
        states=np.asarray(states,'<u4'),potentials=np.asarray(potentials,'<i4'),
        task_outputs=np.asarray(task,'u1'),stateless_outputs=np.asarray(stateless,'u1'))
