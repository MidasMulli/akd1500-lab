"""Sealed input-driven plasticity task; no data-dependent teacher classes/modes.

Context10 + outcome6 + feedback-gate16 =32 active bits. A prediction probe
omits the gate and can match at most16, below learning threshold24. Outcomes
permit a native update; CPU does not select a learning record or weight edit.
"""
from pathlib import Path
import json
import numpy as np
from pong.relearn.model import Reference

P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-predictive-feedback'
N,U=512,16;SEEDS=[108101,108201,108301];GATE=np.arange(160,176);OUT=[np.arange(176,182),np.arange(182,188)]
def initial(seed):
    r=np.random.default_rng(seed);w=np.zeros((N,U),np.int8)
    for j in range(U):
        w[j*10:j*10+10,j]=1;w[GATE,j]=1
        w[r.choice(OUT[0],3,False),j]=1;w[r.choice(OUT[1],3,False),j]=1
    return w
def event(context,outcome,feedback):
    x=np.zeros(N,np.uint8);x[context*10:context*10+10]=1;x[OUT[outcome]]=1
    if feedback:x[GATE]=1
    return x
def schedule(seed,arm):
    r=np.random.default_rng(seed+1);truth=r.integers(0,2,U);changed=r.choice(U,2,False);next_truth=truth.copy();next_truth[changed]^=1
    xs=[];rows=[]
    def append(c,o,teach,phase,target):
        i=len(rows);gate=teach and not(arm=='disconnected' and i>=128)
        xs.append(event(c,o,gate));rows.append(dict(index=i,context=int(c),candidate=int(o),feedback=bool(teach),gate_present=bool(gate),phase=phase,target=int(target),constant_teacher_class=0))
    def transition(c,t,phase,feedback):
        append(c,0,False,phase,t);append(c,1,False,phase,t)
        if feedback:append(c,t,True,phase,t)
    for _ in range(2):
        for c in r.permutation(U):transition(int(c),int(truth[c]),'initial_learning',True)
    for c in r.permutation(U):transition(int(c),int(truth[c]),'baseline_query',False)
    assert len(rows)==128
    for _ in range(4):
        for c in r.permutation(U):transition(int(c),int(next_truth[c]),'changed_rule',True)
    for c in r.permutation(U):transition(int(c),int(next_truth[c]),'post_adaptation',False)
    for c in r.permutation(U):transition(int(c),int(next_truth[c]),'retention_query',False)
    assert len(rows)==384
    # New temporal trajectories: history is the last4binary events. The sparse
    # encoder is fixed; it does not expose the rule or future outcome in a query.
    for start in r.permutation(U):
        context=int(start)
        for step in range(12):
            target=int(next_truth[context]);transition(context,target,'sealed_trajectory_query',False)
            context=((context<<1)|target)&15
    assert len(rows)==768
    return np.asarray(xs),rows,dict(initial_truth=truth.tolist(),changed_truth=next_truth.tolist(),changed_contexts=sorted(map(int,changed)),seeds=SEEDS)
def simulate(ram,x,rows,arm):
    cpu=Reference(ram,n=N,u=U,nw=32,classes=1);ys=[];states=[]
    for i,(a,row) in enumerate(zip(x,rows)):
        learning=not(arm=='frozen' and i>=128)
        if learning:y=cpu.forward(a,0)
        else:
            bits=sum(1<<int(k) for k in np.flatnonzero(a));y=[(w&bits).bit_count() for w in cpu.weights]
        ys.append(y);states.append(cpu.ram[:])
    return np.asarray(ys,'<i4'),np.asarray(states,'<u4')
def metrics(outputs,rows,truth):
    correct={};pred=[]
    for i,row in enumerate(rows[:-1]):
        if row['feedback'] or row['candidate']!=0:continue
        if i+1>=len(outputs):break
        # Strict output gate, same for raw potentials and native fixed reducer.
        a,b=outputs[i],outputs[i+1]
        aa=int(np.any(a>13)) if np.ndim(a) else int(a)
        bb=int(np.any(b>13)) if np.ndim(b) else int(b)
        guess=0 if aa and not bb else 1 if bb and not aa else -1
        group=row['phase']+('_changed' if row['context'] in truth['changed_contexts'] else '_unchanged')
        correct.setdefault(group,[0,0]);correct[group][0]+=int(guess==row['target']);correct[group][1]+=1
        pred.append(dict(index=i,context=row['context'],guess=guess,target=row['target'],phase=row['phase']))
    return dict(groups={k:dict(correct=v[0],total=v[1]) for k,v in correct.items()},predictions=pred)
