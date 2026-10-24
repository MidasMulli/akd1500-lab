"""Frozen multi-stage native learning task; teachers stay outside input features."""
import numpy as np
from pong.relearn.model import Reference

N,U,NW,C,T=512,64,32,4,27
COUNTS=[1,2,4,6]
SEEDS=[99101,99102,99103]
ARMS=['repair','frozen','wrong_bank']


def initial(seed,count):
    rng=np.random.default_rng(seed);result=[]
    for k in range(count):
        ram=[0]*(2+U*20)
        for j in range(U):
            ram[2+j*20+3]=16
            for a in rng.choice(N,NW,replace=False):ram[2+j*20+4+int(a)//32]|=1<<(int(a)%32)
        result.append(ram)
    return result


def permutations(count):return [np.arange(4)]+[np.random.default_rng(99300+k).permutation(4) for k in range(1,count)]


def target(identity,count):
    for p in permutations(count):identity=int(p[identity])
    return identity


def schedule(count,arm,smoke=False):
    assert count in COUNTS and arm in ARMS
    rng=np.random.default_rng(99201);signatures=[[rng.choice(N,N//2,replace=False) for c in range(C)] for domain in range(2)]
    xs,rows=[],[]
    def add(phase,domain,per_class,seed,mask,depth):
        r=np.random.default_rng(seed);order=np.repeat(np.arange(4),per_class);r.shuffle(order)
        for identity in order:
            a=np.zeros(N,np.uint8);a[signatures[domain][int(identity)]]=1
            remove=r.choice(np.flatnonzero(a),8,replace=False);insert=r.choice(np.flatnonzero(1-a),8,replace=False);a[remove]=0;a[insert]=1
            xs.append(a);rows.append(dict(index=len(rows),phase=phase,identity=int(identity),destination=target(int(identity),count),
                label=target(int(identity),depth),mask=mask,domain=domain))
    for k in range(count):add(f'train{k}',0,32,99400+k,1<<k,k+1)
    for phase,domain,seed in [('old',0,99501),('changed_before',1,99502)]:add(phase,domain,2 if smoke else 16,seed,0,count)
    mask={'repair':1,'frozen':0,'wrong_bank':1<<(count-1)}[arm]
    add('repair',1,4 if smoke else 32,99503,mask,count if arm=='wrong_bank' else 1)
    for phase,domain,seed in [('changed_after',1,99504),('retention',0,99505)]:add(phase,domain,2 if smoke else 16,seed,0,count)
    return np.asarray(xs),rows


class Network:
    def __init__(self,ram):self.banks=[Reference(r,N,U,NW,C) for r in ram]
    def forward(self,x,label,mask):
        details=[]
        for k,bank in enumerate(self.banks):
            bits=sum(1<<int(i) for i in np.flatnonzero(x))
            scores=bank.forward(x,label) if mask&(1<<k) else [(w&bits).bit_count() for w in bank.weights]
            support=(np.asarray(scores)>T).reshape(C,16).sum(axis=1);winner=int(support.argmax());details.append(winner)
            x=np.zeros(N,np.uint8);x[winner*128:(winner+1)*128]=1
        return support.tolist(),details


def replay(seed,count,arm,smoke=False):
    net=Network(initial(seed,count));x,rows=schedule(count,arm,smoke);outputs=[];frozen_start=None
    for a,row in zip(x,rows):
        if row['phase']=='old' and frozen_start is None:frozen_start=[b.ram[:] for b in net.banks]
        y,_=net.forward(a,row['label'],row['mask']);outputs.append(y)
    pred=np.asarray(outputs).argmax(axis=1)
    metrics={phase:dict(correct=sum(pred[i]==r['destination'] for i,r in enumerate(rows) if r['phase']==phase),
        count=sum(r['phase']==phase for r in rows)) for phase in ['old','changed_before','changed_after','retention']}
    for m in metrics.values():m['correct']=int(m['correct']);m['accuracy']=m['correct']/m['count']
    protected=[k for k in range(count) if k!=(count-1 if arm=='wrong_bank' else 0)]
    return dict(seed=seed,banks=count,arm=arm,smoke=smoke,metrics=metrics,
        protected_banks_unchanged=all(net.banks[k].ram==frozen_start[k] for k in protected),
        outputs=outputs)


def gate(candidate,frozen):
    a,b=candidate['metrics'],frozen['metrics']
    checks=dict(old=a['old']['accuracy']>=.85,changed=a['changed_after']['accuracy']>=.85,
        gain=a['changed_after']['accuracy']-b['changed_after']['accuracy']>=.30,
        retention=a['retention']['accuracy']>=.80,protected_memory=candidate['protected_banks_unchanged'])
    return dict(status='PASS' if all(checks.values()) else 'STOP_TASK_GATE',checks=checks)
