"""Three native plastic populations with class-presence OR communication."""
import numpy as np
from resident.design import schedule as parent_schedule,target
from pong.relearn.model import Reference
from resident.run import P
S=P/'studies/20261023-fnp-or-relay'
WIDTHS=[64,256,512];SEEDS=[102101,102102,102103];ARMS=['repair','frozen','wrong_bank']


def schedule(arm,smoke=False):
    x,rows=parent_schedule(4,arm,smoke);keep=[i for i,r in enumerate(rows) if r['phase']!='train3'];result=[]
    for i in keep:
        r=dict(rows[i]);r['index']=len(result);r['destination']=target(r['identity'],3)
        if r['phase'].startswith('train'):depth=int(r['phase'][5:])+1;r['label']=target(r['identity'],depth)
        elif r['phase']=='repair':r['label']=target(r['identity'],3 if arm=='wrong_bank' else 1);r['mask']={'repair':1,'frozen':0,'wrong_bank':4}[arm]
        else:r['label']=r['destination']
        result.append(r)
    return x[keep],result


def initial(seed,width):
    rng=np.random.default_rng(seed);banks=[]
    for k in range(3):
        ram=[0]*(2+width*20)
        for j in range(width):
            ram[2+j*20+3]=16
            for bit in rng.choice(512,32,replace=False):ram[2+j*20+4+int(bit)//32]|=1<<(int(bit)%32)
        banks.append(ram)
    return banks


class Network:
    def __init__(self,ram,width):self.banks=[Reference(r,512,width,32,4) for r in ram];self.width=width
    def forward(self,x,label,mask):
        details=[]
        for k,b in enumerate(self.banks):
            bits=sum(1<<int(j) for j in np.flatnonzero(x))
            p=b.forward(x,label) if mask&(1<<k) else [(w&bits).bit_count() for w in b.weights]
            supports=(np.asarray(p)>27).reshape(4,self.width//4).sum(1);presence=supports>0;details.append(dict(supports=supports.tolist(),presence=presence.tolist()))
            x=np.repeat(presence,128).astype(np.uint8)
        return supports.tolist(),details


def replay(seed,width,arm,smoke=False):
    n=Network(initial(seed,width),width);x,rows=schedule(arm,smoke);ys=[];protected_start=None
    for a,r in zip(x,rows):
        if r['phase']=='old' and protected_start is None:protected_start=[b.ram[:] for b in n.banks]
        y,_=n.forward(a,r['label'],r['mask']);ys.append(y)
    pred=np.asarray(ys).argmax(1);metrics={}
    for phase in ['old','changed_before','changed_after','retention']:
        ii=[i for i,r in enumerate(rows) if r['phase']==phase];correct=int(sum(pred[i]==rows[i]['destination'] for i in ii));metrics[phase]=dict(correct=correct,count=len(ii),accuracy=correct/len(ii))
    protected=[k for k in range(3) if k!=(2 if arm=='wrong_bank' else 0)]
    return dict(seed=seed,width=width,arm=arm,metrics=metrics,protected_banks_unchanged=all(n.banks[k].ram==protected_start[k] for k in protected))


if __name__=='__main__':
    import json
    from resident.design import gate
    results=[]
    for width in WIDTHS:
        arms={a:replay(SEEDS[0],width,a) for a in ARMS};g=gate(arms['repair'],arms['frozen']);results.append(dict(width=width,gate=g,arms=arms));print(json.dumps(results[-1]),flush=True)
        (S/'evidence/CPU_SCREEN.json').write_text(json.dumps(dict(status='CPU_FORECAST_ONLY',results=results),indent=2)+'\n')
