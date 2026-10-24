"""Pixel-only observations; no simulator state or future outcome in the encoder."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'studies/20261024-streaming-prediction'
N,U,NW,CLASSES,QUERY=1024,160,64,5,4
SEEDS=[105101,105102,105103]
ROLE='T202'
CHECKPOINT_SHA='3342f73cf939f689abfb1189faf3580719acbf05ca2d439001ee13887ea5bc65'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def locate(image,strict=True):
    a=np.asarray(image)
    assert a.ndim==3 and a.shape[2]==3
    mask=(a[:,:,0]>180)&(a[:,:,1]<80)&(a[:,:,2]<80)
    yy,xx=np.where(mask)
    if len(xx)<4:
        if strict:raise ValueError('disk unresolved by declared pixel tracker')
        return None
    return dict(x=float((xx+.5).mean()/a.shape[1]),y=float((yy+.5).mean()/a.shape[0]),pixels=len(xx))

def band(image,strict=True):
    p=locate(image,strict)
    return -1 if p is None else min(3,int(p['y']*4))

def observation(previous,current,action,context):
    a,b=locate(previous),locate(current)
    # Context is the visible wall palette, independently checked from pixels.
    leaf=np.asarray(current)[20,112]
    inferred=0 if np.max(leaf)<100 else 1
    assert inferred==context,('visible context mismatch',leaf.tolist(),context)
    return dict(y=b['y'],x=b['x'],vy=float(np.clip((b['y']-a['y'])*8,-1,1)),
                action_y=float(action[1]),context=int(context))

def encode(o):
    x=np.zeros(N,np.uint8);base=int(o['context'])*512
    values=[o['y'],(o['vy']+1)/2,(o['action_y']+1)/2,o['x']]
    for j,v in enumerate(values):
        start=int(np.clip(round(float(v)*112),0,112));x[base+j*128+start:base+j*128+start+16]=1
    assert int(x.sum())==64
    return x

def decode(scores):
    s=np.asarray(scores).reshape(5,32)[:4].max(1)
    order=np.sort(s)
    return int(s.argmax()),int(order[-1]),int(order[-1]-order[-2])

def episodes(smoke=False,probe=False):
    if probe:
        return [dict(phase='probe',context=c,episode=j,scene=202610249100000+j,command_seed=106000+j,
                     reversed=False,teach=False) for j,c in enumerate([0,1,0,1])]
    if smoke:
        return [dict(phase='smoke',context=0,episode=0,scene=202610249100010,command_seed=106010,reversed=False,teach=True)]
    rows=[]
    def add(phase,count,offset,contexts,changed,teach):
        for j in range(count):
            c=contexts[j%len(contexts)]
            rows.append(dict(phase=phase,context=c,episode=len(rows),scene=202610249200000+offset+j,
                             command_seed=107000+offset+j,reversed=bool(changed and c==0),teach=teach))
    add('prefix',24,0,[0,1],False,True)
    add('baseline',8,100,[0,1],False,False)
    add('changed_before',8,200,[0],True,False)
    add('adapt',24,300,[0],True,True)
    add('changed_after',12,400,[0],True,False)
    add('retention',12,500,[1],False,False)
    return rows

def commands(seed):
    rng=np.random.default_rng(seed)
    # Eight five-step command blocks. No target/outcome based action selection.
    return [[.15,float(v)] for v in rng.choice([-1.,-.5,.5,1.],8)]

def plan():
    return dict(version=1,role=ROLE,checkpoint_sha256=CHECKPOINT_SHA,seeds=SEEDS,
        inputs=N,neuron_records=U,connections_per_record=NW,classes=CLASSES,
        query_teacher=QUERY,forecast_horizon_steps=5,
        representation='64 active bits in 1024; current/prior pixel disk position, observed displacement, command and visible context; no future or simulator state',
        target='four equal vertical bands of the disk in the subsequent rendered RGB image',
        drift='Disclosed host actuator routing reversal for context A only; not a change to world-model weights or underlying simulator law',
        model='Frozen T202 Nano CUDA BF16 recurrent RGB world model; one requested five-step action rollout and decode per prediction; not CEM/planner',
        learning='Query first with protected teacher4. Advance the stream only after the query record is flushed. Teach the saved past input after the future image arrives. Host derives teacher from pixels; native engine selects connections.',
        controls=['identical native prefix then frozen weights; feedback slots remain protected queries',
                  'exact native-rule compiled CPU and independent Python reconstruction',
                  '64-entry per-context nearest-neighbor CPU with the same delayed labels',
                  'pixel persistence and observed-velocity extrapolation; frozen Nano model band'],
        stages=['model/pixel-interface probe; no AKD access','compiler/CPU smoke','one native smoke',
                'one matched native learning/frozen pair and independent audit','only after pilot gate, two further initialization seeds',
                'only after every sealed gate, consider a separately recorded skipped-forecast arm'],
        gates=dict(minimum_class_count=2,minimum_baseline_accuracy=.65,minimum_changed_accuracy=.75,
                   minimum_gain_over_frozen=.10,minimum_gain_over_persistence=.05,maximum_retention_loss=.05,
                   skip_accuracy_min=.95,skip_margin_min=8,skip_score_min=56,skip_min_eligible=32,skip_error_max=.02),
        skipping_initially_enabled=False,
        limits=['one chip; initialization seeds are not independent hardware','first native graph one plastic FNP; no full-chip or collective-network claim',
                'disclosed pixel tracker and host feedback, not pixel learning or self-selected teaching',
                'bounded band prediction cannot replace full RGB forecasts, event scores or planning',
                'native RAM is not power-loss persistence','no electrical spike trace or energy advantage inferred'])
