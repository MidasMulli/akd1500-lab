"""Software-only feasibility rehearsal. Does not import SDK or access hardware."""
import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent/'source')]
from core import Layout,initial
from pong.relearn.model import Reference
B=Path(__file__).resolve().parent;spec=json.loads((B/'SPEC_FROZEN.json').read_text());g=Layout('oct9');calls=0;started=time.monotonic()
def forward(r,x,label,cap):
    global calls
    assert time.monotonic()-started<spec['timeout_seconds'] and calls<spec['max_CPU_reference_forwards']
    before=r.weights.copy();scores=r.forward(x,label);calls+=1
    for j,(a,b)in enumerate(zip(before,r.weights)):
        if a!=b:
            k=2+j*r.stride;r.ram[k+3]=(min(int(scores[j]),cap)<<16)|(r.ram[k+3]&65535)
def cue(rng):
    k=g.key(rng.integers(0,g.bins,g.d),int(rng.integers(0,6)))
    return g.value_key(k,int(rng.integers(0,3)))
def physical_input(k):
    x=np.zeros(g.n1,'u1');x[4*np.flatnonzero(k)]=1;return x
def recall(r,x):
    bits=sum(1<<int(i)for i in np.flatnonzero(x));return [int((w&bits).bit_count()>31)for w in r.weights]
def unit_ram(r,j):return r.ram[2+j*r.stride:2+(j+1)*r.stride]
all_results=[]
for seed in spec['seeds']:
    rng=np.random.default_rng(seed);weights,_,lts=g.parameters(seed);r=Reference(initial(weights[1],lts[1]),g.n1,32,32,32)
    current=[cue(rng)for _ in range(32)]
    for j,k in enumerate(current):
        for _ in range(8):forward(r,physical_input(k),j,30)
    assert all(recall(r,physical_input(k))==[int(i==j)for i in range(32)]for j,k in enumerate(current))
    initial_model=copy.deepcopy(r)
    new=[cue(rng)for _ in range(64)]
    assert len({x.tobytes()for x in current+new})==96
    wildcard=np.zeros(g.n1,'u1');wildcard[::4]=1;wildcard[4*g.canon]=0
    for arm in spec['arms']:
        m=copy.deepcopy(initial_model);keys=[x.copy()for x in current];correct=protected=0;hidden_changes=0
        for n,k in enumerate(new):
            victim=n%32;before=[unit_ram(m,j).copy()for j in range(32)];h0=(m.rot,m.cursor)
            if arm=='wildcard-retire':
                for clock_class in range(32):forward(m,wildcard if clock_class==victim else np.zeros(g.n1,'u1'),clock_class,1)
            for _ in range(8):
                for clock_class in range(32):forward(m,physical_input(k)if arm!='permission-blocked'and clock_class==victim else np.zeros(g.n1,'u1'),clock_class,30)
            hidden_changes+=int((m.rot,m.cursor)!=h0)
            ok=all(unit_ram(m,j)==before[j]for j in range(32)if j!=victim);protected+=int(ok);assert ok
            predicted=recall(m,physical_input(k));correct+=int(predicted==[int(i==victim)for i in range(32)])
            if arm=='wildcard-retire':
                assert predicted==[int(i==victim)for i in range(32)];keys[victim]=k.copy()
                assert all(recall(m,physical_input(old))==[int(i==j)for i in range(32)]for j,old in enumerate(keys))
            if arm=='permission-blocked':assert all(unit_ram(m,j)==before[j]for j in range(32))
        all_results.append(dict(seed=seed,arm=arm,replacements=len(new),new_exact_recall=correct,all31_other_record_RAM_protected=protected,global_selector_state_changed_rounds=hidden_changes))
result=dict(status='PASS_SOFTWARE_FEASIBILITY_ONLY',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reference_sha256=hashlib.sha256((ROOT/'pong/relearn/model.py').read_bytes()).hexdigest(),CPU_reference_forwards=calls,physical=False,SDK_compiler_qualified=False,native_selector_implemented=False,results=all_results,limits=spec['limits'])
(B/'REPLAY_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
