"""Outcome-grounded native admission policy; no fixed source is always trusted.

Reuses the qualified transfer/revision primitives. The first plastic bank learns
source/context reliability. Its response gates constant-label student learning.
Feedback is externally grounded; host role clocks and the objective are designed.
"""
import itertools
import numpy as np
from admission import transfer as base
from admission.recycle import Reference

N, D, KEYS = base.N, base.D, base.KEYS
PT = list(range(8)) + list(range(32,40))
HT = [j for j in range(64) if j not in PT]
PS, PR, HS, HR = base.PS, base.PR, base.HS, base.HR
DIMENSIONS = [(64,2,PT,HT), *base.DIMENSIONS[1:]]
SEEDS = [151101, 151201, 151301]
ARMS = ['native', 'manager_frozen', 'open', 'students_frozen', 'cut']
STAGES = ['support', 'pilot', 'sealed']
ROLES = ['query', 'feedback', 'transfer', 'exam', 'feedback_negative']
TRUTH = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]


def matrices(arm):
    assert arm in ARMS
    mats = base.matrices('native')
    enc = np.zeros((N, D), 'i1'); et = np.zeros(D, 'i4')
    enc[[64, 65, 69], :25] = 1; enc[70, :25] = -1
    for key in range(KEYS):
        for bit in range(4):
            # Management identity is independent of the student's fact identity.
            policy_bit = 96 + bit*8 + ((key >> bit) & 1)*4
            memory_bit = policy_bit - 64
            enc[policy_bit, 25+key*7:32+key*7] = 1
            enc[memory_bit, 128+key*7:135+key*7] = 1
        et[25+key*7:32+key*7] = 3
        enc[70, 25+key*7:32+key*7] = -1
        enc[68, 31+key*7] = -1
        et[128+key*7:135+key*7] = 3
        enc[65,128+key*7:135+key*7] = -1
    enc[65, 109] = enc[68, 109] = 1; et[109] = 1
    enc[64, 256:288] = 1; enc[69, 288:320] = 1
    enc[69, 320:352] = enc[88, 320:352] = 1; et[320:352] = 1
    for j in range(8):
        enc[69, 352+j*4:356+j*4] = enc[PR[j], 352+j*4:356+j*4] = 1
        et[352+j*4:356+j*4] = 1
    enc[69, 384:416] = enc[71, 384:416] = 1; et[384:416] = 1
    enc[72, 416:448] = 1; enc[73, 448:480] = 1
    enc[65,416:480] = -1
    cmp = np.zeros((64, 64), 'i1'); ct = np.full(64, 127, 'i4')
    for key in range(KEYS): cmp[HT[key], key] = 1; ct[key] = 0
    cmp[HT[12], 32] = 1; ct[32] = 0
    for val, group in enumerate([PT[:8], PT[8:]]):
        cmp[group, 28+val] = 1; ct[28+val] = 0
    # Echo the learned permit and candidate separately through read-only helpers.
    for val in range(2):
        cmp[HT[16+val], 24+val] = 1; ct[24+val] = 0
        for col in [16+2*val, 17+2*val]:
            cmp[HT[13], col] = cmp[HT[16+val], col] = 1
            if arm != 'open':
                cmp[PT[8:], col] = 1; cmp[PT[:8], col] = -1
            if arm == 'cut': cmp[HT[18], col] = -1
        prev, other = HT[14+val], HT[15-val]
        cmp[prev, 16+2*val] = -1
        cmp[[prev, other], 17+2*val] = 1
        ct[16+2*val] = 1 if arm == 'open' else 2
        ct[17+2*val] = 3 if arm == 'open' else 4
    # Previous memory responses are available during transfer, never host-decoded.
    mats[0] = enc, et; mats[1] = cmp, ct
    code, tt = mats[2]
    for c in range(2):
        code[32, 224+c*32:256+c*32] = code[28+c, 224+c*32:256+c*32] = 1
        tt[224+c*32:256+c*32] = 1
    bridge, bt = mats[3]
    for c in range(2): bridge[HS[17+c], 128+c*32:160+c*32] = 1
    return mats


def weights(seed, k):
    w = base.weights(seed, k)
    if k == 0:
        w = np.zeros((D,64),'i1'); rng=np.random.default_rng(seed)
        for j in PT:
            w[:25,j]=1; w[rng.choice(np.arange(25,109),7,False),j]=1
        for j in HT: w[D-32:,j]=1
        for key in range(KEYS):
            w[:, HT[key]] = 0
            w[128+key*7:135+key*7, HT[key]] = 1
            w[D-25:, HT[key]] = 1
        for j, start in [(HT[16], 416), (HT[17], 448), (HT[18], 384)]:
            w[:, j] = 0; w[start:start+32, j] = 1
        for j,start in zip(HT[12:16],[256,288,320,352]):
            w[:,j]=0; w[start:start+32,j]=1
    elif k == 1:
        for c in range(2):
            w[:, HS[17+c]] = 0; w[224+c*32:256+c*32, HS[17+c]] = 1
    else:
        for c, j in enumerate([84, 85]):
            w[:, j] = 0; w[128+c*32:160+c*32, j] = 1
    assert np.all(w.sum(0) == 32)
    return w


def thresholds(k):
    if k: return base.thresholds(k)
    lt=np.full(64,32,'i4'); lt[PT]=0
    t=np.full(64,31,'i4'); t[HT[:12]]=6; t[HT[15]]=3
    return lt,t


def packed(w,k):
    lt,_=thresholds(k); ram=[0,0]
    for j in range(w.shape[1]):
        ram.extend([0,0,0,(int(lt[j])<<16)|16])
        ram.extend(sum(int(w[a*32+b,j])<<b for b in range(32)) for a in range(D//32))
    return ram


def observation(last, e):
    x = last.copy(); role = e['role']
    for start, value in [(96, e['policy_key']), (32, e['key'])]:
        if start == 96 or role != 'transfer':
            x[start:start+32] = 0
            for bit in range(4):
                j = start + bit*8 + ((value >> bit) & 1)*4; x[j:j+4] = 1
    x[72:76] = 0; x[72+e['candidate']] = 1
    x[64:72] = 0
    x[65 if role in ['feedback', 'feedback_negative'] else 69 if role == 'transfer' else 64] = 1
    x[68] = role == 'feedback_negative'; x[70] = role == 'exam'; x[71] = e['block']
    return x


def schedule(stage, arm, orientation=0):
    assert stage in STAGES and arm in ARMS and orientation in [0, 1]
    rows=[]; truth=TRUTH.copy(); reliable=[1^orientation, orientation, orientation, 1^orientation]
    def add(key, policy, candidate, role, phase, block, outcome=None):
        good = int(candidate == outcome) if outcome is not None else -1
        wire = (1-good if role == 'feedback_negative' else good) if role.startswith('feedback') else 0
        labels=[wire, -1, -1] if role.startswith('feedback') else [-1, 0, 0] if role == 'transfer' else [-1, -1, -1]
        if block and arm == 'manager_frozen': labels[0] = -1
        if block and arm == 'students_frozen': labels[1:] = [-1, -1]
        rows.append(dict(index=len(rows),key=key,policy_key=policy,candidate=candidate,role=role,
                         phase=phase,block=block,actual=truth[key],outcome=outcome,quality=good,
                         expected_permit=reliable[policy] if policy<4 else 0,wire=wire,labels=labels))
    def feedback(policies, phase, block):
        # Both candidate values occur for every identity; outcome arrives as feedback.
        for rep in range(3):
            for p in policies:
                cand=(p+rep)&1; outcome=cand if reliable[p] else 1-cand
                add(8+p, p, cand, 'feedback', phase, block, outcome)
                if block: add(8+p, p, cand, 'feedback_negative', phase, block, outcome)
    def propose(key, policy, candidate, phase, block, reps=3):
        for _ in range(reps):
            add(key, policy, candidate, 'query', phase, block)
            add(key, policy, candidate, 'transfer', phase, block)
    def exam(keys, phase, block, repeat=8):
        order=np.tile(keys,repeat); np.random.default_rng(152000+len(rows)+orientation).shuffle(order)
        for key in order.tolist(): add(key, 0, 0, 'exam', phase, block)
    feedback(range(4),'bootstrap_feedback',0)
    for key in range(4): propose(key, reliable.index(1), truth[key], 'bootstrap_memory',0)
    exam(range(4),'bootstrap_exam',0)
    for p in range(4):
        for key in range(4): add(key,p,truth[key],'query','policy_support',0)
    if stage == 'support': return rows
    cycles=1 if stage=='pilot' else 4
    for cycle in range(1,cycles+1):
        suffix=f'_{cycle}'
        # Proposals without new outcomes must not silently teach the manager.
        bad=reliable.index(0)
        for key in range(4): propose(key,bad,1-truth[key],'bad_proposals'+suffix,1)
        exam(range(4),'protected_exam'+suffix,1)
        reliable[0]^=1; reliable[1]^=1
        feedback([0,1],'changed_feedback'+suffix,1)
        for p in range(4):
            for key in range(4): add(key,p,truth[key],'query','changed_policy_probe'+suffix,1)
        truth[0]^=1; truth[1]^=1
        good=0 if reliable[0] else 1
        for key in range(4): propose(key,good,truth[key],'legitimate_proposals'+suffix,1)
        exam(range(4),'updated_exam'+suffix,1)
        for key in [0,1]: propose(key,4,1-truth[key],'unknown_source'+suffix,1)
        exam(range(4),'unknown_protected_exam'+suffix,1)
    if stage=='sealed':
        good=0 if reliable[0] else 1
        for key in [4,5]: propose(key,good,truth[key],'new_memory',1)
        exam(range(6),'new_exam',1,16)
    return rows


def cpu_controls(rows):
    """Task-level online controls see feedback only, never future reliability/truth."""
    policy={}; frozen_policy={}; scores={}; memory={}; results=[]
    names=['adaptive_dictionary','adaptive_ema','fixed_prefix_policy','always_accept','never_after_prefix']
    memories={name:{} for name in names}; initialized=False
    for e in rows:
        if e['block'] and not initialized: frozen_policy=policy.copy(); initialized=True
        p=e['policy_key']; cand=e['candidate']; role=e['role']; key=e['key']
        if role=='feedback':
            observed=int(cand==e['outcome']); policy[p]=observed
            scores[p]=.25*scores.get(p,.5)+.75*observed
        decisions=[policy.get(p,0),int(scores.get(p,0)>.5),frozen_policy.get(p,policy.get(p,0)) if e['block'] else policy.get(p,0),1,int(not e['block'])]
        if role=='transfer':
            for name, permit in zip(names,decisions):
                if permit: memories[name][key]=cand
        results.append([memories[name].get(key,-1) for name in names])
    return names,np.asarray(results,'<i4')


def simulate(rams,stage,arm,orientation=0):
    mats=matrices(arm); refs=[Reference(rams[k],n=D,u=u,nw=32,classes=c) for k,(u,c,_,_) in enumerate(DIMENSIONS)]
    rows=schedule(stage,arm,orientation); last=np.zeros(N,'u1'); xs=[]; ys=[]
    states=[[],[],[]]; ps=[[],[],[]]; fs=[[],[],[]]
    for e in rows:
        x=observation(last,e); z=(x.astype('i4')@mats[0][0].astype('i4')>mats[0][1]).astype('u1')
        for k in range(3):
            fs[k].append(z.copy()); p=np.asarray(refs[k].forward(z,e['labels'][k]),'<i4')
            z=(p>thresholds(k)[1]).astype('u1'); states[k].append(refs[k].ram[:]); ps[k].append(p)
            if k<2:
                for w,t in mats[1:3] if k==0 else mats[3:4]: z=(z.astype('i4')@w.astype('i4')>t).astype('u1')
        if e['role']!='transfer': last=z.copy()
        xs.append(x);ys.append(z)
    names,cpu=cpu_controls(rows)
    a=dict(events=rows,inputs=np.asarray(xs,'u1'),outputs=np.asarray(ys,'u1'),cpu=cpu,cpu_names=names)
    for k in range(3):
        a[f'states{k}']=np.asarray(states[k],'<u4');a[f'potentials{k}']=np.asarray(ps[k],'<i4');a[f'features{k}']=np.asarray(fs[k],'u1')
    return a


def assess(a):
    out={}
    for phase in dict.fromkeys(e['phase'] for e in a['events']):
        ii=[e['index'] for e in a['events'] if e['phase']==phase and e['role'] in ['exam','query']]
        if not ii: continue
        y=a['outputs'][ii]; pred=np.stack([y[:,88],y[:,PR].any(1)],1)
        actual=np.asarray([a['events'][i]['actual'] for i in ii]); permit=y[:,84:86]
        expected=np.asarray([a['events'][i]['expected_permit'] for i in ii])
        out[phase]=dict(queries=len(ii),memory_correct=sum(int(v[t] and not v[1-t]) for v,t in zip(pred,actual)),
            policy_correct=sum(int(v[t] and not v[1-t]) for v,t in zip(permit,expected)),
            ambiguous=int(pred.all(1).sum()),missing=int((~pred.any(1)).sum()),
            stable_correct=sum(int(pred[j,actual[j]] and not pred[j,1-actual[j]]) for j,i in enumerate(ii) if a['events'][i]['key'] in [2,3]),
            stable_queries=sum(a['events'][i]['key'] in [2,3] for i in ii),
            cpu={name:int((a['cpu'][ii,j]==actual).sum()) for j,name in enumerate(a['cpu_names'])})
    return out


def predicate_screen():
    w,t=matrices('native')[1]; count=0
    for deny,allow,candidate,prev0,prev1,teach in itertools.product(range(2),repeat=6):
        v=np.zeros(64,'i4');v[PT[0]]=deny;v[PT[8]]=allow;v[HT[16+candidate]]=1
        v[HT[14]]=prev0;v[HT[15]]=prev1;v[HT[13]]=teach
        got=(v@w>t)[16:20];want=np.zeros(4,bool)
        if allow and not deny and teach:
            prev=[prev0,prev1]
            if not prev[candidate]:want[2*candidate]=1
            elif prev0 and prev1:want[2*candidate+1]=1
        assert np.array_equal(got,want),(deny,allow,candidate,prev0,prev1,teach,got,want)
        count+=1
    return count


if __name__=='__main__':
    import json
    predicate_screen()
    for arm in ARMS:
        a=simulate([packed(weights(SEEDS[0],k),k) for k in range(3)],'pilot',arm)
        print(json.dumps(dict(arm=arm,assessment=assess(a))),flush=True)
