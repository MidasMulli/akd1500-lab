"""Native learned teacher -> two valence-specific student memories.

The host supplies outcomes only while grounding the source teacher. During
transfer every wire label is zero; fixed native routes encode the teacher's
prediction as positive/negative features for two single-class student banks.
"""
import numpy as np
from admission.recycle import Reference,OUTCOME
N=128;D=512;KEYS=12
SEEDS=[118101,118201,118301]
ARMS=['native','teacher_cut','students_frozen']
STAGES=['gate','extended']
PT=list(range(4))+list(range(32,36));PS=list(range(8));PR=list(range(8))
HT=[j for j in range(64) if j not in PT];HS=[j for j in range(64) if j not in PS];HR=[j for j in range(128) if j not in PR]
DIMENSIONS=[(64,2,PT,HT),(64,2,PS,HS),(128,2,PR,HR)]

def matrices(arm):
    enc=np.zeros((N,D),'i1');et=np.zeros(D,'i4')
    enc[[64,65,69],:25]=1;enc[70,:25]=-1
    for key in range(KEYS):
        for bit in range(4):
            j=32+bit*8+((key>>bit)&1)*4
            enc[j,25+key*7:25+(key+1)*7]=1
            enc[j,128+key*7:128+(key+1)*7]=1
        enc[70,25+key*7:25+(key+1)*7]=-1;et[25+key*7:25+(key+1)*7]=3
        enc[70,128+key*7:128+(key+1)*7]=1;et[128+key*7:128+(key+1)*7]=4
    enc[64,256:288]=1;enc[69,288:320]=1
    enc[69,320:352]=enc[88,320:352]=1;et[320:352]=1
    for j in range(8):enc[69,352+j*4:356+j*4]=enc[PR[j],352+j*4:356+j*4]=1;et[352+j*4:356+j*4]=1
    cmp=np.zeros((64,64),'i1');ct=np.full(64,127,'i4')
    for k in range(KEYS):cmp[HT[k],k]=1;ct[k]=0
    cmp[HT[12],32]=1;ct[32]=0
    for val,own,other in [(0,PT[:4],PT[4:]),(1,PT[4:],PT[:4])]:
        cmp[own,24+val]=1;ct[24+val]=0
        prev=HT[14+val];opp=HT[15-val]
        for col in [16+val*2,17+val*2]:cmp[own,col]=1;cmp[other,col]=-1;cmp[HT[13],col]=1
        cmp[prev,16+val*2]=-1;ct[16+val*2]=1
        cmp[[prev,opp],17+val*2]=1;ct[17+val*2]=3
    if arm=='teacher_cut':cmp[:,16:20]=0;ct[16:20]=127
    code=np.zeros((64,D),'i1');tt=np.zeros(D,'i4')
    code[[32,16,17,18,19],:25]=1
    for key in range(KEYS):
        sl=slice(25+key*7,25+key*7+6);code[key,sl]=1;code[[32,16,17,18,19],sl]=1;tt[sl]=1
        j=25+key*7+6;code[key,j]=1;code[[32,16,17],j]=1;tt[j]=1
    code[[18,19],109]=1
    code[32,128:160]=1
    for val in range(2):code[32,160+val*32:192+val*32]=code[24+val,160+val*32:192+val*32]=1;tt[160+val*32:192+val*32]=1
    bridge=np.zeros((64,D),'i1');bt=np.zeros(D,'i4')
    common,neg,q,tc0,tc1=HS[12:17]
    bridge[common,:25]=1
    for key in range(KEYS):
        bridge[HS[key],25+key*7:25+key*7+6]=1
        j=25+key*7+6;bridge[HS[key],j]=bridge[q,j]=bridge[neg,j]=1;bt[j]=1
    bridge[common,109]=1;bridge[[q,neg],109]=-1
    for j in range(8):bridge[PS[j],256+j*4:260+j*4]=bridge[q,256+j*4:260+j*4]=1;bt[256+j*4:260+j*4]=1
    bridge[tc0,320:352]=1;bridge[tc1,352:384]=1
    for bit in range(4):
        for val in range(2):
            sl=slice(400+(bit*2+val)*4,404+(bit*2+val)*4)
            bridge[q,sl]=1
            for key in range(KEYS):
                if ((key>>bit)&1)==val:bridge[HS[key],sl]=1
            bt[sl]=1
    return [(enc,et),(cmp,ct),(code,tt),(bridge,bt)]

def weights(seed,k):
    u,classes,plastic,h=DIMENSIONS[k];w=np.zeros((D,u),'i1');rng=np.random.default_rng(seed+71*k)
    for j in plastic:w[:25,j]=1;w[rng.choice(np.arange(25,109),7,False),j]=1
    used=set()
    def helper(j,inds):
        inds=list(inds);assert len(inds)<=32 and len(set(inds))==len(inds)
        w[inds,j]=1
        if len(inds)<32:w[D-(32-len(inds)):,j]=1
        used.add(j)
    if k==0:
        for key in range(KEYS):helper(h[key],[*range(25+key*7,25+(key+1)*7),*range(128+key*7,128+(key+1)*7)])
        for j,start in zip(h[12:16],[256,288,320,352]):helper(j,range(start,start+32))
    elif k==1:
        for key in range(KEYS):helper(h[key],range(25+key*7,25+key*7+6))
        for j,inds in zip(h[12:17],[range(25),[109],range(128,160),range(160,192),range(192,224)]):helper(j,inds)
    else:
        for bit in range(4):
            for val in range(2):
                for rep in range(4):helper(32+bit*8+val*4+rep,range(400+(bit*2+val)*4,404+(bit*2+val)*4))
        helper(84,range(320,352));helper(85,range(352,384));helper(88,range(256,288))
        for j in range(8):helper(104+j,range(256+j*4,260+j*4))
        helper(80,range(25));helper(81,[109])
    for j in h:
        if j not in used:w[D-32:,j]=1
    assert np.all(w.sum(0)==32)
    return w

def thresholds(k):
    u,c,p,h=DIMENSIONS[k];lt=np.full(u,32,'i4');lt[p]=0;t=np.full(u,31,'i4')
    if k==0:t[h[:12]]=6;t[h[15]]=3
    elif k==1:t[h[:12]]=5;t[h[12]]=24;t[h[13]]=0
    else:t[32:64]=3;t[88]=3;t[104:112]=3;t[80]=24;t[81]=0
    return lt,t

def packed(w,k):
    lt,_=thresholds(k);ram=[0,0]
    for j in range(w.shape[1]):
        ram += [0,0,0,(int(lt[j])<<16)|16]
        ram += [sum(int(w[p*32+b,j])<<b for b in range(32)) for p in range(D//32)]
    return ram

def schedule(stage,arm):
    rows=[]
    def add(key,role,phase):
        labels=[OUTCOME[key],-1,-1] if role=='ground' else [-1,0,0] if role=='transfer' else [-1,-1,-1]
        if arm=='students_frozen':labels[1:]=[-1,-1]
        rows.append(dict(index=len(rows),key=key,role=role,phase=phase,labels=labels,wire=OUTCOME[key] if role=='ground' else 0,actual=OUTCOME[key]))
    def ground(keys,phase):
        for _ in range(3):
            for key in keys:add(key,'ground',phase)
    def probe(keys,repeat,phase,exam=False):
        order=np.tile(keys,repeat);np.random.default_rng(119000+len(rows)).shuffle(order)
        for key in order.tolist():add(key,'exam' if exam else 'query',phase)
    def transfer(keys,repeat,phase):
        for _ in range(repeat):
            for key in keys:add(key,'query',phase);add(key,'transfer',phase)
    ground(range(4),'source_ground');probe(range(4),4,'source_probe')
    transfer(range(4),3,'transfer_known');transfer([6,7],2,'unknown_transfer')
    probe(range(4),12,'silent_exam',True)
    if stage=='extended':
        ground([4,5],'new_source_ground');probe(range(6),4,'new_source_probe')
        transfer(range(6),3,'transfer_extended');probe(range(6),16,'silent_final_exam',True)
    return rows

def simulate(rams,stage,arm):
    mats=matrices(arm);refs=[Reference(rams[k],n=D,u=u,nw=32,classes=c) for k,(u,c,_,_) in enumerate(DIMENSIONS)]
    rows=schedule(stage,arm);last=np.zeros(N,'u1');xs=[];ys=[];ss=[[],[],[]];ps=[[],[],[]];fs=[[],[],[]];cpu=[];teacher={};student={}
    for e in rows:
        role=e['role'];x=last.copy()
        if role!='transfer':
            x[32:64]=0
            for bit in range(4):x[32+bit*8+((e['key']>>bit)&1)*4:36+bit*8+((e['key']>>bit)&1)*4]=1
        x[64:72]=0;x[65 if role=='ground' else 69 if role=='transfer' else 64]=1
        if role=='exam':x[70]=1
        if role=='ground':teacher[e['key']]=e['actual']
        if role=='transfer' and e['key'] in teacher:student[e['key']]=teacher[e['key']]
        cpu.append([teacher.get(e['key'],-1),student.get(e['key'],-1)])
        z=(x.astype('i4')@mats[0][0].astype('i4')>mats[0][1]).astype('u1')
        for k in range(3):
            fs[k].append(z.copy());p=np.asarray(refs[k].forward(z,e['labels'][k]),'<i4');z=(p>thresholds(k)[1]).astype('u1');ps[k].append(p);ss[k].append(refs[k].ram[:])
            if k<2:
                for w,t in mats[1:3] if k==0 else mats[3:4]:z=(z.astype('i4')@w.astype('i4')>t).astype('u1')
        if role!='transfer':last=z.copy()
        xs.append(x);ys.append(z)
    a=dict(events=rows,inputs=np.asarray(xs,'u1'),outputs=np.asarray(ys,'u1'),cpu=np.asarray(cpu,'i4'))
    for k in range(3):a[f'states{k}']=np.asarray(ss[k],'<u4');a[f'potentials{k}']=np.asarray(ps[k],'<i4');a[f'features{k}']=np.asarray(fs[k],'u1')
    return a

def assess(a):
    out={}
    for phase in dict.fromkeys(e['phase'] for e in a['events']):
        ii=[e['index'] for e in a['events'] if e['phase']==phase and e['role'] in ['query','exam']]
        if not ii:continue
        y=a['outputs'][ii];truth=np.asarray([a['events'][i]['actual'] for i in ii]);p=np.stack([y[:,88],y[:,PR].any(1)],1).astype('u1');t=y[:,84:86]
        out[phase]=dict(queries=len(ii),student_correct=int(sum(v[c] and not v[1-c] for v,c in zip(p,truth))),teacher_correct=int(sum(v[c] and not v[1-c] for v,c in zip(t,truth))),teacher_silent=int((~t.any(1)).sum()),ambiguous=int(p.all(1).sum()),missing=int((~p.any(1)).sum()),cpu_student_correct=int((a['cpu'][ii,1]==truth).sum()))
    return out

def predicate_screen():
    w,t=matrices('native')[1];rows=[]
    for tc in [(0,0),(1,0),(0,1),(1,1)]:
        for prev in [(0,0),(1,0),(0,1),(1,1)]:
            for teach in range(2):
                v=np.zeros(64,'i4');v[PT[0]]=tc[0];v[PT[4]]=tc[1];v[HT[13]]=teach;v[HT[14]]=prev[0];v[HT[15]]=prev[1]
                out=(v@w>t)[16:20];expected=np.zeros(4,bool)
                if teach and sum(tc)==1:
                    c=tc.index(1)
                    if not prev[c]:expected[2*c]=1
                    elif all(prev):expected[2*c+1]=1
                assert np.array_equal(out,expected),(tc,prev,teach,out,expected)
                rows.append(dict(teacher=tc,student=prev,self_teach=teach,error=out.tolist()))
    return rows

if __name__=='__main__':
    import json
    predicate_screen()
    for seed in SEEDS:
        for arm in ARMS:
            a=simulate([packed(weights(seed,k),k) for k in range(3)],'extended',arm)
            print(json.dumps(dict(seed=seed,arm=arm,assessment=assess(a))))
