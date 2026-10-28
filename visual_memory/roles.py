"""Two fixed native memory roles; all source/execution/quality gates explicit."""
from pathlib import Path
import argparse,json,traceback
import numpy as np
from visual_memory.data import save,sha,RESERVED
from pong.relearn.model import Reference

P=Path(__file__).resolve().parents[1]
S=P/'studies/20261026-visual-memory-roles'
V=P/'studies/20261026-visual-memory-v3'
T=P/'studies/20261026-visual-memory-v2/teacher-diagnostic/features.npz'

def memory(n=1024,u=352,dual=True):
    rng=np.random.default_rng(20261029);width=u//11;stride=4+n//32;ram=np.zeros(2+u*stride,'<u4')
    for j in range(u):
        b=2+j*stride;ram[b+3]=((64 if j>=width*10 else 0)<<16)|32
        bank=(j%width)//(width//2) if dual else 0
        for k in rng.choice(510,64,False)+bank*512:ram[b+4+int(k)//32]|=np.uint32(1<<(int(k)%32))
    return ram

def schedule(features,labels,sessions,frames,old_session,new_session,transfer=None):
    xx=[];yy=[];cpu=[];rows=[]
    def add(i,phase,role):
        host=np.zeros(512,'u1');host[:510]=features[i,:510];host[510:]=[int(role in ['stable','query']),int(role in ['plastic','query'])]
        routed=np.zeros(1024,'u1')
        if host[510]:routed[:510]=features[i,:510]
        if host[511]:routed[512:1022]=features[i,:510]
        xx.append(np.r_[host,np.zeros(512,'u1')]);cpu.append(routed);yy.append(10 if role=='query' else int(labels[i]));rows.append(dict(index=len(rows),phase=phase,role=role,item=int(i),label=int(labels[i]),session=int(sessions[i]),frame=int(frames[i])))
    for i in np.flatnonzero((sessions==old_session)&(frames<=80)):add(i,'prefix','stable');add(i,'prefix','plastic')
    for phase,sess,lo,hi in [('old',old_session,90,200),('before',new_session,90,140)]:
        for i in np.flatnonzero((sessions==sess)&(frames>=lo)&(frames<=hi)):add(i,phase,'query')
    for i in np.flatnonzero((sessions==new_session)&(frames<=80)):add(i,'adapt','plastic')
    for phase,sess,lo in [('after',new_session,150),('retention',old_session,90)]:
        for i in np.flatnonzero((sessions==sess)&(frames>=lo)):add(i,phase,'query')
    if transfer:
        for i in np.flatnonzero(sessions==transfer):add(i,'transfer','query')
    return np.stack(xx),np.array(yy,'<i4'),np.stack(cpu),rows

def external_features():
    t=np.load(T);z=np.load(V/'encoder01/teacher-transform.npz');d=np.load('/Users/midas/Models/core50-selected01/core50_selected.npz');ids=np.flatnonzero(np.isin(d['object'],RESERVED)&np.isin(d['session'],[1,2]));lookup={int(v):i for i,v in enumerate(t['ids'])}
    f=((t['normalized'][[lookup[int(i)] for i in ids]]-z['mean'])@z['projection']>z['cut']).astype('u1')
    return f,np.searchsorted(RESERVED,d['object'][ids]),d['session'][ids],d['frame'][ids],ids

def simulate(x,y,rows,initial,cut=False):
    ref=Reference(initial,n=1024,u=352,nw=64,classes=11);scores=[];boundaries=[]
    for i,(v,label,r) in enumerate(zip(x,y,rows)):
        vv=v.copy()
        if cut and r['phase']=='adapt':vv[:]=0
        before=list(ref.ram);got=ref.forward(vv,int(label));scores.append(got);boundaries.append(ref.ram.copy())
        if r['role']=='query' or (cut and r['phase']=='adapt'):assert ref.ram==before,'read-only RAM invariant'
    return np.array(scores,'<i4'),np.array(boundaries,'<u4')

def review(scores,rows):
    pred=scores[:,:320].reshape(-1,10,32).max(2).argmax(1);out={}
    for phase in ['old','before','after','retention','transfer']:
        ii=[i for i,r in enumerate(rows) if r['phase']==phase]
        if ii:
            yy=np.array([rows[i]['label'] for i in ii]);out[phase]=dict(correct=int((pred[ii]==yy).sum()),total=len(ii),accuracy=float((pred[ii]==yy).mean()))
    return out

def development():
    out=S/'development';out.mkdir(exist_ok=False);f,y,s,frame,ids=external_features();host,teacher,routed,rows=schedule(f,y,s,frame,1,2);assert len(rows)==600
    initial=memory();a,ar=simulate(routed,teacher,rows,initial);b,br=simulate(routed,teacher,rows,initial,True);ra,rb=review(a,rows),review(b,rows)
    # Larger single-region exact-rule CPU: identical experiences/teachers/queries.
    one=Reference(memory(512,704,False),n=512,u=704,nw=64,classes=11);single=[]
    for label,r in zip(teacher,rows):
        v=np.r_[f[r['item'],:510],np.zeros(2,'u1')];single.append(one.forward(v,int(label)))
    ss=np.array(single);single_pred=ss[:,:640].reshape(-1,10,64).max(2).argmax(1);single_metrics={}
    for phase in ra:
        ii=np.array([i for i,r in enumerate(rows) if r['phase']==phase]);yy=np.array([rows[i]['label'] for i in ii]);single_metrics[phase]=dict(correct=int((single_pred[ii]==yy).sum()),total=len(ii),accuracy=float((single_pred[ii]==yy).mean()))
    gates=dict(old=ra['old']['accuracy']>=.70,after=ra['after']['accuracy']>=.65,gain=ra['after']['accuracy']-rb['after']['accuracy']>=.10-1e-10,retention=ra['old']['accuracy']-ra['retention']['accuracy']<=.05+1e-10)
    np.savez(out/'STREAM.npz',host_inputs=host,teacher=teacher,routed_inputs=routed,initial_ram=initial,open_scores=a,cut_scores=b,open_ram=ar,cut_ram=br,ids=ids)
    save(out/'ROWS.json',rows);r=dict(status='ROLE_CPU_GATE_PASS' if all(gates.values()) else 'ROLE_CPU_GATE_FAILED_STOP',gates=gates,open=ra,cut=rb,larger_single=single_metrics,physical_sessions=0,official_test_evaluated=False,
        input='External teacher-derived recorded features; NOT own RGB encoder',query='Class10protected; no capturedRAMchange, selectorstateevolves',hashes=dict(stream=sha(out/'STREAM.npz'),rows=sha(out/'ROWS.json'),source=sha(Path(__file__)),protocol=sha(S/'PROTOCOL.md')));save(out/'REVIEW.json',r);print(json.dumps(r,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['development']);a=p.parse_args();development()
