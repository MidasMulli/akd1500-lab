"""Exact native recruitment hypothesis with bounded descriptor replay."""
from pathlib import Path
import argparse,datetime as dt,fcntl,json,os,subprocess,time,traceback
import numpy as np
from pong.relearn.model import Reference
from visual_memory.roles import external_features
from visual_memory.data import save,sha

P=Path(__file__).resolve().parents[1];S=P/'studies/20261026-native-recruitment';U=352;N=512

class RecruitmentReference(Reference):
    def __init__(self,ram,cap):super().__init__(ram,n=N,u=U,nw=64,classes=11);self.cap=cap
    def forward(self,v,label,learning=True):
        x=sum(1<<int(k) for k in np.flatnonzero(v));ps=[(w&x).bit_count() for w in self.weights]
        if not learning:return ps
        eligible=[j for j in range(label*32,(label+1)*32) if ps[j]>(self.ram[2+j*20+3]>>16)]
        win=max(eligible,key=lambda j:(ps[j],-j)) if eligible else None;count=0
        if win is not None:
            w=self.weights[win];count=min(self.ram[2+win*20+3]&65535,(w&~x).bit_count(),(x&~w).bit_count())
        got=super().forward(v,label)
        if count:self.ram[2+win*20+3]=(min(ps[win],self.cap)<<16)|(self.ram[2+win*20+3]&65535)
        return got

def initial_weights():
    rng=np.random.default_rng(20261026);w=np.zeros((N,U),'i1')
    for j in range(U):w[rng.choice(N,64,False),j]=1
    return w

def schedule(f,y,s,frame,old,new,transfer=None):
    ids=[];rows=[];labels=[]
    for phase,ss,lo,hi,teach in [('prefix',old,10,80,True),('old',old,90,200,False),('before',new,90,140,False),('adapt',new,10,80,True),('after',new,150,200,False),('retention',old,90,200,False)]+([('transfer',transfer,10,200,False)] if transfer else []):
        for i in np.flatnonzero((s==ss)&(frame>=lo)&(frame<=hi)):
            ids.append(i);labels.append(int(y[i]) if teach else 10);rows.append(dict(index=len(rows),phase=phase,label=int(y[i]),session=int(s[i]),frame=int(frame[i]),item=int(i),teacher=teach))
    return f[ids],np.array(labels,'<i4'),rows

def simulate(x,labels,ram,cap):
    r=RecruitmentReference(ram,cap);scores=[];states=[]
    for v,label in zip(x,labels):
        before=r.ram.copy();scores.append(r.forward(v,int(label),label!=10));states.append(r.ram.copy())
        if label==10:assert r.ram==before
    return np.array(scores,'<i4'),np.array(states,'<u4')

def metrics(scores,states,initial,rows):
    pred=scores[:,:320].reshape(-1,10,32).max(2).argmax(1);out={};seen=set();edits=0;timeline=[];before=initial
    for i,ram in enumerate(states):
        a=before[2:].reshape(U,20)[:,4:];b=ram[2:].reshape(U,20)[:,4:];changed=np.flatnonzero((a!=b).any(1));seen.update(map(int,changed));bits=sum(int(v).bit_count() for v in np.bitwise_xor(a,b).ravel());assert bits%2==0;edits+=bits//2;timeline.append(dict(index=i,edited_records=changed.tolist(),ever_edited_records=len(seen),connection_replacements_this_step=bits//2,cumulative_connection_replacements=edits));before=ram
    for phase in ['old','before','after','retention','transfer']:
        ii=np.array([i for i,r in enumerate(rows) if r['phase']==phase])
        if len(ii):
            yy=np.array([rows[i]['label'] for i in ii]);out[phase]=dict(correct=int((pred[ii]==yy).sum()),total=len(ii),accuracy=float((pred[ii]==yy).mean()),ever_edited_records=timeline[int(ii[-1])]['ever_edited_records'])
    return out,timeline

def mechanism(default,recruit):
    d,r=default,recruit
    forgetting_d=d['old']['accuracy']-d['retention']['accuracy'];forgetting_r=r['old']['accuracy']-r['retention']['accuracy']
    gates=dict(old_floor=min(d['old']['accuracy'],r['old']['accuracy'])>=.70,retention_gain=r['retention']['accuracy']-d['retention']['accuracy']>=.10-1e-10,forgetting_reduction=forgetting_d-forgetting_r>=.05-1e-10,changed_nonharm=r['after']['accuracy']>=d['after']['accuracy']-.05-1e-10,recruitment_growth=r['retention']['ever_edited_records']>=1.5*d['retention']['ever_edited_records'])
    return dict(mechanism_pass=all(gates.values()),gates=gates,changed_application_floor_pass=r['after']['accuracy']>=.65,forgetting_default=forgetting_d,forgetting_recruiting=forgetting_r)

def audit(d):
    owner=json.loads((d/'OWNER.json').read_text());f=Path(owner['fixture']);ref=np.load(f/'REFERENCE.npz');scores=np.fromfile(d/'result.scores','<i4').reshape(-1,U);final=np.fromfile(d/'result.ram','<u4');assert np.array_equal(scores,ref['scores']);assert np.array_equal(final,ref['ram'][-1])
    rows=json.loads((f/'ROWS.json').read_text());initial=np.fromfile(f/'cell.ram','<u4')
    if owner['physical']:
        assert owner['device_free']
        for i,want in enumerate(ref['ram']):assert np.array_equal(np.fromfile(d/f'result.snapshot_{i}.ram','<u4'),want),('full capturedRAM',i)
    m,t=metrics(scores,ref['ram'],initial,rows);save(d/'TIMELINE.json',t);r=dict(status='EXACT_NATIVE_PASS' if owner['physical'] else 'EXACT_COMPILED_CPU_PASS',metrics=m,every_potential_exact=True,every_captured_RAM_exact=owner['physical'],physical=owner['physical'],count=len(rows),hashes={p.name:sha(p) for p in d.iterdir() if p.is_file()});save(d/'AUDIT.json',r);return r

def run(mode,cell):
    assert mode in ['cpu','native'];f=S/'fixtures'/cell;assert json.loads((S/'fixtures/BUILD.json').read_text())['status']=='RECRUITMENT_COMPILER_PASS';count=len(np.fromfile(f/'cell.labels','<i4'));physical=mode=='native'
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,26,20,58,tzinfo=dt.timezone.utc)
    if physical:
        frozen=json.loads((S/'FROZEN.json').read_text())
        for rel,h in frozen['hashes'].items():assert sha(P/rel)==h,('source/fixture drift',rel)
        if cell.startswith('fresh'):
            extra=json.loads((S/'FRESH_FROZEN.json').read_text())
            for rel,h in extra['hashes'].items():assert sha(P/rel)==h,('fresh input drift',rel)
        assert (S/'raw'/('cpu-'+cell)/'AUDIT.json').exists()
        if cell!='smoke':assert (S/'raw/native-smoke/AUDIT.json').exists()
        if cell.startswith('fresh'):assert json.loads((S/'DEVELOPMENT_REVIEW.json').read_text())['mechanism_pass']
    owners=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')] if (S/'raw').exists() else []
    assert sum(r['physical'] for r in owners)+physical<=6 and sum(r['count'] for r in owners if r['physical'])+(count if physical else 0)<=6000
    d=S/'raw'/(mode+'-'+cell);d.mkdir(parents=True,exist_ok=False);r=dict(status='ADMITTED',fixture=str(f),mode=mode,cell=cell,count=count,physical=physical,owner='codex:01a11d87-0208-7613-bbc0-e5e612110c50',utc=dt.datetime.now(dt.timezone.utc).isoformat());save(d/'OWNER.json',r);lock=None
    try:
        if physical:
            lock=open('/tmp/akd1500-research.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
        env=dict(os.environ,AKD_LAB_SNAPSHOTS='1')
        with (d/'STDOUT').open('x') as out,(d/'STDERR').open('x') as err:
            proc=subprocess.Popen([str(P/'build/native_recruitment_actor'),mode,str(f/'cell'),str(d/'result'),str(count),'1','1','0'],cwd=P,stdout=out,stderr=err,env=env);r['pid']=proc.pid;save(d/'OWNER.json',r)
            try:code=proc.wait(timeout=90)
            except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5);raise
        assert code==0,('actor failure',code);r['status']='EXECUTION_PASS'
    except BaseException as e:r.update(status='FAILED_PRESERVED',error=repr(e));raise
    finally:
        if lock:lock.close()
        r.update(finish_utc=dt.datetime.now(dt.timezone.utc).isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1);save(d/'OWNER.json',r)
    result=audit(d);print(json.dumps({k:v for k,v in result.items() if k!='hashes'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','native']);p.add_argument('cell');a=p.parse_args();run(a.mode,a.cell)
