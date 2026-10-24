"""Bounded admissions and independent saved-state checks for input-gated learning."""
from pathlib import Path
import datetime as dt,fcntl,hashlib,json,subprocess,sys
import numpy as np
from feedback.design import P,S,metrics
F=S/'fixtures02'
def now():return dt.datetime.now(dt.timezone.utc)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,r):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2)+'\n')
def freeze():
    target=S/'evidence/FROZEN.json';assert not target.exists()
    paths=[*F.glob('*/*'),* (P/'feedback').glob('*'),P/'src/readout_bench.cpp',P/'src/mmio_driver.hpp',P/'pong/relearn/model.py',P/'tools/prepare_reduction.py',P/'tools/program_codec.py',P/'tools/prepare.py',P/'tools/prepare_network.py',P/'build/predictive_feedback',S/'PROTOCOL.md',S/'AUTHORITY.json']
    save(target,dict(utc=now().isoformat(),hashes={str(p.relative_to(P)):sha(p) for p in paths if p.is_file()}))
def name(mode,seed,variant,arm,count,batch,sleep):return f'{mode}_s{seed}_{variant}_{arm}_n{count}_b{batch}_s{sleep}'
def passed(n):
    p=S/'raw'/n/'AUDIT.json';return p.exists() and json.loads(p.read_text())['status']=='SAVED_ARRAY_PASS'
def audit(d):
    r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0,(d/'stderr').read_text()
    for p,h in r['hashes'].items():assert sha(d/p)==h
    f=F/f's{r["seed"]}';n=r['count'];raw=r['variant']=='raw';dim=16 if raw and r['mode']!='task' else 1
    y=np.fromfile(d/'result.scores','<i4').reshape(n,dim);want=np.fromfile(f/(r['arm']+'.potentials'),'<i4').reshape(768,16)[:n]
    expected=want if raw else (want>13).sum(axis=1,keepdims=True)
    if r['mode']!='task':
        assert np.array_equal(y,expected),'independent Python potentials/native reducer'
        states=np.fromfile(d/'result.snapshots','<u4').reshape(-1,322);bounds=np.fromfile(d/'result.boundaries','<i4');want_states=np.fromfile(f/(r['arm']+('.rawstates' if raw else '.states')),'<u4').reshape(768,322)
        assert np.array_equal(states,want_states[bounds]);assert np.array_equal(np.fromfile(d/'result.ram','<u4'),states[-1])
        if r['mode']=='native':
            assert r['device_free']
            for j in range(int((f/(r['variant']+'.layout')).read_text())):
                original=np.fromfile(f/(r['variant']+f'.packet{j}.ram'),'<u4')
                for tag in ['initial','final']:assert np.array_equal(original,np.fromfile(d/f'result.packet{j}.{tag}','<u4'))
            if r['sleep']:
                a=np.fromfile(d/'result.blocked_intervals','<i8').reshape(-1,5);assert np.all(a[:,1]==32) and np.all(a[:,2]==r['sleep']) and np.all(a[:,3]==0) and np.all(a[:,4]==1)
    rows=json.loads((f/(r['arm']+'.rows.json')).read_text())[:n];truth=json.loads((f/'BUILD.json').read_text())['truth']
    result=metrics(y if raw and r['mode']!='task' else y[:,0],rows,truth)
    times=np.fromfile(d/'result.latency_ns','<i8')/r['batch'];cpu=np.fromfile(d/'result.caller_ns','<i8')/r['batch']
    assert len(times)>0 and np.all(times>0)
    a=dict(status='SAVED_ARRAY_PASS',name=d.name,mode=r['mode'],seed=r['seed'],variant=r['variant'],arm=r['arm'],count=n,batch=r['batch'],sleep_ms=r['sleep'],median_event_us=float(np.median(times)/1000),median_caller_us=float(np.median(cpu)/1000),metrics=result['groups'],constant_teacher_class=0,primary_native_mode_changes=0,queries_preserve_visible_connection_memory=True)
    if n==768:
        for k,v in a['metrics'].items():
            if k.startswith(('baseline_query','retention_query_unchanged','post_adaptation_unchanged','sealed_trajectory_query_unchanged')):assert v['correct']==v['total']
        ch=a['metrics']['post_adaptation_changed'];assert ch['correct']==(2 if r['arm']=='native' else 0)
    save(d/'AUDIT.json',a);save(d/'predictions.json',result['predictions']);return a
def run(mode,seed,variant,arm,count,batch,sleep):
    assert mode in ['native','cpu','task'] and seed in [108101,108201,108301] and variant in ['raw','reduced'] and arm in ['native','frozen','disconnected'] and count in [3,128,768] and batch in [1,32] and sleep in [0,5,20]
    state=json.loads((S/'STATE.json').read_text());auth=json.loads((S/'AUTHORITY.json').read_text());assert not state['closed'] and not state.get('failure');cut=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert now()<cut
    for p,h in json.loads((S/'evidence/FROZEN.json').read_text())['hashes'].items():assert sha(P/p)==h,('freeze drift',p)
    nm=name(mode,seed,variant,arm,count,batch,sleep);d=S/'raw'/nm;assert not d.exists()
    if mode=='native':
        assert passed(name('cpu',seed,variant,arm,768,1,0))
        if variant=='raw':
            if count==128:assert passed(name('native',108101,'raw','native',3,1,0))
            if count==768:assert passed(name('native',108101,'raw','native',128,1,0))
        else:
            assert passed(name('native',seed,'raw','native',768,1,0))
            if count==128:assert passed(name('native',108101,'reduced','native',3,1,0))
            if count==768:assert passed(name('native',108101,'reduced','native',128,1,0))
        if seed!=108101:
            for a in ['native','frozen','disconnected']:assert passed(name('native',108101,'reduced',a,768,1,0))
        if batch==32:
            assert count==768 and arm=='native' and sleep>0 and passed(name('native',seed,'reduced','native',768,1,0))
    else:assert count==768 and batch==1 and sleep==0
    prior=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')];physical=[q for q in prior if q['mode']=='native'];assert len(physical)+(mode=='native')<=auth['maximum_physical_sessions'];assert sum(q['count'] for q in physical)+(count if mode=='native' else 0)<=auth['maximum_forwards']
    d.mkdir(parents=True);r=dict(mode=mode,seed=seed,variant=variant,arm=arm,count=count,batch=batch,sleep=sleep,repeats=1 if mode=='native' else 50,start_utc=now().isoformat(),source_freeze_sha256=sha(S/'evidence/FROZEN.json'));save(d/'OWNER.json',r)
    def execute():
        cmd=['taskset','-c','2','build/predictive_feedback',mode,str(F/f's{seed}'),arm,str(d/'result'),str(count),variant,str(batch),str(sleep),str(r['repeats'])]
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            p=subprocess.Popen(cmd,cwd=P,stdout=out,stderr=err);r['pid']=p.pid;save(d/'OWNER.json',r)
            try:code=p.wait(timeout=min(120,(cut-now()).total_seconds()))
            except subprocess.TimeoutExpired:
                p.terminate()
                try:code=p.wait(timeout=3)
                except subprocess.TimeoutExpired:p.kill();code=p.wait()
        r.update(exit_code=code,finish_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1);r['hashes']={p.name:sha(p) for p in d.iterdir() if p.is_file()};save(d/'RECEIPT.json',r)
        try:a=audit(d)
        except Exception:state['failure']=nm;save(S/'STATE.json',state);raise
        state.setdefault('completed',[]).append(nm);save(S/'STATE.json',state);return a
    if mode=='native':
        with open('/tmp/akd1500-research.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1;result=execute()
    else:result=execute()
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='audit':print(json.dumps(audit(S/'raw'/sys.argv[2])))
    else:run(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4],*map(int,sys.argv[5:]))
