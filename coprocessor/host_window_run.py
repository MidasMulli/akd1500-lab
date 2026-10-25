"""Finite common-clock Nano model/co-processor windows, with deferred audits."""
from pathlib import Path
import argparse,datetime as dt,fcntl,json,os,select,signal,subprocess,time
import numpy as np
from coprocessor.service_run import S,P,sha,save,environment,good,cell,MODES
from coprocessor.service_audit import expected_states
from coprocessor.extent_audit import load_fixture

BLAS='/home/nicklomeli/akida-cc-venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-0f683016.so'

def freeze():
    paths=[P/'coprocessor'/f for f in ['service_window.cpp','window_source.py','host_window_run.py','window_model.py','build_host_window.sh']]
    paths += [P/'streaming/nano.py',P/'streaming/design.py',P/'build/host_window',S/'WINDOW_AMENDMENT.md',S/'WINDOW_SOURCE.json',S/'SELECTION.json',S/'SELECTION_TIMESTAMP_CORRECTION.json',S/'WINDOW_PATH_REPAIR.md',S/'WINDOW_SETUP_REPAIR.md']
    world=Path('/home/nicklomeli/world_model_live_20261016')
    external=[world/'demo_worker.py',world/'PORT_MANIFEST.json',world/'CHECKPOINTS.json',world/'checkpoints/T202.pt']+list((world/'source').glob('*.py'))
    target=S/'evidence/WINDOW_FROZEN_V3.json';assert not target.exists()
    save(target,dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),hashes={str(p.relative_to(P)):sha(p) for p in paths},external={str(p):sha(p) for p in external}))

def line(proc,log,seconds=35):
    assert select.select([proc.stdout],[],[],seconds)[0],'owned child response timeout'
    s=proc.stdout.readline();log.write(s);log.flush();assert s,('owned child closed',proc.poll())
    return s.strip()

def audit(d):
    r=json.loads((d/'RECEIPT.json').read_text());assert r['status']=='EXECUTION_PASS' and r['device_free']
    for name,h in r['hashes'].items():assert sha(d/name)==h,('window result hash',name)
    result=dict(status='EXACT_WINDOW_PASS',mode=r['mode'],busy=r['busy'],seed=r['seed'],arm=r['arm'],count=r['count'])
    if r['mode']!='none':
        f=S/'fixtures'/f's64_s{r["seed"]}_potentials';m,a=load_fixture(f);count=r['count'];words=len(a['ram'])
        canon=S/'reference'/f's64_s{r["seed"]}_a{r["arm"]}'
        want=np.fromfile(canon.with_suffix('.scores'),'<i4').reshape(64,128)[:count]
        states=np.fromfile(canon.with_suffix('.snapshots'),'<u4').reshape(64,words)
        states=expected_states(a['ram'],states,m['n'])
        get=lambda k,dt:np.fromfile(d/('result.'+k),dt)
        assert np.array_equal(get('scores','<i4').reshape(count,128),want),'window outputs'
        assert np.array_equal(get('ram','<u4'),states[count-1]),'window final full RAM'
        assert np.array_equal(get('boundary_index','<u4'),[count]),'window audit boundary'
        assert np.array_equal(get('snapshots','<u4'),states[count-1]),'window captured full RAM'
        starts,ends=get('started_ns','<i8'),get('finished_ns','<i8');assert len(starts)==len(ends)==count
        assert np.all(starts>=r['start_ns']+np.arange(count)*r['period_ns']) and np.all(ends>starts)
        assert np.all(starts[1:]>=ends[:-1])
        wall=get('latency_ns','<i8');cpu=get('caller_ns','<i8');assert np.all(wall>0) and np.all(cpu>0)
        result.update(median_event_us=float(np.median(wall)/1000),p95_event_us=float(np.percentile(wall,95)/1000),total_actor_cpu_ms=float(cpu.sum()/1e6),total_actor_wall_ms=float(wall.sum()/1e6))
        if r['period_ns']:
            lateness=ends-(r['start_ns']+(np.arange(count)+1)*r['period_ns'])
            result.update(missed_event_deadlines=int((lateness>0).sum()),maximum_lateness_us=float(lateness.max()/1000))
        if r['mode']=='native':
            assert np.array_equal(get('initial.ram','<u4'),a['ram'])
            for j in range(len(m['fixed_packets'])):
                want=np.fromfile(f/f'cell.packet{j}.ram','<u4')
                for tag in ['initial','final']:assert np.array_equal(get(f'packet{j}.{tag}','<u4'),want),'fixed native memory changed'
    if r['busy']:
        model=json.loads((d/'MODEL_RECEIPT.json').read_text());assert model['weights_unchanged'] and model['start_ns']==r['start_ns'] and model['end_ns']==r['end_ns']
        t=np.load(d/'foreground-timing.npy',allow_pickle=False);assert t.ndim==2 and t.shape[1]==3 and len(t)>0
        assert np.all(t[:,1]>t[:,0]) and np.all(t[:,2]>0)
        assert np.all(t[:,0]>=r['start_ns']) and np.all(t[:,0]<r['end_ns'])
        assert np.all(t[1:,0]>=t[:-1,1]);assert t[0,0]-r['start_ns']<5_000_000
        completed=t[:,1]<=r['end_ns'];latency=(t[:,1]-t[:,0])/1e6
        result.update(foreground_calls_complete_in_window=int(completed.sum()),foreground_fps=float(completed.sum()/((r['end_ns']-r['start_ns'])/1e9)),foreground_median_ms=float(np.median(latency[completed])),foreground_p95_ms=float(np.percentile(latency[completed],95)),foreground_process_cpu_ms=model['process_cpu_ns']/1e6)
        if r['mode']!='none':assert r['audit_released_ns']>=t[-1,1],'deferred oracle overlapped measured foreground'
    save(d/'AUDIT.json',result);return result

def run(a):
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text());assert not state['closed'] and not state['failure']
    assert (dt.datetime.fromisoformat(auth['scientific_cutoff_utc'])-dt.datetime.now(dt.timezone.utc)).total_seconds()>125
    for fn in ['FROZEN.json','WINDOW_FROZEN_V3.json']:
        frozen=json.loads((S/'evidence'/fn).read_text())
        for p,h in frozen['hashes'].items():assert sha(P/p)==h,('source drift',p)
        for p,h in frozen.get('external',frozen.get('external_runtime_hashes',{})).items():assert sha(Path(p))==h,('external drift',p)
    assert a.seed in auth['seeds'] and a.mode in MODES+['none'] and a.sleep in [0,100] and a.count in [8,64] and a.arm in [0,1]
    assert a.period in [0,50000] and a.repeat in [0,1,2]
    assert not a.busy or a.count==64 and a.period==50000 and a.arm==1
    if a.busy:
        assert json.loads((S/'evidence/WINDOW_QUALIFIED.json').read_text())['status']=='WINDOW_QUALIFIED'
        assert a.mode in ['none','native','cpu-blas1','cpu-blas4']
    if a.mode=='native':
        assert all(good(cell('native',64,seed,'potentials',arm,64,rep,sleep,1,1,1)) for seed in auth['seeds'] for arm in [0,1] for rep in [1,2] for sleep in [0,100]),'confirmation first'
        for mode in MODES[1:]:
            for arm in [0,1]:assert (S/'window_raw'/f'q_{mode}_s107501_a{arm}_n64_w0_r2'/'AUDIT.json').exists(),'all new CPU paths qualify first'
        if a.count==64:assert (S/'window_raw/q_native_s107501_a0_n8_w0_r1/AUDIT.json').exists(),'new read-only smoke first'
    owners=[json.loads(p.read_text()) for sub in ['raw','window_raw'] for p in (S/sub).glob('*/OWNER.json')]
    physical=a.mode=='native';assert sum(v['physical'] for v in owners)+physical<=auth['maximum_physical_sessions']
    assert sum(v['count'] for v in owners if v['physical'])+(a.count if physical else 0)<=auth['maximum_forwards']
    name=f'{"b3" if a.busy else "q"}_{a.mode}_s{a.seed}_a{a.arm}_n{a.count}_w{a.sleep}_r{a.repeat}';d=S/'window_raw'/name;d.mkdir(parents=True,exist_ok=False)
    r=dict(status='ADMITTED',mode=a.mode,physical=physical,busy=a.busy,seed=a.seed,arm=a.arm,count=a.count if a.mode!='none' else 0,sleep_us=a.sleep,repeat=a.repeat,period_ns=a.period*1000,start_utc=dt.datetime.now(dt.timezone.utc).isoformat(),environment_before=environment(),freeze_sha256=sha(S/'evidence/WINDOW_FROZEN_V3.json'))
    save(d/'OWNER.json',r);children=[];locks=[];handles=[]
    def spawn(command,stem):
        err=(d/(stem+'.stderr')).open('w');log=(d/(stem+'.stdout')).open('w');handles.extend([err,log])
        proc=subprocess.Popen(command,cwd=P,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1,env=dict(os.environ,AKD_OPENBLAS_LIBRARY=BLAS))
        children.append(proc);r[stem+'_pid']=proc.pid;save(d/'OWNER.json',r);return proc,log
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('owned120-second window cap')));signal.alarm(120)
    try:
        for path in (['/tmp/akd1500-research.lock'] if physical else [])+(['/tmp/world_model_e2e_demo.lock'] if a.busy else []):
            lock=open(path,'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
        model=None;actor=None
        if a.busy:
            model,mlog=spawn(['taskset','-c','2-5','/home/nicklomeli/wm-venv/bin/python','-m','coprocessor.window_model',str(d)],'model')
            assert json.loads(line(model,mlog,90))['kind']=='ready'
        if a.mode!='none':
            actor,alog=spawn(['taskset','-c','2-5' if a.mode in ['cpu-parallel','cpu-blas4'] else '2','build/host_window',a.mode,str(S/'fixtures'/f's64_s{a.seed}_potentials/cell'),str(d/'result'),str(a.count),str(a.arm),'1',str(a.sleep),'1','1','0'],'actor')
            assert line(actor,alog)=='READY'
        r['start_ns']=time.monotonic_ns()+500_000_000;r['end_ns']=r['start_ns']+3_200_000_000
        r['window_controller_cpu_before']=time.process_time_ns()
        if model:model.stdin.write(json.dumps(dict(start_ns=r['start_ns'],end_ns=r['end_ns']))+'\n');model.stdin.flush()
        if actor:actor.stdin.write(f'{r["start_ns"]} {r["period_ns"]}\n');actor.stdin.flush()
        time.sleep(max(0,(r['start_ns']-time.monotonic_ns())/1e9))
        r['actor_stat_before']=Path(f'/proc/{actor.pid}/stat').read_text() if actor else None;r['host_stat_before']=Path('/proc/stat').read_text();r['host_stat_before_ns']=time.monotonic_ns()
        if actor:assert line(actor,alog)=='WINDOW_DONE'
        if model:assert json.loads(line(model,mlog))['kind']=='closed';assert model.wait(timeout=5)==0
        r['actor_stat_after']=Path(f'/proc/{actor.pid}/stat').read_text() if actor else None;r['host_stat_after']=Path('/proc/stat').read_text();r['host_stat_after_ns']=time.monotonic_ns();r['window_controller_cpu_after']=time.process_time_ns()
        r['audit_released_ns']=time.monotonic_ns()
        if actor:
            actor.stdin.write('AUDIT\n');actor.stdin.flush();assert json.loads(line(actor,alog))['status']=='PASS';assert actor.wait(timeout=5)==0
        r['status']='EXECUTION_PASS'
    except BaseException as error:
        r.update(status='FAILED_PRESERVED',error=repr(error));state['failure']=name;save(S/'STATE.json',state);raise
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        signal.alarm(0)
        for h in handles:h.close()
        for lock in locks:lock.close()
        r.update(end_utc=dt.datetime.now(dt.timezone.utc).isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,environment_after=environment())
        save(d/'OWNER.json',r);r['hashes']={p.name:sha(p) for p in d.iterdir() if p.is_file()};save(d/'RECEIPT.json',r)
    try:result=audit(d)
    except BaseException:
        state['failure']=name;save(S/'STATE.json',state);raise
    state.setdefault('window_completed',[]).append(name);state['stage']='WINDOW_CHARACTERIZATION';save(S/'STATE.json',state);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=MODES+['none','freeze']);p.add_argument('--seed',type=int,default=107501);p.add_argument('--arm',type=int,default=1);p.add_argument('--count',type=int,default=64);p.add_argument('--sleep',type=int,default=0);p.add_argument('--repeat',type=int,default=0);p.add_argument('--period',type=int,default=0);p.add_argument('--busy',action='store_true');a=p.parse_args();freeze() if a.mode=='freeze' else run(a)
