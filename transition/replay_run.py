"""Separate source-frozen finite replay admissions, after primary endpoints pass."""
from transition.run import *
F2=S/'replay_fixtures03'
def freeze2():
    p=S/'evidence'/f'REPLAY_FROZEN{len(list((S/"evidence").glob("REPLAY_FROZEN*.json")))+1:02d}.json'
    files=[P/'transition/replay.cpp',P/'transition/replay_pack.py',P/'transition/replay_run.py',P/'src/readout_bench.cpp',P/'src/mmio_driver.hpp',P/'tools/program_codec.py',P/'build/transition_replay',S/'REPLAY-PROTOCOL.md',*F2.glob('*')]
    save(p,dict(utc=now().isoformat(),hashes={str(q.relative_to(P)):sha(q) for q in files if q.is_file()}))
def audit2(d):
    r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0,(d/'stderr').read_text()
    for k,v in r['hashes'].items():assert sha(d/k)==v
    h=r['horizon'];y=np.fromfile(d/'result.scores','u1').reshape(8,256);want=np.fromfile(F2/f'h{h}.expected','u1').reshape(8,256);assert np.array_equal(y,want)
    q=np.fromfile(d/'result.quiet','<i8').reshape(8,7);assert np.all(q[:,1]-q[:,0]>=5_000_000) and np.all(q[:,3]==0) and np.all(q[:,4]==1) and np.all(q[:,6]>>16==q[:,5])
    desc=np.fromfile(d/'result.descriptors','<u4').reshape(8,16);assert np.all((desc[:,8]>>16&1023)==0)
    for i in range(8):assert np.array_equal(np.fromfile(d/f'result.packet{i}','u1')[32:],y[i])
    assert r['device_free'];a=dict(status='SAVED_ARRAY_PASS',horizon=h,terminal_states_correct=8,total=8,host_descriptors_per_seed=1,quiet_wall_ns=(q[:,1]-q[:,0]).tolist(),quiet_worker_cpu_ns=q[:,2].tolist(),learning_enabled=False);save(d/'AUDIT.json',a);return a
def run2(h):
    assert h in [1,2,4,8];a=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text());cut=dt.datetime.fromisoformat(a['scientific_cutoff_utc']);assert not state['closed'] and not state.get('failure') and now()<cut
    assert all(passed('native',seed,arm,32) for seed in SEEDS for arm in ARMS),'complete primary gate'
    if h>1:assert json.loads((S/f'replay_raw/h{h//2}/AUDIT.json').read_text())['status']=='SAVED_ARRAY_PASS'
    frozen=sorted((S/'evidence').glob('REPLAY_FROZEN*.json'))[-1]
    for p,v in json.loads(frozen.read_text())['hashes'].items():assert sha(P/p)==v,p
    owners=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json') if json.loads(p.read_text())['mode']=='native'];rp=list((S/'replay_raw').glob('*/OWNER.json'));assert len(owners)+len(rp)+1<=a['maximum_physical_sessions'];assert sum(26+8*x['horizon'] for x in owners)+sum(8*json.loads(p.read_text())['horizon'] for p in rp)+8*h<=a['maximum_forwards']
    d=S/f'replay_raw/h{h}'
    if d.exists():d=S/'replay_raw'/(d.name+'_attempt'+str(len(list((S/'replay_raw').glob(d.name+'*')))+1))
    d.mkdir(parents=True,exist_ok=False);r=dict(mode='native_finite_replay',horizon=h,start_utc=now().isoformat(),freeze_sha256=sha(frozen));save(d/'OWNER.json',r)
    with open('/tmp/akd1500-research.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            proc=subprocess.Popen(['taskset','-c','2','build/transition_replay',str(F2),str(h),str(d/'result')],stdout=out,stderr=err,cwd=P);r['pid']=proc.pid;save(d/'OWNER.json',r)
            try:code=proc.wait(timeout=min(20,(cut-now()).total_seconds()))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:code=proc.wait(timeout=2)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
        r.update(exit_code=code,finish_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,hashes={p.name:sha(p) for p in d.iterdir() if p.is_file()});save(d/'RECEIPT.json',r)
        try:result=audit2(d)
        except Exception:state['failure']='replay_h'+str(h);save(S/'STATE.json',state);raise
    state.setdefault('replay_completed',[]).append(h);save(S/'STATE.json',state);print(json.dumps(result),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze2()
    elif sys.argv[1]=='audit':print(json.dumps(audit2(S/'replay_raw'/sys.argv[2])))
    else:run2(int(sys.argv[1]))
