"""Frozen sequential admissions and saved-array verification of native feedback."""
from pathlib import Path
import datetime as dt, fcntl, hashlib, json, subprocess, sys
import numpy as np
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-native-feedback';F=S/'fixtures01'
def now():return dt.datetime.now(dt.timezone.utc)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n')
def name(mode,seed,arm,h):return f'{mode}_s{seed}_{arm}_h{h}'
def passed(mode,seed,arm,h):
    p=S/'raw'/name(mode,seed,arm,h)/'AUDIT.json'
    return p.exists() and json.loads(p.read_text())['status']=='SAVED_ARRAY_PASS'
def freeze():
    p=S/'evidence/FROZEN01.json';assert not p.exists()
    files=[* (P/'recurrence').glob('*'),*F.glob('*/*'),P/'src/readout_bench.cpp',P/'src/mmio_driver.hpp',P/'pong/relearn/model.py',P/'tools/program_codec.py',P/'tools/prepare_reduction.py',P/'tools/prepare_network.py',P/'build/native_feedback',S/'PROTOCOL.md',S/'AUTHORITY.json']
    save(p,dict(utc=now().isoformat(),hashes={str(q.relative_to(P)):sha(q) for q in files if q.is_file()}))
def audit(d):
    r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0,(d/'stderr').read_text()
    for p,h in r['hashes'].items():assert sha(d/p)==h,p
    f=F/f's{r["seed"]}';arm=r['arm'];h=r['horizon'];prefix=f/f'{arm}_{h}'
    def read(p,t):return np.fromfile(p,t)
    y=read(d/'result.scores','u1').reshape(h,128);want=read(prefix.with_suffix('.outputs'),'u1').reshape(h,128);assert np.array_equal(y,want),'independent Python output replay'
    ram=read(d/'result.ram','<u4');states=read(prefix.with_suffix('.states'),'<u4').reshape(h,1026);assert np.array_equal(ram,states[-1]),'independent complete terminal SRAM'
    pq=read(d/'result.probe_scores','u1').reshape(4,128);assert np.array_equal(pq,read(prefix.with_suffix('.probe_outputs'),'u1').reshape(4,128))
    assert np.array_equal(read(d/'result.probe_ram','<u4'),read(prefix.with_suffix('.probe_states'),'<u4').reshape(4,1026)[-1])
    q=idle=None
    if r['mode']=='native':
        assert r['device_free'];q=read(d/'result.quiet','<i8');idle=read(d/'result.idle','<i8')
        assert q[1]-q[0]>=5_000_000 and q[3]==0 and q[4]==1 and q[6]>>16==q[5]
        assert idle[1]-idle[0]>=20_000_000 and idle[3]==idle[4]==0 and idle[5]==q[6]
        assert np.array_equal(read(d/'result.idle.ram','<u4'),ram)
        initial=read(f/'initial.ram','<u4');assert np.array_equal(read(d/'result.initial.ram','<u4'),initial)
        for tag in ['initial','final']:assert np.array_equal(read(d/f'result.fixed.{tag}','<u4'),read(f/'fixed.ram','<u4'))
        desc=read(d/'result.descriptors','<u4').reshape(h,16);outs=read(d/'result.output_addresses','<u4');assert np.array_equal(outs,desc[:,7])
        assert np.all(desc[:,1]>=0x20000100) and np.all(desc[:,1].astype('u8')+128<=0x20100000)
        for i in range(1,h):
            if arm in ['native','frozen'] or arm=='severed' and i>1:assert desc[i,1]==outs[i-1]+32
            elif arm=='repeated_seed':assert desc[i,1]==desc[0,1]
            else:assert desc[i,1]!=outs[i-1]+32
        for i in range(h):
            packet=read(d/f'result.output_packet{i}','u1');assert np.array_equal(packet[32:160],y[i]),'raw SRAM output is next payload'
    initial=read(f/'initial.ram','<u4');before=initial.reshape(-1);delta=(ram[2:].reshape(128,8)[:,4:]^initial[2:].reshape(128,8)[:,4:]);edits=sum(int(v).bit_count() for v in delta.flat)//2
    a=dict(status='SAVED_ARRAY_PASS',name=d.name,mode=r['mode'],seed=r['seed'],arm=arm,horizon=h,output_active=y.sum(1).tolist(),output_groups=[np.flatnonzero(a).tolist() for a in y],post_loop_partial_cue_support=pq.sum(1).tolist(),terminal_connection_replacements=edits,full_terminal_ram_exact=True)
    if q is not None:a.update(quiet_wall_ns=int(q[1]-q[0]),quiet_worker_cpu_ns=int(q[2]),driver_reads_during_quiet=int(q[3]),release_writes=int(q[4]),idle_wall_ns=int(idle[1]-idle[0]),idle_worker_cpu_ns=int(idle[2]),post_exhaustion_job_state_stable=True,post_exhaustion_ram_stable=True)
    save(d/'AUDIT.json',a);return a
def run(mode,seed,arm,h):
    assert mode in ['cpu','native'] and seed in [109101,109201,109301] and arm in ['native','frozen','repeated_seed','severed'] and h in [1,2,3,4,5,8,16,32]
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text());cut=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert not state['closed'] and not state.get('failure') and now()<cut
    frozen=S/'evidence/FROZEN01.json'
    for p,v in json.loads(frozen.read_text())['hashes'].items():assert sha(P/p)==v,('source drift',p)
    if mode=='native':
        assert passed('cpu',seed,arm,h),'CPU gate'
        if seed==109101:
            if arm=='frozen' and h in [2,4]:assert passed('native',seed,'frozen',h//2)
            elif arm=='native' and h in [1,2,4]:assert passed('native',seed,'frozen',4) if h==1 else passed('native',seed,'native',h//2)
            elif arm=='native':assert passed('native',seed,'native',4)
            elif h==32:assert passed('native',seed,'native',32)
        else:
            assert h==32 and all(passed('native',109101,a,32) for a in ['native','frozen','repeated_seed','severed'])
    prior=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')];physical=[r for r in prior if r['mode']=='native'];assert len(physical)+(mode=='native')<=auth['maximum_physical_sessions'];assert sum(r['horizon']+4 for r in physical)+(h+4 if mode=='native' else 0)<=auth['maximum_forwards']
    d=S/'raw'/name(mode,seed,arm,h);d.mkdir(parents=True,exist_ok=False);r=dict(mode=mode,seed=seed,arm=arm,horizon=h,start_utc=now().isoformat(),freeze_sha256=sha(frozen));save(d/'OWNER.json',r)
    def execute():
        cmd=['taskset','-c','2','build/native_feedback',mode,str(F/f's{seed}'),arm,str(h),str(d/'result')]
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            proc=subprocess.Popen(cmd,cwd=P,stdout=out,stderr=err);r['pid']=proc.pid;save(d/'OWNER.json',r)
            try:code=proc.wait(timeout=min(30,(cut-now()).total_seconds()))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:code=proc.wait(timeout=2)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
        r.update(exit_code=code,finish_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1);r['hashes']={p.name:sha(p) for p in d.iterdir() if p.is_file()};save(d/'RECEIPT.json',r)
        try:a=audit(d)
        except Exception:state['failure']=d.name;save(S/'STATE.json',state);raise
        state.setdefault('completed',[]).append(d.name);save(S/'STATE.json',state);return a
    if mode=='native':
        with open('/tmp/akd1500-research.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1;result=execute()
    else:result=execute()
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='audit':print(json.dumps(audit(S/'raw'/sys.argv[2])))
    else:run(sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
