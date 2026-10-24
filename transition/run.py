"""Frozen sequential gates and independent saved-array audit; owns only its process."""
from pathlib import Path
import datetime as dt, fcntl, hashlib, json, subprocess, sys
import numpy as np
from pong.relearn.model import Reference
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-learned-transition';F=S/'fixtures01'
SEEDS=[110101,110201,110301,110401,110501];ARMS=['native','frozen','disconnected'];HS=[0,1,2,4,32]
def now():return dt.datetime.now(dt.timezone.utc)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,a):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(a,indent=2)+'\n')
def name(mode,seed,arm,h):return f'{mode}_s{seed}_{arm}_h{h}'
def passed(mode,seed,arm,h):
    ps=list((S/'raw').glob(name(mode,seed,arm,h)+'*/AUDIT.json'));return any(json.loads(p.read_text())['status']=='SAVED_ARRAY_PASS' for p in ps)
def freeze():
    dest=S/'evidence'/f'FROZEN{len(list((S/"evidence").glob("FROZEN*.json")))+1:02d}.json'
    files=[* (P/'transition').glob('*'),*F.glob('*/*'),P/'src/readout_bench.cpp',P/'src/mmio_driver.hpp',P/'pong/relearn/model.py',P/'tools/prepare.py',P/'tools/program_codec.py',P/'tools/prepare_reduction.py',P/'tools/prepare_network.py',P/'coprocessor/resident_prepare.py',P/'build/learned_transition',S/'PROTOCOL.md',S/'AUTHORITY.json']
    save(dest,dict(utc=now().isoformat(),hashes={str(p.relative_to(P)):sha(p) for p in files if p.is_file()}))
def audit(d):
    r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0,(d/'stderr').read_text()
    for p,h in r['hashes'].items():assert sha(d/p)==h,p
    f=F/f's{r["seed"]}';arm=r['arm'];h=r['horizon'];rd=lambda p,t:np.fromfile(p,t)
    codes=rd(f/'codes','u1').reshape(8,256);partial=rd(f/'partial','u1').reshape(8,256);old=rd(f/'old','<i4');new=rd(f/'new','<i4');changed=rd(f/'changed','<i4');initial=rd(f/'initial.ram','<u4');states=rd(d/'result.states','<u4').reshape(26,98);y=rd(d/'result.scores','u1').reshape(26,256);roll=rd(d/'result.rollout','u1').reshape(8,h,256)
    prefix=f/f'{arm}_{h}';inputs=rd(prefix.with_suffix('.inputs'),'u1').reshape(26,256);labels=rd(prefix.with_suffix('.labels'),'<i4');ref=Reference(initial,n=256,u=8,nw=32,classes=8)
    for i,x in enumerate(inputs):
        bits=sum(1<<int(k) for k in np.flatnonzero(x));pot=ref.forward(x,int(labels[i])) if labels[i]>=0 else [(b&bits).bit_count() for b in ref.weights]
        active=np.flatnonzero(np.asarray(pot)>24);want=np.maximum.reduce(codes[active],axis=0) if len(active) else np.zeros(256,'u1')
        assert np.array_equal(y[i],want),(i,'scalar replay output');assert np.array_equal(states[i],ref.ram),(i,'scalar replay all visible RAM')
    terminal=rd(d/'result.ram','<u4');assert np.array_equal(terminal,ref.ram)
    assert np.array_equal(y[8:16],codes[old]);post=np.array([next((j for j in range(8) if np.array_equal(row,codes[j])),-1) for row in y[18:26]])
    assert np.array_equal(post,new if arm=='native' else old)
    assert np.array_equal(states[7],initial) and np.array_equal(states[15],initial)
    assert all(np.array_equal(s,states[17]) for s in states[18:]),'query prefix changed RAM'
    delta=terminal[2:].reshape(8,12)[:,4:]^initial[2:].reshape(8,12)[:,4:];replacements=sum(int(v).bit_count() for v in delta.flat)//2
    assert replacements==(32 if arm=='native' else 0)
    learned=old if arm!='native' else new;truth=[];own=[]
    for s in range(8):
        a=b=s
        for t in range(h):a=int(new[a]);b=int(learned[b]);truth.append(a);own.append(b);assert np.array_equal(roll[s,t],codes[b]),(s,t,'sealed trajectory')
    correct=sum(np.array_equal(v,codes[a]) for v,a in zip(roll.reshape(-1,256),truth))
    extra={}
    if r['mode']=='native':
        assert r['device_free'];assert np.array_equal(rd(d/'result.initial.ram','<u4'),initial);assert np.array_equal(rd(d/'result.rollout.ram','<u4'),terminal)
        for tag in ['initial','final']:assert np.array_equal(rd(d/f'result.fixed.{tag}','<u4'),rd(f/'fixed.ram','<u4'))
        if h:
            q=rd(d/'result.quiet','<i8').reshape(8,7);desc=rd(d/'result.descriptors','<u4').reshape(8,h,16);outs=rd(d/'result.output_addresses','<u4').reshape(8,h)
            assert np.all(q[:,1]-q[:,0]>=5_000_000) and np.all(q[:,3]==0) and np.all(q[:,4]==1) and np.all(q[:,6]>>16==q[:,5])
            assert np.array_equal(desc[:,:,7],outs) and np.all((desc[:,:,8]>>16&1023)==0)
            for s in range(8):
                assert np.array_equal(rd(d/f'result.seed{s}','u1'),partial[s])
                for t in range(h):
                    if t:assert desc[s,t,1]==outs[s,t-1]+32
                    assert 0x20000100<=desc[s,t,1] and int(desc[s,t,1])+256<=0x20100000
                    assert np.array_equal(rd(d/f'result.packet{s*h+t}','u1')[32:],roll[s,t])
            extra=dict(quiet_wall_ns=(q[:,1]-q[:,0]).tolist(),quiet_worker_cpu_ns=q[:,2].tolist(),driver_reads_during_quiet=0,release_writes_per_chain=1,rollout_labels_delivered=0,direct_device_sram_dependencies=True)
    a=dict(status='SAVED_ARRAY_PASS',name=d.name,mode=r['mode'],seed=r['seed'],arm=arm,horizon=h,connection_replacements=replacements,changed_recall=int(np.sum(post[changed]==new[changed])),changed_total=2,unchanged_retained=int(np.sum(post[np.setdiff1d(range(8),changed)]==new[np.setdiff1d(range(8),changed)])),unchanged_total=6,postquery_states=post.tolist(),rollout_correct_vs_new_graph=correct,rollout_total=8*h,all_outputs_and_visible_states_exact=True,**extra)
    save(d/'AUDIT.json',a);return a
def run(mode,seed,arm,h):
    assert mode in ['cpu','native'] and seed in SEEDS and arm in ARMS and h in HS
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text());cut=dt.datetime.fromisoformat(auth['scientific_cutoff_utc']);assert not state['closed'] and not state.get('failure') and now()<cut
    fr=sorted((S/'evidence').glob('FROZEN*.json'))[-1]
    for p,v in json.loads(fr.read_text())['hashes'].items():assert sha(P/p)==v,('source drift',p)
    if mode=='native':
        assert passed('cpu',seed,arm,h),'CPU gate'
        if seed==SEEDS[0] and arm=='native':
            if h:assert passed('native',seed,arm,{1:0,2:1,4:2,32:4}[h]),'previous stage'
        elif seed==SEEDS[0]:assert h==32 and passed('native',seed,'native',32)
        else:assert h==32 and all(passed('native',SEEDS[0],a,32) for a in ARMS)
    prior=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')];physical=[a for a in prior if a['mode']=='native'];assert len(physical)+(mode=='native')<=auth['maximum_physical_sessions'];assert sum(26+8*a['horizon'] for a in physical)+(26+8*h if mode=='native' else 0)<=auth['maximum_forwards']
    d=S/'raw'/name(mode,seed,arm,h)
    if d.exists():d=S/'raw'/(d.name+'_attempt'+str(len(list((S/'raw').glob(d.name+'*')))+1))
    d.mkdir(parents=True,exist_ok=False);r=dict(mode=mode,seed=seed,arm=arm,horizon=h,start_utc=now().isoformat(),freeze_sha256=sha(fr));save(d/'OWNER.json',r)
    def execute():
        cmd=['taskset','-c','2','build/learned_transition',mode,str(F/f's{seed}'),arm,str(h),str(d/'result')]
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
        state['completed'].append(d.name);save(S/'STATE.json',state);return a
    if mode=='native':
        with open('/tmp/akd1500-research.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1;result=execute()
    else:result=execute()
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='audit':print(json.dumps(audit(S/'raw'/sys.argv[2])))
    else:run(sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
