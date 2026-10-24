"""Frozen staged admissions, exclusive ownership, measured outputs and cleanup."""
import datetime as dt
import fcntl
import json
import subprocess
import sys
import numpy as np
from capability.design import P,S,CELLS,inputs,smoke_indices
from capability import audit
from metal.run import digest
from resident.or_design import schedule


def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')


def freeze():
    f=list((P/'capability').glob('*.py'))+[P/p for p in ['src/primitive_bench.cpp','src/primitive_cpu.cpp','src/quiet_relay.cpp','src/readout_bench.cpp','src/mmio_driver.hpp','tools/build.sh','build/primitive_bench','build/primitive_cpu','build/quiet_relay','resident/or_design.py','resident/design.py','pong/relearn/model.py']]
    f+=list((S/'fixtures01').iterdir())+list((S/'fixtures-quiet').iterdir())+[S/'PROTOCOL.md',S/'AUTHORITY.json']
    target=S/'evidence/FROZEN.json';assert not target.exists();save(target,dict(status='FROZEN_BEFORE_PHYSICAL',utc=dt.datetime.now(dt.timezone.utc).isoformat(),hashes={str(p.relative_to(P)):digest(p) for p in f if p.is_file()}))


def fixed_name(cell,smoke=False,repeat=0):return f'{"smoke" if smoke else "full"}_{cell}_r{repeat}'


def passed(name):
    p=S/'raw'/name/'AUDIT.json';return p.exists() and json.loads(p.read_text())['status'].startswith('INDEPENDENT_')


def admit(kind,**kw):
    state=json.loads((S/'STATE.json').read_text());assert not state['research_admissions_closed'] and kind not in state.get('halted_branches',[])
    if kind=='fixed':
        cell=kw['cell'];smoke=kw['smoke'];repeat=kw['repeat'];assert cell in [f'{k}{u}' for k,u in CELLS] and repeat in [0,1,2]
        if smoke:
            assert cell in ['fnp128','cnp128'] and repeat==0
            if cell=='cnp128':assert passed(fixed_name('fnp128',True))
        else:
            assert passed(fixed_name('cnp128',True))
            pos=[f'{k}{u}' for k,u in CELLS].index(cell)
            if repeat:assert all(passed(fixed_name(f'{k}{u}',False,0)) for k,u in CELLS)
            elif pos:assert passed(fixed_name(f'{CELLS[pos-1][0]}{CELLS[pos-1][1]}'))
        assert (S/'cpu/u128/AUDIT.json').exists(),'CPU numerical screen before physical'
        name=fixed_name(cell,smoke,repeat)
    else:
        assert kind=='quiet';chunk=kw['chunk'];sleep=kw['sleep_ms'];repeat=kw['repeat'];assert (chunk,sleep) in [(8,0),(8,20),(32,100),(128,250)] and repeat in [0,1,2]
        if repeat:assert chunk==128 and passed('quiet_b128_s250_r0')
        elif chunk==32:assert passed('quiet_b8_s0_r0') and passed('quiet_b8_s20_r0')
        elif chunk==128:assert passed('quiet_b32_s100_r0')
        elif sleep:assert passed('quiet_b8_s0_r0')
        name=f'quiet_b{chunk}_s{sleep}_r{repeat}'
    assert name not in [r['name'] for r in state['admitted_runs']];state['admitted_runs'].append(dict(name=name,kind=kind,**kw,admitted_utc=dt.datetime.now(dt.timezone.utc).isoformat()));save(S/'STATE.json',state);return name


def check_freeze():
    f=S/'evidence/FROZEN.json'
    for p,h in json.loads(f.read_text())['hashes'].items():assert digest(P/p)==h,('executed source/fixture drift',p)
    a=json.loads((S/'AUTHORITY.json').read_text());now=dt.datetime.now(dt.timezone.utc);cut=dt.datetime.fromisoformat(a['scientific_cutoff_utc']);assert (cut-now).total_seconds()>10
    return a,now,cut


def run(name):
    state=json.loads((S/'STATE.json').read_text());assert not state['research_admissions_closed'];entry=next(r for r in state['admitted_runs'] if r['name']==name);assert entry['kind'] not in state.get('halted_branches',[])
    auth,now,cut=check_freeze();d=S/'raw'/name
    if entry['kind']=='fixed':
        f=S/'fixtures01'/entry['cell'];meta=json.loads(f.with_suffix('.json').read_text());x,rows=inputs();ii=smoke_indices(rows) if entry['smoke'] else list(range(160));x=x[ii];rows=[rows[i] for i in ii];count=len(x);forwards=count+48
    else:f=S/'fixtures-quiet/u512_s102101';meta=json.loads(f.with_suffix('.json').read_text());x,rows=schedule('repair');count=forwards=len(rows)
    prior=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json')];assert len(prior)<auth['maximum_physical_sessions'] and sum(r['forwards'] for r in prior)+forwards<=auth['maximum_forwards']
    with open('/tmp/akd1500-research.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
        d.mkdir(exist_ok=False);stream=d/'stream';stream.with_suffix('.inputs').write_bytes(x.tobytes());save(stream.with_suffix('.rows.json'),rows)
        if entry['kind']=='fixed':
            stream.with_suffix('.labels').write_bytes(np.zeros(count,'<i4').tobytes());stream.with_suffix('.modes').write_bytes(bytes(count));want=np.fromfile(f.with_suffix('.expected'),'<i4').reshape(160,2)[ii];stream.with_suffix('.expected').write_bytes(want.tobytes())
            args=['build/primitive_bench',str(f),str(stream),str(d/'native'),str(count),'fixed']
        else:
            stream.with_suffix('.labels').write_bytes(np.asarray([r['label'] for r in rows],'<i4').tobytes());stream.with_suffix('.masks').write_bytes(bytes(r['mask'] for r in rows));args=['build/quiet_relay',str(f),str(stream),str(d/'native'),str(entry['chunk']),str(entry['sleep_ms'])]
        rec=dict(**entry,fixture=str(f.relative_to(P)),fixture_info=meta,samples=count,forwards=forwards,start_utc=now.isoformat(),source_freeze_sha256=digest(S/'evidence/FROZEN.json'));save(d/'OWNER.json',rec)
        with (d/'native.stdout').open('w') as out,(d/'native.stderr').open('w') as err:
            p=subprocess.Popen(['taskset','-c','2']+args,cwd=P,stdout=out,stderr=err);rec['native_pid']=p.pid;save(d/'OWNER.json',rec)
            try:code=p.wait(timeout=min(auth['maximum_individual_seconds'],int((cut-now).total_seconds())))
            except subprocess.TimeoutExpired:
                p.terminate()
                try:code=p.wait(timeout=3)
                except subprocess.TimeoutExpired:p.kill();code=p.wait()
        rec.update(status='PASS' if code==0 else 'FAILED_PRESERVED',exit_code=code,device_released=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,finish_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        if code==0:rec['result']=json.loads(next(l for l in (d/'native.stdout').read_text().splitlines() if l.startswith('{"status"')))
        rec['hashes']={p.name:digest(p) for p in d.iterdir() if p.is_file() and p.name!='RECEIPT.json'};save(d/'RECEIPT.json',rec)
        try:
            assert code==0 and rec['device_released'],(d/'native.stderr').read_text();report=audit_run(d)
        except Exception:
            state.setdefault('halted_branches',[]).append(entry['kind']);state['failure']=name;save(S/'STATE.json',state);raise
    print(json.dumps(report),flush=True)


def arrays(d,rec):
    z={'inputs':np.fromfile(d/'stream.inputs',np.uint8).reshape(rec['samples'],512)}
    def add(k,suffix,dtype,shape=None):
        a=np.fromfile(d/('native.'+suffix),dtype);z[k]=a.reshape(shape) if shape else a
    if rec['kind']=='fixed':
        for k,dtp,shape in [('scores','<i4',(-1,2)),('cycles','<u4',None),('counter_bounds','<u4',(-1,2)),('config_bounds','<u4',(-1,2)),('latency_ns','<i8',None),('caller_ns','<i8',None),('idle_counter','<u4',None),('serial_suffix','<i4',(3,8,2)),('burst_suffix','<i4',(3,8,2)),('burst_cycles','<u4',(3,2)),('burst_latency_ns','<i8',(3,2))]:add(k,k,dtp,shape)
        f=P/rec['fixture']
        for j in range(len(rec['fixture_info']['packets'])):
            for suffix in ['initial','final']:add(f'packet{j}.{suffix}',f'packet{j}.{suffix}','<u4')
            z[f'packet{j}.expected']=np.fromfile(f.parent/f'{f.name}.packet{j}.ram','<u4')
    else:
        z['labels']=np.fromfile(d/'stream.labels','<i4');z['masks']=np.fromfile(d/'stream.masks',np.uint8)
        for k,dtp,shape in [('scores','<i4',(768,4)),('boundary_indices','<i4',None),('released_intervals_ns','<i8',(-1,5)),('host_activity','<i8',(-1,5)),('counter_jobs','<u4',(-1,4))]:add(k,k,dtp,shape)
        for k in range(3):
            for suffix in ['initial','final','snapshots']:add(f'bank{k}.{suffix}',f'bank{k}.{suffix}','<u4',(-1,10242) if suffix=='snapshots' else None)
            for suffix in ['initial','final']:add(f'fixed{k}.{suffix}',f'fixed{k}.{suffix}','<u4')
            z[f'fixed{k}.expected']=np.fromfile(P/(rec['fixture']+f'.fixed{k}.ram'),'<u4')
    return z


def audit_run(d):
    rec=json.loads((d/'RECEIPT.json').read_text());assert rec['status']=='PASS'
    for f,h in rec['hashes'].items():assert digest(d/f)==h,('raw file drift',f)
    if rec['kind']=='quiet':rec['rows']=json.loads((d/'stream.rows.json').read_text())
    if rec['kind']=='fixed':rec['units']=rec['fixture_info']['units']
    result=(audit.fixed if rec['kind']=='fixed' else audit.quiet)(rec,arrays(d,rec));save(d/'AUDIT.json',result);return result


def cpu_run(u):
    check_freeze();assert u in [128,256,512,1024,2048];f=S/'fixtures01'/f'cnp{u}';d=S/'cpu'/f'u{u}';d.mkdir(parents=True,exist_ok=False)
    args=['taskset','-c','2','build/primitive_cpu',str(f),str(d/'cpu'),str(u)]
    with (d/'cpu.stdout').open('w') as out,(d/'cpu.stderr').open('w') as err:r=subprocess.run(args,cwd=P,stdout=out,stderr=err,timeout=120)
    rec=dict(status='PASS' if r.returncode==0 else 'FAILED_PRESERVED',units=u,physical_device_opened=False,result=json.loads((d/'cpu.stdout').read_text()) if r.returncode==0 else None)
    save(d/'RECEIPT.json',rec);assert r.returncode==0,(d/'cpu.stderr').read_text()
    z=cpu_arrays(d,rec);result=audit.cpu(rec,z);save(d/'AUDIT.json',result);print(json.dumps(result),flush=True)


def cpu_arrays(d,rec):
    f=S/'fixtures01'/f'cnp{rec["units"]}'
    return dict(inputs=np.fromfile(f.with_suffix('.inputs'),np.uint8).reshape(160,512),scores=np.fromfile(d/'cpu.scores','<i4').reshape(3,160,2),wall_ns=np.fromfile(d/'cpu.wall_ns','<i8').reshape(3,3,5),cpu_ns=np.fromfile(d/'cpu.cpu_ns','<i8').reshape(3,3,5))


if __name__=='__main__':
    a=sys.argv[1:]
    if a[0]=='freeze':freeze()
    elif a[0]=='admit-fixed':print(admit('fixed',cell=a[1],smoke=a[2]=='smoke',repeat=int(a[3])))
    elif a[0]=='admit-quiet':print(admit('quiet',chunk=int(a[1]),sleep_ms=int(a[2]),repeat=int(a[3])))
    elif a[0]=='cpu':cpu_run(int(a[1]))
    elif a[0]=='audit':print(json.dumps(audit_run(S/'raw'/a[1])))
    else:run(a[0])
