"""Serial source-swap admission; exact parent program, independent saved audit."""
from pathlib import Path
import contextlib,datetime as dt,fcntl,io,json,shutil,subprocess,sys,types
import numpy as np
from admission import consensus_crossed as model,consensus_run as parent
from admission.consensus_crossed import *
from admission.run import now,save,sha
P=Path(__file__).resolve().parents[1];S=P/'studies/20261025-consensus-crossed';F=S/'fixtures'
check=types.FunctionType(parent.check.__code__,dict(parent.check.__globals__,simulate=simulate,assess=assess),name='crossed_saved_array_check')
def fixture(seed,arm='native'):return F/f's{seed}_{arm}'

def prepare(seed):
    source=parent.fixture(seed,'native');target=fixture(seed);target.mkdir(parents=True,exist_ok=False)
    meta=json.loads((source/'BUILD.json').read_text());assert meta['status']=='VIRTUAL_COMPILER_PASS'
    files=[p for p in source.iterdir() if p.suffix in ['.bin','.ram','.infer','.learn','.threshold','.weights']]
    for p in files:shutil.copyfile(p,target/p.name);assert sha(p)==sha(target/p.name)
    rows=schedule('crossed','native');np.asarray([[e['key'],ROLES.index(e['role']),*e['labels'],e['wire'],e['block']] for e in rows],'<i4').tofile(target/'crossed.plan')
    meta.update(parent_build_sha256=sha(source/'BUILD.json'),inherited_files={p.name:sha(p) for p in files},cells=dict(crossed=dict(events=len(rows))),scope='Exact reviewed parent program/matrices; prospective source-identity swap plan only.')
    save(target/'BUILD.json',meta)

def audit(d):
    d=Path(d);r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0,(d/'stderr').read_text()
    for k,v in r['hashes'].items():assert sha(d/k)==v,('receipt hash',k)
    def get(k):return np.fromfile(d/('result.'+k),'u1' if k in ['inputs','scores'] else '<i4' if k.startswith('potentials') or k=='learning_modes' else '<u4')
    r['directory']=str(d);a=check(r,get,lambda k:np.fromfile(fixture(r['seed'])/k,'<u4'));save(d/'AUDIT.json',a);return a

def run(mode,seed):
    assert mode in ['cpu','native'] and seed in SEEDS
    auth=json.loads((S/'AUTHORITY.json').read_text());state=json.loads((S/'STATE.json').read_text());cut=dt.datetime.fromisoformat(auth['scientific_cutoff_utc'])
    assert not state['closed'] and not state.get('failure') and not state.get('scientific_stop') and now()<cut
    f=fixture(seed);meta=json.loads((f/'BUILD.json').read_text())
    for name,h in meta['inherited_files'].items():assert sha(f/name)==h,'unchanged inherited program/matrix'
    if mode=='native':
        assert (S/'raw'/f'cpu_s{seed}_native_crossed/AUDIT.json').exists()
        for earlier in SEEDS[:SEEDS.index(seed)]:assert (S/'raw'/f'native_s{earlier}_native_crossed/AUDIT.json').exists()
    physical=list(S.glob('raw/native_*/OWNER.json'));n=len(schedule('crossed','native'))
    assert len(physical)+int(mode=='native')<=auth['maximum_physical_sessions']
    assert sum(json.loads(p.read_text())['reserved_events'] for p in physical)+(n if mode=='native' else 0)<=auth['maximum_forwards']
    def execute():
        d=S/'raw'/f'{mode}_s{seed}_native_crossed';d.mkdir(parents=True,exist_ok=False)
        own=[P/'admission'/p for p in ['consensus_crossed.py','consensus_crossed_run.py','consensus.py','consensus_run.py','consensus_bench.cpp','transfer.py','revision.py','model.py','recycle.py','error.py','run.py']]+[P/'src/readout_bench.cpp',P/'src/mmio_driver.hpp',P/'build/native_consensus',*f.glob('*'),*S.glob('*.md'),S/'AUTHORITY.json']
        save(d/'FROZEN.json',dict(utc=now().isoformat(),hashes={str(p.relative_to(P)):sha(p) for p in own if p.is_file()}))
        r=dict(mode=mode,seed=seed,arm='native',stage='crossed',start_utc=now().isoformat(),reserved_events=n,freeze_sha256=sha(d/'FROZEN.json'));save(d/'OWNER.json',r)
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            proc=subprocess.Popen(['taskset','-c','2','build/native_consensus',mode,str(f),'native','crossed',str(d/'result')],cwd=P,stdout=out,stderr=err);r['pid']=proc.pid;save(d/'OWNER.json',r)
            try:code=proc.wait(timeout=min(90,max(1,(cut-now()).total_seconds())))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:code=proc.wait(timeout=2)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
        r.update(exit_code=code,finish_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,hashes={p.name:sha(p) for p in d.iterdir() if p.is_file()});save(d/'RECEIPT.json',r)
        try:a=audit(d)
        except Exception:state['failure']=d.name;save(S/'STATE.json',state);raise
        state['completed'].append(d.name)
        if mode=='native' and not a['useful_gate']:state['scientific_stop']=d.name
        save(S/'STATE.json',state);return a
    if mode=='native':
        with open('/tmp/akd1500-research.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
            result=execute()
    else:result=execute()
    return result

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare(int(sys.argv[2]))
    elif sys.argv[1]=='audit':print(json.dumps(audit(S/'raw'/sys.argv[2])))
    else:print(json.dumps(run(sys.argv[1],int(sys.argv[2]))))
