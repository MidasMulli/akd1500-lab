"""One bounded cell, explicit ownership and immutable raw directory per attempt."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,subprocess,time
P=Path(__file__).resolve().parents[1];S=P/os.environ.get('AKD_LAB_STUDY','studies/20261023-measurement')
def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('fixture');p.add_argument('--samples',type=int,default=64);p.add_argument('--learning',type=int,choices=[0,1],default=1);p.add_argument('--cpu-repeats',type=int,default=128);p.add_argument('--resident',action='store_true');a=p.parse_args()
    auth=json.loads((S/'AUTHORITY.json').read_text());now=datetime.datetime.now(datetime.timezone.utc)
    assert now<datetime.datetime.fromisoformat(auth['scientific_cutoff_utc'])
    assert len(list((S/'raw').glob('*/OWNER.json')))<auth['maximum_physical_sessions']
    assert sum(json.loads(x.read_text())['samples'] for x in (S/'raw').glob('*/OWNER.json'))+a.samples<=auth['maximum_forwards']
    (S/'raw').mkdir(exist_ok=True)
    d=S/'raw'/a.name;d.mkdir(exist_ok=False);prefix=P/a.fixture
    assert prefix.with_suffix('.bin').exists()
    rec=dict(name=a.name,fixture=a.fixture,samples=a.samples,learning=a.learning,cpu_repeats=a.cpu_repeats,start_utc=now.isoformat(),binary_sha256=hashlib.sha256((P/'build/learning_bench').read_bytes()).hexdigest(),program_sha256=hashlib.sha256(prefix.with_suffix('.bin').read_bytes()).hexdigest())
    (d/'OWNER.json').write_text(json.dumps(rec,indent=2)+'\n')
    with open('/tmp/akd1500-research.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
        for mode in ['cpu','native'] if sum(map(ord,a.name))%2 else ['native','cpu']:
            args=['taskset','-c','2',str(P/'build/learning_bench'),'resident' if mode=='native' and a.resident else mode,str(prefix),str(d/mode),str(a.samples),str(a.learning),str(a.cpu_repeats if mode=='cpu' else 1),'0']
            with (d/(mode+'.stdout')).open('w') as out,(d/(mode+'.stderr')).open('w') as err:
                pr=subprocess.Popen(args,stdout=out,stderr=err);rec[mode+'_pid']=pr.pid;(d/'OWNER.json').write_text(json.dumps(rec,indent=2)+'\n')
                try:code=pr.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    pr.terminate()
                    try:code=pr.wait(timeout=3)
                    except subprocess.TimeoutExpired:pr.kill();code=pr.wait()
                rec[mode+'_exit_code']=code
            if code:
                rec['status']='FAILED_PRESERVED';(d/'RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n');raise RuntimeError((d/(mode+'.stderr')).read_text()[-2000:])
        assert (d/'native.scores').read_bytes()==(d/'cpu.scores').read_bytes(),'raw potential vectors differ'
        assert (d/'native.ram').read_bytes()==(d/'cpu.ram').read_bytes(),'raw complete learning RAM differs'
        rec.update(status='PASS',finish_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),device_released=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1)
        rec['results']={m:json.loads((d/(m+'.stdout')).read_text()) for m in ['native','cpu']}
        rec['hashes']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in d.iterdir() if f.is_file()}
        (d/'RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec['results']),flush=True)
if __name__=='__main__':main()
