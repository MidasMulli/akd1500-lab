"""One owned multi-bank session, bounded by its frozen study envelope."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,subprocess
P=Path(__file__).resolve().parents[1];S=P/os.environ.get('AKD_LAB_STUDY','studies/20261023-resident-network')
def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('fixture');p.add_argument('--samples',type=int,default=16);p.add_argument('--mask',type=int,required=True);p.add_argument('--gdb',action='store_true');p.add_argument('--compiler-inference',action='store_true');a=p.parse_args()
    auth=json.loads((S/'AUTHORITY.json').read_text());now=datetime.datetime.now(datetime.timezone.utc)
    assert not json.loads((S/'STATE.json').read_text()).get('research_admissions_closed',False)
    assert now<datetime.datetime.fromisoformat(auth['scientific_cutoff_utc']) and 0<a.samples<=256
    (S/'raw').mkdir(exist_ok=True)
    assert len(list((S/'raw').glob('*/OWNER.json')))<auth['maximum_physical_sessions']
    assert sum(json.loads(x.read_text())['samples'] for x in (S/'raw').glob('*/OWNER.json'))+a.samples<=auth['maximum_forwards']
    f=P/a.fixture;assert f.with_suffix('.bin').exists() and f.with_suffix('.layout').exists(),'missing fixture before device admission'
    d=S/'raw'/a.name;d.mkdir(exist_ok=False)
    rec=dict(name=a.name,fixture=a.fixture,samples=a.samples,learning_mask=a.mask,compiler_inference=a.compiler_inference,start_utc=now.isoformat(),binary_sha256=hashlib.sha256((P/'build/bank_bench').read_bytes()).hexdigest(),fixture_hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in f.parent.glob(f.name+'.*')})
    (d/'OWNER.json').write_text(json.dumps(rec,indent=2)+'\n')
    with open('/tmp/akd1500-research.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
        with (d/'native.stdout').open('w') as out,(d/'native.stderr').open('w') as err:
            args=[str(P/'build/bank_bench'),str(f),str(d/'native'),str(a.samples),str(a.mask)]
            if a.compiler_inference:args.append('compiler-inference')
            if a.gdb:args=['gdb','--batch','--return-child-result','-ex','run','-ex','bt','--args']+args
            proc=subprocess.Popen(['taskset','-c','2']+args,stdout=out,stderr=err);rec['native_pid']=proc.pid;(d/'OWNER.json').write_text(json.dumps(rec,indent=2)+'\n')
            try:code=proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:code=proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
        rec.update(status='PASS' if code==0 else 'FAILED_PRESERVED',exit_code=code,device_released=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,finish_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if code==0:rec['results']=json.loads(next(line for line in (d/'native.stdout').read_text().splitlines() if line.startswith('{"status"')))
        rec['hashes']={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in d.iterdir() if x.is_file()}
        (d/'RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec.get('results',rec)),flush=True)
        assert code==0 and rec['device_released'],(d/'native.stderr').read_text()
if __name__=='__main__':main()
