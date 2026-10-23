"""One owned scratch-transfer test; no model forward or installed-driver change."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess
P=Path(__file__).resolve().parents[1];S=P/os.environ.get('AKD_LAB_STUDY','studies/20261023-host-offload')

def main():
    auth=json.loads((S/'AUTHORITY.json').read_text());assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(auth['scientific_cutoff_utc'])
    assert len(list((S/'raw').glob('*/OWNER.json')))<auth['maximum_physical_sessions']
    d=S/'raw/transport_smoke';d.mkdir();fixture=P/'studies/20261023-compact-readout/fixtures/n128_u8_potentials'
    owner=dict(name=d.name,samples=0,native_forwards=0,binary_sha256=hashlib.sha256((P/'build/offload_bench').read_bytes()).hexdigest(),start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (d/'OWNER.json').write_text(json.dumps(owner,indent=2)+'\n')
    with open('/tmp/akd1500-research.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
        with (d/'native.stdout').open('w') as out,(d/'native.stderr').open('w') as err:
            args=['taskset','-c','2',str(P/'build/offload_bench'),'transport',str(fixture),str(d/'native'),'1','0','1','0']
            proc=subprocess.Popen(args,stdout=out,stderr=err);owner['native_pid']=proc.pid;(d/'OWNER.json').write_text(json.dumps(owner,indent=2)+'\n')
            try:code=proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:code=proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
        owner.update(status='PASS' if code==0 else 'FAILED_PRESERVED',exit_code=code,device_released=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1)
        if code==0:owner['results']=json.loads((d/'native.stdout').read_text())
        (d/'RECEIPT.json').write_text(json.dumps(owner,indent=2)+'\n');print(json.dumps(owner))
        assert code==0 and owner['device_released'],(d/'native.stderr').read_text()

if __name__=='__main__':main()
