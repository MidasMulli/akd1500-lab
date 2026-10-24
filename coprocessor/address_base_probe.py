"""Zero-forward stock-runtime storage provenance, read-only observations."""
import datetime as dt,fcntl,hashlib,json,mmap,os,subprocess
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-fnp2-base'
def main():
 import akida
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 a=json.loads((S/'AUTHORITY.json').read_text());assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(a['scientific_cutoff_utc'])
 assert not (S/'STATE.json').exists()
 f=P/'studies/20261024-fnp2-sdk/fixtures01/above/learner'
 program=f.with_suffix('.bin').read_bytes();initial=f.with_suffix('.raw').read_bytes()
 freeze=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),hashes={str(p.relative_to(P)):sha(p) for p in [Path(__file__),f.with_suffix('.bin'),f.with_suffix('.raw'),S/'PROTOCOL.md',S/'AUTHORITY.json']})
 (S/'evidence/FROZEN.json').write_text(json.dumps(freeze,indent=2)+'\n')
 with open('/tmp/akd1500-research.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
  dev=None;result=dict(forwards=0,sdk=akida.__version__,custom_transport=False)
  try:
   dev=akida.devices()[0];dev.program=program;dev.learn_enabled=True
   fd=os.open('/dev/akd1500_0',os.O_RDONLY|os.O_CLOEXEC)
   regs=mmap.mmap(fd,0x100000,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ,offset=0xFCC00000)
   address=int.from_bytes(regs[0x10:0x14],'little');regs.close()
   if 0xC0000000<=address and address+len(initial)<=0xC0200000:
    mm=mmap.mmap(fd,0x200000,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ,offset=0xC0000000);direct=mm[address-0xC0000000:address-0xC0000000+len(initial)];mm.close();domain='host_DDR'
   elif 0x20000100<=address and address+len(initial)<=0x20100000:
    mm=mmap.mmap(fd,0x100000,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ,offset=0x20000000);direct=mm[address-0x20000000:address-0x20000000+len(initial)];mm.close();domain='card_SRAM'
   else:raise AssertionError(('unqualified address',address))
   os.close(fd)
   assert bytes(dev.learn_mem)==initial==direct
   result.update(status='ZERO_FORWARD_STORAGE_PROVENANCE_EXACT',record_address=address,record_bytes=len(initial),domain=domain,api_direct_initial_sha256=hashlib.sha256(direct).hexdigest(),runtime_memory=list(dev.memory),desc=str(dev.desc))
  finally:
   if dev is not None:dev.unprogram()
  del dev
 assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
 result.update(device_free=True,utc=dt.datetime.now(dt.timezone.utc).isoformat());(S/'evidence/RESULT.json').write_text(json.dumps(result,indent=2)+'\n');(S/'STATE.json').write_text(json.dumps(dict(closed=True,status=result['status'],owned_jobs_remaining=0),indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
