"""Prospectively frozen stock-SDK comparison; no custom device transport."""
import datetime as dt,fcntl,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from coprocessor.fnp2_boundary_run import trajectory
P=Path(__file__).resolve().parents[1]
S=P/'studies/20261024-fnp2-start'
CASES=['below','above']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def now():return dt.datetime.now(dt.timezone.utc)
def prefix(case,learn):
 f=S/'fixtures01'/case/'learner';meta=json.loads(f.with_suffix('.json').read_text());n,u=meta['input_features'],meta['records']
 a=np.fromfile(f.with_suffix('.raw'),'<u4');x=np.fromfile(f.with_suffix('.inputs'),np.uint8).reshape(2,n)
 from pong.relearn.model import Reference
 ref=Reference([0,0]+a.tolist(),n=n,u=u,nw=32,classes=2);ys=[];states=[]
 for i,v in enumerate(x):
  if learn:y=ref.forward(v,i%2)
  else:
   bits=sum(1<<int(k) for k in np.flatnonzero(v));y=[(w&bits).bit_count() for w in ref.weights]
  ys.append(y);states.append(ref.ram[2:].copy())
 return f,a,x,np.asarray(ys,'<i4'),np.asarray(states,'<u4')
def preflight():
 for case in CASES:
  for learn in [False,True]:
   f,a,x,ys,states=prefix(case,learn);m=json.loads(f.with_suffix('.json').read_text());t=m['target'];u=m['records'];stride=m['record_words']
   assert np.array_equal(ys,np.fromfile(f.with_suffix('.expected'),'<i4').reshape(2,u))
   assert ys[1,t]==31 and ys[1,u//2:].max()==31 and np.count_nonzero(ys[1,u//2:]==31)==1
   changed=np.flatnonzero(np.any(states[-1].reshape(u,stride)!=a.reshape(u,stride),axis=1)).tolist();assert changed==([t] if learn else [])
 save(S/'evidence/PREFLIGHT.json',dict(status='CPU_AND_VIRTUAL_EXACT',physical_device_opened=False))
def freeze():
 assert not (S/'evidence/FROZEN.json').exists()
 import akida
 assert akida.__version__=='2.19.3'
 preflight()
 files=[p for p in (S/'fixtures01').rglob('*') if p.is_file()]
 files += [P/'coprocessor/address_start.py',P/'coprocessor/address_start_prepare.py',P/'pong/relearn/model.py',S/'PROTOCOL.md',S/'AUTHORITY.json']
 sdk=Path(akida.__file__).parent
 save(S/'evidence/FROZEN.json',dict(utc=now().isoformat(),hashes={str(p.relative_to(P)):sha(p) for p in files},sdk_version=akida.__version__,sdk_binary_hashes={str(p):sha(p) for p in [sdk/'core.so',sdk/'libakida.so.2']}))
def worker(case,learn,d):
 import akida
 assert akida.__version__=='2.19.3'
 f,initial,x,ys,states=prefix(case,learn)
 dev=None;programmed=False
 meta=json.loads(f.with_suffix('.json').read_text());n,u,stride=meta['input_features'],meta['records'],meta['record_words']
 seen=[];snap=[];mm=None;fd=None
 info=dict(sdk_version=akida.__version__,default_driver=True,custom_transport_used=False)
 def capture():
  dev.learn_enabled=True
  raw=dev.learn_mem
  arr=np.asarray(raw) if not isinstance(raw,(bytes,bytearray,memoryview)) else np.frombuffer(raw,np.uint8)
  info['api_memory_type']=str(type(raw));info['api_memory_dtype']=str(arr.dtype)
  assert arr.dtype in [np.dtype('uint8'),np.dtype('uint32')],str(arr.dtype)
  result=np.ascontiguousarray(arr).view('<u4').reshape(-1).copy()
  dev.learn_enabled=bool(learn)
  direct=np.frombuffer(mm[address-host:address-host+initial.nbytes],'<u4').copy()
  assert np.array_equal(result,direct),'stock API and host-DDR records differ'
  return result
 try:
  devices=akida.devices();assert len(devices)==1,f'one owned device expected, got {len(devices)}'
  dev=devices[0];info.update(device_type=str(type(dev)),desc=str(dev.desc),soc=str(dev.soc))
  program=f.with_suffix('.bin').read_bytes();dev.program=program;programmed=True
  import mmap,os,struct
  fd=os.open('/dev/akd1500_0',os.O_RDONLY|os.O_CLOEXEC);regs=mmap.mmap(fd,0x100000,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ,offset=0xFCC00000);address=struct.unpack_from('<I',regs,0x10)[0];regs.close();host=0xC0000000
  assert host<=address and address+initial.nbytes<=host+0x200000
  mm=mmap.mmap(fd,0x200000,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ,offset=host)
  info.update(record_address=address,record_bytes=initial.nbytes,storage_domain='verified_host_DDR',record_words=stride,input_features=n,records=u)
  assert capture().tobytes()==initial.tobytes(),'stock initial RAM differs'
  save(d/'DEVICE.json',info)
  initial.tofile(d/'initial.raw')
  for i in range(2):
   a=x[i:i+1].reshape(1,1,1,n)
   y=np.asarray(dev.fit(a,int(i%2)) if learn else dev.forward(a)).reshape(-1)
   assert y.size==u and np.issubdtype(y.dtype,np.integer),'stock output shape/type'
   seen.append(y.astype('<i4'));np.asarray(seen,'<i4').tofile(d/'scores.raw')
   ram=capture();snap.append(ram);np.asarray(snap,'<u4').tofile(d/'snapshots.raw');ram.tofile(d/'final.raw');states[i].tofile(d/'expected.raw')
   save(d/'OBSERVED.json',dict(observed_forwards=i+1,score_match=bool(np.array_equal(y,ys[i])),ram_match=bool(np.array_equal(ram,states[i])),changed_records=np.flatnonzero(np.any(ram.reshape(u,stride)!=initial.reshape(u,stride),axis=1)).tolist()))
   assert np.array_equal(y,ys[i]),'stock exact potentials differ'
   assert np.array_equal(ram,states[i]),'stock exact learning state differs'
  save(d/'PASS.json',dict(status='STOCK_SDK_PREFIX_EXACT',forwards=2,learning=bool(learn)))
 finally:
  if programmed:dev.unprogram()
  save(d/'DEVICE.json',info)
  if mm is not None:mm.close()
  if fd is not None:os.close(fd)
def run(case,learn):
 assert case in CASES and learn in [0,1]
 st=json.loads((S/'STATE.json').read_text());a=json.loads((S/'AUTHORITY.json').read_text())
 assert not st['closed'] and not st.get('failure')
 assert (dt.datetime.fromisoformat(a['scientific_cutoff_utc'])-now()).total_seconds()>100
 for n,h in json.loads((S/'evidence/FROZEN.json').read_text())['hashes'].items():assert sha(P/n)==h,n
 for n,h in json.loads((S/'evidence/FROZEN.json').read_text())['sdk_binary_hashes'].items():assert sha(Path(n))==h,n
 name=f'{case}_l{learn}';d=S/'raw'/name;assert not d.exists()
 sequence=['below_l0','below_l1','above_l0','above_l1'];i=sequence.index(name)
 assert st.get('completed',[])==sequence[:i]
 assert len(list((S/'raw').glob('*/OWNER.json')))<4
 with open('/tmp/akd1500-research.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1
  d.mkdir(parents=True);receipt=dict(case=case,learn=bool(learn),admitted_utc=now().isoformat(),freeze_sha256=sha(S/'evidence/FROZEN.json'))
  save(d/'OWNER.json',receipt)
  with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
   p=subprocess.Popen([sys.executable,'-m','coprocessor.address_start','worker',case,str(learn),str(d)],cwd=P,stdout=out,stderr=err)
   receipt['pid']=p.pid;save(d/'OWNER.json',receipt)
   try:code=p.wait(timeout=90)
   except subprocess.TimeoutExpired:
    p.terminate()
    try:code=p.wait(timeout=3)
    except subprocess.TimeoutExpired:p.kill();code=p.wait()
  receipt.update(exit_code=code,finished_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,hashes={q.name:sha(q) for q in d.iterdir() if q.is_file()})
  save(d/'RECEIPT.json',receipt)
  if code or not receipt['device_free']:
   st.update(closed=True,failure=name,status='FAILED_PRESERVED',owned_jobs_remaining=0,closed_utc=now().isoformat());save(S/'STATE.json',st)
   raise RuntimeError((d/'stderr').read_text())
  st.setdefault('completed',[]).append(name)
  if name=='above_l1':st.update(closed=True,status='COMPLETED_BOUNDED_COMPARISON',owned_jobs_remaining=0,closed_utc=now().isoformat())
  save(S/'STATE.json',st)
  print(json.dumps(receipt))
if __name__=='__main__':
 if sys.argv[1]=='freeze':freeze()
 elif sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]),Path(sys.argv[4]))
 else:run(sys.argv[1],int(sys.argv[2]))
