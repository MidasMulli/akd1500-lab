"""Bounded native class-window stages and independent full-model verification."""
import datetime as dt,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from pong.relearn.model import Reference
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-fnp2-window'
SEEDS=[107101,107301,107401]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def now():return dt.datetime.now(dt.timezone.utc)
def audit(d,allow_failure=False):
 r=json.loads((d/'RECEIPT.json').read_text());f=S/'fixtures01'/f's{r["seed"]}';m=json.loads((f/'meta.json').read_text());u,n,stride,tile=m['records'],m['input_features'],m['record_words'],m['tile_records'];a=np.fromfile(f/'initial.raw','<u4');xs=np.fromfile(f/'inputs.raw',np.uint8).reshape(8,n);labels=np.fromfile(f/'labels.raw','<i4');mode=r['mode'];count=r['count']
 for name,h in r['hashes'].items():assert sha(d/name)==h,name
 if mode.startswith('cpu'):
  ref=Reference([0,0]+a.tolist(),n=n,u=u,nw=32,classes=2);ys=[];qs=[]
  for i in range(count):ys.append(ref.forward(xs[i],int(labels[i])) if mode=='cpu-learn' else [(w&sum(1<<int(k) for k in np.flatnonzero(xs[i]))).bit_count() for w in ref.weights]);qs.append([(w&sum(1<<int(k) for k in np.flatnonzero(xs[i]))).bit_count() for w in ref.weights])
  assert np.array_equal(np.fromfile(d/'result.pre','<i4').reshape(count,u),ys)
  assert np.array_equal(np.fromfile(d/'result.recall','<i4').reshape(count,u),qs)
  assert np.array_equal(np.fromfile(d/'result.final','<u4'),ref.ram[2:])
  assert len(np.fromfile(d/'result.timing','<i8'))==count*5
  result=dict(status='EXACT_CPP_AND_PYTHON_FULL_MODEL',physical_device_opened=False,cpu_model_forwards=count*2*5)
 else:
  guard=np.fromfile(d/'result.guard','<u8');assert len(guard)==3 and guard[0]==0 and guard[2]==2,'zero host writes to live bank after two initialization writes'
  storage=np.fromfile(d/'result.storage','<u4');assert storage.tolist()==[m['record_address'],u*stride*4,m['full_data_address'],m['tile_data_address'],m['host_DDR_base'],m['host_DDR_extent']]
  states=np.fromfile(d/'result.snapshots','<u4').reshape(-1,u*stride)
  if mode=='inspect':
   assert len(states)==3 and np.all(states==a)
   result=dict(status='EXACT_DISJOINT_STORAGE_AND_THREE_VIEW_PROVENANCE',physical_forwards=0,complete_bank_captures=3,host_record_writes=0)
  else:
   width=u if mode=='direct' else tile
   yp=d/'result.pre';qp=d/'result.recall'
   ys=np.fromfile(yp,'<i4').reshape(-1,width);qs=np.fromfile(qp,'<i4').reshape(-1,u) if qp.exists() else np.empty((0,u),'<i4')
   ref=Reference([0,0]+a.tolist(),n=n,u=u,nw=32,classes=2);i_state=0;mismatch=None
   for i,y in enumerate(ys):
    want=np.asarray(ref.forward(xs[i],int(labels[i])) if mode!='frozen' else [(w&sum(1<<int(k) for k in np.flatnonzero(xs[i]))).bit_count() for w in ref.weights])
    actualwant=want if mode=='direct' else want[int(labels[i])*tile:(int(labels[i])+1)*tile]
    assert np.array_equal(y,actualwant),'all native view potentials exact'
    ram=states[i_state];i_state+=1
    if not np.array_equal(ram,ref.ram[2:]):
     mismatch=dict(input_index=i,changed_records=np.flatnonzero(np.any(ram.reshape(u,stride)!=a.reshape(u,stride),axis=1)).tolist(),changed_word_indices=np.flatnonzero(ram!=a).tolist(),target=m['rows'][i]['target']);break
    assert i<len(qs),'recall follows each completed teaching stage'
    wantq=[(w&sum(1<<int(k) for k in np.flatnonzero(xs[i]))).bit_count() for w in ref.weights]
    assert np.array_equal(qs[i],wantq),'full model read-only recall exact'
    assert np.array_equal(states[i_state],ref.ram[2:]);i_state+=1
   assert i_state==len(states),'no omitted complete snapshots'
   if mismatch:
    assert allow_failure and mode=='direct' and r['exit_code']!=0 and len(ys)==1 and not len(qs),'only preserved direct negative branch may mismatch'
    result=dict(status='DIRECT_NATIVE_FAILURE_PRESERVED',physical_forwards=len(ys),complete_bank_captures=len(states),host_record_writes=0,failure=mismatch)
   else:
    assert r['exit_code']==0 and len(ys)==len(qs)==count
    result=dict(status='EXACT_NATIVE_WINDOW_AND_FULL_RECALL',physical_forwards=len(ys)+len(qs),complete_bank_captures=len(states),host_record_writes=0,teaching_inputs=count,learning=mode=='window',targets=[v['target'] for v in m['rows'][:count]],post_recall_target_scores=[int(qs[i,m['rows'][i]['target']]) for i in range(count)])
 assert r['device_free'];save(d/'AUDIT.json',result);return result

def freeze():
 assert not (S/'evidence/FROZEN.json').exists()
 files=[p for p in (S/'fixtures01').rglob('*') if p.is_file()]+[P/n for n in ['coprocessor/window_prepare.py','coprocessor/window_bench.cpp','coprocessor/window_driver.hpp','coprocessor/window_run.py','coprocessor/build_window.sh','coprocessor/record_range.hpp','src/mmio_driver.hpp','src/readout_bench.cpp','pong/relearn/model.py','build/fnp2_window']]+[S/'PROTOCOL.md',S/'AUTHORITY.json']
 save(S/'evidence/FROZEN.json',dict(utc=now().isoformat(),hashes={str(p.relative_to(P)):sha(p) for p in files}))
def passed(name):
 p=S/'raw'/name/'AUDIT.json';return p.exists() and not json.loads(p.read_text())['status'].endswith('FAILURE_PRESERVED')
def run(seed,mode,count):
 assert seed in SEEDS and mode in ['cpu-learn','cpu-frozen','inspect','direct','window','frozen'] and count in [0,1,8]
 assert (count==0)==(mode=='inspect') and (mode!='direct' or count==1)
 st=json.loads((S/'STATE.json').read_text());a=json.loads((S/'AUTHORITY.json').read_text());assert not st['closed'] and not st.get('failure')
 assert (dt.datetime.fromisoformat(a['scientific_cutoff_utc'])-now()).total_seconds()>125
 for n,h in json.loads((S/'evidence/FROZEN.json').read_text())['hashes'].items():assert sha(P/n)==h,n
 name=f's{seed}_{mode}_n{count}';d=S/'raw'/name;assert not d.exists()
 native=not mode.startswith('cpu');physical=[json.loads(p.read_text()) for p in (S/'raw').glob('*/OWNER.json') if not json.loads(p.read_text())['mode'].startswith('cpu')]
 assert len(physical)+native<=16 and sum(x['count']*2 for x in physical)+count*2<=256
 if native:
  assert passed(f's{seed}_cpu-learn_n8') and passed(f's{seed}_cpu-frozen_n8')
  if mode!='inspect':assert passed(f's{seed}_inspect_n0')
  if seed==107101:
   if mode in ['window','frozen']:assert st.get('direct_negative_reviewed')
   if count==8 and mode in ['window','frozen']:assert passed('s107101_window_n1') and passed('s107101_frozen_n1')
  else:assert passed('s107101_window_n8') and passed('s107101_frozen_n8')
 rec=dict(seed=seed,mode=mode,count=count,admitted_utc=now().isoformat(),freeze_sha256=sha(S/'evidence/FROZEN.json'))
 def execute():
  d.mkdir(parents=True);save(d/'OWNER.json',rec)
  with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
   p=subprocess.Popen(['taskset','-c','2','build/fnp2_window',mode,str(S/'fixtures01'/f's{seed}'),str(d/'result'),str(count)],cwd=P,stdout=out,stderr=err);rec['pid']=p.pid;save(d/'OWNER.json',rec)
   try:code=p.wait(timeout=120)
   except subprocess.TimeoutExpired:
    p.terminate()
    try:code=p.wait(timeout=3)
    except subprocess.TimeoutExpired:p.kill();code=p.wait()
  rec.update(exit_code=code,finished_utc=now().isoformat(),device_free=subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1,hashes={p.name:sha(p) for p in d.iterdir() if p.is_file()});save(d/'RECEIPT.json',rec)
  try:r=audit(d,allow_failure=mode=='direct')
  except Exception:
   st.update(closed=True,failure=name,status='FAILED_POSITIVE_GATE_PRESERVED',owned_jobs_remaining=0);save(S/'STATE.json',st);raise
  if mode=='direct':st['direct_negative']=name
  else:st.setdefault('completed',[]).append(name)
  save(S/'STATE.json',st);return r
 if native:
  with open('/tmp/akd1500-research.lock','a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert subprocess.run(['fuser','/dev/akd1500_0'],capture_output=True).returncode==1;r=execute()
 else:r=execute()
 print(json.dumps(r),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='freeze':freeze()
 else:run(int(sys.argv[1]),sys.argv[2],int(sys.argv[3]))
