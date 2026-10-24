"""Independent public replay of guarded native windows and preserved diagnostics."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from coprocessor.fnp2_archive import compact32
from resident.or_archive import expand
from pong.relearn.model import Reference
P=Path(__file__).resolve().parents[1];S=P/'studies/20261024-fnp2-window';BASE=P/'results/2026-10-24-fnp2-window'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def export():
 assert json.loads((S/'STATE.json').read_text())['closed']
 z={};m=dict(status='GUARDED_NATIVE_CLASS_WINDOWS_WITH_PRESERVED_ADDRESS_FAILURES',fixtures={},runs=[],forensics=[],executed_sources={},attempts=[])
 def freeze_sources(root):
  f=json.loads((root/'evidence/FROZEN.json').read_text())
  for p,h in f['hashes'].items():
   if p.startswith(('coprocessor/','src/','pong/','tools/')):assert sha(P/p)==h;m['executed_sources'][p]=h
 def encode(name,a,states):
  for k,v in compact32(a,states).items():z[name+'__delta.'+k]=v
 for seed in [107101,107301,107401]:
  f=S/'fixtures01'/f's{seed}';fid=f's{seed}';meta=json.loads((f/'meta.json').read_text());m['fixtures'][fid]=meta
  for key,filename,dtype in [('initial','initial.raw','<u4'),('inputs','inputs.raw','u1'),('labels','labels.raw','<i4')]:z[fid+'__'+key]=np.fromfile(f/filename,dtype).reshape((-1,512) if key=='inputs' else -1)
 freeze_sources(S)
 for d in sorted((S/'raw').iterdir()):
  r=json.loads((d/'RECEIPT.json').read_text());assert r['device_free']
  for f,h in r['hashes'].items():assert sha(d/f)==h
  name=d.name;r['name']=name;r['fixture']=f's{r["seed"]}';r['error']=(d/'stderr').read_text() if r['exit_code'] else None
  for key,dtype in [('pre','<i4'),('recall','<i4'),('final','<u4'),('guard','<u8'),('storage','<u4'),('timing','<i8')]:
   p=d/f'result.{key}'
   if p.exists():z[name+'__'+key]=np.fromfile(p,dtype)
  if not r['mode'].startswith('cpu'):
   a=z[r['fixture']+'__initial'];states=np.fromfile(d/'result.snapshots','<u4').reshape(-1,len(a));r['snapshots']=len(states);encode(name,a,states)
  m['runs'].append(r)
 for study in ['sdk','width','word','start','ddr']:
  root=P/'studies'/f'20261024-fnp2-{study}';freeze_sources(root)
  for d in sorted((root/'raw').iterdir()):
   if not (d/'scores.raw').exists():continue
   r=json.loads((d/'RECEIPT.json').read_text())
   for f,h in r['hashes'].items():assert sha(d/f)==h
   case=r['case'];fid=f'{study}_{case}';name=f'{study}_{d.name}';f=root/'fixtures01'/case/'learner'
   n,u,stride=(512,8192,20) if study in ['sdk','ddr'] else (1024,4096,36)
   target=({'below':6500,'above':6600} if study in ['sdk','ddr'] else {'below':3600,'above':3700} if study=='width' else {'below':3640,'above':3641} if study=='start' else {'below':3640,'above':3640})[case]
   initial=np.fromfile(d/'initial.raw','<u4');x=np.fromfile(f.with_suffix('.inputs'),np.uint8).reshape(-1,n)[:2]
   z[fid+'__initial']=initial;z[fid+'__inputs']=x
   scores=np.fromfile(d/'scores.raw','<i4').reshape(-1,u);states=np.fromfile(d/'snapshots.raw','<u4').reshape(-1,u*stride)
   z[name+'__scores']=scores;z[name+'__final']=np.fromfile(d/'final.raw','<u4');encode(name,initial,states)
   m['forensics'].append(dict(name=name,fixture=fid,study=study,n=n,u=u,stride=stride,target=target,learn=r['learn'],count=len(scores),exit_code=r['exit_code'],device_free=r['device_free'],hashes=r['hashes'],device=json.loads((d/'DEVICE.json').read_text()),error=(d/'stderr').read_text() if r['exit_code'] else None,disposition='FROZEN_PLACEMENT_GATE_FAILURE_NOT_LEARNING_RESULT' if study=='ddr' else 'STRICT_PREFIX_RESULT'))
 for study,status in [('placement','COPY_GATE_SUPERSEDED_BEFORE_FORWARD'),('ddr','ALIGNED_COPY_GATE'),('base','ZERO_FORWARD_OBSERVABILITY_FAILURE'),('base2','ZERO_FORWARD_PROVEN_HOST_DDR')]:
  root=P/'studies'/f'20261024-fnp2-{study}';freeze_sources(root);m['attempts'].append(dict(study=study,physical_sessions=1,forwards=0,status=status))
 m['storage_provenance']=json.loads((P/'studies/20261024-fnp2-base2/evidence/RESULT.json').read_text())
 m['limitations']=['Host dispatches class windows from supplied teacher labels; one physical FNP2, not autonomous scheduling','Records are in host DDR; scratch/control buffers are in card SRAM','Single-swap unique-winner probes only; hidden selector behavior across arbitrary window switches unqualified','Eight selected targets across three initializations; not all11264records exercised for learning','Direct failure preserved; configured-path root cause unresolved, no silicon erratum claim','No CPU or calibrated electrical-energy advantage','Manual external placement fails frozen state gate and cannot establish a learning outcome; original zero-forward observability error retained']
 np.savez_compressed(BASE.with_suffix('.npz'),**z);m['archive_sha256']=sha(BASE.with_suffix('.npz'));save(BASE.with_suffix('.json'),m)
def decode(z,name,a,count):return expand(a,{k:z[name+'__delta.'+k] for k in ['offsets','indices','values','sha256']},count)
def bits(x):return sum(1<<int(k) for k in np.flatnonzero(x))
def recount(m,z):
 for p,h in m['executed_sources'].items():assert sha(P/p)==h,('executed source',p)
 assert len(m['runs'])==18,'all CPU, pilot, negative and repeated window runs retained'
 assert {r['name'] for r in m['runs'] if r['exit_code']}=={'s107101_direct_n1'},'direct failure cannot be promoted or removed'
 totals=dict(physical_sessions=0,native_forwards=0,full_bank_captures=0,cpu_model_forwards=0)
 main_updates=[]
 for r in m['runs']:
  name=r['name'];seed=r['seed'];f=m['fixtures'][r['fixture']];u,n,tile,stride=f['records'],f['input_features'],f['tile_records'],f['record_words'];a=z[r['fixture']+'__initial'];xs=z[r['fixture']+'__inputs'];labels=z[r['fixture']+'__labels'];mode=r['mode'];count=r['count'];get=lambda k:z[name+'__'+k]
  assert r['device_free'];ref=Reference([0,0]+a.tolist(),n=n,u=u,nw=32,classes=2)
  if mode.startswith('cpu'):
   yy=[];qq=[]
   for i in range(count):
    b=bits(xs[i]);yy.append(ref.forward(xs[i],int(labels[i])) if mode=='cpu-learn' else [(w&b).bit_count() for w in ref.weights]);qq.append([(w&b).bit_count() for w in ref.weights])
   assert np.array_equal(get('pre').reshape(count,u),yy) and np.array_equal(get('recall').reshape(count,u),qq)
   assert np.array_equal(get('final'),ref.ram[2:]);assert len(get('timing'))==5*count
   totals['cpu_model_forwards']+=count*2*5;continue
  totals['physical_sessions']+=1
  g=get('guard');assert len(g)==3 and g[0]==0 and g[2]==2,'no host writes to the live record bank'
  assert get('storage').tolist()==[f['record_address'],u*stride*4,f['full_data_address'],f['tile_data_address'],f['host_DDR_base'],f['host_DDR_extent']]
  states=decode(z,name,a,r['snapshots']);totals['full_bank_captures']+=len(states)
  if mode=='inspect':assert len(states)==3 and np.all(states==a);continue
  width=u if mode=='direct' else tile;ys=get('pre').reshape(-1,width);qs=get('recall').reshape(-1,u) if name+'__recall' in z else np.empty((0,u),'<i4');totals['native_forwards']+=len(ys)+len(qs)
  cursor=0
  for i,y in enumerate(ys):
   b=bits(xs[i]);want=np.array(ref.forward(xs[i],int(labels[i])) if mode!='frozen' else [(w&b).bit_count() for w in ref.weights]);expected=want if mode=='direct' else want[int(labels[i])*tile:(int(labels[i])+1)*tile]
   assert np.array_equal(y,expected),'native potentials match full-model mathematical rule'
   actual=states[cursor];cursor+=1
   if mode=='direct':
    assert r['exit_code']!=0 and len(ys)==1 and not len(qs)
    ids=np.flatnonzero(np.any(actual.reshape(u,stride)!=a.reshape(u,stride),axis=1)).tolist();assert ids==[2295] and not np.array_equal(actual,ref.ram[2:]),'direct off-target result preserved';continue
   assert np.array_equal(actual,ref.ram[2:]),'complete native learning state exact'
   expectedq=[(w&b).bit_count() for w in ref.weights];assert np.array_equal(qs[i],expectedq)
   assert np.array_equal(states[cursor],ref.ram[2:]),'read-only full recall preserves all records';cursor+=1
   target=f['rows'][i]['target'];assert qs[i,target]==(32 if mode=='window' else 31),'matched recall response'
  assert cursor==len(states)
  if count==8 and mode=='window':main_updates.extend((seed,row['target']) for row in f['rows'])
 assert totals==dict(physical_sessions=12,native_forwards=101,full_bank_captures=110,cpu_model_forwards=480),totals
 assert len(main_updates)==24 and {s for s,t in main_updates}=={107101,107301,107401},'three sealed initializations'
 assert len(m['forensics'])==17,'all stock-SDK, stride, adjacent, same-record and invalid-placement cases retained'
 forensic_forwards=0;findings=[]
 for r in m['forensics']:
  name=r['name'];a=z[r['fixture']+'__initial'];x=z[r['fixture']+'__inputs'];n,u,stride=r['n'],r['u'],r['stride'];states=decode(z,name,a,r['count']);ys=z[name+'__scores'];ref=Reference([0,0]+a.tolist(),n=n,u=u,nw=32,classes=2);forensic_forwards+=r['count'];mismatch=False
  for i,y in enumerate(ys):
   b=bits(x[i]);want=ref.forward(x[i],i%2) if r['learn'] else [(w&b).bit_count() for w in ref.weights]
   assert np.array_equal(y,want),'forensic potentials'
   if not np.array_equal(states[i],ref.ram[2:]):
    assert r['exit_code'] and i==len(ys)-1;ids=np.flatnonzero(np.any(states[i].reshape(u,stride)!=a.reshape(u,stride),axis=1)).tolist();mismatch=True
    if r['study']!='ddr':assert ids==({'sdk':[46],'width':[59],'start':[0]}[r['study']])
    findings.append(dict(name=name,actual_changed_records=ids,intended_record=r['target'],disposition=r['disposition']))
  assert bool(r['exit_code'])==mismatch,'forensic failures remain failures'
 assert forensic_forwards==33
 assert m['storage_provenance']['domain']=='host_DDR' and m['storage_provenance']['record_address']==0xC0000040
 return dict(status='INDEPENDENT_NATIVE_WINDOW_REPLAY_WITH_PRESERVED_FAILURES',window=totals,main_sealed_updates=24,initializations=3,host_live_record_writes=0,forensic_physical_sessions=17,forensic_forwards=forensic_forwards,zero_forward_attempts=m['attempts'],findings=findings,scope='Native learning workaround for selected single-swap probes; host schedules teacher windows; no CPU/energy advantage')
def audit_public():
 m=json.loads(BASE.with_suffix('.json').read_text());assert sha(BASE.with_suffix('.npz'))==m['archive_sha256']
 with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:return recount(m,z)
if __name__=='__main__':
 if sys.argv[1]=='export':export()
 result=audit_public();save(P/'results/2026-10-24-fnp2-window-audit.json',result);print(json.dumps(result))
