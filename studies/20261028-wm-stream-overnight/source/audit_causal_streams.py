"""Independent scalar policy/fusion audit, without importing the learning circuit."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('base',type=Path);a=p.parse_args();b=a.base
refs={};report={}
for K in [8,16]:
 d=b/f'evidence/streams-dev/R_DEV_K{K}_CPU_STREAM';out=b/f'evidence/CPU_DEV_K{K}'
 query=[json.loads(l) for l in(d/'QUERY.jsonl').read_text().splitlines()];audit=[json.loads(l) for l in(d/'AUDIT.jsonl').read_text().splitlines()];truth=[json.loads(l) for l in(d/'FEEDBACK.jsonl').read_text().splitlines()];ack=[json.loads(l) for l in(d/'PREDICTION_ACK.jsonl').read_text().splitlines()];applied=[json.loads(l) for l in(d/'FEEDBACK_APPLIED.jsonl').read_text().splitlines()];rows=[json.loads(l) for l in(out/'RECORDS.jsonl').read_text().splitlines()]
 assert len(rows)==len(query)==len(audit)==len(truth)==len(ack)==len(applied)==9*K
 mem={n:{} for n in ['error-only','all-admit','S1AB-frozen']}
 for i,(q,t,f,r,ak,ap)in enumerate(zip(query,audit,truth,rows,ack,applied)):
  assert q['seq']==t['seq']==f['seq']==r['seq']==ak['seq']==ap['seq']==i
  assert len({v['event_id'] for v in [q,t,f,r,ak,ap]})==1
  assert hashlib.sha256(json.dumps(q['query'],sort_keys=True).encode()).hexdigest()==q['query_sha']==r['query_sha']==ak['query_sha']==ap['query_sha']
  ident=(t['scene_id'],t['twin'],t['program_id'],t['regime'])
  if ident in refs:assert refs[ident]==q['query_sha'],('GPU recompute differs',ident)
  else:refs[ident]=q['query_sha']
  key=r['key_hex'];y={'D':0,'S':1,'C':2}[f['outcome']];wm=[q['query']['wm_pT'][j] for j in [2,0,1]]
  assert r['truth']==y and r['wm_class']==int(np.argmax(wm))
  for name,store in mem.items():
   value=store.get(key);assert value==r['answers'][name],('scalar memory answer',K,i,name,value,r['answers'][name])
   expected=[float(v) for v in wm] if value is None else [.25*float(v)+.75*(j==value) for j,v in enumerate(wm)]
   assert np.allclose(expected,r['predictions'][name],atol=1e-14,rtol=0)
   if name=='S1AB-frozen' and i>=3*K:continue
   if key in store and store[key]!=y:del store[key]
   if name=='all-admit' or r['wm_class']!=y:
    if key in store or len(store)<32:store[key]=y
  assert all(x<y for x,y in zip(r['times_ns'],r['times_ns'][1:]))
 report[f'K{K}']=dict(episodes=len(rows),scalar_policies_and_all_fusion_values_exact=True,causal_sequences_and_keys_exact=True,final_store_sizes={k:len(v) for k,v in mem.items()},files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in d.iterdir() if f.is_file()})
r=dict(status='PASS',physical=False,audited_episodes=sum(v['episodes'] for v in report.values()),distinct_context_gain_pairs=len(refs),GPU_recomputes_bitwise_equal=True,runs=report);(b/'ROOT_CAUSAL_DEV_AUDIT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='runs'}))
