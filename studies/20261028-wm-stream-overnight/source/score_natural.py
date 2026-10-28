"""Prospectively frozen N usefulness screen; evaluation-only AUDIT access."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('consumer',type=Path);p.add_argument('stream',type=Path);p.add_argument('spec',type=Path);a=p.parse_args()
spec=json.loads(a.spec.read_text());st=json.loads((a.consumer/'STATUS.json').read_text());assert st['status']=='N_CONSUMER_DONE_AWAITING_SCORE'
read=lambda p:[json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
rows=read(a.consumer/'RECORDS.jsonl');skips=read(a.consumer/'SKIPS.jsonl');audit=read(a.stream/'AUDIT.jsonl');query=read(a.stream/'QUERY.jsonl');feedback=read(a.stream/'FEEDBACK.jsonl')
assert len(audit)==st['episodes']==128 and len(rows)+len(skips)==128 and len(rows)==len(query)==len(feedback)
assert sorted([r['seq'] for r in rows]+[r['seq'] for r in skips])==list(range(128))
for i,(r,q,f) in enumerate(zip(rows,query,feedback)):
 t=audit[r['seq']];assert t['query'] and r['event_id']==q['event_id']==f['event_id']==t['event_id']
 assert r['query_row']==q['query_row']==f['query_row']==f['release_index']==i and r['seq']==q['seq']==f['seq']
 assert all(x<y for x,y in zip(r['times_ns'],r['times_ns'][1:]))
for s in skips:assert not audit[s['seq']]['query'] and not s['taught']
y=np.array([r['truth'] for r in rows],dtype=int);B=np.array([audit[r['seq']]['regime']=='B' for r in rows],dtype=bool);masks={'all':np.ones(len(y),bool),'A':~B,'B':B}
def metric(prob,take):
 z=prob[take];yy=y[take]
 return dict(n=len(yy),accuracy=float((z.argmax(1)==yy).mean()),brier=float(np.square(z-np.eye(3)[yy]).sum(1).mean())) if len(yy) else dict(n=0,accuracy=None,brier=None)
names=list(rows[0]['answers']) if rows else ['error-only','all-admit','after-A-frozen','exact32','Hamming32','float32','FIFO32','LRU32']
summary={n:{m:metric(np.array([r['predictions'][n] for r in rows]).reshape(-1,3),take) for m,take in masks.items()} for n in names+['WM']}
pure={n:{m:dict(n=int(take.sum()),hits=int(sum(r['answers'][n] is not None for r,k in zip(rows,take) if k)),correct=int(sum(r['answers'][n]==r['truth'] for r,k in zip(rows,take) if k))) for m,take in masks.items()} for n in names}
g=spec['gates'];s=summary['error-only'];wm=summary['WM'];coverage=pure['error-only']['B']['hits']/s['B']['n'] if s['B']['n'] else 0
checks={'minimum_queryable_B':s['B']['n']>=g['minimum_queryable_B']}
if s['B']['n'] and s['A']['n']:
 checks.update(B_accuracy_gain_vs_WM=s['B']['accuracy']>=wm['B']['accuracy']+g['B_accuracy_gain_vs_WM']-1e-12,B_Brier_reduction_vs_WM=s['B']['brier']<=wm['B']['brier']-g['B_absolute_Brier_reduction_vs_WM']+1e-12,B_vs_after_A_frozen=s['B']['accuracy']>=summary['after-A-frozen']['B']['accuracy']+g['B_accuracy_gain_vs_after_A_frozen']-1e-12,A_no_harm=s['A']['brier']<=wm['A']['brier']+g['A_Brier_harm_max']+1e-12,B_strong_float_noninferiority=s['B']['brier']<=summary['float32']['B']['brier']+g['B_Brier_vs_strong_float_CPU_max']+1e-12,B_memory_coverage=coverage>=g['B_memory_hit_coverage_min']-1e-12)
else:checks['nonempty_A_B']=False
keys={};raw=set();repeated=0
for r,q in zip(rows,query):
 k=r['key_hex'];repeated+=int(k in keys);keys.setdefault(k,set()).add(r['truth']);raw.add(hashlib.sha256(np.array(q['query']['h0_pool']+q['query']['h5_pool'],'<f4').tobytes()+q['query']['program_id'].encode()).hexdigest())
r=dict(status='PASS' if all(checks.values()) else 'FAILED_USEFULNESS_GATE',physical=False,episodes=128,query_rows=len(rows),observed_terminal_skips=len(skips),checks=checks,summary=summary,pure_memory=pure,B_memory_coverage=coverage,distinct_scenes=len({t['scene_id'] for t in audit}),unique_encoded_keys=len(keys),conflicting_encoded_keys=sum(len(v)>1 for v in keys.values()),repeated_key_queries=repeated,unique_full_float_state_program_pairs=len(raw),causal_query_before_feedback=True,skips_not_taught=True,limits=['Natural simulator scenes, not camera video','Exact key reuse under quantization is reported separately','No native admission unless identical DEV and HELD gates both pass'])
(a.consumer/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['status','episodes','query_rows','observed_terminal_skips','checks','unique_encoded_keys','conflicting_encoded_keys','B_memory_coverage']},indent=2))
