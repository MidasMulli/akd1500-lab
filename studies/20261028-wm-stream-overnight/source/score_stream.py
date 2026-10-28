"""Post-run evaluation only. This opens AUDIT after the consumer has finished."""
import argparse, json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('consumer',type=Path);p.add_argument('stream',type=Path);p.add_argument('--K',type=int,required=True);p.add_argument('--stage',choices=['mechanism','primary','diagnostic'],required=True);a=p.parse_args()
status=json.loads((a.consumer/'STATUS.json').read_text());assert status['status']=='CONSUMER_DONE_AWAITING_SCORE'
rows=[json.loads(l) for l in(a.consumer/'RECORDS.jsonl').read_text().splitlines()];audit=[json.loads(l) for l in(a.stream/'AUDIT.jsonl').read_text().splitlines()]
assert len(rows)==len(audit)==9*a.K
for i,(r,t)in enumerate(zip(rows,audit)):
    assert r['seq']==t['seq']==i and r['event_id']==t['event_id']
    assert all(x<y for x,y in zip(r['times_ns'],r['times_ns'][1:])),i
phase=np.array([t['phase'] for t in audit]);repeat=np.array([t['repeat'] for t in audit]);y=np.array([r['truth'] for r in rows]);wmc=np.array([r['wm_class'] for r in rows])
masks={'all':np.ones(len(y),bool),'S2B_second':(phase==3)&(repeat==1),'S1B_first_return':(phase==5)&(repeat==0),'returned_A_B':np.isin(phase,[4,5]),'all_A':np.isin(phase,[0,2,4])}
admitted={r['key_hex'] for r,t in zip(rows,audit) if t['phase']==1 and r['wm_class']!=r['truth']}
masks['S1B_admitted_first_return']=masks['S1B_first_return']&np.array([r['key_hex'] in admitted for r in rows])
def metric(probs,take):
    q=probs[take];yy=y[take]
    if not len(yy):return dict(n=0,accuracy=None,brier=None)
    return dict(n=len(yy),accuracy=float((q.argmax(1)==yy).mean()),brier=float(np.square(q-np.eye(3)[yy]).sum(1).mean()))
summary={name:{m:metric(np.array([r['predictions'][name] for r in rows]),take) for m,take in masks.items()} for name in rows[0]['predictions']}
pure={name:{m:dict(n=int(take.sum()),hits=int(sum(r['answers'][name] is not None for r,k in zip(rows,take) if k)),correct=int(sum(r['answers'][name]==r['truth'] for r,k in zip(rows,take) if k)),accuracy=float(np.mean([r['answers'][name]==r['truth'] for r,k in zip(rows,take) if k])) if take.any() else None) for m,take in masks.items()} for name in rows[0]['answers']}
by_key={}
for r in rows:by_key.setdefault(r['key_hex'],set()).add(r['truth'])
checks={}
if a.stage=='mechanism':
    checks['all_admit_returned_pure_exact']=pure['all-admit']['returned_A_B']['accuracy']==1.
    checks['known_source_capacity_fits']=len(by_key)<=32
elif a.stage=='primary':
    b='S2B_second';r='S1B_admitted_first_return';s=summary['error-only'];w=summary['WM']
    checks['at_least_four_WM_errors']=int((wmc[masks[b]]!=y[masks[b]]).sum())>=4
    checks['S2B_accuracy_gain_vs_WM']=s[b]['accuracy']>=w[b]['accuracy']+.25-1e-12
    checks['S2B_Brier_ratio']=s[b]['brier']<=.8*w[b]['brier']+1e-12
    for control in ['all-admit','S1AB-frozen']:
        checks[f'S2B_accuracy_vs_{control}']=s[b]['accuracy']>=summary[control][b]['accuracy']+.10-1e-12
        checks[f'S2B_Brier_vs_{control}']=s[b]['brier']<=summary[control][b]['brier']+1e-12
    checks['retained_prior_native_corrections']=pure['error-only'][r]['n']>0 and pure['error-only'][r]['accuracy']>=.95
    checks['A_no_harm']=s['all_A']['brier']<=w['all_A']['brier']+.005+1e-12
    checks['returned_strong_float_noninferiority']=s[r]['n']>0 and s[r]['brier']<=summary['float32'][r]['brier']+.005+1e-12
result=dict(status='DESCRIPTIVE_ONLY' if a.stage=='diagnostic' else 'PASS' if all(checks.values()) else 'FAILED_GATE',stage=a.stage,K=a.K,events=len(rows),checks=checks,summary=summary,pure_memory=pure,unique_keys=len(by_key),conflicting_keys=sum(len(v)>1 for v in by_key.values()),source='actual GPU fp32 query stream; online CPU ensemble',physical=status['physical'])
(a.consumer/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','stage','K','events','checks','unique_keys','conflicting_keys']},indent=2))
