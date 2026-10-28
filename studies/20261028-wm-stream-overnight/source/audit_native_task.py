"""Independently match actual native decisions to the qualified causal CPU run."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('cell');a=p.parse_args();b=a.base
spec=next(v for v in json.loads((b/'NATIVE_PROTOCOL.json').read_text())['cells'] if v['id']==a.cell)
out=b/f'native-{a.cell}';qa=json.loads((out/'fixture/RAW.QA.json').read_text());assert qa['status']=='PASS' and qa['physical'] and qa['observed']==spec['forwards']
rows=[json.loads(l) for l in(out/'RECORDS.jsonl').read_text().splitlines()]
ref=[json.loads(l) for l in(b/f"CPU_{spec['tranche']}_K{spec['K']}/RECORDS.jsonl").read_text().splitlines()]
assert len(rows)==spec['events'];merged=[]
for i,(r,c)in enumerate(zip(rows,ref)):
 assert r['seq']==c['seq']==i and r['query_sha']==c['query_sha'] and r['key_hex']==c['key_hex'] and r['truth']==c['truth'] and r['wm_class']==c['wm_class'],i
 for name in r['answers']:
  assert r['answers'][name]==c['answers'][name],(i,name,'query')
  assert np.allclose(r['predictions'][name],c['predictions'][name],atol=1e-14,rtol=0),(i,name,'forecast')
 assert all(x<y for x,y in zip(r['times_ns'],r['times_ns'][1:])),i
 item=dict(r);item['answers']=dict(c['answers'])|r['answers'];item['predictions']=dict(c['predictions'])|r['predictions'];merged.append(item)
result=dict(status='PASS',physical=True,cell=a.cell,arm=spec['arm'],events=len(rows),forwards=qa['observed'],native_feedback_windows=qa['native_feedback_windows_verified'],all_query_keys_answers_and_forecasts_match_qualified_CPU=True,all_prediction_before_feedback_sequences_pass=True,native_arms=[spec['arm']],other_controls='qualified matched software reference; separately run native controls remain individually labeled')
(out/'TASK_QA.json').write_text(json.dumps(result,indent=2)+'\n')
if spec['events']==9*spec['K']:
 d=out/'matched-score';d.mkdir(exist_ok=False);(d/'STATUS.json').write_text(json.dumps(dict(status='CONSUMER_DONE_AWAITING_SCORE',physical=True)));(d/'RECORDS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in merged))
 stage='primary' if spec['arm']=='error-only' else 'diagnostic'
 subprocess.run([sys.executable,str(b/'source/score_stream.py'),str(d),spec['stream_dir'],'--K',str(spec['K']),'--stage',stage],check=True)
 r=json.loads((d/'RESULT.json').read_text());r['source']='Actual native arm plus qualified matched software control references';r['native_arms']=[spec['arm']];r['other_controls_are_software']=True;(out/'RESULT_NATIVE.json').write_text(json.dumps(r,indent=2)+'\n')
 if stage=='primary':assert r['status']=='PASS',r['checks']
print(json.dumps(result))
