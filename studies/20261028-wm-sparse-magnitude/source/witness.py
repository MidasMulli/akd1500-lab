from pathlib import Path
import json
import numpy as np
from core import Layout,Circuit

B=Path(__file__).resolve().parents[1];results=[]
for profile in ['quad13','oct9']:
    g=Layout(profile);c=Circuit(g);c.bootstrap();keys=[]
    for j in range(8):
        levels=np.zeros(g.d,'i4');levels[0]=j%g.bins;levels[1]=j//g.bins;keys.append(g.key(levels,0))
    for j,k in enumerate(keys):c.feedback(k,j%3,j%3,admit_all=True)
    before=[c.query(k)for k in keys];ram=c.memory.ram.copy()
    for _ in range(8):
        for k in keys:c.query(k)
    readonly=bool(np.array_equal(ram,c.memory.ram));c.feedback(keys[0],1,1,admit_all=True);after=[c.query(k)for k in keys]
    checks=dict(all_eight=before==[j%3 for j in range(8)],one_revised_seven_retained=after==[1]+[j%3 for j in range(1,8)],readonly64=readonly,max_distinct_overlap30=max(int(g.value_key(a,0)@g.value_key(b,0))for i,a in enumerate(keys)for b in keys[i+1:])<=30)
    result=dict(profile=profile,status='PASS'if all(checks.values())else'FAILED_STOP',checks=checks,physical=False);results.append(result);print(json.dumps(result),flush=True)
(B/'SOURCE_WITNESS.json').write_text(json.dumps(results,indent=2)+'\n');assert all(r['status']=='PASS'for r in results)
