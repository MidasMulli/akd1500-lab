"""Prospective discrete outcome forecast; not a hardware or worker result."""
from pathlib import Path
import json,sys
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'image_service/service'))
from controller import CPUController

def forecast(seed,arm):
    decode={71011:['cpu','gstreamer'],71012:['gstreamer','cpu'],71013:['cpu','gstreamer']}[seed]
    export={71011:['cpu','gstreamer'],71012:['gstreamer','cpu'],71013:['gstreamer','cpu']}[seed]
    primary=decode[0];other=decode[1];schedule=['all']*8+[other]*4+['all']*8+[primary]*4
    c=CPUController();offset={71011:0,71012:1,71013:2}[seed];remaining=80;rows=[];complete=0;failed=0
    for job,available in enumerate(schedule):
        st=3
        if job==8 and arm=='frozen':c.learn_transition=False
        for local in range(min(8,remaining)):
            probe=(job%2==1) if job<8 else (arm!='reactive' and st==3 and local==0 and (job-8-offset)%4==0)
            a=c.choose(st,probe=probe)['action'];backend=(decode if st==3 else export)[a]
            ok=available=='all' or available==backend
            out=((1 if backend=='cpu' else 2) if st==3 else 0) if ok else st
            rows.append(dict(job=job,local_attempt=local,state=st,action=a,outcome=out,probe=probe,ok=ok))
            failed+=not ok;c.observe(st,a,out);remaining-=1;st=out
            if st==0:complete+=1;break
        if remaining==0:break
    return dict(seed=seed,arm=arm,completed=complete,attempts=len(rows),failures=failed,events=rows)
if __name__=='__main__':
    obj=dict(status='MODEL_FORECAST_ONLY',rows=[forecast(s,a) for s in [71011,71012,71013] for a in ['probe','reactive','frozen','cpu']])
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='events'} for r in obj['rows']]))
