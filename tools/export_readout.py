"""Sanitized raw archive plus independently recounted compact-readout results."""
from pathlib import Path
import argparse,hashlib,json,statistics
import numpy as np
from verify import verify
P=Path(__file__).resolve().parents[1]

def main(studies,out):
    rows=[];arrays={};fixtures={};states=[]
    for study in studies:
        state=json.loads((study/'STATE.json').read_text());assert state['research_admissions_closed']
        states.append({k:state[k] for k in ['status','physical_sessions','passing_sessions','failed_sessions','forwards']})
        for run in sorted((study/'raw').iterdir()):
            r=json.loads((run/'RECEIPT.json').read_text());row={k:r[k] for k in ['name','status','samples','learning','binary_sha256','program_sha256']};row['study']=study.name
            row['output_contract']=r.get('output_contract',int(r.get('activated',False)));row['threshold']=r['threshold'];row['cpu_fast_threshold']=r.get('cpu_fast_threshold',False)
            if r['status']=='PASS':
                audit=verify(P,run);audit['fixture']=Path(audit['fixture']).name
                row['verification']=audit;row['results']=r['results']
                for mode in ['native','cpu']:
                    v=row['results'][mode];v['mean_us_per_input']=v['wall_ns']/v['samples']/v['repeats']/1000
                for phase in ['input','submit','fetch']:
                    f=run/f'native.{phase}_ns'
                    if f.exists():arrays[study.name+'__'+run.name+'__'+phase+'_ns']=np.fromfile(f,dtype='<i8')
                f=P/r['fixture'];key=f.parent.parent.name+'__'+f.name;row['fixture_key']=key
                if key not in fixtures:
                    n,u,nw,classes=map(int,f.with_suffix('.meta').read_text().split());fixtures[key]=dict(inputs=n,neurons=u,num_weights=nw,classes=classes,output_contract=row['output_contract'],threshold=row['threshold'])
                    for ext,dtype in [('ram','<u4'),('inputs','u1'),('labels','<i4')]:arrays[key+'__'+ext]=np.fromfile(f.with_suffix('.'+ext),dtype=dtype)
                for mode in ['native','cpu']:
                    for ext in ['scores','ram']:arrays[study.name+'__'+run.name+'__'+mode+'_'+ext]=np.fromfile(run/f'{mode}.{ext}',dtype='<i4' if ext=='scores' else '<u4')
            else:row['failure']=(run/'native.stderr').read_text()
            rows.append(row)
    out.parent.mkdir(parents=True,exist_ok=True);archive=out.with_suffix('.npz');np.savez_compressed(archive,**arrays)
    report=dict(schema=1,status='INDEPENDENTLY_VERIFIED',date='2026-10-23',physical_chips=1,studies=states,fixtures=fixtures,runs=rows,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),power='NO_VALIDATED_DIRECT_CARD_SENSOR',platform=dict(cpu='Jetson Orin Nano Cortex-A78AE',core_affinity=2,governor='schedutil',AKD_PCIe='5 GT/s x2',clocks_changed=False),limitations=['Synthetic threshold-association workload, not image-service speed','Threshold responses discard potential magnitudes; group counts further discard neuron identity','One-bit dense output occupies one byte per neuron on this transport','Temporal repeats on one physical chip','CPU remains a required matched comparison; no electrical advantage established','Resident interval is host-observed completion, not core-only timing'])
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(runs=len(rows),archive_bytes=archive.stat().st_size)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('studies',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();main(a.studies,a.output)
