"""Recount, verify, and export compact path-free public evidence."""
from pathlib import Path
import argparse,hashlib,json,statistics,struct,sys
from verify import verify
def main():
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1];rows=[];audits=[]
    for d in sorted((a.study/'raw').iterdir()):
        if not (d/'RECEIPT.json').exists():continue
        r=json.loads((d/'RECEIPT.json').read_text());row=dict(run=d.name,status=r['status'],samples=r['samples'],learning=bool(r['learning']),fixture=Path(r['fixture']).name,binary_sha256=r['binary_sha256'],program_sha256=r['program_sha256'])
        if r['status']=='PASS':
            audits.append(verify(root,d));row['native']=r['results']['native'];row['cpu']=r['results']['cpu']
            for mode in ['native','cpu']:
                m=row[mode];m['mean_us_per_sample']=m['wall_ns']/m['samples']/m['repeats']/1000
            for phase in ['input','submit','fetch']:
                f=d/f'native.{phase}_ns'
                if f.exists():
                    b=f.read_bytes();xs=struct.unpack('<'+'q'*(len(b)//8),b);row[f'mean_{phase}_us']=sum(xs)/len(xs)/1000
            row['cpu_to_native_latency_ratio']=row['native']['mean_us_per_sample']/row['cpu']['mean_us_per_sample']
            row['verification']=audits[-1]
            # Verification paths become fixture basenames in public artifacts.
            row['verification']['fixture']=Path(row['verification']['fixture']).name
        else:
            row['native_exit_code']=r.get('native_exit_code');row['failure']=(d/'native.stderr').read_text().strip()
        rows.append(row)
    capacity=json.loads((a.study/'fixtures_v2/CAPACITY.json').read_text())
    for c in capacity['cells']:c.pop('traceback',None)
    mesh=json.loads((a.study/'mesh_v6/MESH.json').read_text())
    for c in mesh['cells']:
        for k in ['traceback','calls']:c.pop(k,None)
    z=dict(schema=1,status='VERIFIED_BOUNDED_STUDY',date='2026-10-23',physical_programming_sessions=len(rows),passing_sessions=sum(r['status']=='PASS' for r in rows),failed_sessions=sum(r['status']!='PASS' for r in rows),
           physical_chip_samples=1,power_measurement='NO_VALIDATED_CARD_SENSOR_CONNECTED',
           platform=dict(cpu='Jetson Orin Nano Cortex-A78AE',affinity_core=2,governor='schedutil',pcie='5 GT/s x2',clocks_changed=False),
           capacity=capacity,mesh=mesh,runs=rows,
           limitations=['No energy advantage established','No peak utilization or full capacity claim','Mesh permutations are fixed and algebraically composable on CPU','FNP2 mappings are decoder gaps, not silicon failures','Timing replicates on one chip are not independent chips','Native timings include host interface costs; no core-only latency measurement'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(z,indent=2)+'\n')
    (a.study/'evidence/FINAL_INDEPENDENT_VERIFICATION.json').write_text(json.dumps(dict(status='PASS',runs=len(audits),results=audits),indent=2)+'\n')
    print(json.dumps({k:z[k] for k in ['physical_programming_sessions','passing_sessions','failed_sessions']}))
if __name__=='__main__':main()
