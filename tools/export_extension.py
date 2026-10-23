"""Sanitized evidence export: include failed attempts; never include vendor images."""
from pathlib import Path
import hashlib,json,statistics
import numpy as np
from verify import verify
from verify_network import verify as verify_network
P=Path(__file__).resolve().parents[1]

def export(name,network=False):
    study=P/'studies'/name;state=json.loads((study/'STATE.json').read_text());assert state['research_admissions_closed']
    rows=[];fixtures={};arrays={}
    for run in sorted((study/'raw').iterdir()):
        if not (run/'RECEIPT.json').exists():continue
        rec=json.loads((run/'RECEIPT.json').read_text())
        row={k:rec[k] for k in ['name','status','samples','learning','learning_mask','compiler_inference','binary_sha256','program_sha256','wait_us','dma_read_min','dma_write_min'] if k in rec}
        row['exit_code']=rec.get('exit_code',rec.get('native_exit_code',0));row['device_released']=rec.get('device_released',True)
        if rec['status']=='PASS' and rec.get('fixture'):
            row['verification']=(verify_network if network else verify)(P,run);row['verification'].pop('fixture',None)
        if rec.get('results'):row['results']=rec['results']
        if rec['status']!='PASS':
            row['failure']=(run/'native.stderr').read_text().replace(str(P),'<repo>').replace('/home/nicklomeli/akd1500-lab','<repo>')
            row['diagnostic_stdout']=(run/'native.stdout').read_text().replace(str(P),'<repo>').replace('/home/nicklomeli/akd1500-lab','<repo>')
        if rec.get('fixture') and (P/rec['fixture']).with_suffix('.bin').exists():
            f=P/rec['fixture'];key=f.parent.name+'__'+f.name;row['fixture_key']=key
            if key not in fixtures:
                if network:
                    z=json.loads(f.with_suffix('.json').read_text());fixtures[key]={k:z[k] for k in ['seed','inputs','neurons','banks','threshold','classes','weights']};fixtures[key]['encoder']=z.get('encoder','dual_rail' if f.parent.name=='fixtures_v4' else 'positive_8_copy');fixtures[key]['fixed_placements']=sorted({tuple(v['np']) for v in z['fixed']})
                    exts=[f'.bank{k}.ram' for k in range(len(z['banks']))]+[f'.fixed{k}.ram' for k in range(len(z['fixed']))]
                else:
                    n,u,nw,classes=map(int,f.with_suffix('.meta').read_text().split());contract,threshold=map(int,f.with_suffix('.output').read_text().split());fixtures[key]=dict(inputs=n,neurons=u,num_weights=nw,classes=classes,output_contract=contract,threshold=threshold);exts=['.ram']
                for ext in ['.inputs','.labels']+exts:
                    arrays['fixture__'+key+'__'+ext[1:].replace('.','_')]=np.fromfile(Path(str(f)+ext),dtype='u1' if ext=='.inputs' else '<u4')
            for file in run.glob('native.*'):
                if file.suffix not in ['.ram','.scores','.latency_ns','.input_ns','.submit_ns','.fetch_ns']:continue
                arrays[run.name+'__'+file.name.replace('.','_')]=np.fromfile(file,dtype='<i8' if file.suffix.endswith('_ns') else '<u4')
            if not network and rec['status']=='PASS':
                for ext in ['ram','scores']:arrays[run.name+'__cpu_'+ext]=np.fromfile(run/f'cpu.{ext}',dtype='<u4')
        rows.append(row)
    label='balanced-network' if 'balanced' in name else 'network' if network else 'offload';out=P/'results'/f'2026-10-23-{label}';np.savez_compressed(out.with_suffix('.npz'),**arrays)
    result=dict(schema=1,status='RECOUNTED_WITH_FAILURES_PRESERVED',physical_chips=1,study=name,execution=state,fixtures=fixtures,runs=rows,archive_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest())
    if not network:
        result['confirmation']={}
        for arm in ['mmio','dma','resident_busy','resident_sleep']:
            rr=[r for r in rows if r['name'].startswith('confirm_') and r['name'].endswith('_'+arm)];assert len(rr)==3
            result['confirmation'][arm]={label:statistics.median(r['results']['native'][metric]/r['samples']/1000 for r in rr) for label,metric in [('wall_us_per_input','wall_ns'),('calling_thread_cpu_us_per_input','thread_cpu_ns')]}
        result['limitations']=['Calling-thread CPU is not whole-system CPU/IRQ accounting','No direct electrical energy measurement','Initial 16-input DMA smoke was much slower; cause unresolved; retained in raw results','Confirmation means are temporal repeats on one chip with dynamic clocks','Synthetic association workload; not image-service throughput','Available driver source was audited; exact loaded-module binary/source identity not established','Sparse-input metadata-only format adapters timed out; no claim of silicon sparse-input incapability']
    else:result['limitations']=['Three connected learning banks, not three independent input streams','Fixed encoders and two-count readout are supplied by us','Same external class label at all stages; local updates, no backpropagation','Intermediate responses reconstructed; per-event bank RAM and final output counts measured','No task accuracy or energy advantage follows from exact mechanism checks','CNP-interleaved larger family failed; no 4/7-bank physical expansion','No hidden-state restart or power-loss durability result']
    out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    for name,network in [('20261023-host-offload',False),('20261023-resident-network',True),('20261023-balanced-network',True)]:
        z=export(name,network);print(name,len(z['runs']),z.get('confirmation',{}))
