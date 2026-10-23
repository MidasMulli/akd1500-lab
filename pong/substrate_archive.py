"""Export exact-stream cost results without licensed runtime or compiled programs."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from verify import verify,read_words
from pong.experiment import save,sha


def summary(rows,kinds=None):
    result={}
    for kind in (kinds or ['mmio','dma','queued_dma','queued_dma_sleep']):
        r=[x for x in rows if x['configuration']==kind]
        assert len(r)==3
        metrics={}
        for name in ['native_elapsed_us_per_input','native_thread_cpu_us_per_input','exact_CPU_elapsed_us_per_input','exact_CPU_thread_us_per_input']:
            a=[x[name] for x in r];metrics[name]=dict(mean=float(np.mean(a)),min=min(a),max=max(a),runs=a)
        result[kind]=metrics
    return result


def export(study,stem='2026-10-23-pong-substrate'):
    plan=json.loads((study/'PROTOCOL.json').read_text());f=study/'fixtures-replay01/pong';arrays={};rows=[];runs=[];fixtures={}
    shared={suffix:np.frombuffer(f.with_suffix('.'+suffix).read_bytes(),dtype).copy() for suffix,dtype in [('inputs','u1'),('labels','<u4'),('ram','<u4')]}
    n,u,nw,classes=map(int,f.with_suffix('.meta').read_text().split())
    for raw in sorted((study/'raw').iterdir()):
        if raw.name=='transport_smoke':continue
        rec=json.loads((raw/'RECEIPT.json').read_text());checked=verify(ROOT,raw)
        assert (raw/'native.scores').read_bytes()==(study/'reference/native.scores').read_bytes()
        assert (raw/'native.ram').read_bytes()==(study/'reference/native.ram').read_bytes()
        key=raw.name
        fixtures[key]=dict(inputs=n,neurons=u,num_weights=nw,classes=classes)
        for suffix,value in shared.items():arrays[f'fixture__{key}__{suffix}']=value
        for mode in ['native','cpu']:
            for suffix in ['scores','ram']:arrays[f'{key}__{mode}_{suffix}']=np.array(read_words(raw/f'{mode}.{suffix}'),np.uint32)
        # The generic materializer supports this empty per-step capture. Every
        # score/update is independently replayed to the observed final RAM.
        arrays[key+'__snapshots']=np.empty((0,2+u*(4+n//32)),np.uint32)
        for path in raw.glob('native.*_ns'):arrays[key+'__'+path.name.replace('.','_')]=np.frombuffer(path.read_bytes(),'<i8').copy()
        runs.append(dict(name=key,fixture_key=key,samples=rec['samples'],learning=1,audit=checked,
            binary_sha256=rec['binary_sha256'],program_sha256=rec['program_sha256']))
        source=json.loads((study/'evidence'/f'CELL_{key}.json').read_text())
        native,cpu=rec['results']['native'],rec['results']['cpu'];count=rec['samples']
        row=dict(name=key,configuration=source['configuration'],
            native_elapsed_us_per_input=native['wall_ns']/count/1000,
            native_thread_cpu_us_per_input=native['thread_cpu_ns']/count/1000,
            exact_CPU_elapsed_us_per_input=cpu['wall_ns']/count/cpu['repeats']/1000,
            exact_CPU_thread_us_per_input=cpu['thread_cpu_ns']/count/cpu['repeats']/1000,
            results=rec['results'],profile_before=source['profile_before'],profile_after=source['profile_after'])
        rows.append(row)
    assert len(rows)==len(plan['configurations'])*plan['repeats']
    archive=ROOT/f'results/{stem}.npz';np.savez_compressed(archive,**arrays)
    result=dict(status='EXACT_MATCHED_STREAM_CPU_REPLAY_LOWER_COST',physical_chips=1,physical_sessions=len(rows)+1,
        verified_native_forwards=sum(r['samples'] for r in runs),transport_smoke=json.loads((study/'raw/transport_smoke/RECEIPT.json').read_text()),
        summary=summary(rows,list(plan['configurations'])),rows=rows,archive_sha256=sha(archive),
        fixtures=fixtures,runs=runs,
        original_sequence_scores_sha256=sha(study/'reference/native.scores'),original_final_RAM_sha256=sha(study/'reference/native.ram'),
        protocol=plan,paddle_control_admitted=False,learning_gate_unchanged='Pong relearning gate failed; this is a separate cost study',
        interpretation='DMA and queued waiting reduce caller CPU compared with MMIO; all tested native paths still cost more elapsed and calling-thread CPU time than exact CPU replay on this recorded workload',
        limitations=plan['boundaries']+[
            'Elapsed/input for queued modes is batch-amortized cost, not individual response latency',
            'Native per-step RAM snapshots are not taken in this throughput comparison; every output and complete final RAM match the independently reconstructed original stream',
            'Runs are temporal repeats of one recorded stream, not independent game environments',
            'This characterizes current model and host/runtime path, not silicon peak performance or all native learning banks'])
    save(ROOT/f'results/{stem}.json',result)
    print(json.dumps(dict(status=result['status'],summary=result['summary'],verified_native_forwards=result['verified_native_forwards'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('--stem',default='2026-10-23-pong-substrate');a=p.parse_args();export(a.study.resolve(),a.stem)
