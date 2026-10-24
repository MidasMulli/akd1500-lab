"""Portable repeated-state service replay and GPU resource-result audit."""
from pathlib import Path
import argparse,hashlib,json,re
import numpy as np
from feedback.archive import BASE
from feedback.design import P,S
from pong.relearn.model import Reference
OUT=P/'results/20261024-predictive-corun'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):p.write_text(json.dumps(r,indent=2)+'\n')
def build():
    arrays={};runs=[]
    for d in sorted((S/'gpu').iterdir()):
        rec=json.loads((d/'RECEIPT.json').read_text());aud=json.loads((d/'AUDIT.json').read_text());assert aud['status']=='GPU_AND_SERVICE_AUDIT_PASS';files={}
        for p in d.iterdir():
            if not(p.name.endswith('.frames_ns') or p.name.startswith('service.') and p.suffix in ['.scores','.ram','.batch_ns']):continue
            typ='<i8' if p.name.endswith('_ns') else '<u4' if p.suffix=='.ram' else '<i4';key=d.name+'__'+p.name.replace('.','_');arrays[key]=np.fromfile(p,typ);files[p.name]=dict(key=key,sha256=sha(p))
        log=(d/'tegrastats.log').read_text();emc=[int(x) for x in re.findall(r'EMC_FREQ\s+(\d+)%',log)];freq=[int(x) for x in re.findall(r'EMC_FREQ\s+\d+%@([0-9]+)',log)];temps=[float(x) for x in re.findall(r'(?:cpu|soc0|soc1|soc2|gpu)@([0-9.]+)C',log)]
        runs.append(dict(name=d.name,receipt=rec,audit=aud,arrays=files,gpu_ready=json.loads((d/'GPU_READY.json').read_text()),telemetry_summary=dict(samples=len(log.splitlines()),emc_percent_range=[min(emc),max(emc)] if emc else None,emc_frequency_mhz=sorted(set(freq)),temperature_range_c=[min(temps),max(temps)] if temps else None),runtime_warning='CC8.7 PyTorch build support warning present' if 'No published PyTorch' in (d/'gpu.stderr').read_text() else None))
    np.savez_compressed(OUT.with_suffix('.npz'),**arrays);save(OUT.with_suffix('.json'),dict(archive_sha256=sha(OUT.with_suffix('.npz')),learning_archive_sha256=sha(BASE.with_suffix('.npz')),runs=runs,source_freeze=json.loads((S/'evidence/GPU_FREEZE.json').read_text()),scope='Fixed T202 GPU service and paced separate learner workload; no per-client DRAM bandwidth or calibrated energy measurement'))
    return audit()
def audit():
    meta=json.loads(OUT.with_suffix('.json').read_text());assert sha(OUT.with_suffix('.npz'))==meta['archive_sha256'];learn=json.loads(BASE.with_suffix('.json').read_text());f=learn['fixtures']['s108101'];results=[]
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as src,np.load(OUT.with_suffix('.npz'),allow_pickle=False) as z:
        x=src[f['arrays']['native.inputs']['key']].reshape(768,512);initial=src[f['arrays']['reduced.ram']['key']];rows=f['rows']['native'];cpu=Reference(initial,n=512,u=16,nw=32,classes=1);expected=[]
        for i in range(3072):expected.append(sum(v>13 for v in cpu.forward(x[i%768],0)))
        expected=np.array(expected,'<i4');q=np.array([not rows[i%768]['feedback'] for i in range(3072)])
        assert len(meta['runs'])==12 and len({r['name'] for r in meta['runs']})==12
        outputs=set()
        for r in meta['runs']:
            rec=r['receipt'];a=r['audit'];get=lambda k:z[r['arrays'][k]['key']]
            assert rec['gpu_exit']==0 and rec['service_exit']==0 and rec['gpu_result']['receipt']['weights_unchanged'];outputs.add(a['output_sha256'])
            for v in r['arrays'].values():assert hashlib.sha256(z[v['key']].tobytes()).hexdigest()==v['sha256']
            frames=get('gpu.frames_ns').reshape(-1,3);assert len(frames)==a['frames'] and np.all(frames[:,1]>frames[:,0])
            if a['arm']!='alone':
                got=get('service.scores');assert got.shape==(3072,)
                if a['arm'] in ['native','cpu']:
                    assert np.array_equal(got,expected),'continuous exact outputs';assert np.array_equal(get('service.ram'),cpu.ram),'continuous full SRAM replay'
                else:assert np.array_equal(got[q],expected[q]),'optimized task CPU prediction equivalence'
                batch=get('service.batch_ns').reshape(-1,4);assert len(batch)==96 and np.all(batch[:,2]>batch[:,1])
                assert int(np.sum(batch[:,2]>batch[:,3]+2000000))==a['service_late_batches'],'late batches independently recounted'
            results.append(a)
    assert len(outputs)==1
    summary={}
    for arm in ['alone','cpu','native','task']:
        rr=[r for r in results if r['arm']==arm];summary[arm]=dict(gpu_fps=[r['gpu_forecasts_per_second'] for r in rr],median_gpu_fps=float(np.median([r['gpu_forecasts_per_second'] for r in rr])),gpu_median_ms=[r['gpu_median_ms'] for r in rr],gpu_p95_ms=[r['gpu_p95_ms'] for r in rr])
        if arm!='alone':summary[arm]['service_thread_cpu_ms']=[r['service_thread_cpu_ms'] for r in rr];summary[arm]['median_service_cpu_percent_of_one_core']=float(np.median([100*r['service_thread_cpu_ms']/r['service_wall_ms'] for r in rr]))
    late=sum(r.get('service_late_batches',0) for r in results)
    report=dict(status='INDEPENDENT_CONTINUOUS_REPLAY_PASS_WITH_DEADLINE_EXCEPTION' if late else 'INDEPENDENT_CONTINUOUS_REPLAY_PASS',runs=12,native_forwards=9216,gpu_outputs_identical=True,model_weights_unchanged=True,all_service_deadlines_met=late==0,late_batches_over_2ms=late,summary=summary,interpretation='No reproducible GPU-service or host-CPU advantage established for this small cache-friendly prediction task; native input-gated learning remains qualified separately. Retain the one late optimized-CPU batch as a scheduling/timing exception; no rerun or cause attribution.',archive_sha256=meta['archive_sha256']);save(OUT.with_suffix('.audit.json'),report);return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','audit']);a=p.parse_args();print(json.dumps(build() if a.action=='build' else audit()))
