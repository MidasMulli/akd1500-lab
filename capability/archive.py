"""Public exact-output evidence, including the failed prefix and zero-forward cleanup."""
import json
import numpy as np
from capability.design import P,S,CELLS,DENSITIES,weights
from capability import audit,failure
from capability.run import arrays,audit_run,cpu_arrays,save,digest,fixed_name
from resident.or_archive import compact,expand
BASE=P/'results/2026-10-24-native-primitives'
ALLOC=P/'studies/20261024-queue-allocation'
QUIET_PASS=['quiet_b8_s0_r0','quiet_b8_s20_r0','quiet_b32_s100_r0']
FAILED='quiet_b128_s250_r0'


def names():return {fixed_name(f'{k}{u}',False,r) for k,u in CELLS for r in range(3)}|{fixed_name(k,True,0) for k in ['fnp128','cnp128']}|set(QUIET_PASS)|{FAILED}


def decode(rec,z):
    z=dict(z)
    if rec['kind']=='quiet':
        count=rec['snapshot_boundaries']
        for k in range(3):z[f'bank{k}.snapshots']=expand(z[f'bank{k}.initial'],{key:z[f'bank{k}.delta.{key}'] for key in ['offsets','indices','values','sha256']},count)
    return z


def export():
    assert json.loads((S/'STATE.json').read_text())['research_admissions_closed'];assert {p.name for p in (S/'raw').iterdir()}==names()
    records=[];data={}
    for d in sorted((S/'raw').iterdir()):
        rec=json.loads((d/'RECEIPT.json').read_text());rec.pop('native_pid',None);rec['receipt_sha256']=digest(d/'RECEIPT.json');rec['requested_forwards']=rec['forwards']
        if d.name==FAILED:rec,z,result=failure.raw(d);rec.pop('native_pid',None);rec['receipt_sha256']=digest(d/'RECEIPT.json');rec['requested_forwards']=768;rec['forwards']=512;rec['samples']=512
        else:
            result=audit_run(d);z=arrays(d,rec)
            if rec['kind']=='quiet':rec['rows']=json.loads((d/'stream.rows.json').read_text())
            else:rec['units']=rec['fixture_info']['units']
        rec['audit']=result;rec.pop('hashes',None)
        if rec['kind']=='quiet':
            rec['snapshot_boundaries']=len(z['bank0.snapshots'])
            for k in range(3):
                enc=compact(z[f'bank{k}.initial'],z.pop(f'bank{k}.snapshots'))
                for key,v in enc.items():z[f'bank{k}.delta.{key}']=v
        records.append(rec)
        for key,v in z.items():data[d.name+'__'+key]=v
    cpus=[]
    for u in [128,256,512,1024,2048]:
        d=S/'cpu'/f'u{u}';rec=json.loads((d/'RECEIPT.json').read_text());z=cpu_arrays(d,rec);rec['audit']=audit.cpu(rec,z);rec['name']=f'cpu{u}';cpus.append(rec)
        for k,v in z.items():data[f'cpu{u}__'+k]=v
    alloc=json.loads((ALLOC/'raw/allocation01/AUDIT.json').read_text());ad=ALLOC/'raw/allocation01';data['allocation__requested_effective']=np.fromfile(ad/'native.requested_effective','<u4').reshape(6,2)
    for j in range(6):
        for suffix in ['initial','final']:data[f'allocation__packet{j}.{suffix}']=np.fromfile(ad/f'native.packet{j}.{suffix}','<u4')
    fr=json.loads((S/'evidence/FROZEN.json').read_text());sources={p:dict(sha256=h,public_path='capability/executed/quiet_relay_v1.cpp' if p=='src/quiet_relay.cpp' else p) for p,h in fr['hashes'].items() if p.startswith(('capability/','src/','tools/','resident/','pong/'))}
    meta=dict(status='NATIVE_SCALING_AND_BLOCKED_HOST_LEARNING_WITH_PRESERVED_LIMIT',physical_experiment_sessions=30,allocation_cleanup_sessions=1,native_forwards=7984,complete_learning_bank_snapshots=7404,
        runs=records,cpu_runs=cpus,allocation=alloc,executed_sources=sources,source_audit=json.loads((ALLOC/'evidence/SOURCE_AUDIT.json').read_text()),repairs=json.loads((S/'evidence/REPAIRS.json').read_text()),operations=json.loads((S/'evidence/OPERATIONS.json').read_text()),
        post_close_safety=json.loads((S/'evidence/HARDENED_REFUSAL.json').read_text()),
        host_configuration={k:v for k,v in json.loads((S/'evidence/HOST.json').read_text()).items() if k!='cpuinfo'},
        limitations=['Fixed paths: one frozen weight/input bank, three temporal repeats on one chip.','Quiet learning: one initialization, one run per qualified arm;128-event stage failed before repair and then crashed during cleanup.','Host thread sleeps; not whole-host sleep, interrupt-driven completion or autonomous teaching.','Sleep bounds completion by first wake; it is not the actual device completion timestamp.','Released-interval caller CPU excludes preload/drain and diagnostics.','DMA monitor cycles are buffer service, not per-NPU arithmetic or electrical spike traces.','Matched dense input/two-count CPU implementations are faster here; no CPU speed or calibrated energy advantage.','CNP/FNP readout placements differ; same mathematics does not isolate transistor-level circuit causes.'])
    np.savez_compressed(BASE.with_suffix('.npz'),**data);meta['archive_sha256']=digest(BASE.with_suffix('.npz'));BASE.with_suffix('.json').write_text(json.dumps(meta,separators=(',',':'))+'\n');return audit_public()


def recount(meta,z):
    assert meta['status']=='NATIVE_SCALING_AND_BLOCKED_HOST_LEARNING_WITH_PRESERVED_LIMIT'
    assert {r['name'] for r in meta['runs']}==names(),'complete primary set including failed128 stage'
    assert meta['physical_experiment_sessions']==30 and meta['allocation_cleanup_sessions']==1 and meta['native_forwards']==7984 and meta['complete_learning_bank_snapshots']==7404
    for rec in meta['runs']:
        data=decode(rec,{k.split('__',1)[1]:v for k,v in z.items() if k.startswith(rec['name']+'__')})
        result=failure.recount(rec,data) if rec['name']==FAILED else (audit.fixed(rec,data) if rec['kind']=='fixed' else audit.quiet(rec,data))
        assert result==rec['audit'],'public native recount'
        assert rec['device_released']
        if rec['name']!=FAILED:assert rec['status']=='PASS' and rec['exit_code']==0
        if rec['kind']=='fixed':
            assert __import__('hashlib').sha256(weights()[:,:rec['units']].T.tobytes()).hexdigest()==rec['fixture_info']['weight_sha256'],'frozen identical primitive weights'
            assert len({tuple(q['np']) for l in rec['fixture_info']['mapping'] for q in l['nps']})>=2
            for j,p in enumerate(rec['fixture_info']['packets']):assert __import__('hashlib').sha256(data[f'packet{j}.expected'].astype('<u4').tobytes()).hexdigest()==p['sha256']
    assert {r['units'] for r in meta['cpu_runs']}=={128,256,512,1024,2048}
    for rec in meta['cpu_runs']:
        data={k.split('__',1)[1]:v for k,v in z.items() if k.startswith(rec['name']+'__')};assert audit.cpu(rec,data)==rec['audit']
    assert np.array_equal(z['allocation__requested_effective'],[[1,1],[9,9],[33,33],[65,65],[128,128],[129,128]])
    for j in range(6):assert np.array_equal(z[f'allocation__packet{j}.initial'],z[f'allocation__packet{j}.final'])
    assert meta['allocation']['status']=='INDEPENDENT_ALLOCATION_CLEANUP_PASS' and meta['allocation']['physical_forwards']==0
    for name in QUIET_PASS[1:]:
        assert np.array_equal(z[name+'__scores'],z[QUIET_PASS[0]+'__scores']),'same queued and polled outputs'
        for k in range(3):assert np.array_equal(z[name+f'__bank{k}.final'],z[QUIET_PASS[0]+f'__bank{k}.final']),'same queued and polled final RAM'
    return dict(status='INDEPENDENT_PRIMITIVE_ARCHIVE_PASS',physical_sessions=31,native_forwards=7984,complete_learning_bank_snapshots=7404,cpu_timed_calls=sum(r['audit']['timed_calls'] for r in meta['cpu_runs']),failed_stage_preserved=True)


def audit_public():
    m=json.loads(BASE.with_suffix('.json').read_text());assert digest(BASE.with_suffix('.npz'))==m['archive_sha256']
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:return recount(m,{k:z[k] for k in z.files})


if __name__=='__main__':
    import sys
    print(json.dumps(export() if sys.argv[1]=='export' else audit_public()))
