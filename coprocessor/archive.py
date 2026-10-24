"""Public, lossless numeric evidence; independent CPU replay needs no SDK."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from coprocessor.reference import features,trajectory

P=Path(__file__).resolve().parents[1]
BASE=P/'results/2026-10-24-pcie-fabric'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def export():
    arrays={};runs=[];sources={};binaries={};fixtures=[]
    s=P/'studies/20261024-pcie-fabric';f=s/'fixtures03'
    build=json.loads((f/'BUILD.json').read_text())
    for row in build['cells']:
        cell=row['name'];q=f/cell;w=int(cell.split('_c')[1]);z={}
        for key,dt,shape in [('inputs','u1',(96,32,32,3)),('features','u1',(96,512)),('w0','i1',(3,3,3,w)),('w1','i1',(3,3,w,8)),('t0','<i4',(w,)),('t1','<i4',(8,)),('ram','<u4',(2562,))]:
            z[key]=np.fromfile(q.with_suffix('.'+key),dt).reshape(shape);arrays[cell+'__'+key]=z[key]
        fixtures.append(dict(**row,program_sha256=digest(q.with_suffix('.bin')),file_hashes={p.name:digest(p) for p in f.glob(cell+'.*')}))
    for campaign in ['20261024-pcie-fabric','20261024-pcie-transport']:
        root=P/'studies'/campaign;frozen=json.loads((root/'evidence/FROZEN.json').read_text())
        for name,h in frozen['hashes'].items():
            if name.startswith(('coprocessor/','src/')):assert digest(P/name)==h;sources[name]=h
            if name.startswith('build/'):binaries[name]=h
        for d in sorted((root/'raw').iterdir()):
            rec=json.loads((d/'RECEIPT.json').read_text());assert rec['exit_code']==0 and rec['device_free']
            for name,h in rec['hashes'].items():assert digest(d/name)==h
            name=campaign.split('pcie-')[1]+'_'+d.name
            summary=json.loads(next(l for l in (d/'stdout').read_text().splitlines() if l.startswith('{')))
            row={k:rec[k] for k in ['mode','cell','learn','count','batch','repeat','repeats','source_freeze_sha256','admitted_utc','finish_utc']}
            row.update(name=name,campaign=campaign,result=summary,profile=rec.get('profile'),raw_hashes=rec['hashes'])
            if rec['count']:
                for key,dt in [('scores','<i4'),('ram','<u4'),('latency_ns','<i8'),('caller_ns','<i8')]:
                    a=np.fromfile(d/('result.'+key),dt)
                    if key=='scores':a=a.reshape(rec['count'],128)
                    arrays[name+'__'+key]=a
                if rec['mode']=='native':arrays[name+'__snapshots']=np.fromfile(d/'result.snapshots','<u4').reshape(rec['count']//rec['batch'],2562)
            runs.append(row)
    document=json.loads((P/'coprocessor/DOCUMENTS.json').read_text())
    metadata=dict(status='MEASURED_BOUNDED_CHARACTERIZATION',fixtures=fixtures,runs=runs,executed_sources=sources,executed_binary_sha256=binaries,compiler='g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0',compile_flags=['-std=c++17','-O3','-mcpu=native','AKIDA_VERSION=2.19.3'],
        inventory=json.loads((s/'evidence/INVENTORY.json').read_text()),compiler_frontier=json.loads((s/'evidence/COMPILER_FRONTIER.json').read_text()),documents=document,
        preserved_failures=[dict(attempt='compiler01',reason='CNP is convolution, not the HPC cross-correlation convention',physical_forwards=0),dict(attempt='compiler02',reason='Spatial-to-FC feature order is channel-column-row',physical_forwards=0)],
        limitations=['Synthetic seeded RGB/weights characterize execution; not recognition or generalization quality','One chip and initialization; temporal repeats are not independent scientific seeds','Timing includes input transfer/enqueue/fetch; excludes programming/preallocation/full-RAM diagnostics, verification and surrounding I/O','Calling-thread CPU excludes kernel workers/interrupts/other host threads; not electrical energy','SDK descriptor and compiler format boundaries are not silicon maxima','No independent simultaneous teachers, power-cycle durability, total SRAM occupancy or router-hop timing measured'])
    np.savez_compressed(BASE.with_suffix('.npz'),**arrays);metadata['numeric_archive_sha256']=digest(BASE.with_suffix('.npz'));save(BASE.with_suffix('.json'),metadata)
    return metadata

def recount(m,z,verify_sources=True):
    assert len(m['fixtures'])==3
    if verify_sources:
        for name,h in m['executed_sources'].items():assert digest(P/name)==h,('executed source drift',name)
    cached={}
    for f in m['fixtures']:
        cell=f['name'];a={k:z[cell+'__'+k] for k in ['inputs','features','w0','w1','t0','t1','ram']}
        assert np.array_equal(features(a),a['features']),'independent RGB-to-FC feature order'
        for learn in [False,True]:cached[cell,learn]=trajectory(a['ram'],a['features'],learn)
    expected_primary={f'fabric_cpu-{mode}_rgb32_c{w}_l{l}' for mode in ['dense','sparse'] for w in [16,64,128] for l in [0,1]}
    expected_primary|={'fabric_native_rgb32_c16_l1_b1_n8_r0'}
    expected_primary|={f'fabric_native_rgb32_c{w}_l{l}_b{b}_n96_r0' for w in [16,64,128] for l in [0,1] for b in [1,8]}
    expected_primary|={f'fabric_native_rgb32_c128_l{l}_b8_n96_r{r}' for l in [0,1] for r in [1,2]}
    expected_follow={'transport_copy_gate'}|{f'transport_p{p}_l1_n8_r0' for p in [1,2,3]}|{f'transport_p{p}_l{l}_n96_r{r}' for p in range(4) for l in [0,1] for r in [0,1,2]}
    assert len(m['runs'])==57 and {r['name'] for r in m['runs']}==expected_primary|expected_follow,'all staged controls and profiles retained'
    totals=dict(physical_sessions=0,native_forwards=0,full_learning_ram_snapshots=0,cpu_sample_executions=0);metrics=[]
    for r in m['runs']:
        name=r['name'];n=r['count'];physical=r['mode']=='native'
        totals['physical_sessions']+=physical
        if not n:
            assert r['profile']==-1 and r['result']['forwards']==0 and r['result']['bytes_verified_per_direction']==3712,'zero-forward transport gate'
            continue
        get=lambda key:z[name+'__'+key]
        expected,states=cached[r['cell'],r['learn']]
        assert np.array_equal(get('scores'),expected[:n]),('independent all-potential replay',name)
        assert np.array_equal(get('ram'),states[n-1]),('independent final full RAM',name)
        if physical:
            totals['native_forwards']+=n;totals['full_learning_ram_snapshots']+=n//r['batch']
            assert np.array_equal(get('snapshots'),states[r['batch']-1:n:r['batch']]),('independent full RAM boundaries',name)
            t=get('latency_ns');c=get('caller_ns');assert len(t)==len(c)==n//r['batch']
            if n>16:t=t[16//r['batch']:];c=c[16//r['batch']:]
            t=t/r['batch'];c=c/r['batch']
        else:
            assert r['repeats']==5;totals['cpu_sample_executions']+=n*5
            t=get('latency_ns').reshape(5,n)[:,16:].reshape(-1);c=get('caller_ns').reshape(5,n)[:,16:].reshape(-1)
        assert np.all(t>0) and np.all(c>0)
        metrics.append(dict(name=name,mode=r['mode'],cell=r['cell'],learn=r['learn'],batch=r['batch'],profile=r['profile'],repeat=r['repeat'],count=n,median_us=float(np.median(t)/1000),p95_us=float(np.percentile(t,95)/1000),median_caller_us=float(np.median(c)/1000)))
    assert totals==dict(physical_sessions=45,native_forwards=3872,full_learning_ram_snapshots=1016,cpu_sample_executions=5760),totals
    assert len(m['inventory']['descriptor']['nps'])==32
    return dict(status='INDEPENDENT_PUBLIC_REPLAY_PASS',**totals,metrics=metrics)

def audit_public():
    m=json.loads(BASE.with_suffix('.json').read_text());assert digest(BASE.with_suffix('.npz'))==m['numeric_archive_sha256']
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:return recount(m,z)
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='export':export()
    r=audit_public();save(P/'results/2026-10-24-pcie-fabric-audit.json',r);print(json.dumps({k:v for k,v in r.items() if k!='metrics'}))
