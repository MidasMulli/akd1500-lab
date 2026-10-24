"""Exact external-record evidence, including both halted learning prefixes."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from coprocessor.run import P,digest,save
from coprocessor.fnp2_capacity_run import trajectory
from resident.or_archive import expand
BASE=P/'results/2026-10-24-fnp2-contract'
CAMPAIGNS=['contract','capacity','guard','boundary']

def compact32(initial,states):
    # The existing complete-state decoder accepts uint32 indices. The older
    # uint16 encoder is intentionally not used for buffers above 65536 words.
    previous=initial.copy();offsets=[0];indices=[];values=[];hashes=[]
    for row in states:
        ii=np.flatnonzero(previous!=row);indices.extend(ii.tolist());values.extend(row[ii].tolist());offsets.append(len(indices));hashes.append(np.frombuffer(hashlib.sha256(row.astype('<u4').tobytes()).digest(),np.uint8));previous=row
    return dict(offsets=np.asarray(offsets,'<u4'),indices=np.asarray(indices,'<u4'),values=np.asarray(values,'<u4'),sha256=np.stack(hashes))

def export():
    data={};runs=[];fixtures={};sources={};binaries={}
    for campaign in CAMPAIGNS:
        root=P/'studies'/f'20261024-fnp2-{campaign}';assert json.loads((root/'STATE.json').read_text())['closed']
        frozen=json.loads((root/'evidence/FROZEN.json').read_text())
        for p,h in frozen['hashes'].items():
            if p.startswith(('coprocessor/','src/','pong/')):assert digest(P/p)==h;sources[p]=h
            if p.startswith('build/'):binaries[p]=h
        for d in sorted((root/'raw').iterdir()):
            r=json.loads((d/'RECEIPT.json').read_text());assert r['device_free']
            for p,h in r['hashes'].items():assert digest(d/p)==h
            u=r.get('units',544);case=r.get('case');fixture=root/('fixtures02' if campaign=='contract' else 'fixtures01')
            if campaign!='contract':fixture=fixture/(case if campaign=='boundary' else f'u{u}')
            f=fixture/'learner';fid=campaign+'_'+(case or str(u));name=campaign+'_'+d.name
            if fid not in fixtures:
                initial=np.fromfile(f.with_suffix('.raw'),'<u4');x=np.fromfile(f.with_suffix('.inputs'),'u1').reshape(64,512)
                data[fid+'__initial']=initial;data[fid+'__inputs']=x
                fixtures[fid]=dict(units=u,compiler=json.loads(f.with_suffix('.json').read_text()),program_sha256=digest(f.with_suffix('.bin')))
            initial=data[fid+'__initial'];scores=np.fromfile(d/'result.scores','<i4').reshape(-1,u);n=len(scores)
            row=dict(name=name,campaign=campaign,fixture=fid,units=u,mode=r['mode'],learn=r['learn'],intended_count=r['count'],observed_count=n,exit_code=r['exit_code'],device_free=r['device_free'],source_freeze_sha256=r['source_freeze_sha256'],raw_hashes=r['hashes'],admitted_utc=r['admitted_utc'],finish_utc=r['finish_utc'],error=(d/'stderr').read_text() if r['exit_code'] else None)
            data[name+'__scores']=scores;data[name+'__final']=np.fromfile(d/'result.raw','<u4')
            if r['mode']=='native':
                states=np.fromfile(d/'result.snapshots','<u4').reshape(n,u*20)
                for k,v in compact32(initial,states).items():data[name+'__delta.'+k]=v
                data[name+'__storage']=np.fromfile(d/'result.storage','<u4')
                if (d/'result.guard_counts').exists():data[name+'__guard_counts']=np.fromfile(d/'result.guard_counts','<u8')
            if not r['exit_code']:
                row['result']=json.loads(next(l for l in (d/'stdout').read_text().splitlines() if l.startswith('{')))
                for key in ['latency_ns','caller_ns']:data[name+'__'+key]=np.fromfile(d/('result.'+key),'<i8')
            runs.append(row)
    evidence=P/'studies/20261024-fnp2-boundary/evidence'
    meta=dict(status='FNP2_DECODE_AND_PRESERVED_OFF_TARGET_UPDATE_FAILURE',fixtures=fixtures,runs=runs,executed_sources=sources,executed_binary_sha256=binaries,
        source_audit=json.loads((P/'studies/20261024-fnp2-contract/evidence/SOURCE_AUDIT.json').read_text()),diagnosis=json.loads((evidence/'DIAGNOSIS.json').read_text()),
        rejected_explanatory_models={p.name:json.loads(p.read_text()) for p in (P/'studies/20261024-fnp2-guard/evidence').glob('*HYPOTHESIS.json')},
        limitations=['Successful eight-event capacity prefixes do not establish address coverage or converged learning','11,264-record frozen inference passes; teaching stops after four completed forwards','Controlled record6500 learns correctly; record6600 probe produces off-target record46 changes','Guard observes writes issued through this harness, not arbitrary kernel/device writers','512KiB-related metadata alias is an observation; internal cause and universal silicon limit unresolved','Current records reside in the card SRAM aperture; host DDR and multipass are not qualified','No CPU speed or electrical-energy win. Timings are unwarmed except the 544-record 64-event control'])
    np.savez_compressed(BASE.with_suffix('.npz'),**data);meta['numeric_archive_sha256']=digest(BASE.with_suffix('.npz'));save(BASE.with_suffix('.json'),meta)

def recount(m,z):
    for p,h in m['executed_sources'].items():assert digest(P/p)==h,('source identity',p)
    assert len(m['runs'])==34,'all contract/capacity/failed diagnostic runs retained'
    failed={'guard_native_u11264_l1':4,'boundary_native_above_l1':2}
    assert {r['name'] for r in m['runs'] if r['exit_code']}==set(failed),'both failures must remain failures'
    cache={};totals=dict(physical_sessions=0,native_forwards=0,full_record_snapshots=0,cpu_sample_executions=0,failed_physical_sessions=0);findings=[]
    for r in m['runs']:
        fid=r['fixture'];u=r['units'];n=r['observed_count'];name=r['name'];initial=z[fid+'__initial'];x=z[fid+'__inputs'][:r['intended_count']]
        key=fid,r['learn']
        if key not in cache:cache[key]=trajectory(initial,x,u,r['learn'])
        expected,states=cache[key];get=lambda k:z[name+'__'+k]
        assert np.array_equal(get('scores'),expected[:n]),('all FNP2 potentials',name)
        native=r['mode']=='native';totals['physical_sessions']+=native
        if native:
            snapshots=expand(initial,{k:get('delta.'+k) for k in ['offsets','indices','values','sha256']},n)
            assert np.array_equal(get('final'),snapshots[-1]);address,size,base,extent=map(int,get('storage'));assert size==u*80 and base+256<=address and address+size<=base+extent,'actual card SRAM domain'
            totals['native_forwards']+=n;totals['full_record_snapshots']+=n
        if r['exit_code']:
            assert native and n==failed[name] and 'learning-state mismatch' in r['error']
            assert np.array_equal(snapshots[:-1],states[:n-1]),'qualified failed prefix'
            assert not np.array_equal(snapshots[-1],states[n-1]),'failure cannot be promoted to pass'
            actual_ids=np.flatnonzero((snapshots[-1].reshape(u,20)!=snapshots[-2].reshape(u,20)).any(1)).tolist();want_ids=np.flatnonzero((states[n-1].reshape(u,20)!=states[n-2].reshape(u,20)).any(1)).tolist()
            assert (actual_ids,want_ids)==(([2295,2296],[8849]) if name.startswith('guard_') else ([46],[6600])),'off-target update location'
            totals['failed_physical_sessions']+=1;findings.append(dict(name=name,actual_updated_records=actual_ids,intended_records=want_ids))
        else:
            assert n==r['intended_count'] and np.array_equal(get('final'),states[n-1])
            if native:assert np.array_equal(snapshots,states[:n]),'all complete FNP2 states'
            else:totals['cpu_sample_executions']+=5*n
        if r['campaign']=='boundary' and native:
            counts=get('guard_counts');assert int(counts[0])==0 and int(counts[1])>0,'host record write observer'
    assert totals==dict(physical_sessions=18,native_forwards=246,full_record_snapshots=246,cpu_sample_executions=1200,failed_physical_sessions=2),totals
    return dict(status='INDEPENDENT_FNP2_RECOUNT_WITH_PRESERVED_FAILURES',**totals,findings=findings,largest_inference_population=11264,maximum_learning_population='Not established; full address coverage fails in the configured path')

def audit_public():
    m=json.loads(BASE.with_suffix('.json').read_text());assert digest(BASE.with_suffix('.npz'))==m['numeric_archive_sha256']
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:return recount(m,z)
if __name__=='__main__':
    if sys.argv[1]=='export':export()
    result=audit_public();save(P/'results/2026-10-24-fnp2-audit.json',result);print(json.dumps(result))
