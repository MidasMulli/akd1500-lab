"""Public causal NPU parameter interventions and measured output differences."""
import json
import hashlib
import numpy as np
from capability.design import P
from capability.run import arrays,digest
from capability.contribution_run import S,recount as recount_run
BASE=P/'results/2026-10-24-cnp-contribution'
NAMES={'baseline'}|{f'suppress{i}' for i in range(11)}


def export():
    assert json.loads((S/'STATE.json').read_text())['research_admissions_closed'];assert {d.name for d in (S/'raw').iterdir()}==NAMES
    records=[];z={}
    for d in sorted((S/'raw').iterdir()):
        r=json.loads((d/'RECEIPT.json').read_text());assert r['status']=='PASS' and r['device_released']
        for f,h in r['hashes'].items():assert digest(d/f)==h,f
        a=arrays(d,r);r['audit']=recount_run(r,a);r['receipt_sha256']=digest(d/'RECEIPT.json');r.pop('native_pid',None);r.pop('hashes');records.append(r)
        for k,v in a.items():z[d.name+'__'+k]=v
    fr=json.loads((S/'evidence/FROZEN.json').read_text());meta=dict(status='ALL11_CNP_PARAMETER_CONTRIBUTIONS_VERIFIED',physical_sessions=12,native_forwards=1056,runs=records,
        executed_sources={p:h for p,h in fr['hashes'].items() if p.startswith(('capability/','src/'))},
        limitations=['Ordinary compiler threshold suppression, not physical NPU power gating.','Causal parameter-region contribution, not same-clock or transistor-level execution tracing.','One chip, one frozen synthetic workload.','No CPU speed or calibrated energy advantage.'])
    np.savez_compressed(BASE.with_suffix('.npz'),**z);meta['archive_sha256']=digest(BASE.with_suffix('.npz'));BASE.with_suffix('.json').write_text(json.dumps(meta,separators=(',',':'))+'\n');return audit_public()


def recount(meta,z):
    assert meta['status']=='ALL11_CNP_PARAMETER_CONTRIBUTIONS_VERIFIED' and meta['physical_sessions']==12 and meta['native_forwards']==1056
    assert {r['name'] for r in meta['runs']}==NAMES,'all11 causal treatments plus baseline'
    baseline=next(r for r in meta['runs'] if r['name']=='baseline');seen=[]
    for r in meta['runs']:
        assert r['status']=='PASS' and r['exit_code']==0 and r['device_released']
        a={k.split('__',1)[1]:v for k,v in z.items() if k.startswith(r['name']+'__')};assert recount_run(r,a)==r['audit']
        changes=[]
        for j,p in enumerate(r['fixture_info']['packets']):
            actual=a[f'packet{j}.initial'];base=z[f'baseline__packet{j}.initial'];assert len(base)==len(actual) and np.array_equal(base[:2],actual[:2])
            assert hashlib.sha256(a[f'packet{j}.expected'].astype('<u4').tobytes()).hexdigest()==p['sha256']
            header=int(actual[0]);mask=(header>>8)&15;assert mask>0 and not mask&(mask-1);npid=[header>>24,(header>>16)&255,mask.bit_length()-1];assert npid==p['np'],'physical packet coordinates'
            ii=np.flatnonzero(base!=actual).tolist()
            if ii:changes.append(dict(packet=j,np=p['np'],changed_words=ii))
        assert changes==r['fixture_info']['changed_packets'],'actual single-region intervention'
        if r['name']!='baseline':
            assert len(changes)==1;seen.append(tuple(changes[0]['np']))
            block=r['fixture_info']['suppressed_block'];assert np.array_equal(np.asarray(r['fixture_info']['thresholds'])[block[0]:block[1]],np.full(block[1]-block[0],32))
    assert len(seen)==len(set(seen))==11,'11 distinct physical CNP parameter regions'
    assert set(seen)=={tuple(p['np']) for p in baseline['fixture_info']['mapping'][0]['nps']}
    return dict(status='INDEPENDENT_ALL11_CNP_CAUSAL_PASS',physical_sessions=12,native_forwards=1056,distinct_causal_regions=11)


def audit_public():
    m=json.loads(BASE.with_suffix('.json').read_text());assert digest(BASE.with_suffix('.npz'))==m['archive_sha256']
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:return recount(m,{k:z[k] for k in z.files})


if __name__=='__main__':
    import sys
    print(json.dumps(export() if sys.argv[1]=='export' else audit_public()))
