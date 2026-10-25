"""Host-service audit against saved exact-rule traces, with explicit boundaries."""
from pathlib import Path
import hashlib,json
import numpy as np
from coprocessor.extent_audit import load_fixture,masks

P=Path(__file__).resolve().parents[1]
S=P/'studies/20261025-host-service'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def expected_states(initial,baseline,n):
    """Only fixed inference words may differ across potential/binary programs."""
    states=baseline.copy();states[:,:2]=initial[:2]
    states[:,2:].reshape(-1,128,4+n//32)[:,:,:3]=initial[2:].reshape(128,4+n//32)[None,:,:3]
    return states

def audit(d):
    r=json.loads((d/'RECEIPT.json').read_text());assert r['exit_code']==0
    for name,h in r['hashes'].items():assert sha(d/name)==h,('result hash',name)
    f=S/'fixtures'/r['fixture'];m,a=load_fixture(f);n=m['n'];words=len(a['ram']);count=r['count'];batch=r['batch']
    canon=S/'reference'/f's{m["side"]}_s{m["seed"]}_a{r["arm"]}'
    cs=np.fromfile(canon.with_suffix('.scores'),'<i4').reshape(64,128)
    cr=np.fromfile(canon.with_suffix('.snapshots'),'<u4').reshape(64,words)
    get=lambda k,dtype:np.fromfile(d/('result.'+k),dtype)
    y=get('scores','<i4').reshape(count,128);state=get('snapshots','<u4').reshape(-1,words);idx=get('boundary_index','<u4')
    assert np.array_equal(y,(cs[:count]>24).astype('i4') if m['output_contract']=='mask' else cs[:count]),'exact outputs'
    want=expected_states(a['ram'],cr,n)
    expected_idx=np.arange(batch,count+1,batch) if r['full_audit'] or r['mode']!='native' else np.array([count])
    assert np.array_equal(idx,expected_idx),'missing/unexpected RAM boundary'
    assert np.array_equal(state,want[idx-1]),'complete RAM boundary differs from exact CPU trace'
    assert np.array_equal(get('ram','<u4'),want[count-1]),'complete final RAM'
    before=np.concatenate([a['ram'][None],want[:count-1]])
    bm,am=masks(before,n),masks(want[:count],n)
    potentials=np.einsum('nk,njk->nj',a['features'].reshape(64,n)[:count].astype('i4'),bm.astype('i4'))
    assert np.array_equal(cs[:count],potentials),'independent potentials from stored connections'
    changed=np.any(before!=want[:count],axis=1)
    if r['arm'] and count>8:assert changed.any()
    else:assert not changed.any()
    for i in range(count):
        if not(r['arm'] and 8<=i<40):assert not changed[i],'query/frozen memory mutation'
    t=get('latency_ns','<i8').reshape(r['repeats'],count//batch)
    cpu=get('caller_ns','<i8').reshape(t.shape);assert np.all(t>0) and np.all(cpu>0)
    # Remove the same first eight events, irrespective of batch size.
    timed=t[:,8//batch:] if count>8 else t;used=cpu[:,8//batch:] if count>8 else cpu
    result=dict(status='EXACT_SERVICE_PASS',mode=r['mode'],fixture=r['fixture'],arm=r['arm'],count=count,batch=batch,
        sleep_us=r['sleep_us'],profile=r['profile'],full_audit=r['full_audit'],
        median_batch_us=float(np.median(timed)/1000),p95_batch_us=float(np.percentile(timed,95)/1000),
        amortized_wall_us_per_image=float(np.median(timed)/batch/1000),
        amortized_process_cpu_us_per_image=float(np.median(used)/batch/1000),
        observed_memory_boundaries=len(idx),learning_edit_events_in_reference=int(changed.sum()),
        masks_added_in_reference=int((am>bm).sum()),masks_removed_in_reference=int((bm>am).sum()))
    if r['mode']=='native':
        assert r['device_free']
        for j in range(len(m['fixed_packets'])):
            want_packet=np.fromfile(f/f'cell.packet{j}.ram','<u4')
            for tag in ['initial','final']:assert np.array_equal(get(f'packet{j}.{tag}','<u4'),want_packet)
        result['median_polls_per_batch']=float(np.median(get('polls','<i8')))
        result['median_read_bytes_per_batch']=float(np.median(get('read_bytes','<i8')))
        result['median_write_bytes_per_batch']=float(np.median(get('write_bytes','<i8')))
        for name in ['input','enqueue','fetch']:result[f'median_{name}_us_per_batch']=float(np.median(get(name+'_ns','<i8'))/1000)
    save(d/'AUDIT.json',result);return result
