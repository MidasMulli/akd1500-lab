"""Independent pixel/time/native-state recount; fail closed before expansion."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from streaming.design import N,U,NW,QUERY,STUDY,encode,observation,decode,band,plan,save,sha
from pong.relearn.model import Reference,RecentNearest

def words(path):return np.frombuffer(Path(path).read_bytes(),dtype='<u4').astype(np.uint32).tolist()

def audit_run(study,name):
    root=study/'raw'/name;owner=json.loads((root/'OWNER.json').read_text());assert owner['status']=='EXECUTION_PASS' and owner['device_released']
    rows=json.loads((root/'rows.json').read_text());frames=np.load(root/'frames.npz');events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()]
    replies=[json.loads(x) for x in (root/'native.stdout').read_text().splitlines()];forward=[r for r in replies if r['kind']=='forward']
    assert len(events)==len(rows)*3 and len(forward)==len(rows)*2==owner['native_forwards']
    assert all(len(frames[k])==len(rows) for k in ['current','previous','future','forecast'])
    ram=words(root/'fixture.ram');ref=Reference(ram,N,U,NW,5);cpu=RecentNearest();prev=ram[:]
    prefix_ram=None;prefix_scores=[];total_edits=0;post_prefix_edits=0
    scored=np.frombuffer((root/'native.scores').read_bytes(),'<i4').reshape(-1,U)
    xs=np.frombuffer((root/'fixture.inputs').read_bytes(),np.uint8).reshape(-1,N)
    labels=np.frombuffer((root/'fixture.labels').read_bytes(),'<i4')
    for i,row in enumerate(rows):
        q,o,f=events[3*i:3*i+3];assert [x['kind'] for x in [q,o,f]]==['query','outcome','feedback']
        assert q['sample']==o['sample']==f['sample']==i
        assert q['clock_ns']<o['clock_ns']<f['clock_ns']
        assert o['query_clock_ns']==q['clock_ns'] and f['outcome_clock_ns']==o['clock_ns']
        assert not any(k in q for k in ['truth','future','teacher','outcome']),'future field leaked to query'
        assert q['native']['host_monotonic_ns']<q['clock_ns']
        obs=observation(frames['previous'][i],frames['current'][i],q['action'],row['context']);assert obs==row['observation']==q['observation']
        x=encode(obs);assert hashlib.sha256(x.tobytes()).hexdigest()==q['input_sha256']
        truth=band(frames['future'][i]);assert row['truth']==o['truth']==truth
        assert band(frames['forecast'][i],False)==row['nano_prediction']==q['nano_prediction']
        assert row['persistence']==band(frames['current'][i])
        expected_teacher=truth if row['teach'] and (owner['arm']=='learn' or row['phase']=='prefix') else QUERY
        assert row['teacher']==f['teacher']==expected_teacher
        for reply,label in [(q['native'],QUERY),(f['native'],expected_teacher)]:
            j=reply['index'];assert reply==forward[j] and j<len(forward)
            assert np.array_equal(xs[j],x) and labels[j]==label and reply['label']==label
            want=ref.forward(x,label);assert reply['scores']==want==scored[j].tolist()
            actual=words(root/f'native.snapshot_{j}.ram');assert actual==ref.ram,'full RAM differs from independent native-rule replay'
            edits=sum((a^b).bit_count() for a,b in zip(prev,actual));assert edits>=reply['connection_bit_edits']
            if label==QUERY:assert actual==prev,'query altered parameter RAM'
            pred=cpu.forward(x,label,row['context']);assert pred==reply['cpu_task_prediction']
            if label==QUERY and j==row['query_native_index']:assert pred==row['cpu_prediction']
            total_edits+=reply['connection_bit_edits']
            if row['phase']=='prefix':prefix_scores.append(reply['scores']);prefix_ram=actual[:]
            else:
                post_prefix_edits+=reply['connection_bit_edits']
                if owner['arm']=='frozen':assert actual==prefix_ram,'trained frozen RAM changed'
            prev=actual
        prediction,score,margin=decode(q['native']['scores']);assert [prediction,score,margin]==[row['prediction'],row['score'],row['margin']]
    model=json.loads((root/'MODEL_RECEIPT.json').read_text());assert model['weights_unchanged'] and model['model_calls']==len(rows)
    assert owner['skipped_inferences']==0
    metrics={}
    for phase in sorted({r['phase'] for r in rows}):
        rr=[r for r in rows if r['phase']==phase];hist=np.bincount([r['truth'] for r in rr],minlength=4)
        metrics[phase]=dict(n=len(rr),histogram=hist.tolist(),**{k:float(np.mean([r[k]==r['truth'] for r in rr])) for k in ['prediction','cpu_prediction','persistence','velocity_prediction','nano_prediction']})
    eligible=[r for r in rows if r['phase'] in ['changed_after','retention'] and r['score']>=56 and r['margin']>=8]
    result=dict(status='INDEPENDENT_STREAM_RAM_AND_CAUSAL_ORDER_PASS',name=name,seed=owner['seed'],arm=owner['arm'],
                samples=len(rows),native_forwards=len(forward),complete_ram_snapshots=len(forward),
                connection_bit_edits=total_edits,post_prefix_connection_bit_edits=post_prefix_edits,metrics=metrics,
                prefix_ram_sha256=None if prefix_ram is None else hashlib.sha256(np.asarray(prefix_ram,'<u4').tobytes()).hexdigest(),
                prefix_scores_sha256=hashlib.sha256(np.asarray(prefix_scores,'<i4').tobytes()).hexdigest(),
                timing={k:float(np.median([r[k] for r in forward])) for k in ['transaction_ns','native_thread_cpu_ns','cpu_reference_ns','cpu_task_ns']},
                model_ns_median=float(np.median([r['model_ns'] for r in rows])),
                skip_shadow=dict(eligible=len(eligible),errors=sum(r['prediction']!=r['truth'] for r in eligible)),
                weights_unchanged=True,skipped_inferences=0,rows_sha256=sha(root/'rows.json'))
    save(root/'AUDIT.json',result);print(json.dumps(result));return result

def pair(study,seed):
    a=audit_run(study,f's{seed}_learn');b=audit_run(study,f's{seed}_frozen');g=plan()['gates'];checks={}
    for phase in ['prefix','baseline','changed_before','adapt','changed_after','retention']:
        assert a['metrics'][phase]['n']==b['metrics'][phase]['n'] and a['metrics'][phase]['histogram']==b['metrics'][phase]['histogram']
    for key in ['current','previous','future','forecast']:
        aa=np.load(study/'raw'/f's{seed}_learn/frames.npz')[key];bb=np.load(study/'raw'/f's{seed}_frozen/frames.npz')[key]
        assert np.array_equal(aa,bb),('matched model/stream pixel mismatch',key)
    checks['identical_trained_prefix']=a['prefix_ram_sha256']==b['prefix_ram_sha256'] and a['prefix_scores_sha256']==b['prefix_scores_sha256']
    checks['frozen_ram_unchanged']=b['post_prefix_connection_bit_edits']==0
    checks['native_adaptation_occurred']=a['post_prefix_connection_bit_edits']>0
    checks['baseline_quality']=a['metrics']['baseline']['prediction']>=g['minimum_baseline_accuracy']
    checks['changed_quality']=a['metrics']['changed_after']['prediction']>=g['minimum_changed_accuracy']
    checks['gain_over_frozen']=a['metrics']['changed_after']['prediction']-b['metrics']['changed_after']['prediction']>=g['minimum_gain_over_frozen']
    checks['gain_over_persistence']=a['metrics']['changed_after']['prediction']-a['metrics']['changed_after']['persistence']>=g['minimum_gain_over_persistence']
    checks['retention']=a['metrics']['retention']['prediction']>=b['metrics']['retention']['prediction']-g['maximum_retention_loss']
    checks['class_coverage']=all(min(a['metrics'][p]['histogram'])>=g['minimum_class_count'] for p in ['baseline','changed_after','retention'])
    skip=a['skip_shadow'];skip_checks=dict(prediction_gate=all(checks.values()),
        eligible=skip['eligible']>=g['skip_min_eligible'],error=skip['errors']/max(1,skip['eligible'])<=g['skip_error_max'],
        accuracy=a['metrics']['changed_after']['prediction']>=g['skip_accuracy_min'])
    result=dict(status='PASS_EXPANSION' if all(checks.values()) else 'STOP_PREDICTION_GATE',seed=seed,checks=checks,
                skip_checks=skip_checks,skipping_enabled=False,learn=a,frozen=b)
    save(study/'evidence'/f'PAIR_s{seed}.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('--name');p.add_argument('--seed',type=int);a=p.parse_args()
    if a.name:audit_run(a.study.resolve(),a.name)
    else:print(json.dumps({k:v for k,v in pair(a.study.resolve(),a.seed).items() if k not in ['learn','frozen']}))
