"""Recount raw native choices, process faults and image artifacts without builders."""
from pathlib import Path
import hashlib,json,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'service'))
from artifacts import verify as artifact, sha
from verify_native import verify as native_audit

def verify(d):
    s=json.loads((d/'SUMMARY.json').read_text());assert s['status']=='EXECUTION_COMPLETE' and s['mode']=='recovery'
    native=(d/'RECEIPT.json').exists();nv=native_audit(d) if native else None
    choices={e['chain_id']:e for e in nv['events'] if e['phase']=='choice'} if nv else {}
    obs={e['chain_id']:e for e in nv['events'] if e['phase']=='observation'} if nv else {}
    events=[json.loads(l) for l in (d/'events.jsonl').read_text().splitlines()]
    ops=[e for e in events if e['kind']=='operation'];env={e['job']:e['available'] for e in events if e['kind']=='environment'}
    states=[json.loads(l) for l in (d/'PROCESS_STATES.jsonl').read_text().splitlines()]
    offsets={71011:0,71012:1,71013:2};offset=offsets[s['seed']]
    assignments={71011:{'decode':['cpu','gstreamer'],'export':['cpu','gstreamer']},
        71012:{'decode':['gstreamer','cpu'],'export':['gstreamer','cpu']},
        71013:{'decode':['cpu','gstreamer'],'export':['gstreamer','cpu']}}[s['seed']]
    mapping=[None]*8;history=[None]*4;good=0;recoveries=[];first_return=None;probe_attempts=0;failed=0
    for e in ops:
        j,k,st,a,out=e['job'],e['local_attempt'],e['state'],e['action'],e['outcome']
        expected_probe=(j%2==1) if j<8 else (s['arm']!='reactive' and st==3 and k==0 and (j-8-offset)%4==0)
        assert e['probe']==expected_probe,'clock depends only on admitted counters/state'
        probe_attempts+=bool(e['probe'])
        direct=[mapping[2*st+b]==0 for b in range(2)]
        pred={z for z in range(4) if any(mapping[2*z+b]==0 for b in range(2))}
        hops=[mapping[2*st+b] in pred if mapping[2*st+b] is not None else False for b in range(2)]
        alternate=0 if history[st] and history[st][0]==1 else 1
        want=alternate if e['probe'] else direct.index(True) if any(direct) else hops.index(True) if any(hops) else alternate
        assert a==want,'decision mismatch'
        req=e['rpc']['request'];res=e['rpc']['result'];stage='decode' if st==3 else 'export'
        assert req['stage']==stage and req['backend']==assignments[stage][a]==e['backend']
        assert (env[j]=='all' or env[j]==e['backend'])==bool(res['ok']),'fault schedule versus real operation'
        if native:
            c=choices[e['query_id']];o=obs[e['observation_id']]
            assert (c['state'],c['action'],c['probe'])==(st,a,e['probe'])
            assert (o['state'],o['action'],o['outcome'])==(st,a,out)
            assert e['request_start_id'] in nv['request_starts']
        if res['ok']:
            assert artifact(req['input'],req['output'],stage)==e['verification']
            assert sha(req['input'])==res['input_sha256'] and sha(req['output'])==res['output_sha256']
            assert res['executed'] is True
            assert out==((1 if e['backend']=='cpu' else 2) if stage=='decode' else 0)
            good+=out==0
        else:
            assert out==st and e['verification'] is None
            assert res['error'] in ('actual_worker_RPC_timeout','actual_worker_endpoint_backpressure')
            start=req['deadline_unix']-2.;key=stage+':'+e['backend']
            seen=[x['states'][key] for x in states if start+.2<x['unix_time']<req['deadline_unix']-.1]
            if res['error']=='actual_worker_RPC_timeout':assert len(seen)>=2 and set(seen)=={'T'}
            else:assert res['errno']==11 and res['executed'] is False
            failed+=1
        if 12<=j<20 and st==3 and a==0 and res['ok']:
            if first_return is None:first_return=j
            if mapping[6]==3:recoveries.append(dict(job=j,probe=e['probe'],old_prediction=mapping[6],actual=out))
        if s['arm']!='frozen' or j<8:mapping[2*st+a]=out
        history[st]=[a,out]
    assert len(ops)==s['operation_attempts']<=80 and failed==s['worker_failures']
    assert good==s['completed']==sum(j['complete'] for j in s['jobs'])
    for job in s['jobs']:
        saved=json.loads((d/'jobs'/f'{job["job"]:03}'/'MANIFEST.json').read_text());assert saved==job
        actual=[e for e in ops if e['job']==job['job']]
        assert actual==[dict(kind='operation',**e) for e in job['operations']]
        st=3;src=job['source'];assert sha(src)==job['source_sha256']
        for k,e in enumerate(actual):
            assert e['local_attempt']==k and e['state']==st and e['rpc']['request']['input']==src
            if e['outcome']!=st:src=e['rpc']['request']['output']
            st=e['outcome']
        assert job['complete']==(st==0) and len(actual)<=8
        if st==0:assert sha(job['final_png'])==job['final_sha256']
    if nv:assert len(ops)==nv['native_action_selections']==nv['observations']
    return dict(status='PASS',run=d.name,arm=s['arm'],seed=s['seed'],completed=good,attempts=len(ops),failures=failed,
        probe_attempts=probe_attempts,first_preferred_return_job=first_return,recovery_events=recoveries,
        native=native,native_forwards=nv['native_forwards'] if nv else 0,
        all_actual_artifacts_checked=True,all_native_states_and_actions_exact=bool(nv),
        availability_is_evaluation_only=True,events_sha256=sha(d/'events.jsonl'),raw_sha256=nv['raw_sha256'] if nv else None)

if __name__=='__main__':
    for rel in sys.argv[1:]:
        d=P/rel;result=verify(d);(d/'VERIFICATION_RECOVERY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
