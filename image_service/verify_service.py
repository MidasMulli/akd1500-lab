"""Recount actual artifacts and bind worker actions/outcomes to audited native buffers."""
from pathlib import Path
import hashlib,json,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'service'))
from artifacts import verify as check_artifact,sha
from verify_native import verify as check_native

def verify(d):
    summary=json.loads((d/'SUMMARY.json').read_text());assert summary['status']=='EXECUTION_COMPLETE'
    native=(d/'RECEIPT.json').exists();nv=check_native(d) if native else None
    choices={x['chain_id']:x for x in nv['events'] if x['phase']=='choice'} if native else {}
    observations={x['chain_id']:x for x in nv['events'] if x['phase']=='observation'} if native else {}
    events=[json.loads(l) for l in (d/'events.jsonl').read_text().splitlines()]
    operators=[x for x in events if x['kind']=='operation'];assert len(operators)==sum(x['attempts'] for x in summary['jobs'])
    states=[json.loads(l) for l in (d/'PROCESS_STATES.jsonl').read_text().splitlines()] if (d/'PROCESS_STATES.jsonl').exists() else []
    assignments={71011:dict(decode=['cpu','gstreamer'],export=['cpu','gstreamer']),71012:dict(decode=['gstreamer','cpu'],export=['gstreamer','cpu']),71013:dict(decode=['cpu','gstreamer'],export=['gstreamer','cpu'])}[summary.get('seed',71011)]
    cpu_mapping=[None]*8;cpu_history=[None]*4
    verified_outputs=0;goals=0;timeouts=0;backpressure=0
    for event in operators:
        req=event['rpc']['request'];result=event['rpc']['result'];s,a,out=event['state'],event['action'],event['outcome']
        assert req['stage']==('decode' if s==3 else 'export') and req['backend']==assignments[req['stage']][a]==event['backend']
        if native:
            c=choices[event['query_id']];o=observations[event['observation_id']]
            assert c['state']==s and c['action']==a and c['depth']==event['choice']['depth']
            assert o['state']==s and o['action']==a and o['outcome']==out
            assert event['request_start_id'] in nv['request_starts']
        else:
            direct=[cpu_mapping[2*s+k]==0 for k in range(2)];preds={x for x in range(4) if any(cpu_mapping[2*x+k]==0 for k in range(2))}
            hop=[cpu_mapping[2*s+k] in preds if cpu_mapping[2*s+k] is not None else False for k in range(2)]
            expected=direct.index(True) if any(direct) else hop.index(True) if any(hop) else (0 if cpu_history[s] is not None and cpu_history[s][0]==1 else 1)
            assert a==expected,'CPU baseline deviated from identical supplied algorithm'
            cpu_mapping[2*s+a]=out;cpu_history[s]=[a,out]
        if result['ok']:
            check=check_artifact(req['input'],req['output'],req['stage']);assert check==event['verification']
            assert sha(req['input'])==result['input_sha256'] and sha(req['output'])==result['output_sha256']
            assert result['executed'] is True
            want=(1 if req['backend']=='cpu' else 2) if req['stage']=='decode' else 0;assert out==want
            verified_outputs+=1;goals+=out==0
        else:
            assert out==s and event['verification'] is None
            if result['error']=='actual_worker_RPC_timeout':
                assert event['rpc']['host_elapsed_seconds']>=2.0
                if states:
                    start=req['deadline_unix']-2.0;key=req['stage']+':'+req['backend']
                    seen=[x['states'][key] for x in states if start+.2<x['unix_time']<req['deadline_unix']-.1]
                    assert len(seen)>=2 and all(x=='T' for x in seen),('timeout without recorded stopped worker',req['request_id'],set(seen))
                timeouts+=1
            elif result['error']=='actual_worker_endpoint_backpressure':
                assert result['errno']==11 and result['executed'] is False
                if states:
                    start=req['deadline_unix']-2.0;key=req['stage']+':'+req['backend']
                    seen=[x['states'][key] for x in states if abs(x['unix_time']-start)<.1]
                    assert seen and 'T' in seen,'backpressure without stopped owned worker'
                backpressure+=1
    for job in summary['jobs']:
        f=d/'jobs'/f'{job["job"]:03}'/'MANIFEST.json';saved=json.loads(f.read_text())
        assert saved=={k:v for k,v in job.items() if k!='epoch'}
        actual=[x for x in operators if x['job']==job['job']]
        assert actual==[dict(kind='operation',**x) for x in job['operations']]
        state=3;source=job['source'];assert sha(source)==job['source_sha256']
        for i,x in enumerate(actual):
            assert x['local_attempt']==i and x['state']==state and x['rpc']['request']['input']==source
            if x['outcome']!=state:source=x['rpc']['request']['output']
            state=x['outcome']
        assert job['complete']==(state==0)
        assert job['complete']==bool(job['final_png'])
        if job['complete']:assert sha(job['final_png'])==job['final_sha256']
    assert goals==summary['completed']==sum(j['complete'] for j in summary['jobs'])
    if native:assert len(operators)==nv['native_action_selections']==nv['observations']
    if summary.get('mode')!='watch':
        assert len(operators)<=60 and all(j['attempts']<=8 for j in summary['jobs'])
        forecast=json.loads((P/'evidence/PRE_PHYSICAL_FORECAST.json').read_text());f=next(x for x in forecast['forecasts'] if x['seed']==summary['seed'] and x['arm']==summary['arm'])
        forecast_match=(goals,len(operators),summary['worker_failures'])==(f['completed'],f['attempts'],f['failures'])
    else:forecast_match=None
    return dict(status='PASS',run=d.name,native=native,completed=goals,
                attempts=len(operators),worker_failures=summary.get('worker_failures',timeouts),actual_timeout_failures=timeouts,
                actual_socket_backpressure_failures=backpressure,
                valid_worker_artifacts=verified_outputs,all_goals_have_verified_PNGs=True,
                native_action_and_observation_binding=bool(native),
                recorded_process_states_validate_timeouts=bool(states),prospective_forecast_match=forecast_match,
                per_epoch_completed=summary.get('per_epoch_completed'),per_epoch_attempts=summary.get('per_epoch_attempts'),
                raw_sha256=nv['raw_sha256'] if native else None,
                events_sha256=hashlib.sha256((d/'events.jsonl').read_bytes()).hexdigest())
if __name__=='__main__':
    for rel in sys.argv[1:]:
        d=P/rel;obj=verify(d);(d/'VERIFICATION_SERVICE.json').write_text(json.dumps(obj,indent=2)+'\n');print(json.dumps(obj))
