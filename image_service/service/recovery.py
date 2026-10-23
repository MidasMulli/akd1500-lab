"""Recovery study: fixed exploration clock, native action choice, real workers."""
from pathlib import Path
import argparse, datetime, json, sys, time
from app import ASSIGNMENTS, perform_job
from controller import CPUController
from native_client import NativeClient, P, save
from pool import Workers

def probe_clock(job, local, state, offset=0, enabled=True):
    # Supplied policy. No availability, prediction, outcome or wall-clock input.
    if job < 8:
        return job % 2 == 1
    return enabled and state == 3 and local == 0 and (job - 8 - offset) % 4 == 0

def run(name, stage, seed, arm, count=24):
    assert seed in ASSIGNMENTS and arm in ('probe','reactive','frozen','cpu')
    assert count in (2,24)
    seed_offset={71011:0,71012:1,71013:2}[seed]
    cpu=arm=='cpu'; native=None; workers=None; failed=True
    ctl=CPUController(); primary=ASSIGNMENTS[seed]['decode'][0]
    alternate='gstreamer' if primary=='cpu' else 'cpu'
    if cpu:
        d=P/'cpu_runs'/name;d.mkdir(exist_ok=False)
    else:
        native=NativeClient(name,stage);d=native.d
    images=sorted((P/'inputs'/str(seed)).glob('*.jpg'));assert len(images)==4
    log=(d/'events.jsonl').open('w');jobs=[];remaining=80;start=time.monotonic()
    schedule=['all']*8+[alternate]*4+['all']*8+[primary]*4
    try:
        workers=Workers(d);save(d/'WORKER_OWNERS.json',workers.owners)
        for job in range(count):
            assert time.monotonic()-start<170
            if remaining<=0:break
            available=schedule[job]
            workers.set_available(available)
            if job==8 and arm=='frozen':native.mode(0,1)
            log.write(json.dumps(dict(kind='environment',job=job,available=available))+'\n');log.flush()
            clock=lambda j,k,s:probe_clock(j,k,s,seed_offset,arm!='reactive')
            result=perform_job(images[job%4],job,d,workers,native,ctl,ASSIGNMENTS[seed],remaining,log,clock)
            jobs.append(result);remaining-=result['attempts']
        summary=dict(status='EXECUTION_COMPLETE',mode='recovery',arm=arm,seed=seed,probe_offset=seed_offset,
            offered_requests=count,completed=sum(j['complete'] for j in jobs),attempted_requests=len(jobs),
            operation_attempts=80-remaining,worker_failures=sum(j['failures'] for j in jobs),
            host_elapsed_seconds=time.monotonic()-start,jobs=jobs,
            supplied='fixed probe clock, opposite-last-action rule, task codes and lookahead',
            claim='Real job completion and observed recovery; no energy or CPU speed advantage')
        save(d/'SUMMARY.json',summary);failed=False
        print(json.dumps({k:v for k,v in summary.items() if k!='jobs'}),flush=True)
    finally:
        if workers:workers.close()
        log.close()
        if native:native.close(failed=failed)
        else:save(d/'CPU_RECEIPT.json',dict(failed=failed,finish=datetime.datetime.now(datetime.timezone.utc).isoformat()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('stage');p.add_argument('seed',type=int)
    p.add_argument('arm');p.add_argument('--count',type=int,default=24);a=p.parse_args()
    run(a.name,a.stage,a.seed,a.arm,a.count)
