"""Bounded watch-folder image service and staged real-worker experiments."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,sys,time,traceback
from PIL import Image
from artifacts import fixtures,verify,sha
from controller import CPUController
from native_client import NativeClient,save,P
from pool import Workers

AVAILABILITY=['all','all','cpu','cpu','gstreamer','all']
ASSIGNMENTS={71011:dict(decode=['cpu','gstreamer'],export=['cpu','gstreamer']),
             71012:dict(decode=['gstreamer','cpu'],export=['gstreamer','cpu']),
             71013:dict(decode=['cpu','gstreamer'],export=['gstreamer','cpu'])}

def perform_job(src,job_id,d,workers,native,controller,assignments,budget,events,probe_clock=None):
    src=Path(src).resolve()
    with Image.open(src) as im:
        assert im.format=='JPEG';width,height=im.size;assert 1<=width<=2048 and 1<=height<=2048
    jobdir=d/'jobs'/f'{job_id:03}';jobdir.mkdir(parents=True,exist_ok=False)
    state=3;current=src;attempts=[];complete=False;final=None
    request_start=native.request_start() if native else None
    for local in range(min(8,budget)):
        probe=bool(probe_clock and probe_clock(job_id,local,state))
        query=native.query(probe=probe) if native else None
        if native:
            y=query['slots'][-1]['after'][32:64];codes=[i for i,v in enumerate(y[:16]) if v]
            assert len(codes)==1 and 8<=codes[0]<16
            ctx=codes[0]-8;s,action=divmod(ctx,2);assert s==state,'native task state disagrees with actual artifacts'
            choice=dict(state=s,action=action,depth=1 if y[20] else 2 if y[21] else 0)
            native.snapshot()  # Positive protected-query isolation at every action.
        else:choice=controller.choose(state,probe=probe);action=choice['action']
        stage='decode' if state==3 else 'export';backend=assignments[stage][action];target=jobdir/f'operation_{local:02}_{stage}_{backend}.npy'
        req=dict(request_id=f'{job_id}:{local}',stage=stage,backend=backend,input=str(current),output=str(target),width=width,height=height)
        rpc=workers.call(req);check=None;actual=state
        if rpc['result']['ok']:
            check=verify(current,target,stage)
            assert rpc['result']['input_sha256']==sha(current) and rpc['result']['output_sha256']==sha(target)
            actual=(1 if backend=='cpu' else 2) if stage=='decode' else 0
        observation=native.observe(actual) if native else None
        if native:
            y=observation['slots'][0]['after'][32:64];expected_state=3 if actual==0 else actual
            assert y[expected_state]==1 and sum(y[:16])==1 and bool(y[23])==(actual==0)
            native.snapshot()
        else:controller.observe(state,action,actual)
        event=dict(job=job_id,local_attempt=local,state=state,action=action,probe=probe,choice=choice,stage=stage,backend=backend,
                   rpc=rpc,verification=check,outcome=actual,query_id=query['id'] if query else None,
                   observation_id=observation['id'] if observation else None,request_start_id=request_start['id'] if request_start else None)
        attempts.append(event);events.write(json.dumps(dict(kind='operation',**event))+'\n');events.flush()
        if actual==0:complete=True;final=target.with_suffix('.png');break
        if actual!=state:current=target
        state=actual
    manifest=dict(job=job_id,source=str(src),source_sha256=sha(src),complete=complete,attempts=len(attempts),
                  failures=sum(not x['rpc']['result']['ok'] for x in attempts),final_png=str(final) if final else None,
                  final_sha256=sha(final) if final else None,operations=attempts)
    save(jobdir/'MANIFEST.json',manifest)
    return manifest

def run_series(name,stage,seed,arm):
    assert seed in ASSIGNMENTS and arm in ['twohop','onestep','frozen_transition','cpu']
    native=None;workers=None;failed=True;controller=CPUController(twohop=arm!='onestep');images=sorted((P/'inputs'/str(seed)).glob('*.jpg'));assert len(images)==4
    if arm!='cpu':native=NativeClient(name,stage,'onestep' if arm=='onestep' else 'twohop');d=native.d
    else:d=P/'cpu_runs'/name;d.mkdir(parents=True,exist_ok=False)
    jobs=[];remaining=60;start=time.monotonic();log=(d/'events.jsonl').open('w')
    try:
        workers=Workers(d);save(d/'WORKER_OWNERS.json',workers.owners)
        for epoch,available in enumerate(AVAILABILITY):
            workers.set_available(available)
            if arm=='frozen_transition' and epoch==1:
                native.mode(0,1)
            log.write(json.dumps(dict(kind='epoch',epoch=epoch,available=available,operator_PID=__import__('os').getpid()))+'\n');log.flush()
            for k,src in enumerate(images):
                if remaining<=0:break
                assert time.monotonic()-start<400
                manifest=perform_job(src,epoch*4+k,d,workers,native,controller,ASSIGNMENTS[seed],remaining,log)
                manifest['epoch']=epoch;jobs.append(manifest);remaining-=manifest['attempts']
            if remaining<=0:break
        summary=dict(status='EXECUTION_COMPLETE',arm=arm,seed=seed,completed=sum(j['complete'] for j in jobs),offered_requests=24,
                     attempted_requests=len(jobs),operation_attempts=60-remaining,worker_failures=sum(j['failures'] for j in jobs),
                     per_epoch_completed=[sum(j['complete'] for j in jobs if j['epoch']==e) for e in range(6)],
                     per_epoch_attempts=[sum(j['attempts'] for j in jobs if j['epoch']==e) for e in range(6)],
                     host_elapsed_seconds=time.monotonic()-start,claim='Task outcome and operation budget; not hardware throughput or energy comparison',jobs=jobs)
        save(d/'SUMMARY.json',summary);failed=False;print(json.dumps({k:v for k,v in summary.items() if k!='jobs'}),flush=True)
    finally:
        if workers:workers.close()
        log.close()
        if native:native.close(failed=failed)
        else:save(d/'CPU_RECEIPT.json',dict(failed=failed,finish=datetime.datetime.now(datetime.timezone.utc).isoformat()))

def watch(name,inbox,outbox,max_jobs,max_seconds):
    assert 1<=max_jobs<=24 and 1<=max_seconds<=400
    inbox,outbox=Path(inbox).resolve(),Path(outbox).resolve();inbox.mkdir(parents=True,exist_ok=True);outbox.mkdir(parents=True,exist_ok=True)
    native=NativeClient(name,'watch','twohop');d=native.d;workers=None;log=(d/'events.jsonl').open('w');done=set();start=time.monotonic();jobs=[];failed=True
    try:
        workers=Workers(d);save(d/'WORKER_OWNERS.json',workers.owners)
        while len(jobs)<max_jobs and time.monotonic()-start<max_seconds:
            candidates=[p for p in sorted(inbox.iterdir()) if p.suffix.lower() in ['.jpg','.jpeg'] and p not in done]
            if not candidates:time.sleep(.2);continue
            src=candidates[0];stat=src.stat();time.sleep(.1)
            if src.stat().st_size!=stat.st_size:continue
            destination=outbox/(src.stem+'.png');published=outbox/(src.stem+'.manifest.json')
            if destination.exists() or published.exists():
                try:
                    receipt=json.loads(published.read_text());same=(receipt['source_sha256']==sha(src) and receipt['final_sha256']==sha(destination))
                except (OSError,KeyError,ValueError):same=False
                log.write(json.dumps(dict(kind='input_skipped',source=str(src),reason='already_verified_output' if same else 'output_name_conflict'))+'\n');log.flush();done.add(src)
                print('SKIP',src,'already completed' if same else 'output conflict',flush=True);continue
            try:
                with Image.open(src) as image:
                    assert image.format=='JPEG' and max(image.size)<=2048
                    image.load()
            except Exception as exc:
                log.write(json.dumps(dict(kind='input_rejected',source=str(src),reason=str(exc)))+'\n');log.flush();done.add(src)
                print('REJECT',src,flush=True);continue
            result=perform_job(src,len(jobs),d,workers,native,None,ASSIGNMENTS[71011],8,log);jobs.append(result);done.add(src)
            if result['complete']:
                dest=outbox/(src.stem+'.png');assert not dest.exists(),'refuse output overwrite'
                shutil.copy2(result['final_png'],dest);save(outbox/(src.stem+'.manifest.json'),result)
                print('COMPLETE',dest,flush=True)
        save(d/'SUMMARY.json',dict(status='EXECUTION_COMPLETE',mode='watch',completed=sum(j['complete'] for j in jobs),jobs=jobs,max_seconds=max_seconds,max_jobs=max_jobs))
        failed=False
    finally:
        if workers:workers.close()
        log.close();native.close(failed=failed)

if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    s=sub.add_parser('series');s.add_argument('name');s.add_argument('stage');s.add_argument('seed',type=int);s.add_argument('arm')
    w=sub.add_parser('watch');w.add_argument('--name',required=True);w.add_argument('--inbox',required=True);w.add_argument('--outbox',required=True);w.add_argument('--max-jobs',type=int,default=8);w.add_argument('--max-seconds',type=int,default=180)
    args=parser.parse_args()
    if args.command=='series':run_series(args.name,args.stage,args.seed,args.arm)
    else:watch(args.name,args.inbox,args.outbox,args.max_jobs,args.max_seconds)
