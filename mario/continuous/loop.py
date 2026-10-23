"""Repeated native decisions, actual short outcomes, and unbroken timed gameplay."""
from pathlib import Path
import argparse,hashlib,json,select,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'mario/sealed'))
from mario.sealed.bank import encode,save,sha
from mario.continuous.environment import opened,observation,button,contexts
from mario.continuous.model import Reference
from live import Native as PreviousNative


class Native(PreviousNative):
    def __init__(self,study,name,fixture):
        remote='/home/nicklomeli/akd1500-lab'
        command=f'cd {remote} && timeout --signal=INT --kill-after=5 178 /home/nicklomeli/akida-cc-venv/bin/python mario/continuous/serve.py studies/{study.name} {name} {fixture} --forwards 4096'
        self.process=subprocess.Popen(['ssh','-T','jetson',command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.id=0;self.ready=self.read();assert self.ready['kind']=='ready'


def run(study,fixture,output,split,native_name=None,wall=False,constant=None):
    output.mkdir(parents=True,exist_ok=False);meta=json.loads(fixture.with_suffix('.json').read_text());frozen=meta['arm']=='frozen'
    x=np.fromfile(fixture.with_suffix('.inputs'),'u1').reshape(-1,1024);y=np.fromfile(fixture.with_suffix('.labels'),'<u4')
    machine=Native(study,native_name,str(fixture.relative_to(ROOT))) if native_name else Reference(np.fromfile(fixture.with_suffix('.ram'),'<u4'))
    log=(output/'events.jsonl').open('w');frames=[];times=[];events=[];episodes=[];origin=time.monotonic_ns();count=0
    def record(row):
        events.append(row);log.write(json.dumps(row)+'\n');log.flush()
    def forward(v,label,phase,frame_id=None):
        nonlocal count
        if native_name:reply=machine.forward(v,int(label))
        else:
            before=machine.ram[:];start=time.monotonic_ns();scores=machine.forward(v,int(label));changed=[j for j in range(192) if before[2+36*j:2+36*(j+1)]!=machine.ram[2+36*j:2+36*(j+1)]]
            edits=sum((a^b).bit_count() for j in changed for a,b in zip(before[2+36*j+4:2+36*(j+1)],machine.ram[2+36*j+4:2+36*(j+1)]))
            reply=dict(id=count+1,index=count,label=int(label),scores=scores,changed_records=changed,connection_bit_edits=edits,
                       sent_monotonic_ns=start,received_monotonic_ns=time.monotonic_ns(),input_sha256=hashlib.sha256(v.tobytes()).hexdigest(),model_only=True)
        record(dict(kind='forward',phase=phase,input_frame=frame_id,response=reply));count+=1;return reply
    try:
        for v,label in zip(x,y):forward(v,label,'preparation')
        origin=time.monotonic_ns();record(dict(kind='game_start',mono_ns=origin,physical=bool(native_name)))
        for episode,c in enumerate(contexts(split)):
            env,history,sky,info,valid=opened(c)
            try:
                if not valid:
                    episodes.append(dict(context=c['id'],valid=False));continue
                elapsed=c['wait']+c['start'];start_x=int(info['x_pos']);controlled=0;decisions=0;dead=flag=False
                target=time.monotonic()
                def sample(phase,button_value=None):
                    nonlocal target
                    frame_id=len(frames);frames.append(history[-1].copy());times.append(time.monotonic_ns()-origin)
                    record(dict(kind='frame',frame_id=frame_id,episode=episode,context=c['id'],elapsed=elapsed,controlled=controlled,phase=phase,button=button_value,mono_ns=origin+times[-1]))
                    if wall:
                        target+=1/60;time.sleep(max(0,target-time.monotonic()))
                    return frame_id
                frame_id=sample('decision_observation')
                while controlled<1200 and not dead and not flag:
                    obs=observation(history,sky);inputs=[encode(**obs,action=a,encoder='projection',elapsed=elapsed) for a in [0,1]]
                    replies=[forward(v,2,'action_selection',frame_id) for v in inputs]
                    margins=[max(r['scores'][64:128])-max(r['scores'][:64]) for r in replies]
                    action=int(margins[1]>margins[0]) if constant is None else constant
                    record(dict(kind='decision',episode=episode,context=c['id'],input_frame=frame_id,elapsed=elapsed,controlled=controlled,action=action,margins=margins,native_ids=[r['id'] for r in replies],mono_ns=time.monotonic_ns(),constant_override=constant))
                    previous_x=int(info['x_pos']);source_frame=frame_id
                    for k in range(min(24,1200-controlled)):
                        b=button(action,k);frame,_,terminated,truncated,info=env.step(b);history.append(frame.copy());elapsed+=1;controlled+=1
                        flag=bool(info.get('flag_get'));dead=bool(info.get('death') or ((terminated or truncated) and not flag))
                        frame_id=sample('controlled_gameplay',b)
                        if dead or flag:break
                    advance=int(info['x_pos'])-previous_x;success=bool(not dead and advance>=12)
                    feedback=forward(inputs[action],2 if frozen else int(success),'observed_feedback',source_frame)
                    record(dict(kind='outcome',episode=episode,context=c['id'],input_frame=source_frame,end_frame=frame_id,action=action,success=success,advance_evaluation_only=advance,dead=dead,flag=flag,response_id=feedback['id'],mono_ns=time.monotonic_ns()))
                    decisions+=1
                row=dict(context=c['id'],valid=True,controlled_frames=controlled,decisions=decisions,advance_evaluation_only=int(info['x_pos'])-start_x,dead=dead,flag=flag,alive_at_cap=bool(not dead and not flag and controlled==1200))
                episodes.append(row);record(dict(kind='episode_end',episode=episode,**row,mono_ns=time.monotonic_ns()))
            finally:env.close()
        cleanup=machine.close() if native_name else dict(model_only=True)
        np.savez_compressed(output/'game.npz',frames=np.array(frames,np.uint8),mono_offsets_ns=np.array(times,np.int64))
        report=dict(status='CONTINUOUS_CAPTURE_PASS',physical=bool(native_name),arm=meta['arm'],seed=meta['seed'],split=split,episodes=episodes,forwards=count,
                    frames=len(frames),duration_seconds=(times[-1]-times[0])/1e9,game_sha256=sha(output/'game.npz'),events_sha256=sha(output/'events.jsonl'),
                    source_sha256=sha(__file__),cleanup=cleanup,training_presentations=len(x),constant_override=constant,
                    total_advance=sum(e.get('advance_evaluation_only',0) for e in episodes),surviving_episodes=sum(e.get('alive_at_cap',False) or e.get('flag',False) for e in episodes))
        save(output/'REPORT.json',report);print(json.dumps(report))
    except BaseException as e:
        save(output/'FAILURE.json',dict(error=repr(e),forwards=count,frames=len(frames)))
        if frames:np.savez_compressed(output/'partial-game.npz',frames=np.array(frames,np.uint8),mono_offsets_ns=np.array(times,np.int64))
        raise
    finally:
        log.close()
        if native_name and machine.process.poll() is None:
            try:machine.close()
            except Exception:machine.process.terminate();machine.process.wait(timeout=10)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('fixture',type=Path);p.add_argument('output',type=Path);p.add_argument('--split',choices=['dev','confirm'],default='dev');p.add_argument('--native-name');p.add_argument('--wall',action='store_true');p.add_argument('--constant',type=int,choices=[0,1]);a=p.parse_args();run(a.study.resolve(),a.fixture.resolve(),a.output.resolve(),a.split,a.native_name,a.wall,a.constant)
