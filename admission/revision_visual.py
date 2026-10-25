"""Reuse the qualified connection-replay scene for measured revision cycles."""
from pathlib import Path
import argparse,json,subprocess,time
import numpy as np
from PIL import ImageDraw
from admission.revision_archive import BASE,audit,get_run
from admission.revision import simulate,DIMENSIONS,D,PR
from admission.run import masks,sha
from admission.transfer_visual import Film as TransferFilm,prediction
from coprocessor.neuron_visual import text,BG,INK,MUTED,CYAN,GREEN,AMBER,DIM
FPS=60;DURATION=30

def source():
    r=audit();m=json.loads(BASE.with_suffix('.json').read_text());data=dict(archive_sha256=m['archive_sha256'],seed=122101,summary=r['primary'],cpu=r['cpu_correct'],arms={},banks=[])
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for arm in ['native','students_frozen','teacher_cut','no_source_retire']:
            _,get,fg=get_run(z,m,f'native_s122101_{arm}_repeat');y=get('scores').reshape(-1,128);data['arms'][arm]=y.tolist()
            if arm!='native':continue
            a=simulate([fg(f'bank{k}.ram') for k in range(3)],'repeat',arm);data['events']=a['events']
            for k,(u,c,plastic,helpers) in enumerate(DIMENSIONS):
                initial=fg(f'bank{k}.ram');states=get(f'states{k}').reshape(-1,len(initial));responses=(y[:,104:112] if k==1 else y[:,PR] if k==2 else (a['potentials0'][:,plastic]>31)).astype('u1')
                data['banks'].append(dict(plastic=plastic,initial=masks(initial,D,u)[plastic,:110].tolist(),connections=masks(states,D,u)[:,plastic,:110].tolist(),features=a[f'features{k}'][:,:110].tolist(),responses=responses.tolist()))
    data['scope']='Recorded readback of three plastic groups across seven sequential FNP3s, four cue associations, twelve engineered external-rule reversals. Host supplies actual/complement source labels and fixed role clocks; student transfers have constant class0 and no external outcome. Teacher-silent exams mask the source answer path; its NPU still runs. Stored edges and final student responses measured; source per-record activity/intermediate features derived. Camera/geometry/glow/time schematic. CPU also passes; no efficiency/autonomy claim.'
    return data

class Film(TransferFilm):
    def __init__(self,data):
        self.original=data
        view=dict(data,events=[dict(e,role='ground' if e['role']=='ground_negative' else e['role']) for e in data['events']]);super().__init__(view)
        d=ImageDraw.Draw(self.base);d.rectangle((95,45,2980,235),fill=BG);text(d,(110,60),'Correct the teacher. Revise the network.',80,INK,True)
        text(d,(115,180),'Native source correction → constant-label transfer → independent student recall',36,MUTED)
        d.rectangle((3015,140,3800,185),fill=BG);text(d,(3025,145),'Three initializations · 12 rule reversals',25,MUTED)
        d.rectangle((105,1700,3830,1855),fill=BG)
        cards=[('NATIVE REVISION','native',GREEN),('STUDENTS FROZEN AFTER TRAINING','students_frozen',AMBER),('TEACHING LINK CUT AFTER TRAINING','teacher_cut',MUTED),('CPU DICTIONARY CONTROL',None,CYAN)]
        for k,(label,arm,col) in enumerate(cards):
            x=115+k*925;d.rounded_rectangle((x,1710,x+870,1845),radius=19,fill=(14,27,40),outline=DIM,width=2);text(d,(x+24,1728),label,24,MUTED,True)
            value=data['summary'][arm]['correct'] if arm else data['cpu'];text(d,(x+24,1772),f'{value:,} / 3,456',44,col,True)
        d.rectangle((105,1870,3830,1930),fill=BG);text(d,(115,1885),'Post-transfer teacher-silent probes · 3 seeds × 12 reversals × 96 queries · four cue associations; keys 2/3 stay fixed.',28,MUTED)

    def choose(self,t):
        segments=[(0,3,list(range(72,84))),(3,7,list(range(108,116))),(7,10,list(range(116,132))),(10,13,list(range(132,148))),(13,20,list(range(148,172))),(20,23,list(range(172,196))),(23,27,list(range(268,276))+list(range(308,316))),(27,30,list(range(1932,2028)))]
        for lo,hi,ids in segments:
            if t<hi:return ids[min(len(ids)-1,int((t-lo)/(hi-lo)*len(ids)))]
        return 2027

    def frame(self,t):
        im=super().frame(t);d=ImageDraw.Draw(im);i=self.choose(t);e=self.original['events'][i];role=e['role'];phase=e['phase'];tail=phase.rsplit('_',1)[-1];cycle=int(tail) if tail.isdigit() else 0;y=self.outputs[i]
        d.rectangle((2790,550,3710,670),fill=(11,21,33))
        label='SOURCE LEARNS CHANGED FACT' if role=='ground' else 'RETIRE OLD SOURCE ASSOCIATION' if role=='ground_negative' else 'NATIVE TEACHER REVISES STUDENTS' if role=='transfer' else 'STUDENTS RECALL WITHOUT TEACHER' if role=='exam' else 'QUERY NATIVE SOURCE + STUDENTS'
        if phase.startswith('before_transfer_'):label='BEFORE TRANSFER · OLD ANSWER'
        text(d,(2800,560),label,25,GREEN if role=='transfer' else INK,True);text(d,(2800,615),f'Cue {e["key"]:04b} · reversal {cycle}/12 · event {i+1}/2028',26,MUTED)
        d.rectangle((3240,800,3705,858),fill=(11,21,33));answer=prediction(y);text(d,(3250,806),answer,35,GREEN if answer==str(e['actual']) else AMBER,True)
        text(d,(2800,1000),f'Expected fact: {e["actual"]}  (evaluation only on queries)',25,MUTED)
        if role=='ground_negative':
            d.rectangle((2790,1040,3710,1150),fill=(11,21,33));text(d,(2800,1060),'Complement evidence retires the obsolete response.',25,MUTED);text(d,(2800,1103),'Students are off; native source competition selects it.',24,MUTED)
        d.rectangle((2790,1170,3710,1575),fill=(11,21,33));text(d,(2800,1182),'MATCHED TRAINED CONTROLS · SAME EVENT',25,CYAN,True)
        for j,(arm,label) in enumerate([('native','Native revision'),('students_frozen','Students frozen after training'),('teacher_cut','Teaching link cut after training'),('no_source_retire','Source retirement disabled')]):
            yy=1235+j*62;v=prediction(self.original['arms'][arm][i]);text(d,(2800,yy),label,25,INK if arm=='native' else MUTED,True);text(d,(3500,yy-1),v,31,GREEN if v==str(e['actual']) else AMBER,True)
        ii=[e['index'] for e in self.original['events'] if e['phase'].startswith('exam_') and e['index']<=i];correct=sum(prediction(self.outputs[j])==str(self.original['events'][j]['actual']) for j in ii)
        text(d,(2800,1515),f'Post-transfer so far: {correct} / {len(ii)} correct',27,GREEN,True)
        return im

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);data=source();(a.out/'recorded-events.json').write_text(json.dumps(data,separators=(',',':'))+'\n');film=Film(data)
    for label,t in [('source-retirement',3.8),('old-student-answer',11.2),('native-revision',13.35),('corrected-student',22.1),('repeated-revision',29.6)]:film.frame(t).save(a.out/f'{label}.png')
    if a.video:
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','h264_videotoolbox','-b:v','13M','-profile:v','main','-level:v','5.2','-bf','0','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(a.out/'native-memory-revision-4k60.mp4')]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);start=time.monotonic()
        try:
            for i in range(1800):
                proc.stdin.write(film.frame(i/60).tobytes())
                if i%300==0:print(json.dumps(dict(frame=i,total=1800,elapsed_s=round(time.monotonic()-start,1))),flush=True)
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(a.out/'native-memory-revision-4k60.mp4'),'-vf','scale=1920:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','19','-profile:v','main','-level:v','4.2','-refs','1','-bf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out/'native-memory-revision-1080p60.mp4')],check=True)
    (a.out/'SOURCE_BINDING.json').write_text(json.dumps(dict(archive_sha256=data['archive_sha256'],events_sha256=sha(a.out/'recorded-events.json'),renderer_sha256=sha(__file__),base_renderer_sha256=sha(Path(__file__).with_name('transfer_visual.py')),scope=data['scope'],outputs={p.name:sha(p) for p in a.out.iterdir() if p.suffix in ['.png','.mp4']}),indent=2)+'\n');print(json.dumps(dict(status='RENDERED',out=str(a.out))))
if __name__=='__main__':main()
