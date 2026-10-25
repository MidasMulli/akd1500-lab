"""Source-bound continuous 3D replay: protection and corroborated revision.

Stored edges and returned cohort/student class bits are measured. Per-record
source responses and hidden features are derived; geometry/glow/time schematic.
"""
from pathlib import Path
import argparse,json,math,subprocess,time
import numpy as np
from PIL import ImageDraw
from admission.consensus_archive import BASE,audit,get_run
from admission.consensus_crossed_archive import audit as crossed_audit
from admission.consensus import simulate,DIMENSIONS,D,PR
from admission.run import masks,sha
from admission.transfer_visual import Film as ExistingFilm,prediction
from coprocessor.neuron_visual import text,blend,Scene,BG,INK,MUTED,CYAN,GREEN,AMBER,DIM

def source():
    report=audit();crossed=crossed_audit();m=json.loads(BASE.with_suffix('.json').read_text());data=dict(archive_sha256=m['archive_sha256'],seed=124101,summary=report['primary'],counterbalanced=crossed['endpoints']['protected'],counterbalanced_archive_sha256=crossed['archive_sha256'],arms={},banks=[])
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for arm in ['native','follow_b','students_frozen','teacher_cut']:
            _,get,fg=get_run(z,m,f'native_s124101_{arm}_repeat');y=get('scores').reshape(-1,128);data['arms'][arm]=y.tolist()
            if arm!='native':continue
            a=simulate([fg(f'bank{k}.ram') for k in range(3)],'repeat',arm);data['events']=a['events']
            for k,(u,c,p,h) in enumerate(DIMENSIONS):
                initial=fg(f'bank{k}.ram');states=get(f'states{k}').reshape(-1,len(initial));responses=y[:,104:112] if k==1 else y[:,PR] if k==2 else a['potentials0'][:,p]>31
                data['banks'].append(dict(plastic=p,initial=masks(initial,D,u)[p,:110].tolist(),connections=masks(states,D,u)[:,p,:110].tolist(),features=a[f'features{k}'][:,:110].tolist(),responses=np.asarray(responses,'u1').tolist()))
    data['scope']=report['scope']+' Displayed physical cell: seed124101 repeat. Final cohort class flags are native readback; source per-record response dots are derived. All48 plastic records are shown; helper records and fixed routing are summarized, not drawn as extra plastic neurons. No electrical spike measurement.'
    return data

class Film(ExistingFilm):
    def __init__(self,data):
        view=dict(data,events=[dict(e,role='ground' if e['role']=='ground_negative' else e['role']) for e in data['events']]);super().__init__(view);self.original=data
        d=ImageDraw.Draw(self.base);d.rectangle((95,40,2985,240),fill=BG)
        text(d,(110,60),'A memory network can decline a lesson.',76,INK,True)
        text(d,(115,180),'Disagree → protect stored knowledge. Corroborate → revise native connections.',36,MUTED)
        d.rectangle((3010,135,3805,190),fill=BG);text(d,(3025,145),'3 initializations · 18 corruption / revision cycles',24,MUTED)
        for k,label in [(1,'Sources A + B'),(2,'Native agreement')]:
            x=115+k*522;d.rectangle((x+14,303,x+480,351),fill=(15,26,39));text(d,(x+18,309),label,28,GREEN if k==1 else CYAN,True)
        d.rectangle((135,480,920,591),fill=(10,19,30));text(d,(145,488),'TWO SOURCE COHORTS · ONE NPU',30,INK,True);text(d,(145,543),'32 plastic + 96 fixed helper records',26,MUTED)
        d.rectangle((105,1614,3830,2130),fill=BG)
        text(d,(115,1625),'Actual stored binary connections · Green + added · Amber − removed · Cyan response / active feature',29,MUTED)
        cards=[('PROTECT DURING DISAGREEMENT','native','protected',GREEN),('FOLLOW CORRUPTED SOURCE B','follow_b','protected',AMBER),('ACCEPT CORROBORATED CHANGE','native','updated',GREEN),('STUDENTS FROZEN AFTER TRAINING','students_frozen','updated',AMBER)]
        for k,(label,arm,endpoint,col) in enumerate(cards):
            x=115+k*925;q=data['summary'][arm][endpoint];d.rounded_rectangle((x,1700,x+870,1835),radius=19,fill=(14,27,40),outline=DIM,width=2)
            text(d,(x+24,1718),label,23,MUTED,True);text(d,(x+24,1765),f'{q["correct"]:,} / {q["queries"]:,} recall probes',40,col,True)
        text(d,(115,1866),'Four synthetic cue associations · six cycles per initialization · teacher-silent exams; repeated queries are not independent trials.',27,MUTED)
        text(d,(115,1912),'Host: evidence, raw cues and fixed clocks. Native chip: source responses, agreement/error coding and learned student connection edits.',27,MUTED)
        text(d,(115,1958),'Two sources share one class-partitioned NPU. Both student clocks are enabled during transfer; absence of teaching features protects memory.',27,MUTED)
        text(d,(115,2004),'Measured stored edges + returned class bits; source record dots / hidden features derived. Camera, geometry, glow and slowed replay schematic.',26,MUTED)
        text(d,(115,2050),'CPU agreement also passes. Both sources can agree on a false fact and corrupt memory. No autonomous truth, efficiency or CPU advantage claim.',26,MUTED)
        text(d,(115,2094),'Scope: at most one responding record per source class, verified at every query / transfer. Helpers omitted. Not electrical spikes.',25,MUTED)

    def choose(self,t):
        # Same resident cell throughout. Selected event windows preserve order.
        segments=[(0,3,list(range(64,88))),(3,7,list(range(136,148))),(7,11,list(range(164,188))),(11,14,list(range(188,236))),(14,18,list(range(248,284))),(18,20,list(range(284,300))),(20,26,list(range(332,384))),(26,28,list(range(384,432))),(28,30,list(range(1864,1912)))]
        for lo,hi,ids in segments:
            if t<hi:return ids[min(len(ids)-1,int((t-lo)/(hi-lo)*len(ids)))]
        return 1911

    @staticmethod
    def answer(bits):
        return '0 + 1' if all(bits) else '0' if bits[0] else '1' if bits[1] else '—'

    def frame(self,t):
        i=self.choose(t);e=self.original['events'][i];role=e['role'];phase=e['phase'];y=self.outputs[i];im=self.base.copy();d=ImageDraw.Draw(im);deltas=[]
        for k,b in enumerate(self.banks):
            after=b['conn'][i];before=b['conn'][i-1] if i else b['initial'];add=(after==1)&(before==0);remove=(after==0)&(before==1);changed=(add|remove).any(1);deltas.append((int(add.sum()),int(remove.sum())))
            units=np.array([[.05+(j//8)*.39,.95-(j%8)*.27,(j//8-1.5)*.14] for j in range(32)]) if k==0 else self.units
            fp=Scene.project(self,self.points,t,center=(500+k*870,1010),scale=280);up=Scene.project(self,units,t,center=(500+k*870,1010),scale=280);f=b['features'][i];response=b['responses'][i]
            def edge(p,q,col,width):
                p,q=p[:2],q[:2];v=np.linspace(0,1,7)[:,None];c1=p+[(q[0]-p[0])*.42,0];c2=q-[(q[0]-p[0])*.32,0];v=(1-v)**3*p+3*(1-v)**2*v*c1+3*(1-v)*v**2*c2+v**3*q
                d.line([tuple(vv) for vv in v],fill=col,width=width,joint='curve')
            for j in range(len(b['plastic'])):
                for bit in np.flatnonzero(before[j]):edge(fp[bit],up[j],blend(BG,CYAN,.34 if f[bit] and response[j] else .12),2)
                for bit in np.flatnonzero(remove[j]):edge(fp[bit],up[j],AMBER,5)
                for bit in np.flatnonzero(add[j]):edge(fp[bit],up[j],GREEN,5)
            for bit in np.argsort(fp[:,2]):
                xx,yy,fac=fp[bit];r=(4 if f[bit] else 2)*fac;d.ellipse((xx-r,yy-r,xx+r,yy+r),fill=CYAN if f[bit] else DIM)
            for j,record in enumerate(b['plastic']):
                xx,yy,fac=up[j];Scene.orb(d,xx,yy,(12 if k==0 else 19)*fac,GREEN if changed[j] else CYAN if response[j] else DIM,bool(changed[j] or response[j]));text(d,(xx+17 if k==0 else xx+27,yy-11),f'R{record:02}',16 if k==0 else 23,INK if changed[j] or response[j] else MUTED,True)
            if k==0:
                text(d,(450,675),'A · class 0 / 1',21,CYAN,True);text(d,(685,675),'B · class 0 / 1',21,CYAN,True)
            text(d,(145+k*870,610),f'+{int(add.sum())} / −{int(remove.sum())} stored edits this event',27,GREEN if add.any() else MUTED)
            learning=(k==0 and role in ['ground','ground_negative']) or (k>0 and role=='transfer')
            text(d,(145+k*870,1375),'LEARNING MODE ENABLED' if learning else 'READ ONLY',25,GREEN if learning else MUTED,True)
        # Informative live readback panel, rather than an abstract score counter.
        d.rectangle((2785,545,3710,1575),fill=(11,21,33))
        qi=self.query[i] if role=='transfer' else i;sy=self.outputs[qi];aa=self.answer(sy[86:88]);bb=self.answer([sy[89],sy[90]])
        disagree=aa in ['0','1'] and bb in ['0','1'] and aa!=bb
        title='SOURCES DISAGREE · PROTECT' if disagree else 'CORROBORATED · CAN REVISE' if aa==bb and aa in ['0','1'] else 'SOURCE PATH SILENT' if role=='exam' else 'GROUNDING A SOURCE'
        text(d,(2800,558),title,28,AMBER if disagree else CYAN,True)
        text(d,(2800,615),f'Cue {e["key"]:04b} · event {i+1}/1912',27,MUTED)
        for j,(label,answer) in enumerate([('SOURCE A',aa),('SOURCE B',bb)]):
            x=2800+j*450;d.rounded_rectangle((x,685,x+410,812),radius=12,fill=(16,32,46),outline=AMBER if disagree else DIM,width=2);text(d,(x+20,700),label,24,MUTED,True);text(d,(x+20,746),answer,36,CYAN,True)
        text(d,(2800,840),'Native class flags'+(' · preceding query' if role=='transfer' else ''),23,MUTED)
        recall_i=qi if role=='transfer' else i
        label='STUDENT RECALL · BEFORE UPDATE' if role=='transfer' else 'STUDENT RECALL · READ ONLY'
        answer=prediction(self.outputs[recall_i]) if role in ['query','exam','transfer'] else 'not queried'
        text(d,(2800,900),label,23,MUTED,True);text(d,(3440,890),answer,32 if answer=='not queried' else 44,GREEN if answer==str(e['actual']) else AMBER,True)
        if role=='transfer':
            enabled=any(self.banks[k]['features'][i,:25].any() for k in [1,2]);note='TEACHING FEATURES PRESENT' if enabled else 'NO TEACHING FEATURES · RAM HELD'
        elif role=='ground_negative':note='COMPLEMENT EVIDENCE → SOURCE ONLY'
        elif role=='ground':note=f'EXTERNAL FACT → COHORT {"AB"[e["cohort"]]}'
        elif role=='exam':note='TEACHER SILENT · STORED RECALL'
        else:note='READ-ONLY QUERY'
        text(d,(2800,986),note,24,GREEN if role=='transfer' and not disagree else MUTED,True)
        text(d,(2800,1033),f'Student edits now: +{deltas[1][0]+deltas[2][0]} / −{deltas[1][1]+deltas[2][1]}',27,GREEN if sum(deltas[1])+sum(deltas[2]) else MUTED,True)
        text(d,(2800,1080),f'Evaluation fact: {e["actual"]} · never a transfer label',24,MUTED)
        if phase.startswith('pending_'):text(d,(2800,1124),'One valid report is still uncorroborated: wait.',23,AMBER)
        text(d,(2800,1190),'MATCHED CONTROL · SAME EVENT',25,CYAN,True)
        for j,(arm,label) in enumerate([('native','Native agreement'),('follow_b','Follow B without corroboration'),('students_frozen','Students frozen after training')]):
            yy=1245+j*65;v=prediction(self.original['arms'][arm][recall_i]) if role in ['query','exam','transfer'] else '—';text(d,(2800,yy),label,24,INK if arm=='native' else MUTED,True);text(d,(3505,yy-3),v,32,GREEN if v==str(e['actual']) else AMBER,True)
        text(d,(2800,1470),'SEPARATE SOURCE-SWAP REPLICATION',23,CYAN,True)
        c=self.original['counterbalanced'];text(d,(2800,1515),f'Native {c["correct"]}/{c["queries"]} · CPU always A/B {c["cpu_follow_a_correct"]}/{c["queries"]}',24,MUTED)
        return im

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);data=source();(args.out/'recorded-events.json').write_text(json.dumps(data,separators=(',',':'))+'\n');film=Film(data)
    for name,t in [('bad-teaching',4.2),('protected-memory',8.55),('waiting-for-corroboration',17.25),('native-connection-revision',23.37),('independent-recall',27.0)]:film.frame(t).save(args.out/f'{name}.png')
    if args.video:
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','h264_videotoolbox','-b:v','14M','-profile:v','main','-level:v','5.2','-bf','0','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(args.out/'native-memory-protection-4k60.mp4')]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);start=time.monotonic()
        try:
            for i in range(1800):
                proc.stdin.write(film.frame(i/60).tobytes())
                if i%300==0:print(json.dumps(dict(frame=i,total=1800,elapsed_s=round(time.monotonic()-start,1))),flush=True)
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(args.out/'native-memory-protection-4k60.mp4'),'-vf','scale=1920:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','19','-profile:v','main','-level:v','4.2','-refs','1','-bf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(args.out/'native-memory-protection-1080p60.mp4')],check=True)
    (args.out/'SOURCE_BINDING.json').write_text(json.dumps(dict(archive_sha256=data['archive_sha256'],counterbalanced_archive_sha256=data['counterbalanced_archive_sha256'],events_sha256=sha(args.out/'recorded-events.json'),renderer_sha256=sha(__file__),base_renderer_sha256=sha(Path(__file__).with_name('transfer_visual.py')),scope=data['scope'],outputs={p.name:sha(p) for p in args.out.iterdir() if p.suffix in ['.png','.mp4']}),indent=2)+'\n');print(json.dumps(dict(status='RENDERED',out=str(args.out))))
if __name__=='__main__':main()
