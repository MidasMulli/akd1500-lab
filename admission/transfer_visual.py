"""Source-bound continuous 3D replay of native learned-teacher transfer.

Stored edges are measured. Student response bits are returned through the final
readout, teacher per-record responses and input features are derived. Geometry,
glow, floating camera and slowed event timing are presentation, not electrical data.
"""
from pathlib import Path
import argparse,json,math,subprocess,time
import numpy as np
from PIL import Image,ImageDraw
from admission.transfer_archive import BASE,audit,get_run
from admission.transfer import simulate,DIMENSIONS,D,PR
from admission.run import masks,sha
from coprocessor.neuron_visual import text,blend,Scene,BG,INK,MUTED,CYAN,GREEN,AMBER,DIM
W,H,FPS,DURATION=3840,2160,60,30

def source():
    summary=audit();m=json.loads(BASE.with_suffix('.json').read_text());data=dict(archive_sha256=m['archive_sha256'],seed=118101,summary={k:summary[k] for k in ['primary','cpu_final_correct','source_after_new_facts']},arms={},banks=[])
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for arm in ['native','teacher_cut','students_frozen']:
            r,get,fg=get_run(z,m,f'raw02__native_s118101_{arm}_extended');data['arms'][arm]=get('scores').reshape(-1,128).tolist()
            if arm!='native':continue
            a=simulate([fg(f'bank{k}.ram') for k in range(3)],'extended',arm);data['events']=a['events']
            for k,(u,c,plastic,helpers) in enumerate(DIMENSIONS):
                initial=fg(f'bank{k}.ram');states=get(f'states{k}').reshape(-1,len(initial));conn=masks(states,D,u)[:,plastic,:110]
                returned=(get('scores').reshape(-1,128)[:,104:112] if k==1 else get('scores').reshape(-1,128)[:,PR] if k==2 else (a['potentials0'][:,plastic]>31)).astype('u1')
                data['banks'].append(dict(plastic=plastic,initial=masks(initial,D,u)[plastic,:110].tolist(),connections=conn.tolist(),features=a[f'features{k}'][:,:110].tolist(),responses=returned.tolist(),response_provenance='derived from input/weights' if k==0 else 'returned native response bits'))
    data['scope']='Three plastic groups, seven sequential FNP3s; 24 plastic and 232 fixed helper records. Host grounds source with external facts; transfer wire class always zero, no external outcomes. Native teacher prediction drives designed error/valence coding. Teacher-silent exams mask the source plastic cue but still clock its NPU. Stored connection edges and student response bits measured; teacher record responses and intermediate features derived. Camera, geometry, glow and replay timing schematic. CPU also passes; no efficiency or autonomy claim.'
    return data

def prediction(y,teacher=False):
    p=[y[84],y[85]] if teacher else [y[88],any(y[j] for j in PR)]
    return '0 + 1' if all(p) else '0' if p[0] else '1' if p[1] else '—'

class Film:
    def __init__(self,data):
        self.data=data;self.events=data['events'];self.outputs=data['arms']['native'];self.base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(self.base)
        text(d,(110,60),'A learned memory teaches another.',86,INK,True)
        text(d,(115,180),'AKD1500 · native knowledge transfer · independent recall with the teacher answer silent',37,MUTED)
        text(d,(3025,91),'RECORDED CHIP REPLAY',29,CYAN,True)
        text(d,(3025,145),'Three initializations · six cue associations',24,MUTED)
        d.line((115,258,3725,258),fill=DIM,width=2)
        labels=['Role + cue code','Learned teacher','Native error rule','Valence-0 code','Student 0','Valence bridge','Student 1']
        ids=['1,3,0','2,1,0','2,2,0','2,3,0','3,1,0','3,2,0','3,3,0']
        for k,(label,npid) in enumerate(zip(labels,ids)):
            x=115+k*522;d.rounded_rectangle((x,294,x+494,402),radius=14,fill=(15,26,39),outline=GREEN if k in [1,4,6] else DIM,width=2)
            text(d,(x+18,309),label,28,GREEN if k in [1,4,6] else CYAN,True);text(d,(x+18,357),f'FNP3 [{npid}]'+(' · plastic' if k in [1,4,6] else ' · fixed'),25,MUTED)
        for k,(label,npid,helper) in enumerate([('SOURCE TEACHER','2,1,0',56),('STUDENT · OUTCOME 0','3,1,0',56),('STUDENT · OUTCOME 1','3,3,0',120)]):
            x=115+k*870;d.rounded_rectangle((x,458,x+830,1590),radius=24,fill=(10,19,30),outline=DIM,width=2)
            text(d,(x+30,488),label,32,INK,True);text(d,(x+30,543),f'8 plastic + {helper} fixed helper records',26,MUTED)
            text(d,(x+30,1460),'Record responses: '+('derived' if k==0 else 'native readback'),25,CYAN)
            text(d,(x+30,1510),'32 stored binary connections per record',25,MUTED)
        d.rounded_rectangle((2760,458,3725,1590),radius=24,fill=(11,21,33),outline=DIM,width=2)
        text(d,(2800,490),'MEASURED CAUSAL COMPARISON',28,CYAN,True)
        text(d,(115,1638),'Edges: stored connections   + Green: added   − Amber: removed   Cyan: response / active feature',29,MUTED)
        cards=[('NATIVE TRANSFER','288 / 288',GREEN),('TEACHING LINK CUT','0 / 288',AMBER),('STUDENT LEARNING OFF','0 / 288',MUTED),('CPU DICTIONARY CONTROL','288 / 288',CYAN)]
        for k,(label,value,color) in enumerate(cards):
            x=115+k*925;d.rounded_rectangle((x,1710,x+870,1845),radius=19,fill=(14,27,40),outline=DIM,width=2)
            text(d,(x+24,1728),label,26,MUTED,True);text(d,(x+24,1772),value,45,color,True)
        text(d,(115,1885),'Final teacher-silent exams · three initializations × 96 queries · controls retain the same source-RAM trajectory.',29,MUTED)
        text(d,(115,1932),'Host: raw cues, source-grounding facts, fixed role clocks. Transfer: constant class 0; no host-provided outcome or bank choice.',28,MUTED)
        text(d,(115,1979),'Teacher answer path is masked during exams; its NPU still runs. Designed comparator, feature coding and valence split.',28,MUTED)
        text(d,(115,2026),'Measured connections + returned student bits; teacher record activity and intermediate features derived. Geometry/camera/time schematic.',27,MUTED)
        text(d,(115,2073),'Selected slowed replay, not an electrical spike trace. Host-paced; CPU also passes. No energy, speed or autonomy advantage claimed.',27,MUTED)
        j=np.arange(110);yy=1-2*(j+.5)/110;rr=np.sqrt(1-yy*yy);angle=j*math.pi*(3-math.sqrt(5))
        self.points=np.column_stack((-.8+.29*rr*np.cos(angle),1.0*yy,.58*rr*np.sin(angle)))
        self.units=np.array([[.35+(i//4)*.78,.96-(i%4)*.64,.23 if i//4 else -.23] for i in range(8)])
        self.banks=[dict(initial=np.asarray(b['initial']),conn=np.asarray(b['connections']),features=np.asarray(b['features']),responses=np.asarray(b['responses']),plastic=b['plastic']) for b in data['banks']]
        self.query=[];last=0
        for e in self.events:
            if e['role']=='query':last=e['index']
            self.query.append(last)

    def choose(self,t):
        segments=[(0,3,list(range(0,28))),(3,10,list(range(28,52))),(10,13,list(range(52,72))),(13,18,list(range(108,138))),(18,24,list(range(138,174))),(24,30,list(range(174,270)))]
        for lo,hi,ids in segments:
            if t<hi:return ids[min(len(ids)-1,int((t-lo)/(hi-lo)*len(ids)))]
        return 269

    def frame(self,t):
        i=self.choose(t);e=self.events[i];role=e['role'];y=self.outputs[i];im=self.base.copy();d=ImageDraw.Draw(im)
        for k,b in enumerate(self.banks):
            after=b['conn'][i];before=b['conn'][i-1] if i else b['initial'];added=(after==1)&(before==0);removed=(after==0)&(before==1);changed=(added|removed).any(1)
            fp=Scene.project(self,self.points,t,center=(500+k*870,1010),scale=280);up=Scene.project(self,self.units,t,center=(500+k*870,1010),scale=280)
            f=b['features'][i];response=b['responses'][i]
            def edge(p,q,color,width):
                p,q=p[:2],q[:2];curve=np.linspace(0,1,9)[:,None];c1=p+np.array([(q[0]-p[0])*.42,0]);c2=q-np.array([(q[0]-p[0])*.32,0]);v=(1-curve)**3*p+3*(1-curve)**2*curve*c1+3*(1-curve)*curve**2*c2+curve**3*q
                d.line([tuple(vv) for vv in v],fill=color,width=width,joint='curve')
            for j in range(8):
                for bit in np.flatnonzero(before[j]):edge(fp[bit],up[j],blend(BG,CYAN,.34 if f[bit] and response[j] else .14),2)
                for bit in np.flatnonzero(removed[j]):edge(fp[bit],up[j],AMBER,5)
                for bit in np.flatnonzero(added[j]):edge(fp[bit],up[j],GREEN,5)
            for bit in np.argsort(fp[:,2]):
                xx,yy,fac=fp[bit];rr=(4 if f[bit] else 2)*fac;d.ellipse((xx-rr,yy-rr,xx+rr,yy+rr),fill=GREEN if f[bit] and bit==109 else CYAN if f[bit] else DIM)
            for j in range(8):
                xx,yy,fac=up[j];Scene.orb(d,xx,yy,19*fac,GREEN if changed[j] else CYAN if response[j] else DIM,bool(changed[j] or response[j]));text(d,(xx+27,yy-13),f'R{b["plastic"][j]:02}',23,INK if response[j] or changed[j] else MUTED,True)
            text(d,(145+k*870,610),f'+{int(added.sum())} / −{int(removed.sum())} stored edits this event',27,GREEN if added.any() else MUTED)
            active=(k==0 and role=='ground') or (k>0 and role=='transfer');text(d,(145+k*870,1375),'LEARNING ROLE' if active else 'READ ONLY',27,GREEN if active else MUTED,True)
        headings={'ground':'EXTERNAL FACT → SOURCE','query':'QUERY NATIVE MEMORIES','transfer':'NATIVE TEACHER → STUDENTS','exam':'TEACHER SILENT · STUDENTS RECALL'}
        text(d,(2800,560),headings[role],27,GREEN if role=='transfer' else INK,True)
        text(d,(2800,615),f'Cue {e["key"]:04b}   ·   event {i+1} / 270',29,MUTED)
        source=prediction(self.outputs[self.query[i]],True) if role=='transfer' else prediction(y,True)
        for row,(label,value,col) in enumerate([('SOURCE ANSWER',source,CYAN),('STUDENT ANSWER',prediction(y),GREEN),('HOST TRANSFER CLASS','0 (constant)' if role=='transfer' else 'not transferring',MUTED)]):
            yy=700+row*110;text(d,(2800,yy),label,25,MUTED,True);text(d,(3250,yy-4),value,35,col,True)
        if role=='ground':note=['External outcome '+str(e['wire'])+' grounds the source.','Both student learning clocks are off.']
        elif role=='transfer':note=['Source prediction selects native feature coding.','Host supplies no outcome; both students enabled.']
        elif role=='exam':note=['Source class bits read back as 0 / 0.','Students answer from their stored connections.']
        else:note=['Returned source and student answers are read only.','No learning RAM changes on this query.']
        for j,line in enumerate(note):text(d,(2800,1060+j*43),line,25,MUTED)
        text(d,(2800,1182),'SAME EVENT · MATCHED CONTROLS',25,CYAN,True)
        for j,(arm,label) in enumerate([('native','Native transfer'),('teacher_cut','Teaching link cut'),('students_frozen','Student learning off')]):
            yy=1235+j*62;v=self.data['arms'][arm][i];text(d,(2800,yy),label,27,INK if arm=='native' else MUTED,True);text(d,(3500,yy-1),prediction(v),31,GREEN if arm=='native' else AMBER,True)
        ii=[e['index'] for e in self.events if e['role']=='exam' and e['index']<=i];correct=sum(prediction(self.outputs[j])==str(self.events[j]['actual']) for j in ii)
        text(d,(2800,1460),f'This seed: {correct} / {len(ii)} silent exams so far',28,GREEN,True)
        return im

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);data=source();(a.out/'recorded-events.json').write_text(json.dumps(data,separators=(',',':'))+'\n');film=Film(data)
    for label,t in [('grounding',.15),('native-transfer',3.35),('silent-recall',12.8),('new-facts',14.0),('final-recall',28.2)]:film.frame(t).save(a.out/f'{label}.png')
    if a.video:
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','h264_videotoolbox','-b:v','14M','-profile:v','main','-level:v','5.2','-bf','0','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(a.out/'native-teacher-transfer-4k60.mp4')]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);start=time.monotonic()
        try:
            for i in range(FPS*DURATION):
                proc.stdin.write(film.frame(i/FPS).tobytes())
                if i%300==0:print(json.dumps(dict(frame=i,total=1800,elapsed_s=round(time.monotonic()-start,1))),flush=True)
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(a.out/'native-teacher-transfer-4k60.mp4'),'-vf','scale=1920:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','19','-profile:v','main','-level:v','4.2','-refs','1','-bf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out/'native-teacher-transfer-1080p60.mp4')],check=True)
    (a.out/'SOURCE_BINDING.json').write_text(json.dumps(dict(archive_sha256=data['archive_sha256'],events_sha256=sha(a.out/'recorded-events.json'),renderer_sha256=sha(__file__),scope=data['scope'],outputs={p.name:sha(p) for p in a.out.iterdir() if p.suffix in ['.png','.mp4']}),indent=2)+'\n');print(json.dumps(dict(status='RENDERED',out=str(a.out))))
if __name__=='__main__':main()
