"""Recorded local-SRAM learner responses and edits, with mapped native fabric.

Geometry, camera and playback time are schematic. No invented spike timing.
Only returned FNP3 potentials drive neuron brightness; CNP routes are static.
"""
from pathlib import Path
import argparse,json,math,subprocess,time
import numpy as np
from PIL import Image,ImageDraw
from coprocessor.resident_archive import DEFAULT,audit,mask,sha
from coprocessor.neuron_visual import text,blend,Scene,BG,INK,MUTED,CYAN,GREEN,AMBER,DIM

W,H,FPS,DURATION=3840,2160,60,24

def source_data():
    checked=audit();meta=json.loads(DEFAULT.with_suffix('.json').read_text());cell='c512_s107501';fix=meta['fixtures'][cell]
    with np.load(DEFAULT.with_suffix('.npz'),allow_pickle=False) as z:
        initial=z[fix['arrays']['ram']['key']];features=z[fix['arrays']['features']['key']].reshape(64,128);images=z[fix['arrays']['inputs']['key']].reshape(64,32,32,3)
        get=lambda arm,k:z[meta['runs'][f'native_{cell}_a{arm}_n64_r0']['arrays'][k]['key']]
        states=get(1,'snapshots').reshape(64,1026);frozen=get(0,'snapshots').reshape(64,1026);before=np.concatenate([initial[None],states[:-1]])
        bm,am=mask(before),mask(states);fm=mask(frozen);scores=get(1,'scores').reshape(64,128);fscores=get(0,'scores').reshape(64,128)
        shown=[8,12,16,20,24,28,32,36];units=[int(np.flatnonzero(np.any(bm[i]!=am[i],axis=1))[0]) for i in shown]
        assert len(set(units))==8
        events=[]
        for i in shown+list(range(40,48)):
            added=am[i].astype(int)-bm[i].astype(int);edits=np.flatnonzero(np.any(added!=0,axis=1)).tolist()
            assert np.array_equal(frozen[i],initial)
            events.append(dict(index=i,query=i>=40,input=features[i].tolist(),image=images[i].tolist(),units=units,before=bm[i,units].tolist(),after=am[i,units].tolist(),frozen=fm[i,units].tolist(),response=scores[i].tolist(),frozen_response=fscores[i].tolist(),edited_records=edits,added_connections=int((added==1).sum()),removed_connections=int((added==-1).sum())))
    return dict(seed=107501,units=units,events=events,archive_sha256=meta['archive_sha256'],audit={k:v for k,v in checked.items() if k!='rows'},native_mapping=fix['metadata']['mapping'])

class ResidentScene:
    def __init__(self,data):
        self.data=data;self.base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(self.base)
        text(d,(115,65),'Learning stays on the chip',96,bold=True)
        text(d,(120,195),'Multi-NPU spatial processing → native FNP3 learning → read-only recall',39,MUTED)
        d.line((120,270,3720,270),fill=(35,51,68),width=2)
        text(d,(125,300),'MAPPED NATIVE ROUTE  ·  CONFIGURATION, NOT A LIVE SPIKE TRACE',29,MUTED,True)
        boxes=[(120,370,570,510,'RGB → events','HRC · image input'),(700,345,1300,425,'Spatial CNP 1',''),(700,455,1300,535,'Spatial CNP 2',''),(1450,370,2110,510,'Projection CNP','8 binary channels × 4×4'),(2260,370,2920,510,'FNP3 learner','128 records · local SRAM')]
        for x,y,xx,yy,title,sub in boxes:
            d.rounded_rectangle((x,y,xx,yy),radius=20,fill=(13,24,36),outline=(48,75,94),width=2);text(d,(x+25,y+16),title,33,CYAN,True)
            if sub:text(d,(x+25,y+79),sub,27,MUTED)
        for p,q in [((570,440),(700,385)),((570,440),(700,495)),((1300,385),(1450,440)),((1300,495),(1450,440)),((2110,440),(2260,440))]:Scene.edge(d,p,q,DIM,4)
        text(d,(3100,370),'512 spatial filters',35,INK,True);text(d,(3100,430),'3 CNPs + 1 FNP3',32,CYAN)
        text(d,(3100,485),'One native hardware pass',28,MUTED)
        d.rounded_rectangle((110,590,2570,1770),radius=32,fill=(11,20,31),outline=(34,56,73),width=2)
        d.rounded_rectangle((2630,590,3730,1770),radius=32,fill=(15,21,29),outline=(59,51,40),width=2)
        text(d,(160,625),'NATIVE LEARNING  ·  LOCAL SRAM',38,CYAN,True);text(d,(2690,625),'MATCHED FROZEN CHIP',35,AMBER,True)
        text(d,(160,695),'128 feature bits · 8 sampled learner records · 32 connections each',31,MUTED)
        text(d,(2690,695),'Same input · teaching disabled',31,MUTED)
        text(d,(120,1930),'Measured: returned match counts + complete connection-memory readbacks.',32,MUTED)
        text(d,(120,1987),'Derived: input feature bits from independently verified integer convolution.',30,MUTED)
        text(d,(120,2042),'Schematic: positions, cables, camera and expanded playback timing. No electrical spike times observed.',29,MUTED)
        text(d,(120,2094),'Three fresh initializations verified · CPU controls matched exactly · no speed or energy advantage established',29,MUTED)
        j=np.arange(128);yy=1-2*(j+.5)/128;rr=np.sqrt(1-yy*yy);theta=j*math.pi*(3-math.sqrt(5))
        self.features=np.column_stack((-1.15+.27*rr*np.cos(theta),yy*1.12,.85*rr*np.sin(theta)))
        self.units=np.array([[.45+(i%2)*.92,.93-(i//2)*.62,.22 if data['units'][i]>=64 else -.22] for i in range(8)])
    def project(self,p,t):return Scene.project(self,p,t,center=(1245,1190),scale=580)
    def frame(self,t):
        j=min(7,int(t/2)) if t<16 else 8+min(7,int(t-16));e=self.data['events'][j];fraction=(t%2)/2 if t<16 else t%1;post=e['query'] or fraction>=.48
        im=self.base.copy();d=ImageDraw.Draw(im);fp=self.project(self.features,t);up=self.project(self.units,t);bits=np.array(e['input']);masks=np.array(e['after' if post else 'before'])
        text(d,(160,755),'Query only · learned memory holds' if e['query'] else ('Connection readback · native edit' if post else 'Teaching cue · returned response'),37,GREEN if e['query'] or post else INK,True)
        for k in range(8):
            selected=self.data['units'][k] in e['edited_records']
            for bit in np.flatnonzero(masks[k]):
                added=e['after'][k][bit] and not e['before'][k][bit]
                color=GREEN if post and added else CYAN if bits[bit] else AMBER
                Scene.edge(d,fp[bit],up[k],color if selected else blend(BG,color,.26),5 if selected else 2)
                if selected and post and added:Scene.edge(d,fp[bit],up[k],blend(BG,GREEN,.15),17);Scene.edge(d,fp[bit],up[k],GREEN,5)
        for k in np.argsort(fp[:,2]):
            x,y,fac=fp[k];r=(7 if bits[k] else 3.5)*fac;d.ellipse((x-r,y-r,x+r,y+r),fill=CYAN if bits[k] else DIM)
        for k in np.argsort(up[:,2]):
            unit=self.data['units'][k];score=e['response'][unit];x,y,fac=up[k];color=blend(DIM,CYAN,(score/32)**.6);hot=unit in e['edited_records']
            Scene.orb(d,x,y,(17+score*.65)*fac,GREEN if hot and post else color,hot);text(d,(x+54,y-25),f'R{unit}  ·  {score}/32',29,INK,hot)
        text(d,(260,1575),'DERIVED FEATURE INPUTS',28,MUTED);text(d,(1580,1575),'MEASURED MATCH COUNTS',28,MUTED)
        if e['query']:caption='0 connection edits · teaching disabled · exact full RAM retained'
        else:caption=f'{e["added_connections"]} connections added + {e["removed_connections"]} removed · '+('verified after teaching' if post else 'before native edit')
        text(d,(170,1685),caption,34,GREEN if post else MUTED,True)
        text(d,(2690,765),'MEASURED FROZEN RESPONSES',28,MUTED,True)
        # All128 returned potentials, not a decorative unobserved neuron cloud.
        for unit,score in enumerate(e['frozen_response']):
            x=2720+(unit%16)*61;y=835+(unit//16)*58;r=7+score*.24;col=blend(DIM,AMBER,(score/32)**.7)
            d.ellipse((x-r,y-r,x+r,y+r),fill=col)
        text(d,(2690,1315),'Sampled records · same 0–32 scale',28,MUTED)
        for k,unit in enumerate(self.data['units']):
            y=1370+k*36;fscore=e['frozen_response'][unit];nscore=e['response'][unit]
            text(d,(2690,y),f'R{unit}',25,MUTED);d.rectangle((2835,y+9,3459,y+23),fill=(28,41,53));d.rectangle((2835,y+9,2835+fscore*19.5,y+23),fill=AMBER)
            xx=2835+nscore*19.5;d.line((xx,y+2,xx,y+31),fill=GREEN,width=4);text(d,(3500,y),f'{fscore} / {nscore}',25,INK)
        text(d,(2690,1690),'Frozen RAM: zero edits',33,AMBER,True)
        text(d,(125,1820),f'RECORDED EVENT {e["index"]+1:02d}/64',34,CYAN,True);text(d,(850,1820),'READ-ONLY RECALL' if e['query'] else 'TEACHING · NATIVE CONNECTION CHANGES',34,INK,True)
        text(d,(2690,1820),'AMBER frozen  /  GREEN learned',27,MUTED)
        d.rectangle((120,1895,3720,1901),fill=DIM);d.rectangle((120,1895,120+round(3600*t/24),1901),fill=CYAN)
        return im

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    data=source_data();(a.out/'recorded-events.json').write_text(json.dumps(data,separators=(',',':'))+'\n');scene=ResidentScene(data)
    for name,t in [('teaching',.5),('edit',1.5),('later-edit',9.5),('recall',18.5),('final',23.9)]:scene.frame(t).save(a.out/(name+'.png'))
    if a.video:
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','h264_videotoolbox','-b:v','26M','-profile:v','high','-pix_fmt','yuv420p','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(a.out/'akd-resident-learning-4k60.mp4')]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);start=time.monotonic()
        try:
            for i in range(FPS*DURATION):
                proc.stdin.write(scene.frame(i/FPS).tobytes())
                if i%180==0:print(json.dumps(dict(frame=i,total=1440,elapsed=round(time.monotonic()-start,1))),flush=True)
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(a.out/'akd-resident-learning-4k60.mp4'),'-vf','scale=1920:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out/'akd-resident-learning-1080p60.mp4')],check=True)
    binding=dict(status='RECORDED_LOCAL_SRAM_REPLAY',archive_sha256=data['archive_sha256'],events_sha256=sha(a.out/'recorded-events.json'),renderer_sha256=sha(__file__),helper_renderer_sha256=sha(Path(__file__).with_name('neuron_visual.py')),shown_seed=107501,shown_teaching_events=[9,13,17,21,25,29,33,37],shown_query_events=list(range(41,49)),sampled_records=data['units'],full_frozen_responses=128,internal_spike_timing_observed=False,connection_positions_schematic=True,scope='3 fixed CNPs plus1 FNP3 learner and HRC; native local connection memory, no registered Nano DDR weights',outputs={q.name:sha(q) for q in a.out.iterdir() if q.suffix in ['.png','.mp4']})
    (a.out/'SOURCE_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n');print(json.dumps(dict(status='RENDERED',out=str(a.out))))
if __name__=='__main__':main()
