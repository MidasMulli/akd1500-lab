"""Measured bounded-feedback replay. Geometry and expanded timing schematic."""
from pathlib import Path
import argparse,json,math,subprocess,time
import numpy as np
from PIL import Image,ImageDraw
from recurrence.archive import BASE,audit,masks,sha
from coprocessor.neuron_visual import text,blend,Scene,BG,INK,MUTED,CYAN,GREEN,AMBER,DIM
W,H=3840,2160
PALETTE=[CYAN,GREEN,AMBER,(174,163,241)]
def source():
    checked=audit();meta=json.loads(BASE.with_suffix('.json').read_text());seed=109101;f=meta['fixtures'][f's{seed}'];data=dict(seed=seed,archive_sha256=meta['archive_sha256'],audit={k:v for k,v in checked.items() if k!='rows'},mapping=f['metadata']['mapping'],units=[0,1,32,33,64,65,96,97])
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        fg=lambda k:z[f['arrays'][k]['key']];data['seed_input']=fg('seed.input').tolist();data['initial_connections']=masks(fg('initial.ram')).tolist();data['probes']=fg('probes.inputs').reshape(4,128).tolist();data['arms']={}
        for arm in ['native','frozen','repeated_seed','severed']:
            r=meta['runs'][f'native_s{seed}_{arm}_h32'];get=lambda k:z[r['arrays'][k]['key']]
            data['arms'][arm]=dict(outputs=get('scores').reshape(32,128).tolist(),probe_outputs=get('probe_scores').reshape(4,128).tolist(),connections=masks(get('ram')).tolist(),quiet=get('quiet').tolist(),idle=get('idle').tolist(),descriptors=get('descriptors').reshape(32,16).tolist())
        data['prefixes']={}
        for h in [1,2,3,4,5,8,16,32]:
            key=f'native_s{seed}_native_h{h}';r=meta['runs'][key];a=z[r['arrays']['ram']['key']]
            data['prefixes'][str(h)]=dict(connections=masks(a).tolist(),source_run=key)
    data['scope']='Returned binary outputs are measured in one 32-step run per arm; connection snapshots come from independent prefix runs. Relay identity and fixed SRAM are verified. Camera, geometry and expanded timing are schematic; no electrical spike timestamps.'
    return data

class Film:
    def __init__(self,data):
        self.data=data;self.base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(self.base)
        text(d,(115,65),'One seed. Thirty-two native steps.',94,INK,True)
        text(d,(120,193),'AKD1500 outputs become its next inputs in device SRAM',43,MUTED)
        d.line((120,275,3720,275),fill=DIM,width=2)
        d.rounded_rectangle((120,325,2460,495),radius=23,fill=(13,25,36),outline=(45,75,94),width=2)
        text(d,(160,352),'ONE PLASTIC NPU  →  FIXED RELAY NPU  →  DEVICE SRAM  →  NEXT INPUT',36,CYAN,True)
        text(d,(160,422),'Initial model + finite schedule prepared by Nano · no per-step host copying',32,MUTED)
        d.rounded_rectangle((2520,325,3720,495),radius=23,fill=(13,25,36),outline=(45,75,94),width=2)
        text(d,(2570,352),'RECORDED CONTROL WORKER',33,CYAN,True)
        d.rounded_rectangle((120,545,2460,1750),radius=32,fill=(11,20,31),outline=(36,59,77),width=2)
        d.rounded_rectangle((2520,545,3720,1750),radius=32,fill=(12,21,31),outline=(36,59,77),width=2)
        text(d,(160,585),'CONNECTION MEMORY + RETURNED ACTIVATIONS',36,CYAN,True)
        text(d,(160,650),'128 state bits · 8 sampled learning records of 128 · fixed relay',31,MUTED)
        text(d,(205,735),'STATE INPUT',28,MUTED,True);text(d,(1160,735),'PLASTIC FNP3',28,MUTED,True);text(d,(1930,735),'FIXED FNP3',28,MUTED,True)
        text(d,(2570,585),'MEASURED OUTPUT · ALL 128 CHANNELS',33,CYAN,True)
        text(d,(2570,1110),'MATCHED CONTROLS · SAME INITIAL STATE',31,INK,True)
        text(d,(120,1885),'Measured: returned binary outputs, descriptor addresses and complete learning-SRAM readbacks.',31,MUTED)
        text(d,(120,1940),'Connection changes: separate prefix runs. Same seed and exact trajectory; not simultaneous internal probes.',29,MUTED)
        text(d,(120,1995),'Motion, layout and playback timing are schematic. No electrical spike timings or physical neuron positions.',29,MUTED)
        text(d,(120,2050),'Finite hardware queue · programmed transition ring · CPU reproduces it · no speed or energy advantage claimed.',29,MUTED)
        text(d,(120,2100),'22 physical sessions · 518 native forwards · 3 initializations · independently audited outputs and full SRAM',28,MUTED)
        p=[]
        for g in range(4):
            for j in range(32):
                yy=1-2*(j+.5)/32;r=math.sqrt(1-yy*yy);a=j*math.pi*(3-math.sqrt(5));p.append([-1.9+.18*r*math.cos(a),.94-.63*g+.2*yy,.26*r*math.sin(a)])
        self.features=np.array(p);self.neurons=np.array([[.04+.62*(i%2),.94-.63*(i//2)+.12*(-1)**i,-.1+.2*(i%2)] for i in range(8)]);self.relay=self.neurons+np.array([1.35,0,0])
    def project(self,p,t):
        q=Scene.project(self,p,t,center=(1270,1175),scale=450);q[:,1]=1175+(q[:,1]-1175)*.75;return q
    def frame(self,t):
        idle=17<=t<20;query=t>=20;q=min(3,int(t-20)) if query else None
        step=min(3,int(t/3)) if t<12 else min(31,4+int((t-12)*28/5)) if t<17 else 31
        a=self.data['arms']['native'];x=np.array(self.data['probes'][q] if query else self.data['seed_input'] if step==0 else a['outputs'][step-1]);y=np.array(a['probe_outputs'][q] if query else a['outputs'][step])
        h=32 if query or idle else max(k for k in [1,2,3,4,5,8,16,32] if k<=step+1)
        before=self.data['initial_connections'] if h==1 else self.data['prefixes'][str(max(k for k in [1,2,3,4,5,8,16,32] if k<h))]['connections'];after=self.data['prefixes'][str(h)]['connections']
        show_after=query or idle or t>=12 or t%3>=1.45;m=np.array(after if show_after else before);old=np.array(self.data['initial_connections']);diff=m-old
        im=self.base.copy();d=ImageDraw.Draw(im)
        status='Host resumes · partial-cue recall' if query else 'Queue empty · later quiet observation' if idle else 'Worker sleeping · no driver I/O after release'
        text(d,(2570,422),status,30,GREEN if query else AMBER if idle else CYAN,True)
        caption=f'Query {q+1}/4 · missing half of a state cue' if query else 'Schedule exhausted · last output stored' if idle else f'Step {step+1:02d}/32 · '+('one external seed' if step==0 else 'previous output supplies this input')
        text(d,(160,800),caption,37,GREEN if query else INK,True)
        p,n,r=self.project(self.features,t),self.project(self.neurons,t),self.project(self.relay,t)
        for k,unit in enumerate(self.data['units']):
            for b in np.flatnonzero(m[unit]):
                added=diff[unit,b]>0;hot=bool(x[b]) and not idle;col=GREEN if added else PALETTE[b//32]
                Scene.edge(d,p[b],n[k],col if hot else blend(BG,col,.23),4 if hot else 2)
            Scene.edge(d,n[k],r[k],CYAN if y[unit] and not idle else DIM,5 if y[unit] and not idle else 2)
        for b in np.argsort(p[:,2]):
            xx,yy,fac=p[b];rad=(7.3 if x[b] and not idle else 3.7)*fac;d.ellipse((xx-rad,yy-rad,xx+rad,yy+rad),fill=PALETTE[b//32] if x[b] and not idle else DIM)
        for k,unit in enumerate(self.data['units']):
            for pos,is_relay in [(n,False),(r,True)]:
                xx,yy,fac=pos[k];lit=bool(y[unit]) and not idle;Scene.orb(d,xx,yy,19*fac,GREEN if lit and diff[unit].any() and not is_relay else CYAN if lit else DIM,lit)
                text(d,(xx-25,yy+37),('F' if is_relay else 'R')+str(unit),27,INK if lit else MUTED,lit)
        for g in range(4):
            yy=float(p[g*32:(g+1)*32,1].mean());d.rounded_rectangle((175,yy-23,248,yy+25),radius=7,fill=(11,20,31));text(d,(190,yy-15),f'S{g}',28,PALETTE[g],True)
        # Explicit transport loop through scratch SRAM, not a fictional direct
        # recurrent wire in the neuron fabric. Motion only highlights topology.
        path=[(2210,1490),(2210,1570),(470,1570),(470,1490)]
        for i in range(3):d.line((path[i],path[i+1]),fill=DIM if idle or query else CYAN,width=4)
        d.polygon([(458,1504),(482,1504),(470,1488)],fill=DIM if idle or query else CYAN)
        text(d,(690,1513),'FEEDBACK COMPLETE · HOST SUPPLIES RECALL CUE' if query else 'QUEUE EMPTY · FEEDBACK HAS STOPPED' if idle else 'DEVICE SRAM  ·  OUTPUT PAYLOAD → NEXT INPUT',29,MUTED,True)
        edits=int(((diff==1)).sum());text(d,(160,1650),f'{edits} verified connection replacements',36,GREEN,True)
        text(d,(1240,1658),f'Independent prefix readback: {h} steps',28,MUTED)
        text(d,(2570,650),'Partial-cue response' if query else 'Last completed output' if idle else f'Returned state · step {step+1:02d}',33,INK,True)
        for b in range(128):
            xx=2600+(b%16)*66;yy=735+(b//16)*38;col=PALETTE[b//32] if y[b] else DIM
            d.rounded_rectangle((xx,yy,xx+39,yy+20),radius=5,fill=col)
        count=int(y.sum());text(d,(2570,1060),f'{count} active output '+('channel' if count==1 else 'channels')+(' · retained value' if idle else ''),30,MUTED)
        for j,(arm,label) in enumerate([('native','Native learning'),('frozen','Learning frozen'),('repeated_seed','Seed reused'),('severed','Feedback severed')]):
            row=self.data['arms'][arm];v=np.array(row['probe_outputs'][q] if query else row['outputs'][step]);yy=1190+j*100
            text(d,(2570,yy),label,29,GREEN if arm=='native' else AMBER if arm=='frozen' else MUTED,arm=='native')
            for k in range(4):
                count=int(v[k*32:(k+1)*32].sum());xx=2910+k*175;d.rounded_rectangle((xx,yy,xx+136,yy+40),radius=7,fill=blend(BG,PALETTE[k],.6) if count else (23,36,48));text(d,(xx+15,yy+3),f'S{k}: {count}',26,INK if count else MUTED)
        text(d,(2570,1600),'3 initializations · partial-cue recall',29,MUTED)
        text(d,(2570,1650),'Native 12/12   ·   Frozen 0/12',36,GREEN,True)
        # The progress ruler is an editorial position marker, not a chip timer.
        for i in range(32):
            xx=120+i*112.5;col=CYAN if i<=step else DIM;d.rectangle((xx,1800,xx+98,1812),fill=col)
        elapsed=(a['idle'][1]-a['idle'][0])/1e6 if idle else (a['quiet'][1]-a['quiet'][0])/1e6
        footer='No additional completed jobs or learning-SRAM changes during '+f'{elapsed:.2f} ms' if idle else '32 dependent steps complete by first wake after '+f'{elapsed:.2f} ms' if not query else 'Host supplies four new partial cues; native query outputs measured and SRAM unchanged'
        text(d,(120,1830),footer,31,AMBER if idle else MUTED)
        return im

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);data=source();(a.out/'recorded-events.json').write_text(json.dumps(data,separators=(',',':'))+'\n');film=Film(data)
    for name,t in [('seed',.8),('first-edit',2),('feedback',4.8),('full-memory',11),('queue-end',18.4),('recall',22.5)]:film.frame(t).save(a.out/(name+'.png'))
    if a.video:
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','h264_videotoolbox','-b:v','24M','-profile:v','high','-pix_fmt','yuv420p','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(a.out/'akd-native-feedback-4k60.mp4')]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);start=time.monotonic()
        try:
            for i in range(1440):
                proc.stdin.write(film.frame(i/60).tobytes())
                if i%240==0:print(json.dumps(dict(frame=i,total=1440,elapsed=round(time.monotonic()-start,1))),flush=True)
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(a.out/'akd-native-feedback-4k60.mp4'),'-vf','scale=1920:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out/'akd-native-feedback-1080p60.mp4')],check=True)
    (a.out/'SOURCE_BINDING.json').write_text(json.dumps(dict(status='AUDITED_NATIVE_FEEDBACK_REPLAY',archive_sha256=data['archive_sha256'],events_sha256=sha(a.out/'recorded-events.json'),renderer_sha256=sha(__file__),scope=data['scope'],electrical_spike_timings_observed=False,outputs={p.name:sha(p) for p in a.out.iterdir() if p.suffix in ['.png','.mp4']}),indent=2)+'\n')
    print(json.dumps(dict(status='RENDERED',out=str(a.out))))
if __name__=='__main__':main()
