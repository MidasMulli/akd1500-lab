"""Audited paired-recording replay of native signal-gated plasticity."""
from pathlib import Path
import argparse,json,math,subprocess,time
import numpy as np
from PIL import Image,ImageDraw
from feedback.archive import BASE,audit,sha
from coprocessor.neuron_visual import text,blend,Scene,BG,INK,MUTED,CYAN,GREEN,AMBER,DIM
W,H,FPS,DURATION=3840,2160,60,24
def masks(a):return np.unpackbits(a[...,2:].reshape(*a.shape[:-1],16,20)[...,4:].copy().view('u1'),axis=-1,bitorder='little')
def source():
    result=audit();meta=json.loads(BASE.with_suffix('.json').read_text());f=meta['fixtures']['s108101'];units=[2,8,0,1,3,4,5,6];rows=f['rows']['native']
    unchanged=next(r['index'] for r in rows if r['phase']=='post_adaptation' and r['context']==5 and r['candidate']==r['target']);ids=[104,146,147,148,185,186,unchanged,377]
    captions=['Initial rule · query predicts outcome 0','Rule changes · old response still wins','New outcome probe · no learned match yet','Actual outcome arrives · native connections change','Old outcome probe · response falls','New outcome probe · response rises','Other rule queried · retained without repair','Read-only recall · new association persists']
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        def run(variant,arm):
            r=meta['runs'][f'native_s108101_{variant}_{arm}_n768_b1_s0'];return lambda k:z[r['arrays'][k]['key']]
        raw,fr,red=run('raw','native'),run('reduced','frozen'),run('reduced','native');states=raw('snapshots').reshape(768,322);init=z[f['arrays']['raw.ram']['key']];before=np.vstack([init,states[:-1]]);bm,am=masks(before),masks(states)
        fst=fr('snapshots').reshape(768,322);fm=masks(fst);pot=raw('scores').reshape(768,16);rp=red('scores');fp=fr('scores');inputs=z[f['arrays']['native.inputs']['key']].reshape(768,512)
        # Reduced and raw paths have different output auxiliaries but the exact
        # same selected connections/learning metadata at every sample.
        rs=red('snapshots').reshape(768,322);assert np.array_equal(masks(rs),am)
        events=[]
        for i,title in zip(ids,captions):
            frozen_weights=fm[i];frozen_pot=frozen_weights.astype('i4')@inputs[i].astype('i4')
            # Reconstructed frozen potentials are labeled derived; only support
            # values are directly returned by the frozen reducer recording.
            changes=am[i].astype(int)-bm[i].astype(int)
            events.append(dict(index=i,caption=title,row=rows[i],input=inputs[i].tolist(),before=bm[i,units].tolist(),after=am[i,units].tolist(),potentials=pot[i].tolist(),frozen_potentials_derived=frozen_pot.tolist(),native_support=int(rp[i]),frozen_support=int(fp[i]),added=int((changes==1).sum()),removed=int((changes==-1).sum())))
    return dict(seed=108101,units=units,events=events,archive_sha256=meta['archive_sha256'],scope='Raw native potentials and reduced readout from paired exact-state recordings; frozen potentials independently reconstructed from its captured SRAM',audit={k:v for k,v in result.items() if k!='rows'})
class Film:
    def __init__(self,data):
        self.data=data;self.base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(self.base)
        text(d,(115,65),'The input decides when learning happens',88,bold=True);text(d,(120,180),'Recorded native learning · prediction probes stay below the write threshold',39,MUTED)
        d.line((120,260,3720,260),fill=DIM,width=2)
        boxes=[(120,330,1030,500,'External observation encoder','Context + candidate / actual outcome'),(1250,330,2330,500,'FNP3 · local connection SRAM','Learning enabled · constant class 0'),(2550,330,3720,500,'Fixed FNP3 readout','Native support response · one output')]
        for x,y,xx,yy,title,sub in boxes:
            d.rounded_rectangle((x,y,xx,yy),radius=24,fill=(13,24,36),outline=(48,75,94),width=2);text(d,(x+28,y+28),title,38,CYAN,True);text(d,(x+28,y+95),sub,30,MUTED)
        d.line((1030,415,1250,415),fill=DIM,width=4);d.line((2330,415,2550,415),fill=DIM,width=4)
        d.rounded_rectangle((110,560,2570,1760),radius=32,fill=(11,20,31),outline=(34,56,73),width=2);d.rounded_rectangle((2630,560,3730,1760),radius=32,fill=(15,21,29),outline=(59,51,40),width=2)
        text(d,(160,600),'NATIVE RESPONSES + CAPTURED CONNECTIONS',37,CYAN,True);text(d,(160,670),'8 sampled records of 16 · 32 connections each · schematic geometry',31,MUTED)
        text(d,(2690,600),'MATCHED FROZEN REPLAY',35,AMBER,True);text(d,(2690,670),'Same learned prefix · updates disabled',30,MUTED)
        text(d,(120,1930),'Measured: native potentials, fixed readout responses and complete connection-memory states.',30,MUTED)
        text(d,(120,1985),'Frozen potentials derived from its captured SRAM. Native potentials/readout shown from paired recordings.',28,MUTED)
        text(d,(120,2040),'Host supplies observations + fixed encoding. Native mode and class stay constant. No electrical spike trace.',28,MUTED)
        text(d,(120,2093),'Three seeds · 576/576 sealed predictions · all contexts previously taught · no CPU speed or energy win',29,MUTED)
        j=np.arange(188);yy=1-2*(j+.5)/188;rr=np.sqrt(1-yy*yy);theta=j*math.pi*(3-math.sqrt(5));self.features=np.column_stack((-1.1+.3*rr*np.cos(theta),yy*1.12,.8*rr*np.sin(theta)))
        self.units=np.array([[.45+(i%2)*.88,.96-(i//2)*.63,.22 if i%2 else -.22] for i in range(8)])
    def project(self,p,t):return Scene.project(self,p,t,center=(1230,1200),scale=580)
    def frame(self,t):
        j=min(7,int(t/3));e=self.data['events'][j];post=e['row']['feedback'] and (t%3)>1.4;m=np.array(e['after' if post else 'before']);bits=np.array(e['input']);im=self.base.copy();d=ImageDraw.Draw(im)
        text(d,(160,735),e['caption'],37,GREEN if e['row']['feedback'] or j>=4 else INK,True);p=self.project(self.features,t);q=self.project(self.units,t)
        for k,unit in enumerate(self.data['units']):
            hot=unit==e['row']['context']
            for b in np.flatnonzero(m[k]):
                color=GREEN if b in range(160,176) else AMBER if b>=176 else CYAN
                added=e['after'][k][b] and not e['before'][k][b]
                if post and added:color=GREEN
                Scene.edge(d,p[b],q[k],color if hot and bits[b] else blend(BG,color,.24),5 if hot and bits[b] else 2)
        for b in np.argsort(p[:,2]):
            x,y,fac=p[b];col=GREEN if 160<=b<176 else AMBER if b>=176 else CYAN;r=(8 if bits[b] else 3)*fac;d.ellipse((x-r,y-r,x+r,y+r),fill=col if bits[b] else DIM)
        for k in np.argsort(q[:,2]):
            unit=self.data['units'][k];v=e['potentials'][unit];x,y,fac=q[k];hot=unit==e['row']['context'];Scene.orb(d,x,y,(17+v*.6)*fac,GREEN if hot and post else blend(DIM,CYAN,(v/32)**.6),hot)
            text(d,(x+52,y-25),f'R{unit} · {v}/32',29,INK,hot)
        text(d,(210,1570),'CONTEXT · OUTCOME · FEEDBACK INPUTS',28,MUTED);text(d,(1610,1570),'RETURNED MATCH COUNTS',28,MUTED)
        gate='16 feedback anchors present' if e['row']['gate_present'] else 'Feedback anchors absent'
        caption=f'{e["added"]} added + {e["removed"]} removed' if e['row']['feedback'] else 'Query: 0 connection edits'
        text(d,(165,1665),gate+'  ·  '+caption,35,GREEN if post or not e['row']['feedback'] else MUTED,True)
        text(d,(2690,745),'MATCH THRESHOLD FOR LEARNING: 24',29,MUTED,True)
        text(d,(2690,795),'Bars: captured-state frozen / native',27,MUTED)
        for k,unit in enumerate(self.data['units']):
            y=875+k*70;v=e['potentials'][unit];fv=e['frozen_potentials_derived'][unit];text(d,(2690,y),f'R{unit}',28,MUTED)
            d.rectangle((2820,y+14,3444,y+34),fill=DIM);d.rectangle((2820,y+14,2820+fv*19.5,y+34),fill=AMBER)
            xx=2820+v*19.5;d.line((xx,y+5,xx,y+46),fill=GREEN,width=5);threshold=2820+24*19.5;d.line((threshold,y+5,threshold,y+46),fill=(110,125,140),width=2)
            text(d,(3480,y+3),f'{fv} / {v}',29,INK)
        text(d,(2690,1485),'FIXED READOUT · MEASURED',29,MUTED,True);text(d,(2690,1542),f'Frozen {e["frozen_support"]}  /  Native {e["native_support"]}',39,INK,True)
        text(d,(2690,1610),f'Candidate outcome {e["row"]["candidate"]} · context {e["row"]["context"]}',30,MUTED)
        text(d,(2690,1675),'Outcome input' if e['row']['feedback'] else 'Prediction probe',33,GREEN if e['row']['feedback'] else CYAN,True)
        text(d,(120,1810),f'RECORDED EVENT {e["index"]+1:03d}/768',34,CYAN,True);text(d,(1080,1810),'INPUT-GATED PLASTICITY · HOST LEARNING MODE UNCHANGED',32,INK)
        d.rectangle((120,1890,3720,1896),fill=DIM);d.rectangle((120,1890,120+round(3600*t/24),1896),fill=CYAN);return im
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);data=source();(a.out/'recorded-events.json').write_text(json.dumps(data,separators=(',',':'))+'\n');scene=Film(data)
    for label,t in [('baseline',1),('wrong',4),('feedback',10.8),('adapted',16),('retained',23.9)]:scene.frame(t).save(a.out/(label+'.png'))
    if a.video:
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','h264_videotoolbox','-b:v','26M','-profile:v','high','-pix_fmt','yuv420p','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(a.out/'akd-input-gated-learning-4k60.mp4')]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);start=time.monotonic()
        try:
            for i in range(1440):
                proc.stdin.write(scene.frame(i/60).tobytes())
                if i%240==0:print(json.dumps(dict(frame=i,total=1440,elapsed=round(time.monotonic()-start,1))),flush=True)
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(a.out/'akd-input-gated-learning-4k60.mp4'),'-vf','scale=1920:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out/'akd-input-gated-learning-1080p60.mp4')],check=True)
    binding=dict(status='AUDITED_INPUT_GATED_REPLAY',archive_sha256=data['archive_sha256'],events_sha256=sha(a.out/'recorded-events.json'),renderer_sha256=sha(__file__),paired_recordings=True,observed_spike_timing=False,logical_geometry=True,shown_events=[e['index']+1 for e in data['events']],scope=data['scope'],outputs={p.name:sha(p) for p in a.out.iterdir() if p.suffix in ['.png','.mp4']});(a.out/'SOURCE_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
    print(json.dumps(dict(status='RENDERED',out=str(a.out))))
if __name__=='__main__':main()
