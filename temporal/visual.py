"""Source-bound carried-context replay; geometry and playback are schematic."""
from pathlib import Path
import argparse,json,math,subprocess,time
import numpy as np
from PIL import Image,ImageDraw
from temporal.archive import BASE,audit,masks,sha
from temporal.model import schedule
from coprocessor.neuron_visual import text,blend,Scene,BG,INK,MUTED,CYAN,GREEN,AMBER,DIM
W,H=3840,2160;FPS=60;DURATION=24;NAMES=list('ABCDPQRS')

def source():
    checked=audit();m=json.loads(BASE.with_suffix('.json').read_text());f=m['fixtures']['s112101'];data=dict(archive_sha256=checked['archive_sha256'],audit={k:v for k,v in checked.items() if k!='rows'},arms={})
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        fg=lambda k:z[f['arrays'][k]['key']]
        data.update(codes=fg('codes').reshape(8,256).tolist(),cues=fg('cues').reshape(6,256).tolist(),initial_connections=masks(fg('initial.ram')).tolist(),changed=fg('changed').tolist())
        for arm in ['native','frozen','severed']:
            name=f'native_s112101_{arm}_e8';r=m['runs'][name];get=lambda k:z[r['arrays'][k]['key']]
            states=get('states').reshape(-1,98);conn=masks(states);before=np.asarray(data['initial_connections']);counts=[];c=0
            for v in conn:c+=int(((v==1)&(before==0)).sum());counts.append(c);before=v
            data['arms'][arm]=dict(source=name,inputs=get('inputs').reshape(-1,256).tolist(),outputs=get('scores').reshape(-1,256).tolist(),connections=conn.tolist(),cumulative=counts,delays_ns=get('delays_ns').tolist())
    data['events']=schedule(112101,8)
    data['scope']='Recorded input/output codes and per-event learning SRAM. Native history is output SRAM, not a hidden neuronal-state trace. Host supplies current symbols and external teaching, and schedules each event. Class activity decoded from returned codes; geometry, flow particles, camera and slowed timing schematic. CPU state machine also passes.'
    return data

class Film:
    def __init__(self,data):
        self.data=data;self.codes=np.asarray(data['codes']);self.base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(self.base)
        text(d,(115,65),'Same X. Different answers.',94,INK,True)
        text(d,(120,205),'The external symbol is identical. The preceding chip output supplies the context.',40,MUTED)
        d.line((120,285,3720,285),fill=DIM,width=2)
        for box in [(120,330,2380,480),(2440,330,3720,480),(120,530,2380,1765),(2440,530,3720,1765)]:d.rounded_rectangle(box,radius=26,fill=(11,20,31),outline=DIM,width=2)
        text(d,(165,565),'RECORDED NATIVE CONNECTIONS + RETURNED CODES',34,CYAN,True)
        text(d,(165,625),'8 records · 2 retaught destinations · fixed 256-channel state decoder',29,MUTED)
        text(d,(185,750),'RETAINED HISTORY · SRAM',27,GREEN,True)
        text(d,(185,1190),'CURRENT SYMBOL · HOST',27,CYAN,True)
        text(d,(1170,750),'FNP3 [1,3,0]',26,MUTED,True);text(d,(1870,750),'FNP3 [2,1,0]',26,MUTED,True)
        for x,label,value,c in [(120,'CONTEXT-DEPENDENT PREDICTIONS','480 / 480',GREEN),(1040,'CHANGED ASSOCIATIONS','240 / 240',CYAN),(1960,'UNCHANGED ASSOCIATIONS','240 / 240',GREEN),(2880,'HISTORY ERASED','0 / 480 responses',AMBER)]:
            d.rounded_rectangle((x,1810,x+840,1930),radius=19,fill=(14,25,36),outline=DIM,width=2);text(d,(x+25,1826),label,24,MUTED,True);text(d,(x+25,1865),value,39,c,True)
        text(d,(120,1970),'Five initializations · eight changes between two synthetic rules · seeded prior · one program load per resident run',29,MUTED)
        text(d,(120,2020),'Host writes only the current-symbol half; prior native output stays in device SRAM. Explicit ablation erases that history.',28,MUTED)
        text(d,(120,2070),'Measured returned codes + selected SRAM connections. Class activity inferred through the fixed decoder; no electrical spike trace.',28,MUTED)
        text(d,(120,2120),'Recorded replay · schematic layout, camera and flow timing · serial host-paced events · CPU state-machine control also passes 480/480',27,MUTED)
        self.features=np.zeros((256,3))
        for part in range(2):
          for j in range(128):
            yy=1-2*(j+.5)/128;r=math.sqrt(1-yy*yy);a=j*math.pi*(3-math.sqrt(5));self.features[part*128+j]=[-1.48+.28*r*math.cos(a),(.65 if part==0 else -.72)+.35*yy,.33*r*math.sin(a)]
        self.units=np.asarray([[.12+.16*(j%2),1.04-.305*j,-.07+.14*(j%2)] for j in range(8)])
        self.outputs=np.asarray([[1.72,1.04-.305*j,0] for j in range(8)])
    def state(self,v):return next((j for j in range(8) if np.array_equal(v,self.codes[j])),None)
    def project(self,v,t):return Scene.project(self,v,t,center=(1280,1170),scale=445)
    def choose(self,t):
        ev=self.data['events']
        if t<4:ids=[e['index'] for e in ev if e['phase']=='old_query' and e['cue']>=4];return ids[min(3,int(t))],'native','RETAINED CONTEXT · BEFORE CHANGE'
        if t<8:ids=[e['index'] for e in ev if e['phase']=='teacher' and e['epoch']==0 and e['cue']>=4];return ids[min(3,int(t-4))],'native','FRESH OBSERVATION · NATIVE UPDATE'
        if t<12:ids=[e['index'] for e in ev if e['phase']=='probe' and e['epoch']==0 and e['variant']=='full' and e['cue']>=4];return ids[min(3,int(t-8))],'native','SAME X · UPDATED PREDICTION'
        if t<14:ids=[e['index'] for e in ev if e['phase']=='probe' and e['epoch']==0 and e['variant']=='long' and e['cue']>=4 and e['context'] in self.data['changed']];return ids[min(1,int(t-12))],'native','INCOMPLETE X · AFTER 20 ms DWELL'
        if t<17:ids=[e['index'] for e in ev if e['phase']=='teacher' and e['epoch']==1 and e['cue']>=4];return ids[min(3,int((t-14)/.75))],'native','RULE REVERSES · SAME RESIDENT MODEL'
        if t<20:ids=[e['index'] for e in ev if e['phase']=='probe' and e['epoch']==1 and e['variant']=='full' and e['cue']>=4];return ids[min(3,int((t-17)/.75))],'native','SAME X · LEARNED AGAIN'
        ids=[e['index'] for e in ev if e['phase']=='probe' and e['epoch']==1 and e['variant']=='full' and e['cue']>=4];return ids[min(3,int(t-20))],'severed','CONTROL · ERASE ONLY THE CARRIED HISTORY'
    @staticmethod
    def grid(d,bits,xy,color):
        for k,v in enumerate(bits):
            xx=xy[0]+(k%32)*32;yy=xy[1]+(k//32)*28;d.rounded_rectangle((xx,yy,xx+22,yy+17),radius=3,fill=color if v else (26,40,54))
    def frame(self,t):
        i,arm,phase=self.choose(t);row=self.data['arms'][arm];e=self.data['events'][i];x=np.asarray(row['inputs'][i]);y=np.asarray(row['outputs'][i]);history=self.state(np.r_[x[:128],np.zeros(128,'u1')]);response=self.state(y);teacher=e['label'] if e['phase']=='teacher' else None
        conn=np.asarray(row['connections'][i]);before=np.asarray(row['connections'][i-1] if i else self.data['initial_connections']);frozen=self.state(self.data['arms']['frozen']['outputs'][i]);intact=self.state(self.data['arms']['native']['outputs'][i]);im=self.base.copy();d=ImageDraw.Draw(im)
        text(d,(160,355),phase,32,AMBER if teacher is not None or arm=='severed' else GREEN,True)
        text(d,(160,417),'Teacher labels the observed outcome; response shown before update' if teacher is not None else 'Current X is byte-identical across contexts within this query variant',28,MUTED)
        text(d,(2480,355),'ONE LOAD · NO WEIGHT / THRESHOLD RESETS',27,CYAN,True)
        text(d,(2480,417),f'Recorded event {i} · '+('old rule' if e['epoch']<0 else f'change {e["epoch"]+1} / 8'),29,MUTED)
        fp=self.project(self.features,t);np_=self.project(self.units,t);op=self.project(self.outputs,t)
        for j in range(8):
            for bit in np.flatnonzero(conn[j]):
                hot=bool(x[bit]);added=bool(not before[j,bit] and teacher is not None);c=GREEN if added else GREEN if bit<128 else CYAN;Scene.edge(d,fp[bit],np_[j],c if hot else blend(BG,c,.20),3 if hot else 1)
                if hot and (j==response or j==teacher):
                    q=fp[bit,:2]+(np_[j,:2]-fp[bit,:2])*((t*1.25+bit*.043)%1);d.ellipse((q[0]-3,q[1]-3,q[0]+3,q[1]+3),fill=INK)
            Scene.edge(d,np_[j],op[j],CYAN if response==j else blend(BG,CYAN,.2),5 if response==j else 2)
        for k in np.argsort(fp[:,2]):
            xx,yy,fac=fp[k];r=(6 if x[k] else 2.6)*fac;c=GREEN if k<128 else CYAN;d.ellipse((xx-r,yy-r,xx+r,yy+r),fill=c if x[k] else DIM)
        for j in range(8):
            xx,yy,fac=np_[j];Scene.orb(d,xx,yy,22*fac,CYAN if response==j else DIM,response==j)
            if teacher==j:d.ellipse((xx-35,yy-35,xx+35,yy+35),outline=AMBER,width=5)
            d.rounded_rectangle((xx+29,yy-23,xx+89,yy+20),radius=5,fill=(11,20,31))
            text(d,(xx+35,yy-18),f'M{j}',26,AMBER if teacher==j else MUTED,True)
            xx,yy,_=op[j];d.rounded_rectangle((xx-66,yy-25,xx+66,yy+25),radius=10,fill=blend(BG,CYAN,.5) if response==j else (22,35,48),outline=CYAN if response==j else DIM,width=2);text(d,(xx-12,yy-18),NAMES[j],29,INK if response==j else MUTED,True)
        path=[(2230,1545),(2230,1630),(375,1630),(375,1060)]
        for k in range(3):d.line((path[k],path[k+1]),fill=AMBER if arm=='severed' else GREEN,width=3)
        d.polygon([(365,1075),(385,1075),(375,1058)],fill=AMBER if arm=='severed' else GREEN)
        text(d,(620,1643),'SRAM output → next descriptor input · history half '+('erased by control' if arm=='severed' else 'untouched by host'),25,AMBER if arm=='severed' else GREEN)
        text(d,(165,1700),f'{row["cumulative"][i]} cumulative native connection replacements in this run',29,GREEN,True)
        cols=[(2500,'RETAINED',NAMES[history] if history is not None else 'Erased',GREEN),(2910,'CURRENT','X−4 bits' if e['cue']==5 else 'X',CYAN),(3340,'RETURNED',NAMES[response] if response is not None else 'No response',CYAN if response is not None else AMBER)]
        for xx,label,value,c in cols:text(d,(xx,580),label,26,MUTED,True);text(d,(xx,630),value,48 if len(value)>4 else 66,c,True)
        text(d,(2500,760),'Intact arm: '+('No response' if intact is None else NAMES[intact])+'   ·   Frozen: '+('No response' if frozen is None else NAMES[frozen]),34,MUTED)
        text(d,(2500,845),'SEALED PROBE RESULTS · FIRST CHANGED RULE',28,INK,True)
        text(d,(2500,915),'Previous context',27,MUTED);text(d,(2940,915),'Native',27,GREEN);text(d,(3300,915),'Frozen',27,AMBER)
        full=[v for v in self.data['events'] if v['phase']=='probe' and v['epoch']==0 and v['variant']=='full' and v['cue']>=4]
        for c in range(4):
            ev=next(v for v in full if v['context']==c);yy=980+80*c;hot=c==e['context'];nr=self.state(self.data['arms']['native']['outputs'][ev['index']]);fr=self.state(self.data['arms']['frozen']['outputs'][ev['index']])
            if hot:d.rounded_rectangle((2485,yy-10,3670,yy+58),radius=12,fill=(22,38,49))
            text(d,(2520,yy),NAMES[c]+' then X',34,INK if hot else MUTED,True);text(d,(2970,yy),NAMES[nr],34,GREEN,True);text(d,(3330,yy),NAMES[fr],34,AMBER,True)
        text(d,(2500,1340),'RETAINED HISTORY · 128 BINARY CHANNELS',27,GREEN,True);self.grid(d,x[:128],(2500,1390),GREEN)
        text(d,(2500,1530),'CURRENT SYMBOL · 128 BINARY CHANNELS',27,CYAN,True);self.grid(d,x[128:],(2500,1580),CYAN)
        text(d,(2500,1710),'History-erased probes: same captured learning RAM',26,AMBER)
        return im

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);data=source();(a.out/'recorded-events.json').write_text(json.dumps(data,separators=(',',':'))+'\n');film=Film(data)
    for name,t in [('old',1.3),('teaching',5.3),('new-rule',9.3),('incomplete',12.5),('reversal',18),('history-erased',22.5)]:film.frame(t).save(a.out/(name+'.png'))
    if a.video:
        proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','h264_videotoolbox','-b:v','20M','-profile:v','high','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(a.out/'akd-temporal-context-4k60.mp4')],stdin=subprocess.PIPE);start=time.monotonic()
        try:
            for i in range(DURATION*FPS):
                proc.stdin.write(film.frame(i/FPS).tobytes())
                if i%300==0:print(json.dumps(dict(frame=i,total=DURATION*FPS,elapsed_s=round(time.monotonic()-start,1))),flush=True)
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(a.out/'akd-temporal-context-4k60.mp4'),'-vf','scale=1920:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out/'akd-temporal-context-1080p60.mp4')],check=True)
    binding=dict(status='AUDITED_TEMPORAL_CONTEXT_REPLAY',archive_sha256=data['archive_sha256'],events_sha256=sha(a.out/'recorded-events.json'),renderer_sha256=sha(__file__),scope=data['scope'],outputs={p.name:sha(p) for p in a.out.iterdir() if p.suffix in ['.png','.mp4']});(a.out/'SOURCE_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n');print(json.dumps(dict(status='RENDERED',out=str(a.out))))
if __name__=='__main__':main()
