"""Source-bound 4K60 replay; connections are measured, motion is schematic."""
from pathlib import Path
import argparse,json,math,subprocess,time
import numpy as np
from PIL import Image,ImageDraw
from discovery.archive import BASE,audit,masks,sha
from discovery.model import simulate,matrices
from coprocessor.neuron_visual import text,blend,Scene,BG,INK,MUTED,CYAN,GREEN,AMBER,DIM
W,H,FPS,DURATION=3840,2160,60,28

def source():
    checked=audit();m=json.loads(BASE.with_suffix('.json').read_text());f=m['fixtures']['s113101'];data=dict(archive_sha256=checked['archive_sha256'],arms={})
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        initial=z[f['arrays']['initial.ram']['key']];data['initial']=masks(initial).tolist()
        for arm in ['unseen','unseen_frozen']:
            run=m['runs'][f'native_s113101_{arm}_c12'];get=lambda k:z[run['arrays'][k]['key']];states=get('states').reshape(-1,482);conn=masks(states);before=masks(initial);total=0;counts=[];focus=[];j=0
            for v in conn:
                changed=np.flatnonzero((v[:32]!=before[:32]).any(1))
                if len(changed):assert len(changed)==1;j=int(changed[0])
                total+=int(((v[:32]==1)&(before[:32]==0)).sum());counts.append(total);focus.append(j);before=v
            a=simulate(initial,12,arm);data['arms'][arm]=dict(inputs=get('inputs').reshape(-1,128).tolist(),outputs=get('scores').reshape(-1,128).tolist(),connections=conn.tolist(),features=a['features'].tolist(),cumulative=counts,focus=focus,events=a['events'])
    data['scope']='Recorded native input/output and per-event learning RAM. Feature activations and helper activity derived from fixed weights and captured inputs. Thirty-two plastic records, eight fixed helpers, three NPUs. Fixed third-order basis and explicit raw history; host writes only current bits, schedules calls and learning mode. Moving camera, flow particles and slowed timing schematic; not electrical spikes. CPU dictionary also succeeds. No outcome prediction or efficiency claim.'
    return data

class Film:
    def __init__(self,data):
        self.data=data;self.base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(self.base)
        text(d,(115,65),'The chip chooses the record.',94,INK,True)
        text(d,(120,205),'Raw bits arrive. Native competition allocates memory. Earlier memories remain.',40,MUTED)
        d.line((120,285,3720,285),fill=DIM,width=2)
        for box in [(120,330,2330,475),(2390,330,3720,475),(120,525,2330,1770),(2390,525,3720,1770)]:d.rounded_rectangle(box,radius=26,fill=(11,20,31),outline=DIM,width=2)
        text(d,(165,555),'MEASURED CONNECTIONS + RETURNED RECORD ACTIVITY',32,CYAN,True)
        text(d,(165,615),'32 competing plastic records · no history ID or record-selection label',28,MUTED)
        text(d,(170,715),'FIXED FEATURE BASIS',27,MUTED,True);text(d,(1200,715),'NATIVE CONTEXT BANK',27,CYAN,True)
        text(d,(170,755),'FNP3 [1,3,0]',24,MUTED);text(d,(1200,755),'FNP3 [2,1,0]',24,MUTED)
        text(d,(1750,715),'FIXED RELAY',26,MUTED,True);text(d,(1750,755),'FNP3 [2,2,0]',24,MUTED)
        for x,label,value,c in [(120,'LEARNED HISTORY RECORDS','80 / 80',CYAN),(1040,'ORIGINAL RECORD IDS RETAINED','20 / 20',GREEN),(1960,'UNSEEN-ORDER RECALL','640 / 640',GREEN),(2880,'TRAINED-FROZEN HISTORY BANK','20 / 80',AMBER)]:
            d.rounded_rectangle((x,1810,x+840,1935),radius=18,fill=(14,25,36),outline=DIM,width=2);text(d,(x+25,1827),label,24,MUTED,True);text(d,(x+25,1864),value,43,c,True)
        for y,line in [(1973,'Five random initializations · four old histories, twelve newly allocated · fixed finite four-symbol representation'),(2020,'One constant training label0: the chip chooses the winner. Host supplies raw symbols, paces forwards and gates learning.'),(2067,'Recorded input/output + connection RAM; feature/helper activity derived. Geometry, camera and flow timing schematic.'),(2114,'Selected events, slowed replay · output precedes each edit · CPU dictionary also passes · no outcome or energy advantage claimed.')]:text(d,(120,y),line,27,MUTED)
        self.features=np.zeros((256,3))
        for j in range(256):
            yy=1-2*(j+.5)/256;r=math.sqrt(1-yy*yy);a=j*math.pi*(3-math.sqrt(5));self.features[j]=[-1.35+.4*r*math.cos(a),.1+.9*yy,.55*r*math.sin(a)]
        self.units=np.asarray([[.15+.2*(j%4),1.05-.29*(j//4),.18*((j%4)-1.5)] for j in range(32)])
        self.helpers=np.asarray([[-.12+j*.2,-1.36,0] for j in range(8)])
    def project(self,v,t):return Scene.project(self,v,t,center=(1230,1170),scale=500)
    def choose(self,t):
        if t<5:return 19+min(63,int(t/5*64)),'LEARN FOUR FAMILIAR HISTORIES'
        if t<15:return 99+min(63,int((t-5)/10*64)),'NEW HISTORIES · NATIVE ALLOCATION'
        if t<20:return 291+min(31,int((t-15)/5*32)),'READ-ONLY RECALL · ALL SIXTEEN'
        return 323+min(63,int((t-20)/8*64)),'NEW ORDER · NO TEACHING OR RESET'
    def frame(self,t):
        i,phase=self.choose(t);a=self.data['arms']['unseen'];c=self.data['arms']['unseen_frozen'];e=a['events'][i];x=np.asarray(a['inputs'][i]);y=np.asarray(a['outputs'][i]);features=np.asarray(a['features'][i]);after=np.asarray(a['connections'][i]);before=np.asarray(a['connections'][i-1] if i else self.data['initial']);edits=(after[:32]!=before[:32]).any(1);active=np.flatnonzero(y[64:96]);frozen=np.flatnonzero(np.asarray(c['outputs'][i])[64:96]);focus=int(active[0]) if len(active)==1 else a['focus'][i];im=self.base.copy();d=ImageDraw.Draw(im)
        text(d,(165,353),phase,32,GREEN if e['label']<0 else CYAN,True)
        text(d,(165,411),'Learning '+('ON · one class, all records compete' if e['label']==0 else 'OFF · connection RAM unchanged'),28,MUTED)
        text(d,(2430,353),f'Recorded event {i+1} / 451 · seed113101',29,INK,True)
        text(d,(2430,412),f'{a["cumulative"][i]} native connection replacements so far',28,GREEN)
        fp=self.project(self.features,t);up=self.project(self.units,t);hp=self.project(self.helpers,t)
        for j in range(32):
            for bit in np.flatnonzero(before[j]):
                hot=bool(features[bit]);col=CYAN if hot and j==focus else blend(BG,CYAN,.23 if hot else .10);Scene.edge(d,fp[bit],up[j],col,3 if j==focus and hot else 1)
                if hot and j in active and bit%3==0:
                    q=fp[bit,:2]+(up[j,:2]-fp[bit,:2])*((t*1.3+bit*.067)%1);d.ellipse((q[0]-3,q[1]-3,q[0]+3,q[1]+3),fill=INK)
            if edits[j]:
                for bit in np.flatnonzero((after[j]==1)&(before[j]==0)):Scene.edge(d,fp[bit],up[j],GREEN,5)
                for bit in np.flatnonzero((after[j]==0)&(before[j]==1)):Scene.edge(d,fp[bit],up[j],blend(BG,AMBER,.5),3)
        # Fixed relay has one real connection per record-response output bit.
        for j in range(32):
            q=(1810+(j%8)*48+18,1185+(j//8)*74+18,1)
            Scene.edge(d,up[j],q,CYAN if j in active else blend(BG,CYAN,.16),4 if j in active else 1)
        text(d,(1790,860),'RETURNED SRAM',25,MUTED,True)
        text(d,(1790,910),'3 retained raw symbols',23,GREEN)
        for k in range(3):
            xx=1810+k*118;v=int(y[k*16+8:k*16+16].sum()>0)
            d.rounded_rectangle((xx,958,xx+86,1020),radius=10,fill=(20,39,48),outline=GREEN,width=2);text(d,(xx+28,962),v,42,GREEN,True)
        text(d,(1790,1090),'32 record-response bits',23,CYAN)
        for j in range(32):
            xx=1810+(j%8)*48;yy=1185+(j//8)*74;hot=j in active
            d.rounded_rectangle((xx,yy,xx+35,yy+35),radius=7,fill=CYAN if hot else (25,40,54),outline=CYAN if hot else DIM,width=2)
            text(d,(xx+2,yy+39),f'{j:02}',18,INK if hot else MUTED)
        for k in np.argsort(fp[:,2]):
            xx,yy,fac=fp[k];r=(6 if features[k] else 2.5)*fac;d.ellipse((xx-r,yy-r,xx+r,yy+r),fill=CYAN if features[k] else DIM)
        for j in np.argsort(up[:,2]):
            xx,yy,fac=up[j];hot=j in active;Scene.orb(d,xx,yy,16*fac,GREEN if edits[j] else CYAN if hot else DIM,hot or edits[j]);text(d,(xx+19,yy-15),f'{j:02}',22,INK if hot or edits[j] else MUTED,True)
        for k,(xx,yy,fac) in enumerate(hp):
            val=bool(x[k*8:(k+1)*8].sum()) if k<8 else False;Scene.orb(d,xx,yy,11*fac,MUTED,val)
        text(d,(945,1615),'8 FIXED RAW-HISTORY HELPERS',25,MUTED,True)
        d.line([(2140,1480),(2140,1670),(320,1670),(320,1510)],fill=GREEN,width=3);text(d,(560,1688),'Native output SRAM → next input · history untouched by host',25,GREEN)
        text(d,(165,1730),'Green: newly stored connection   Amber: removed connection   Cyan: recorded active record',25,MUTED)
        text(d,(2440,565),'CURRENT FOUR-SYMBOL HISTORY',26,MUTED,True)
        for k in range(4):
            bit=int(x[k*16+8:k*16+16].sum()>0);xx=2440+k*135;d.rounded_rectangle((xx,615,xx+108,695),radius=13,fill=(21,38,51),outline=CYAN if k==3 else DIM,width=3);text(d,(xx+38,622),bit,54,CYAN if k==3 else INK,True)
        text(d,(3020,622),'old → current',27,MUTED);text(d,(2440,735),'NATIVE',26,CYAN,True);text(d,(3050,735),'TRAINED + FROZEN',26,AMBER,True)
        text(d,(2440,780),f'M{int(active[0]):02}' if len(active)==1 else 'No response',52,CYAN if len(active) else MUTED,True)
        text(d,(3050,780),f'M{int(frozen[0]):02}' if len(frozen)==1 else 'Unallocated',48,AMBER if len(frozen) else MUTED,True)
        text(d,(2440,890),f'M{focus:02} · '+('RECORD UPDATED NOW' if edits[focus] else '32 STORED CONNECTIONS'),28,INK,True)
        text(d,(2440,938),'256 input-feature slots · measured before/after bits',25,MUTED)
        for bit in range(256):
            xx=2440+(bit%16)*48;yy=1000+(bit//16)*31;old=before[focus,bit];new=after[focus,bit];col=GREEN if new and not old else AMBER if old and not new else CYAN if new and features[bit] else (58,94,112) if new else (24,37,50)
            d.rounded_rectangle((xx,yy,xx+36,yy+21),radius=4,fill=col)
        added=int(((after[focus]==1)&(before[focus]==0)).sum());removed=int(((after[focus]==0)&(before[focus]==1)).sum());text(d,(2440,1525),f'+{added} stored / −{removed} removed in this event',29,GREEN if added else MUTED,True)
        text(d,(2440,1590),'Fixed basis and helpers are engineered.',27,MUTED)
        text(d,(2440,1634),'Record identity is selected by native competition.',27,CYAN)
        text(d,(2440,1678),'The next gate is learning useful consequences.',27,MUTED)
        text(d,(2440,1722),'No electrical spike or physical die-position claim.',25,MUTED)
        return im

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);data=source();(a.out/'recorded-events.json').write_text(json.dumps(data,separators=(',',':'))+'\n');film=Film(data)
    for name,t in [('bootstrap',3.2),('allocation',6.2),('learning',9.4),('recall',17.2),('unseen-order',23.3)]:film.frame(t).save(a.out/(name+'.png'))
    if a.video:
        proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','h264_videotoolbox','-b:v','20M','-profile:v','high','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(a.out/'akd-context-discovery-4k60.mp4')],stdin=subprocess.PIPE);start=time.monotonic()
        try:
            for i in range(DURATION*FPS):
                proc.stdin.write(film.frame(i/FPS).tobytes())
                if i%300==0:print(json.dumps(dict(frame=i,total=DURATION*FPS,elapsed_s=round(time.monotonic()-start,1))),flush=True)
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(a.out/'akd-context-discovery-4k60.mp4'),'-vf','scale=1920:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out/'akd-context-discovery-1080p60.mp4')],check=True)
    (a.out/'SOURCE_BINDING.json').write_text(json.dumps(dict(status='AUDITED_CONTEXT_ALLOCATION_REPLAY',archive_sha256=data['archive_sha256'],events_sha256=sha(a.out/'recorded-events.json'),renderer_sha256=sha(__file__),scope=data['scope'],outputs={p.name:sha(p) for p in a.out.iterdir() if p.suffix in ['.png','.mp4']}),indent=2)+'\n')
    print(json.dumps(dict(status='RENDERED',out=str(a.out))))
if __name__=='__main__':main()
