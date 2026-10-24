"""Measured resident adaptation and a matched visible-state selector control."""
from pathlib import Path
import argparse,bisect,json,math,subprocess,time
import numpy as np
from PIL import Image,ImageDraw
from plasticity.archive import BASE,audit,masks,sha
from plasticity.model import simulate
from transition.visual import Film as PriorFilm,COLORS
from coprocessor.neuron_visual import text,blend,Scene,BG,INK,MUTED,CYAN,GREEN,AMBER,DIM
W,H=3840,2160;FPS=60;DURATION=28
def source():
    checked=audit();meta=json.loads(BASE.with_suffix('.json').read_text());f=meta['fixtures']['s111101'];data=dict(arms={},seed=111101,archive_sha256=checked['archive_sha256'],audit={k:v for k,v in checked.items() if k not in ['rows','superseded_cpu_forecasts','cpu_timings']})
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        fg=lambda k:z[f['arrays'][k]['key']]
        for k in ['codes','partial']:data[k]=fg(k).reshape(8,256).tolist()
        for k in ['old','new','changed']:data[k]=fg(k).tolist()
        data['initial_connections']=masks(fg('native.ram')).tolist();data['selector']={}
        for arm in ['native','baseline']:
            name=f'native_s111101_{arm}_e12';r=meta['runs'][name];get=lambda k:z[r['arrays'][k]['key']];a=simulate(fg(arm+'.ram'),111101,arm,12);ids=get('state_ids');ram=get('states').reshape(-1,98);previous=masks(fg(arm+'.ram'));count=0;counts={}
            for i,s in zip(ids,ram):
                after=masks(s);count+=int(((after==1)&(previous==0)).sum());counts[str(int(i))]=count;previous=after
            data['arms'][arm]=dict(source=name,events=a['events'],inputs=get('inputs').reshape(-1,256).tolist(),outputs=get('scores').reshape(-1,256).tolist(),connections={str(int(i)):masks(s).tolist() for i,s in zip(ids,ram)},metadata={str(int(i)):s[2:].reshape(8,12)[:,3].tolist() for i,s in zip(ids,ram)},cumulative=counts,quiet=get('quiet').reshape(12,9).tolist())
        names=[('0','native_s111101_native_e2_attempt2',0,False),('8-partial','native_s111101_native_e2_q8',8,False),('8-full','native_s111101_native_e2_q8full',8,True)]
        for key,name,q,full in names:
            r=meta['runs'][name];get=lambda k:z[r['arrays'][k]['key']];a=simulate(fg('native.ram'),111101,'native',2,q,full);ev=next(e['index'] for e in a['events'] if e['kind']=='fresh_example' and e['epoch']==1);ids=get('state_ids');ram=get('states').reshape(-1,98);before=ram[np.flatnonzero(ids==ev-1)[0]];after=ram[np.flatnonzero(ids==ev)[0]];data['selector'][key]=dict(source=name,before=masks(before).tolist(),after=masks(after).tolist(),update_event=ev,target=int(a['events'][ev]['label']))
    data['scope']='Returned codes and full learning SRAM readbacks. Class activity inferred through fixed decoder; geometry, flow and slowed timing schematic, not electrical spike tracing. Two alternating synthetic maps, external labels, host-paced bounded feedback; native settings fixed for the resident session.'
    return data
class Film(PriorFilm):
    def __init__(self,data):
        super().__init__(data);self.base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(self.base)
        text(d,(115,65),'Twelve changes. One resident network.',91,INK,True)
        text(d,(120,200),'Fresh observations → native connection edits → chip-fed prediction continues',40,MUTED)
        d.line((120,280,3720,280),fill=DIM,width=2)
        for box in [(120,325,2380,475),(2440,325,3720,475),(120,525,2380,1760),(2440,525,3720,1760)]:d.rounded_rectangle(box,radius=27,fill=(11,20,31),outline=(36,59,77),width=2)
        text(d,(165,565),'RECORDED NATIVE LEARNING MEMORY',36,CYAN,True)
        text(d,(165,625),'8 learning records · 144 used features of 256 · fixed 256-channel decoder',29,MUTED)
        text(d,(185,730),'BINARY STATE CUE',27,MUTED,True);text(d,(990,730),'PLASTIC FNP3 [1,3,0]',27,MUTED,True);text(d,(1740,730),'FNP3 [2,1,0] · STATE CODES',25,MUTED,True)
        text(d,(2500,570),'KEPT PLASTIC · SDK SETTING 0.75',33,GREEN,True);text(d,(2500,620),'Native updates continue after each rule reversal',27,MUTED)
        text(d,(2500,1150),'INHERITED SETTING · 0.25',33,AMBER,True);text(d,(2500,1200),'Learns first change; later reversals are blocked',27,MUTED)
        cards=[(120,'CHANGED-RULE CHECKS','120 / 120 recalled',GREEN),(1040,'RETENTION CHECKS','360 / 360 retained',CYAN),(1960,'POST-UPDATE NATIVE FEEDBACK','960 / 960 next states',GREEN),(2880,'COMPLETED SESSION TOTAL','384 connection swaps',GREEN)]
        for x,label,value,c in cards:d.rounded_rectangle((x,1810,x+840,1930),radius=19,fill=(14,25,36),outline=DIM,width=2);text(d,(x+26,1827),label,24,MUTED,True);text(d,(x+26,1865),value,38,c,True)
        text(d,(120,1970),'Five initializations · two alternating synthetic maps · external outcome labels · one program load per run',30,MUTED)
        text(d,(120,2020),'Measured outputs + visible SRAM. Class activity decoded from returned codes. Quiet bursts complete by first wake after 5ms.',28,MUTED)
        text(d,(120,2070),'Host pauses feedback to insert observations and probes; the last native state stays in device SRAM. No weight/threshold resets.',28,MUTED)
        text(d,(120,2120),'Layout, flow animation and playback timing schematic · no electrical spike trace · finite schedule · CPU control also passes',27,MUTED)
        self.ids={arm:sorted(map(int,row['connections'])) for arm,row in data['arms'].items()}
        self.epoch_events={}
        for e in range(12):self.epoch_events[e]=[v for v in data['arms']['native']['events'] if v['epoch']==e]
    def graph(self,d,center,connections,current,previous,color,t,history):
        # A learned self-transition is a real edge, not a zero-length line.
        c=np.array(connections).copy();loops=[s for s in range(8) if self.codes[s]@c[s]>24]
        for s in loops:c[s]=0
        super().graph(d,center,c,current,previous,color,t,history)
        for s in loops:
            p=np.array(center)+self.graphpos[s]*(.91 if center[1]>1000 else 1);outward=p-np.array(center);outward/=np.linalg.norm(outward);side=np.array([-outward[1],outward[0]]);a=p+outward*25+side*15;b=p+outward*25-side*15;c1=p+outward*130+side*85;c2=p+outward*130-side*85;points=[]
            for u in np.linspace(0,1,41):points.append((1-u)**3*a+3*(1-u)**2*u*c1+3*(1-u)*u*u*c2+u**3*b)
            hot=current==s and previous==s;col=color if hot else blend(BG,color,.5);d.line([tuple(v) for v in points],fill=col,width=6 if hot else 3);v=points[-1]-points[-2];v/=np.linalg.norm(v);side2=np.array([-v[1],v[0]]);d.polygon([tuple(b),tuple(b-v*18+side2*8),tuple(b-v*18-side2*8)],fill=col)
            if hot:
                q=points[int((t*1.7)%1*40)];d.ellipse((q[0]-7,q[1]-7,q[0]+7,q[1]+7),fill=INK)
    def context(self,x):
        return next((j for j in range(8) if np.array_equal(x,self.codes[j]) or np.array_equal(x,self.partial[j])),None)
    def position(self,t):
        if t<2:return self.epoch_events[0][0],-1
        if t<14:e=min(2,int((t-2)/4));f=((t-2)%4)/4
        elif t<26:e=min(11,3+int((t-14)/ (12/9)));f=((t-14)%(12/9))/(12/9)
        else:return self.epoch_events[11][-1],11
        ev=self.epoch_events[e]
        if f<.13:return ev[0],e
        if f<.49:return ev[1+min(3,int((f-.13)/.09))],e
        if f<.65:
            s=self.data['changed'][0 if f<.57 else 1];return next(x for x in ev if x['kind']=='query' and x['context']==s),e
        return [x for x in ev if x['kind']=='feedback'][min(15,int((f-.65)/.35*16))],e
    def frame(self,t):
        event,epoch=self.position(t);i=event['index'];kind=event['kind'];a=self.data['arms']['native'];b=self.data['arms']['baseline'];x=np.asarray(a['inputs'][i]);y=np.asarray(a['outputs'][i]);fy=np.asarray(b['outputs'][i]);ns,fs=self.state(y),self.state(fy);snap=self.ids['native'][bisect.bisect_right(self.ids['native'],i)-1];bsnap=self.ids['baseline'][bisect.bisect_right(self.ids['baseline'],i)-1];conn=np.asarray(a['connections'][str(snap)]);bc=np.asarray(b['connections'][str(bsnap)]);meta=np.asarray(a['metadata'][str(snap)]);teacher=event['label'] if kind=='fresh_example' else None
        prevsnap=self.ids['native'][max(0,bisect.bisect_left(self.ids['native'],snap)-1)];oldconn=np.asarray(a['connections'][str(prevsnap)])
        caption='PREDICT · CARRIED NATIVE STATE' if kind=='pre_prediction' else 'NEW OBSERVATION · NATIVE UPDATE' if kind=='fresh_example' else 'RECALL PROBE · SAME INCOMPLETE CUE' if kind=='query' else 'FEEDBACK · OUTPUT SUPPLIES NEXT INPUT'
        sub='Initial feedback cue; subsequent epochs retain the last chip output' if epoch<0 else 'Teacher supplies observed destination; response precedes update' if teacher is not None else 'Learning enabled; probes do not change visible connection RAM' if kind=='query' else 'Host submits finite schedule; successive state values come from the chip'
        im=self.base.copy();d=ImageDraw.Draw(im);text(d,(160,350),f'CHANGE {max(1,epoch+1):02d} / 12  ·  '+caption,31,AMBER if teacher is not None else GREEN,True);text(d,(160,410),sub,29,MUTED)
        text(d,(2480,350),'ONE LOAD · ZERO WEIGHT RESTORES',28,CYAN,True)
        selected=teacher if teacher is not None else int(self.data['old'][self.data['changed'][0]]);word=int(meta[selected]);text(d,(2480,410),f'M{selected}: learned threshold {word>>16} · swap budget {word&65535}',29,MUTED)
        p=self.project(self.features,t);n=self.project(self.neurons,t);r=self.project(self.decoded,t)
        for j in range(8):
            for bit in np.flatnonzero(conn[j]):
                hot=bool(x[bit]);added=bool(conn[j,bit] and not oldconn[j,bit] and teacher is not None);c=GREEN if added else COLORS[self.group[bit]] if self.group[bit]>=0 else CYAN;Scene.edge(d,p[bit],n[j],c if hot else blend(BG,c,.23),3 if hot else 1)
                if hot and (j==ns or j==teacher):
                    q=p[bit,:2]+(n[j,:2]-p[bit,:2])*((t*1.3+bit*.037)%1);d.ellipse((q[0]-3,q[1]-3,q[0]+3,q[1]+3),fill=INK)
            Scene.edge(d,n[j],r[j],CYAN if ns==j else blend(BG,CYAN,.2),5 if ns==j else 2)
        for bit in sorted(self.used,key=lambda k:p[k,2]):
            xx,yy,fac=p[bit];rad=(6 if x[bit] else 3)*fac;c=COLORS[self.group[bit]] if self.group[bit]>=0 else CYAN;d.ellipse((xx-rad,yy-rad,xx+rad,yy+rad),fill=c if x[bit] else DIM)
        for g,center in enumerate(self.centers):
            xx,yy,_=self.project(np.asarray([center]),t)[0];lx=xx+(-34 if g==0 else -75 if (g-1)%2==0 else 25);ly=yy+65;d.rounded_rectangle((lx-7,ly-3,lx+(90 if g==0 else 44),ly+32),radius=6,fill=(11,20,31));text(d,(lx,ly),'shared' if g==0 else f'S{g-1}',26,MUTED if g==0 else COLORS[g-1],True)
        for j in range(8):
            xx,yy,fac=n[j];Scene.orb(d,xx,yy,21*fac,CYAN if ns==j else DIM,ns==j)
            if teacher==j:d.ellipse((xx-34,yy-34,xx+34,yy+34),outline=AMBER,width=5)
            d.rounded_rectangle((xx+31,yy-22,xx+84,yy+18),radius=5,fill=(11,20,31));text(d,(xx+36,yy-17),f'M{j}',27,AMBER if teacher==j else INK if ns==j else MUTED,True)
            xx,yy,fac=r[j];d.rounded_rectangle((xx-53,yy-24,xx+53,yy+24),radius=10,fill=blend(BG,COLORS[j],.7) if ns==j else (23,36,48),outline=COLORS[j] if ns==j else DIM,width=2);text(d,(xx-18,yy-16),f'S{j}',28,INK if ns==j else MUTED,ns==j)
        path=[(2220,1495),(2220,1585),(365,1585),(365,1470)]
        for j in range(3):d.line((path[j],path[j+1]),fill=CYAN if kind=='feedback' else DIM,width=3)
        d.polygon([(355,1484),(375,1484),(365,1468)],fill=CYAN if kind=='feedback' else DIM)
        text(d,(690,1592),'DEVICE SRAM · last native output carried across new observations',23,CYAN)
        count=a['cumulative'][str(snap)];text(d,(165,1640),f'{count} cumulative connection replacements',34,GREEN,True);text(d,(165,1685),'Each selected connection is read back from native memory',27,MUTED);text(d,(165,1720),'Amber: teacher-selected record · cyan: decoded response',26,MUTED)
        text(d,(1720,1640),'Response before edit' if teacher is not None else 'Returned next state',26,MUTED);text(d,(1720,1680),'No response' if ns is None else f'S{ns} · 32 active bits',33,CYAN,True)
        hist=[];fh=[]
        if kind=='feedback':
            ids=[e['index'] for e in self.epoch_events[event['epoch']] if e['kind']=='feedback' and e['index']<=i];hist=[self.context(a['inputs'][ids[0]])]+[self.state(a['outputs'][k]) for k in ids];fh=[self.context(b['inputs'][ids[0]])]+[self.state(b['outputs'][k]) for k in ids]
        self.graph(d,(3080,880),conn,ns,self.context(x),GREEN,t,hist or ([self.context(x),ns] if ns is not None else []));self.graph(d,(3080,1438),bc,fs,self.context(b['inputs'][i]),AMBER,t,fh or ([self.context(b['inputs'][i]),fs] if fs is not None else []))
        for j in range(12):xx=2500+j*95;d.rounded_rectangle((xx,1698,xx+78,1708),radius=3,fill=GREEN if j<=epoch else DIM)
        text(d,(2500,1720),'Same resident model · each arm carries its own native state',23,MUTED)
        return im
def selector_figure(data,out):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im);text(d,(120,70),'Unchanged visible RAM. Different next edit.',83,INK,True);text(d,(120,190),'A controlled query-history effect on later native learning · one initialization',38,MUTED)
    a=data['selector']['0'];bp=data['selector']['8-partial'];bf=data['selector']['8-full'];assert a['before']==bp['before']==bf['before'];text(d,(120,300),'SAME COMPLETE LEARNING RAM BEFORE UPDATE · SAME NEXT TEACHING EXAMPLE',35,CYAN,True)
    target=a['target'];baseline=np.array(a['after'][target]);columns=[('0','No extra queries'),('8-full','8 full-cue queries'),('8-partial','8 incomplete-cue queries')]
    for col,(key,title) in enumerate(columns):
        x=120+col*1200;row=data['selector'][key];bits=np.array(row['after'][target]);dif=bits!=baseline;d.rounded_rectangle((x,390,x+1080,1730),radius=27,fill=(11,20,31),outline=DIM,width=2);text(d,(x+40,430),title,43,INK,True);text(d,(x+40,505),'Same returned query codes' if col else 'Reference partial learning edit',31,MUTED);text(d,(x+40,570),f'Native record M{target} · 256 mask-bit locations',30,CYAN,True)
        for j in range(256):
            xx=x+60+(j%16)*59;yy=680+(j//16)*48;c=GREEN if dif[j] and bits[j] else CYAN if bits[j] else (26,41,55);d.rounded_rectangle((xx,yy,xx+40,yy+30),radius=5,fill=c,outline=GREEN if dif[j] else c,width=3 if dif[j] else 1)
        text(d,(x+45,1460),'Filled: selected · green outline: differing zero bit',23,MUTED)
        text(d,(x+45,1520),f'{int(dif.sum())} differing mask bits',48,GREEN if dif.any() else CYAN,True);text(d,(x+45,1590),'32 connections remain selected',30,MUTED);text(d,(x+45,1645),'After the same partial native update',29,MUTED)
    text(d,(120,1810),'The two eight-query cases match descriptor count and output codes. Their input detail differs.',36,INK,True)
    text(d,(120,1880),'Visible learning RAM is identical immediately before the next update. Extra queries receive no teaching labels.',31,MUTED)
    text(d,(120,1940),'Eight incomplete cues change six later mask bits; eight full cues match the zero-query case. The partial-query case repeats exactly.',30,MUTED)
    text(d,(120,2000),'A reconstructed score-accumulation model predicts the captured edits. No hidden register was directly measured.',30,MUTED)
    text(d,(120,2060),'Recorded SRAM bits; schematic placement. This qualifies input-dependent query history, not persistence through restore or power loss.',29,MUTED)
    im.save(out/'selector-history-4k.png')
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);data=source();(a.out/'recorded-events.json').write_text(json.dumps(data,separators=(',',':'))+'\n');film=Film(data)
    for name,t in [('start',.8),('first-change',4),('reversal',8.4),('partial-edit',7.5),('feedback',11.2),('final',27.6)]:film.frame(t).save(a.out/(name+'.png'))
    selector_figure(data,a.out)
    if a.video:
        proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','h264_videotoolbox','-b:v','22M','-profile:v','high','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(a.out/'akd-resident-adaptation-4k60.mp4')],stdin=subprocess.PIPE);start=time.monotonic()
        try:
            for i in range(DURATION*FPS):
                proc.stdin.write(film.frame(i/FPS).tobytes())
                if i%300==0:print(json.dumps(dict(frame=i,total=DURATION*FPS,elapsed_s=round(time.monotonic()-start,1))),flush=True)
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(a.out/'akd-resident-adaptation-4k60.mp4'),'-vf','scale=1920:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out/'akd-resident-adaptation-1080p60.mp4')],check=True)
    binding=dict(status='AUDITED_RESIDENT_ADAPTATION_REPLAY',archive_sha256=data['archive_sha256'],events_sha256=sha(a.out/'recorded-events.json'),renderer_sha256=sha(__file__),scope=data['scope'],outputs={p.name:sha(p) for p in a.out.iterdir() if p.suffix in ['.png','.mp4']});(a.out/'SOURCE_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n');print(json.dumps(dict(status='RENDERED',out=str(a.out))))
if __name__=='__main__':main()
