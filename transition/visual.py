"""Recorded changed-trajectory replay in the established neural visual style."""
from pathlib import Path
import argparse,json,math,subprocess,time
import numpy as np
from PIL import Image,ImageDraw
from transition.archive import BASE,audit,masks,sha
from coprocessor.neuron_visual import text,blend,Scene,BG,INK,MUTED,CYAN,GREEN,AMBER,DIM
W,H=3840,2160;DURATION=26;FPS=60
COLORS=[CYAN,(155,192,248),GREEN,(180,164,242),AMBER,(234,153,173),(135,217,213),(208,213,149)]
def source():
    checked=audit();meta=json.loads(BASE.with_suffix('.json').read_text());f=meta['fixtures']['s110101']
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        fg=lambda k:z[f['arrays'][k]['key']];data=dict(seed=110101,codes=fg('codes').reshape(8,256).tolist(),partial=fg('partial').reshape(8,256).tolist(),old=fg('old').tolist(),new=fg('new').tolist(),changed=fg('changed').tolist(),initial_connections=masks(fg('initial.ram')).tolist(),arms={})
        for arm in ['native','frozen','disconnected']:
            run=next(v for v in meta['runs'].values() if v['receipt']['mode']=='native' and v['receipt']['seed']==110101 and v['receipt']['horizon']==32 and v['receipt']['arm']==arm and v['receipt']['exit_code']==0);get=lambda k:z[run['arrays'][k]['key']]
            data['arms'][arm]=dict(source_run=next(k for k,v in meta['runs'].items() if v is run),outputs=get('scores').reshape(26,256).tolist(),connections=masks(get('states').reshape(26,98)).tolist(),rollout=get('rollout').reshape(8,32,256).tolist(),quiet=get('quiet').reshape(8,7).tolist())
    data.update(archive_sha256=checked['archive_sha256'],audit={k:v for k,v in checked.items() if k not in ['rows','cpu_timings']},scope='Recorded diagnostic outputs and complete per-event SRAM readbacks; returned rollout codes and output-to-input pointers. Class responses inferred from the one-to-one fixed decoder. Feature and neuron geometry, motion and slowed timing schematic. No electrical spike trace. One externally taught change episode; finite queue; CPU table also passes.')
    return data

class Film:
    def __init__(self,data):
        self.data=data;self.codes=np.asarray(data['codes']);self.partial=np.asarray(data['partial']);self.initial=np.asarray(data['initial_connections']);self.base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(self.base)
        text(d,(115,65),'Learning changes the loop.',100,INK,True)
        text(d,(120,205),'Native connection edits change which state the AKD1500 predicts next',42,MUTED)
        d.line((120,285,3720,285),fill=DIM,width=2)
        d.rounded_rectangle((120,330,2380,475),radius=22,fill=(13,25,36),outline=(41,68,88),width=2)
        d.rounded_rectangle((2440,330,3720,475),radius=22,fill=(13,25,36),outline=(41,68,88),width=2)
        d.rounded_rectangle((120,525,2380,1760),radius=30,fill=(11,20,31),outline=(36,59,77),width=2)
        d.rounded_rectangle((2440,525,3720,1760),radius=30,fill=(11,20,31),outline=(36,59,77),width=2)
        text(d,(165,565),'RECORDED CONNECTION MEMORY',36,CYAN,True)
        text(d,(165,625),'144 used features of 256  ·  8 learning records  ·  fixed 256-channel decoder',29,MUTED)
        text(d,(185,730),'BINARY STATE CUE',27,MUTED,True)
        text(d,(990,730),'PLASTIC FNP3 [1,3,0]',27,MUTED,True)
        text(d,(1740,730),'FNP3 [2,1,0] · STATE CODES',25,MUTED,True)
        text(d,(2500,570),'WITH NATIVE LEARNING',35,GREEN,True)
        text(d,(2500,1150),'LEARNING FROZEN',35,AMBER,True)
        for x,label,value,color in [(120,'CHANGED ASSOCIATIONS','10 / 10 recalled',GREEN),(1040,'UNCHANGED ASSOCIATIONS','30 / 30 retained',CYAN),(1960,'LEARNED TRAJECTORIES','1,280 / 1,280 states',GREEN),(2880,'FROZEN + UNLABELLED','Each 309 / 1,280',AMBER)]:
            d.rounded_rectangle((x,1810,x+840,1930),radius=19,fill=(14,25,36),outline=DIM,width=2);text(d,(x+28,1826),label,25,MUTED,True);text(d,(x+28,1865),value,39,color,True)
        text(d,(120,1975),'Five initializations · engineered prior · externally taught destination classes · one change episode · one prototype per class',30,MUTED)
        text(d,(120,2026),'Measured outputs + selected-connection SRAM readbacks. Class activity decoded from returned codes; no electrical spike trace.',28,MUTED)
        text(d,(120,2074),'Geometry, camera, flow animation and expanded playback are schematic. Finite queue; CPU table also passes. No efficiency claim.',28,MUTED)
        text(d,(120,2120),'Separate two-pass hardware-generated replay failed during configuration; not shown as working. Earlier FNP2 issue remains open.',26,MUTED)
        # Every displayed feature and connection has an exact archived index.
        self.features=np.zeros((256,3));self.used=[];self.group=np.full(256,-1,int);self.groups=[]
        common=np.flatnonzero(self.codes.sum(0)==8);self.groups.append(common)
        for s in range(8):self.groups.append(np.flatnonzero((self.codes[s]>0)&(self.codes.sum(0)==1)))
        self.centers=[]
        for g,ids in enumerate(self.groups):
            center=np.array([-2.06,.0,-.15]) if g==0 else np.array([-1.56+.28*((g-1)%2),1.04-.64*((g-1)//2),-.14+.28*((g-1)%2)])
            self.centers.append(center)
            for j,b in enumerate(ids):
                yy=1-2*(j+.5)/len(ids);r=math.sqrt(1-yy*yy);theta=j*math.pi*(3-math.sqrt(5));self.features[b]=center+np.array([.13*r*math.cos(theta),.13*yy,.23*r*math.sin(theta)]);self.used.append(int(b));self.group[b]=g-1
        self.neurons=np.array([[-.15+.34*(j%2),1.06-.31*j,-.10+.20*(j%2)] for j in range(8)])
        self.decoded=np.array([[1.45,1.06-.31*j,.05] for j in range(8)])
        order=[0]
        while len(order)<8:order.append(data['old'][order[-1]])
        self.order=order;self.graphpos={}
        for i,s in enumerate(order):a=-math.pi/2+2*math.pi*i/8;self.graphpos[s]=np.array([165*math.cos(a),165*math.sin(a)])
    def project(self,p,t):return Scene.project(self,p,t,center=(1270,1170),scale=440)
    def state(self,y):return next((j for j in range(8) if np.array_equal(y,self.codes[j])),None)
    def graph(self,d,center,connections,current,previous,color,t,history):
        positions={s:tuple(np.array(center)+p*(.91 if center[1]>1000 else 1)) for s,p in self.graphpos.items()}
        for s in range(8):
            active=np.flatnonzero(self.codes[s]@connections.T>24)
            for dest in active:
                a=np.array(positions[s]);b=np.array(positions[int(dest)]);v=b-a;length=float(np.linalg.norm(v));unit=v/length;aa=a+unit*32;bb=b-unit*34;col=GREEN if dest!=self.data['old'][s] else blend(BG,color,.43);hot=s==previous and dest==current
                d.line((tuple(aa),tuple(bb)),fill=color if hot else col,width=7 if hot else 3);side=np.array([-unit[1],unit[0]]);tip=bb;d.polygon([tuple(tip),tuple(tip-unit*17+side*9),tuple(tip-unit*17-side*9)],fill=color if hot else col)
                if hot:
                    p=aa+(bb-aa)*((t*1.7)%1);d.ellipse((p[0]-8,p[1]-8,p[0]+8,p[1]+8),fill=INK)
        for s,p in positions.items():
            lit=s==current;radius=34 if lit else 26;Scene.orb(d,*p,radius,color if lit else DIM,lit);text(d,(p[0]-18,p[1]-16),f'S{s}',27,INK if lit else MUTED,lit)
        if history:
            start=max(0,len(history)-7);value=' → '.join(f'S{x}' for x in history[start:]);text(d,(center[0]-500,center[1]+(212 if center[1]>1000 else 230)),value,30,color,True)
    def frame(self,t):
        a=self.data['arms']['native'];f=self.data['arms']['frozen'];changed=self.data['changed'];teacher=None;step=None;native_history=[];frozen_history=[];previous=None;fp=None
        if t<3:
            s=changed[min(1,int(t/1.5))];i=8+s;x=self.partial[s];y=np.array(a['outputs'][i]);fy=np.array(f['outputs'][i]);conn=self.initial;phase='BEFORE TEACHING · PARTIAL-CUE QUERY';sub=f'Both models still predict the old transition from S{s}';mode='Host supplies the same incomplete cue';previous=fp=s
        elif t<7:
            k=0 if t<5 else 1;s=changed[k];i=16+k;teacher=self.data['new'][s];x=self.codes[s];y=np.array(a['outputs'][i]);fy=np.array(f['outputs'][i]);after=t>=(3 if k==0 else 5)+.8;ci=i if after else i-1;conn=np.array(a['connections'][ci]);phase=f'TEACH CHANGE {k+1}/2 · S{s} → S{teacher}';sub='External example supplies the observed destination class';mode='Native update; response shown before that edit';previous=fp=s
        elif t<10:
            s=changed[min(1,int((t-7)/1.5))];i=18+s;x=self.partial[s];y=np.array(a['outputs'][i]);fy=np.array(f['outputs'][i]);conn=np.array(a['connections'][17]);phase='QUERY AGAIN · NO TEACHING LABEL';sub=f'Same incomplete S{s} cue · changed prediction after learning';mode='Learning remains enabled; visible memory unchanged';previous=fp=s
        else:
            start=changed[0];step=min(31,int((t-10)*2));x=self.partial[start] if step==0 else np.array(a['rollout'][start][step-1]);y=np.array(a['rollout'][start][step]);fy=np.array(f['rollout'][start][step]);conn=np.array(a['connections'][17]);phase=f'CHIP-FED PREDICTION · STEP {step+1:02d} / 32';sub='The preceding chip output supplies the next input in device SRAM';mode='Worker asleep · zero reads / writes after release';native_history=[start]+[self.state(v) for v in a['rollout'][start][:step+1]];frozen_history=[start]+[self.state(v) for v in f['rollout'][start][:step+1]];previous=native_history[-2];fp=frozen_history[-2]
        ns,fs=self.state(y),self.state(fy);edits=int(((conn==1)&(self.initial==0)).sum());im=self.base.copy();d=ImageDraw.Draw(im)
        text(d,(160,353),phase,34,GREEN if step is not None else AMBER if teacher is not None else CYAN,True);text(d,(160,412),sub,31,MUTED)
        text(d,(2480,353),'CONTROL WORKER / TEACHER',29,CYAN,True);text(d,(2480,412),mode,28,MUTED)
        p=self.project(self.features,t);n=self.project(self.neurons,t);r=self.project(self.decoded,t)
        for j in range(8):
            for b in np.flatnonzero(conn[j]):
                hot=bool(x[b]);added=bool(conn[j,b] and not self.initial[j,b]);col=GREEN if added else COLORS[self.group[b]] if self.group[b]>=0 else CYAN
                Scene.edge(d,p[b],n[j],col if hot else blend(BG,col,.23),3 if hot else 1)
                # Presentation motion traces the measured selected synapse.
                if hot and (j==ns or j==teacher):
                    q=p[b,:2]+(n[j,:2]-p[b,:2])*((t*1.2+b*.037)%1);d.ellipse((q[0]-3,q[1]-3,q[0]+3,q[1]+3),fill=INK)
            if ns==j:Scene.edge(d,n[j],r[j],CYAN,5)
            else:Scene.edge(d,n[j],r[j],blend(BG,CYAN,.2),2)
        for b in sorted(self.used,key=lambda b:p[b,2]):
            xx,yy,fac=p[b];hot=bool(x[b]);rad=(6 if hot else 3)*fac;col=COLORS[self.group[b]] if self.group[b]>=0 else CYAN;d.ellipse((xx-rad,yy-rad,xx+rad,yy+rad),fill=col if hot else DIM)
        for g,center in enumerate(self.centers):
            xx,yy,_=self.project(np.asarray([center]),t)[0];lx=xx+(-34 if g==0 else -75 if (g-1)%2==0 else 25);ly=yy+65;d.rounded_rectangle((lx-7,ly-3,lx+(90 if g==0 else 44),ly+32),radius=6,fill=(11,20,31));text(d,(lx,ly),'shared' if g==0 else f'S{g-1}',26,MUTED if g==0 else COLORS[g-1],True)
        for j in range(8):
            xx,yy,fac=n[j];lit=ns==j;Scene.orb(d,xx,yy,21*fac,CYAN if lit else DIM,lit)
            if teacher==j:d.ellipse((xx-34,yy-34,xx+34,yy+34),outline=AMBER,width=5)
            d.rounded_rectangle((xx+31,yy-22,xx+84,yy+18),radius=5,fill=(11,20,31));text(d,(xx+36,yy-17),f'M{j}',27,AMBER if teacher==j else INK if lit else MUTED,True)
            xx,yy,fac=r[j];col=COLORS[j];d.rounded_rectangle((xx-53,yy-24,xx+53,yy+24),radius=10,fill=blend(BG,col,.7) if lit else (23,36,48),outline=col if lit else DIM,width=2);text(d,(xx-18,yy-16),f'S{j}',28,INK if lit else MUTED,lit)
        label='Response before edit' if teacher is not None else 'Returned next state'
        text(d,(1720,1640),label,27,MUTED);text(d,(1720,1680),'No response' if ns is None else f'S{ns} · 32 active bits',34,CYAN,True)

        if step is not None:
            path=[(2220,1495),(2220,1585),(365,1585),(365,1470)]
            for j in range(3):d.line((path[j],path[j+1]),fill=CYAN,width=3)
            d.polygon([(355,1484),(375,1484),(365,1468)],fill=CYAN)
            text(d,(690,1592),'DEVICE SRAM · returned code → next queued input',25,CYAN)
        text(d,(165,1640),f'{edits} verified connection replacements',36,GREEN,True)
        text(d,(165,1685),'Selected synapses in SRAM; camera is schematic',28,MUTED)
        text(d,(165,1720),'Amber ring: teacher-selected record   ·   Cyan: decoded response',27,MUTED)
        self.graph(d,(3080,880),conn,ns,previous,GREEN,t,native_history or ([previous,ns] if ns is not None else []))
        self.graph(d,(3080,1438),self.initial,fs,fp,AMBER,t,frozen_history or ([fp,fs] if fs is not None else []))
        text(d,(2500,620),'Stored successor map · changed connections in green',27,MUTED)
        text(d,(2500,1200),'Same examples and seed · old transition memory retained',27,MUTED)
        if step is not None:
            for i in range(32):x0=2500+i*35;d.rectangle((x0,1696,x0+28,1705),fill=GREEN if i<=step else DIM)
            text(d,(2500,1720),'Finite queue · no labels during rollout',24,MUTED)
        return im
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);data=source();(a.out/'recorded-events.json').write_text(json.dumps(data,separators=(',',':'))+'\n');film=Film(data)
    for name,t in [('before',1),('first-edit',4.5),('second-edit',6.5),('recall',8),('loop',14.6),('final',25.8)]:film.frame(t).save(a.out/(name+'.png'))
    if a.video:
        proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','h264_videotoolbox','-b:v','24M','-profile:v','high','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(a.out/'akd-learned-transition-4k60.mp4')],stdin=subprocess.PIPE);start=time.monotonic()
        try:
            for i in range(DURATION*FPS):
                proc.stdin.write(film.frame(i/FPS).tobytes())
                if i%300==0:print(json.dumps(dict(frame=i,total=DURATION*FPS,elapsed_s=round(time.monotonic()-start,1))),flush=True)
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(a.out/'akd-learned-transition-4k60.mp4'),'-vf','scale=1920:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out/'akd-learned-transition-1080p60.mp4')],check=True)
    (a.out/'SOURCE_BINDING.json').write_text(json.dumps(dict(status='AUDITED_LEARNED_TRANSITION_REPLAY',archive_sha256=data['archive_sha256'],events_sha256=sha(a.out/'recorded-events.json'),renderer_sha256=sha(__file__),scope=data['scope'],electrical_spike_times_observed=False,files={p.name:sha(p) for p in a.out.iterdir() if p.suffix in ['.png','.mp4']}),indent=2)+'\n')
    print(json.dumps(dict(status='RENDERED',out=str(a.out))))
if __name__=='__main__':main()
