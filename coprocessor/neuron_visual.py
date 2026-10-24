"""Recorded native responses and connection masks, drawn as a schematic 3D graph.

No spike times, neuron-to-neuron links, or physical coordinates are synthesized.
The camera and transitions are presentation motion. All visible edges, input
states, numeric responses, and learning changes come from the public archive.
"""
from pathlib import Path
import argparse, hashlib, json, math, subprocess, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from coprocessor.window_archive import BASE, P, audit_public, decode

W, H, FPS, DURATION = 3840, 2160, 60, 24
BG = (7, 12, 20)
INK = (231, 240, 248)
MUTED = (139, 161, 181)
CYAN = (102, 207, 237)
GREEN = (112, 231, 181)
AMBER = (237, 177, 104)
DIM = (39, 58, 76)
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
if not Path(FONT).exists():
    FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(n, bold=False):
    if (n, bold) not in FONTS:
        FONTS[n, bold] = ImageFont.truetype(BOLD if bold else FONT, n)
    return FONTS[n, bold]


def text(d, xy, value, n=34, color=INK, bold=False):
    d.text(tuple(map(round, xy)), str(value), font=font(n, bold), fill=color)


def blend(a, b, f):
    return tuple(round(x + (y-x)*f) for x, y in zip(a, b))


def source_data():
    audit = audit_public()
    meta = json.loads(BASE.with_suffix('.json').read_text())
    seed = 107101
    f = meta['fixtures'][f's{seed}']
    ids = [r['target'] for r in f['rows']]
    with np.load(BASE.with_suffix('.npz'), allow_pickle=False) as z:
        initial = z[f's{seed}__initial'].copy()
        native = f's{seed}_window_n8'
        frozen = f's{seed}_frozen_n8'
        states = decode(z, native, initial, 16)
        controls = decode(z, frozen, initial, 16)
        ys = z[native+'__pre'].reshape(8, f['tile_records'])
        fy = z[frozen+'__pre'].reshape(8, f['tile_records'])
        qs = z[native+'__recall'].reshape(8, f['records'])
        fq = z[frozen+'__recall'].reshape(8, f['records'])
        xs = z[f's{seed}__inputs'].copy()
        def masks(a):
            words = a.reshape(f['records'], f['record_words'])[ids, 4:].copy()
            return np.unpackbits(words.view('u1'), axis=1, bitorder='little').tolist()
        events = []
        before = initial
        for i, row in enumerate(f['rows']):
            after = states[2*i]
            bm, am = masks(before), masks(after)
            assert np.array_equal(states[2*i], states[2*i+1])
            assert np.array_equal(controls[2*i], initial)
            assert np.array_equal(controls[2*i+1], initial)
            added = np.flatnonzero(np.array(am[i])-np.array(bm[i]) == 1).tolist()
            removed = np.flatnonzero(np.array(bm[i])-np.array(am[i]) == 1).tolist()
            assert added == [row['added_bit']] and removed == [row['removed_bit']]
            assert all(sum(mask) == 32 for mask in bm+am)
            assert sum(xs[i]) == 32
            assert int(qs[i, ids[i]]) == 32 and int(fq[i, ids[i]]) == 31
            assert np.flatnonzero(np.any(before.reshape(f['records'], -1) != after.reshape(f['records'], -1), axis=1)).tolist() == [ids[i]]
            pre, fp = [], []
            for target in ids:
                eligible = target//f['tile_records'] == row['label']
                pre.append(int(ys[i, target % f['tile_records']]) if eligible else None)
                fp.append(int(fy[i, target % f['tile_records']]) if eligible else None)
            events.append(dict(index=i, target=ids[i], teacher_label=row['label'],
                               input=xs[i].tolist(), before=bm, after=am,
                               frozen=masks(initial), response=pre, frozen_response=fp,
                               recall=qs[i, ids].tolist(), frozen_recall=fq[i, ids].tolist(),
                               added=added, removed=removed))
            before = states[2*i+1]
    return dict(seed=seed, records=ids, events=events, audit=audit,
                archive_sha256=sha(BASE.with_suffix('.npz')),
                metadata_sha256=sha(BASE.with_suffix('.json')))


class Scene:
    def __init__(self, data):
        self.data = data
        # Feature coordinates are stable, arbitrary schematic positions.
        j = np.arange(512)
        yy = 1-2*(j+.5)/512
        rr = np.sqrt(1-yy*yy)
        theta = j*math.pi*(3-math.sqrt(5))
        self.features = np.column_stack((-1.15+.28*rr*np.cos(theta),
                                         yy*1.12,
                                         .82*rr*np.sin(theta)))
        # Eight actual sampled records; class0 and class1 are spatially separated.
        self.units = np.array([[.44+(i%2)*.91, .93-(i//2)*.62,
                                .22 if row['teacher_label'] else -.22]
                               for i,row in enumerate(data['events'])])
        self.base = Image.new('RGB', (W,H), BG)
        d = ImageDraw.Draw(self.base)
        text(d,(120,75),'Inside native learning',100,bold=True)
        text(d,(125,211),'Recorded AKD1500 responses + connection readbacks',40,MUTED)
        d.line((120,294,3720,294),fill=(35,51,68),width=2)
        d.rounded_rectangle((110,340,2580,1730),radius=34,fill=(11,20,31),outline=(34,56,73),width=2)
        d.rounded_rectangle((2640,340,3730,1730),radius=34,fill=(15,21,29),outline=(59,51,40),width=2)
        text(d,(165,385),'NATIVE LEARNING',42,CYAN,True)
        text(d,(2700,385),'FROZEN STILL RESPONDS',40,AMBER,True)
        text(d,(165,455),'Eight sampled learning records · 512 binary input channels',34,MUTED)
        text(d,(2700,455),'Same cue · learning disabled',32,MUTED)
        text(d,(120,1953),'One FNP2 learner · records in Nano DDR · host selects the teacher class view',33,MUTED)
        text(d,(120,2012),'Recorded values and connections. Camera, geometry and expanded playback timing are schematic.',32,MUTED)
        text(d,(120,2067),'No electrical spike trace · no neuron-to-neuron links shown · no CPU or energy advantage established',29,MUTED)

    def project(self, points, t, center=(1280,1080), scale=605):
        a=.35*math.sin(t/24*math.pi*1.1)-.16
        b=.045*math.sin(t/24*math.pi*2)
        ry=np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
        rx=np.array([[1,0,0],[0,math.cos(b),-math.sin(b)],[0,math.sin(b),math.cos(b)]])
        q=points@ry.T@rx.T
        fac=4.8/(4.8-q[:,2])
        return np.column_stack((center[0]+q[:,0]*scale*fac,
                                center[1]-q[:,1]*305*fac,
                                fac))

    @staticmethod
    def edge(d, p, q, color, width=2, amount=1):
        # A curved schematic cable. Endpoints are the actual input and record.
        p=np.array(p[:2]);q=np.array(q[:2])
        c1=p+np.array([(q[0]-p[0])*.39,0])
        c2=q-np.array([(q[0]-p[0])*.31,0])
        n=22
        points=[]
        for u in np.linspace(0,max(.001,amount),n):
            v=(1-u)**3*p+3*(1-u)**2*u*c1+3*(1-u)*u*u*c2+u**3*q
            points.append(tuple(v))
        d.line(points,fill=color,width=width,joint='curve')

    @staticmethod
    def orb(d,x,y,r,color,hot=False):
        if hot:
            for radius,mix in [(r*1.65,.08),(r*1.35,.12),(r*1.12,.22)]:
                d.ellipse((x-radius,y-radius,x+radius,y+radius),fill=blend(BG,color,mix))
        for k in range(9,0,-1):
            radius=r*k/9
            c=blend((12,25,38),color,.22+.68*(1-k/10))
            dx=-r*.18*(1-k/9);dy=-r*.18*(1-k/9)
            d.ellipse((x-radius+dx,y-radius+dy,x+radius+dx,y+radius+dy),fill=c)
        d.ellipse((x-r,y-r,x+r,y+r),outline=blend(color,INK,.3),width=2)

    def frame(self,t):
        # First update is held for ten seconds; seven following updates get two seconds each.
        i=0 if t<10 else min(7,1+int((t-10)/2))
        elapsed=t if i==0 else (t-10)%2
        fraction=elapsed/(10 if i==0 else 2)
        phase=0 if fraction<.34 else 1 if fraction<.65 else 2
        e=self.data['events'][i]; post=phase>=1
        im=self.base.copy();d=ImageDraw.Draw(im)
        fp=self.project(self.features,t);np_=self.project(self.units,t)
        mask=np.array(e['after' if post else 'before'])
        inputs=np.array(e['input'])
        scores=e['recall'] if phase==2 else e['response']
        captions=['Cue arrives · recorded native response','Native connection edit · verified in RAM','Same cue returns · read-only recall']
        text(d,(165,535),captions[phase],45,INK,True)
        text(d,(2700,535),'Responds to every new cue',37,INK,True)
        # Background edges are actual recorded connection bits of the other sampled records.
        for j in range(8):
            if j==i:continue
            for bit in np.flatnonzero(mask[j]):
                self.edge(d,fp[bit],np_[j],(26,53,67) if inputs[bit] else (20,34,46),2)
        # All 512 input channels exist in this view; active channels encode the saved cue.
        for k in np.argsort(fp[:,2]):
            x,y,factor=fp[k];r=4.8*factor if inputs[k] else 2.5*factor
            d.ellipse((x-r,y-r,x+r,y+r),fill=CYAN if inputs[k] else (41,58,74))
        # The current winner has exactly32 recorded connections, before and after.
        for bit in np.flatnonzero(mask[i]):
            c=CYAN if inputs[bit] else AMBER
            if bit in e['added'] and post:c=GREEN
            if bit in e['added'] and post:
                self.edge(d,fp[bit],np_[i],blend(BG,GREEN,.17),22)
                self.edge(d,fp[bit],np_[i],GREEN,8)
                px,py,_=fp[bit]
                d.ellipse((px-14,py-14,px+14,py+14),outline=GREEN,width=4)
            else:self.edge(d,fp[bit],np_[i],blend(BG,c,.85),4 if inputs[bit] else 6)
        if post:
            for bit in e['removed']:
                # A marked, fading old edge is explicitly the removed RAM bit.
                if phase==1:self.edge(d,fp[bit],np_[i],blend(BG,AMBER,.35),3)
        for j in np.argsort(np_[:,2]):
            x,y,fac=np_[j];s=scores[j]
            strength=0 if s is None else s/32
            hue=GREEN if j==i and phase==2 else CYAN
            col=blend(DIM,hue,strength**.6) if s is not None else DIM
            radius=(17+19*strength)*fac
            self.orb(d,x,y,radius,col,j==i)
            text(d,(x+55,y-29),f'R{self.data["records"][j]}',31,col,j==i)
            text(d,(x+55,y+13),f'{s} matches' if s is not None else 'not queried',26,MUTED)
        text(d,(330,1510),'BINARY INPUT CHANNELS',30,MUTED)
        text(d,(1670,1510),'SAMPLED NEURON RECORDS',30,MUTED)
        # Frozen neurons are not inactive: show the same saved input and all eight
        # sampled scores from the matched frozen recording, on the same phase.
        # No scan cursor or invented firing times are added to the binary input.
        text(d,(2700,650),'EXACT SAME RECORDED INPUT',29,MUTED,True)
        for bit in range(512):
            x=2750+(bit%32)*27;y=710+(bit//32)*15
            radius=5 if inputs[bit] else 3
            d.ellipse((x-radius,y-radius,x+radius,y+radius),fill=CYAN if inputs[bit] else (37,52,65))
        text(d,(2700,1000),'RECORDED NEURON RESPONSES',29,MUTED,True)
        fscores=e['frozen_recall'] if phase==2 else e['frozen_response']
        for j,score in enumerate(fscores):
            by=1070+j*59
            if j==i:
                d.rounded_rectangle((2685,by-10,3685,by+46),radius=10,fill=(35,31,26),outline=(97,73,45),width=2)
            text(d,(2700,by),f'R{self.data["records"][j]}',29,AMBER if j==i else MUTED,j==i)
            d.rounded_rectangle((2880,by+9,3504,by+28),radius=6,fill=(27,41,53))
            if score is not None and score:
                d.rounded_rectangle((2880,by+9,2880+score*19.5,by+28),radius=6,fill=AMBER if j==i else blend(DIM,CYAN,(score/32)**.5))
            text(d,(3540,by),score if score is not None else '—',29,AMBER if j==i else MUTED,j==i)
            if j==i and phase==2:
                # True scale: the native recall's32-count response is one count
                # beyond this frozen31-count response, not an exaggerated gap.
                xx=2880+e['recall'][i]*19.5
                d.line((xx,by+2,xx,by+36),fill=GREEN,width=4)
        fs=e['frozen_recall'][i] if phase==2 else e['frozen_response'][i]
        text(d,(2700,1580),f'Frozen {fs} / 32  ·  native {e["recall"][i]} / 32' if phase==2 else f'Teaching response: {fs} / 32',37,AMBER,True)
        text(d,(2700,1640),'RAM verified: zero connection edits',31,MUTED)
        # Large, direct view of the one measured missing/replaced edge.
        removed=e['removed'][0];added=e['added'][0]
        bx,by=180,1600
        text(d,(bx,by),f'INPUT {removed}',28,AMBER,True)
        d.line((bx+170,by+20,bx+410,by+20),fill=AMBER if not post else (78,58,39),width=6)
        if post:
            d.line((bx+270,by-1,bx+305,by+37),fill=AMBER,width=5)
            d.line((bx+270,by+37,bx+305,by-1),fill=AMBER,width=5)
        text(d,(bx+435,by), 'REMOVED' if post else 'CONNECTED',28,AMBER if post else MUTED)
        bx=1020
        text(d,(bx,by),f'INPUT {added}',28,GREEN if post else CYAN,True)
        d.line((bx+165,by+20,bx+410,by+20),fill=GREEN if post else (37,66,80),width=6)
        text(d,(bx+435,by),'CONNECTED' if post else 'MISSING',28,GREEN if post else MUTED)
        text(d,(1950,by),f'Recall {e["recall"][i]} / 32' if phase==2 else 'Teaching response 31 / 32',32,GREEN if phase==2 else INK,True)
        # Compact event strip makes succession visible without a cut or text-only ending.
        text(d,(120,1790),f'RECORDED UPDATE  {i+1:02d} / 08',35,CYAN,True)
        text(d,(850,1790),f'Teacher class {e["teacher_label"]}  ·  selected R{e["target"]}',36,INK)
        text(d,(120,1850),'24 selected updates / 3 initializations verified · this film replays one eight-update run',33,MUTED)
        text(d,(2750,1790),'HOST WEIGHT WRITES',31,MUTED)
        text(d,(2750,1840),'0 after initialization',40,GREEN,True)
        d.rectangle((120,1910,3720,1916),fill=(30,46,61))
        d.rectangle((120,1910,120+int(3600*t/DURATION),1916),fill=CYAN)
        return im


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video',action='store_true')
    p.add_argument('--out',type=Path,default=P/'visual/neuron-learning')
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    data=source_data()
    (out/'recorded-events.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
    scene=Scene(data)
    for name,t in [('opening',1),('response',3),('edit',5),('recall',8),('later',17.5),('final',23.9)]:
        scene.frame(t).save(out/(name+'.png'))
    if args.video:
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','h264_videotoolbox','-b:v','26M','-profile:v','high','-pix_fmt','yuv420p','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(out/'akd-native-connections-4k60.mp4')]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        start=time.monotonic()
        try:
            for i in range(FPS*DURATION):
                proc.stdin.write(scene.frame(i/FPS).tobytes())
                if i%120==0:print(json.dumps(dict(frame=i,total=FPS*DURATION,elapsed=round(time.monotonic()-start,1))),flush=True)
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(out/'akd-native-connections-4k60.mp4'),'-vf','scale=1920:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'akd-native-connections-1080p60.mp4')],check=True)
    binding=dict(status='RECORDED_RESPONSE_AND_CONNECTION_REPLAY',archive_sha256=data['archive_sha256'],metadata_sha256=data['metadata_sha256'],renderer_sha256=sha(__file__),events_sha256=sha(out/'recorded-events.json'),shown_seed=data['seed'],shown_updates=8,qualified_updates=24,shown_record_ids=data['records'],scope='Selected single-swap updates in one host-scheduled FNP2 learning layer; 512 binary input channels; eight sampled learning records of11264allocated; records in Nano DDR',not_observed=['Electrical spike timing','Neuron-to-neuron communication in this experiment','Physical neuron coordinates','Continuous arbitrary-stream learning','CPU or energy advantage'],motion='Camera and expanded phase timing schematic; no generated numeric responses or extra connections',outputs={q.name:sha(q) for q in out.iterdir() if q.suffix in ['.png','.mp4']})
    (out/'SOURCE_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
    print(json.dumps(dict(status='RENDERED',out=str(out))),flush=True)


if __name__=='__main__':main()
