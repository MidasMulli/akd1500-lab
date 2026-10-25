"""Continuous measured correction replay; camera and geometry are schematic."""
from pathlib import Path
import argparse,json,math,subprocess,time
import numpy as np
from PIL import Image,ImageDraw
from admission.correction_archive import BASE,audit
from admission.recycle import simulate,PLASTIC,SEEDS,D,U
from admission.run import masks,sha
from coprocessor.neuron_visual import text,blend,Scene,BG,INK,MUTED,CYAN,GREEN,AMBER,DIM
W,H,FPS,DURATION=3840,2160,60,30

def source():
    summary=audit();m=json.loads(BASE.with_suffix('.json').read_text());fam=m['families']['correction'];data=dict(seed=SEEDS[0],archive_sha256=m['archive_sha256'],summary={k:summary[k] for k in ['primary','cpu']},arms={})
    with np.load(BASE.with_suffix('.npz'),allow_pickle=False) as z:
        for arm in ['native','no_retire','frozen']:
            r=fam['runs'][f'native_s{SEEDS[0]}_{arm}_repeat'];f=fam['fixtures'][f's{SEEDS[0]}_{arm}'];get=lambda k:z[r['arrays'][k]['key']]
            if arm!='native':data['arms'][arm]=dict(outputs=get('scores').reshape(-1,128).tolist());continue
            initial=z[f['arrays']['bank.ram']['key']];a=simulate(initial,'repeat',arm);conn=masks(get('states').reshape(-1,len(initial)),D,U)[:,PLASTIC,:110];before=masks(initial,D,U)[PLASTIC,:110]
            data['arms'][arm]=dict(events=a['events'],inputs=get('inputs').reshape(-1,128).tolist(),outputs=get('scores').reshape(-1,128).tolist(),features=a['features'][:,:110].tolist(),initial=before.tolist(),connections=conn.tolist())
    data['scope']='Recorded returned binary responses, native output-source pointers and connection readbacks. Fixed features derived from captured inputs. One eight-record learning bank, 24 helpers, five sequential FNP3s. Host paces query/positive/negative roles, supplies external actual/complement labels. Designed comparator and retirement coding. No host error decision, record selection, reset or weight repair. Camera, geometry, glow and replay timing schematic; not physical layout or electrical spikes. CPU dictionary also passes; no efficiency claim.'
    return data

def pred(y):return '0 + 1' if y[84] and y[85] else '0' if y[84] else '1' if y[85] else 'missing'

class Film:
    def __init__(self,data):
        self.data=data;self.a=data['arms']['native'];self.base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(self.base)
        text(d,(110,65),'Correct the memory. Keep the rest.',90,INK,True)
        text(d,(115,190),'AKD1500 · native prediction-error gating + contradictory teaching',39,MUTED)
        text(d,(2980,96),'RECORDED CHIP REPLAY',30,CYAN,True)
        text(d,(2980,151),'Three initializations · 12 reversals each',25,MUTED)
        d.line((115,275,3725,275),fill=DIM,width=2)
        names=['Compare prediction / actual','Gate positive + negative roles','Expand binary features','Native plastic memory','Relay prediction + raw key']
        ids=['[1,3,0]','[2,1,0]','[2,2,0]','[2,3,0]','[3,1,0]']
        for k,(name,npid) in enumerate(zip(names,ids)):
            x=115+k*730;d.rounded_rectangle((x,310,x+690,422),radius=15,fill=(15,26,39),outline=GREEN if k==3 else DIM,width=2)
            text(d,(x+21,326),name,28,GREEN if k==3 else CYAN,True);text(d,(x+21,375),f'FNP3 {npid}'+(' · 8 plastic + 24 helpers' if k==3 else ' · fixed'),25,MUTED)
        for box in [(115,470,2430,1680),(2475,470,3725,1680)]:d.rounded_rectangle(box,radius=26,fill=(11,20,31),outline=DIM,width=2)
        text(d,(165,510),'MEASURED CONNECTIONS + RETURNED RESPONSES',29,CYAN,True)
        text(d,(165,558),'One learning NPU · eight records · 32 connections per record',28,MUTED)
        text(d,(165,1545),'Green: added connection     Amber: retired connection     Cyan: returned response',27,MUTED)
        text(d,(165,1595),'Edges show stored state after update; responses were returned before that update.',25,MUTED)
        text(d,(2520,510),'SAME INPUTS · MATCHED NATIVE CONTROLS',29,CYAN,True)
        primary=data['summary']['primary'];cpu=data['summary']['cpu']
        for x,label,v,color in [(115,'CORRECTION + RETIREMENT',primary['native'],GREEN),(1040,'RETIREMENT PATH REMOVED',primary['no_retire'],AMBER),(1965,'TRAINED THEN FROZEN',primary['frozen'],AMBER),(2890,'CPU LATEST-FACT DICTIONARY',cpu,CYAN)]:
            d.rounded_rectangle((x,1730,x+835,1865),radius=20,fill=(15,27,40),outline=DIM,width=2)
            text(d,(x+24,1747),label,25,MUTED,True);text(d,(x+24,1792),f'{v["probe_correct"]} / {v["probe_total"]}',44,color,True)
        text(d,(115,1905),'Three initial connection seeds · same fixed four-context stream · 48 read-only probes after each reversal.',29,MUTED)
        text(d,(115,1952),'Two unrelated original memories stay byte-identical. Bounded eight-record bank; no resets, weight repairs or program reloads.',28,MUTED)
        text(d,(115,1999),'Host-paced external outcomes + fixed teaching roles. Error gate, shared features and retirement code are designed.',28,MUTED)
        text(d,(115,2046),'Recorded binary outputs + connection RAM; intermediate features derived. Camera, geometry, glow and timing are schematic.',27,MUTED)
        text(d,(115,2093),'Selected slowed replay; not an electrical spike trace. CPU also passes. General eviction, autonomy and efficiency remain unproven.',27,MUTED)
        j=np.arange(110);yy=1-2*(j+.5)/110;rr=np.sqrt(1-yy*yy);theta=j*math.pi*(3-math.sqrt(5))
        self.points=np.column_stack((-.85+.38*rr*np.cos(theta),1.12*yy,.65*rr*np.sin(theta)))
        self.units=np.array([[.6+(i//4)*1.0,1.05-(i%4)*.7,.25 if i//4 else -.25] for i in range(8)])
        self.lookup={};self.query=[];q=0
        for i,e in enumerate(self.a['events']):
            if not e['teacher']:q=i
            self.query.append(q)
        self.initial=np.asarray(self.a['initial']);self.conn=np.asarray(self.a['connections']);self.features=np.asarray(self.a['features'])

    def choose(self,t):
        segments=[(0,4,list(range(0,18))+list(range(36,40))),(4,10,list(range(52,68))),(10,16,list(range(112,128))),
                  (16,22,list(range(172,188))),(22,26,list(range(712,728))),(26,30,list(range(724,772)))]
        for lo,hi,ids in segments:
            if t<hi:return ids[min(len(ids)-1,int((t-lo)/(hi-lo)*len(ids))) ]
        return 771

    def frame(self,t):
        i=self.choose(t);e=self.a['events'][i];qi=self.query[i];q=self.a['outputs'][qi];y=self.a['outputs'][i]
        im=self.base.copy();d=ImageDraw.Draw(im);after=self.conn[i];before=self.conn[i-1] if i else self.initial
        adds=(after==1)&(before==0);removes=(after==0)&(before==1);changed=(adds|removes).any(1)
        phase='QUERY · outcome not yet supplied' if not e['teacher'] else 'NEGATIVE EVIDENCE · retire contradicted response' if e['erase'] else 'POSITIVE EVIDENCE · learn current outcome'
        if e['phase'].startswith('probe_'):phase='READ-ONLY PROBE · no teaching outcome supplied'
        cycle=int(e['phase'].split('_')[-1]) if e['phase'].startswith(('probe_','change_')) else 0
        text(d,(165,624),phase,33,GREEN if e['teacher'] else INK,True)
        text(d,(165,681),f'Key {e["key"]:04b}    ·    reversal {cycle} / 12    ·    event {i+1} / 772',30,MUTED)
        fp=Scene.project(self,self.points,t,center=(1000,1125),scale=530);up=Scene.project(self,self.units,t,center=(1000,1125),scale=530)
        f=self.features[i];responses=np.asarray(y[72:80]);curve=np.linspace(0,1,12)[:,None]
        def edge(p,q,color,width):
            p=p[:2];q=q[:2];c1=p+np.array([(q[0]-p[0])*.42,0]);c2=q-np.array([(q[0]-p[0])*.32,0]);v=(1-curve)**3*p+3*(1-curve)**2*curve*c1+3*(1-curve)*curve**2*c2+curve**3*q
            d.line([tuple(vv) for vv in v],fill=color,width=width,joint='curve')
        for j in range(8):
            for bit in np.flatnonzero(before[j]):edge(fp[bit],up[j],blend(BG,CYAN,.33 if f[bit] and responses[j] else .14),2)
            for bit in np.flatnonzero(removes[j]):edge(fp[bit],up[j],AMBER,5)
            for bit in np.flatnonzero(adds[j]):edge(fp[bit],up[j],GREEN,6)
        for bit in np.argsort(fp[:,2]):
            x,yy,fac=fp[bit];r=(5 if f[bit] else 3)*fac;col=GREEN if bit==109 and f[bit] else CYAN if f[bit] else DIM
            d.ellipse((x-r,yy-r,x+r,yy+r),fill=col)
        for j in range(8):
            x,yy,fac=up[j];Scene.orb(d,x,yy,23*fac,GREEN if changed[j] else CYAN if responses[j] else DIM,bool(changed[j] or responses[j]))
            text(d,(x+34,yy-17),f'R{PLASTIC[j]:02}',27,INK if responses[j] or changed[j] else MUTED,True)
        text(d,(1290,755),'CLASS 0',29,CYAN,True);text(d,(1810,755),'CLASS 1',29,CYAN,True)
        text(d,(180,1495),'25 shared channels + 84 key channels + 1 retirement sentinel',27,MUTED)
        # A dynamic edit readout, not a simulated electrical activity counter.
        text(d,(2520,580),'SAVED QUERY PREDICTION',25,MUTED,True)
        text(d,(2520,626),pred(q),43,GREEN if pred(q)==str(e['actual']) else AMBER,True)
        text(d,(3100,580),'ACTUAL OUTCOME',25,MUTED,True)
        revealed=e['teacher'] or e['phase'].startswith(('probe_','old_probe'))
        text(d,(3100,626),e['actual'] if revealed else 'pending',43,INK,True)
        text(d,(2520,700),'Probe truth is evaluation-only; query actor never receives it.',25,MUTED)
        for row,(arm,label) in enumerate([('native','Error gate + retirement'),('no_retire','Same encoding, no retirement'),('frozen','Trained then frozen')]):
            yy=770+row*88;v=self.data['arms'][arm]['outputs'][qi];text(d,(2520,yy),label,28,INK if arm=='native' else MUTED,True)
            for k in range(2):
                xx=3430+k*100;col=GREEN if v[84+k] else DIM
                d.ellipse((xx,yy-3,xx+38,yy+35),fill=col);text(d,(xx+47,yy-2),k,27,col,True)
        permit=bool(y[80]) if e['teacher'] else None
        text(d,(2520,1055),'TEACHING PASSES' if permit else 'TEACHING BLOCKED' if permit is not None else 'READ ONLY · waiting for outcome',29,GREEN if permit else MUTED,True)
        text(d,(2520,1102),f'Stored edit now: +{int(adds.sum())} / −{int(removes.sum())} connections',28,GREEN if adds.any() else MUTED)
        # Every cell is a measured read-only probe, arranged by reversal.
        text(d,(2520,1185),'READ-ONLY ACCURACY BY REVERSAL · SELECTED SEED',26,CYAN,True)
        colors={'native':GREEN,'no_retire':AMBER,'frozen':MUTED};xx0=2540;yy0=1450;ww=1030;hh=190
        for pct in [0,50,100]:
            yy=yy0-hh*pct/100;d.line((xx0,yy,xx0+ww,yy),fill=DIM,width=1);text(d,(xx0-7,yy-32),f'{pct}%',22,MUTED)
        for arm in ['frozen','no_retire','native']:
            points=[]
            for c in range(1,13):
                ids=[ev['index'] for ev in self.a['events'] if ev['phase']==f'probe_{c}' and not ev['teacher'] and ev['index']<=i]
                if not ids:continue
                hits=sum(pred(self.data['arms'][arm]['outputs'][j])==str(self.a['events'][j]['actual']) for j in ids)
                points.append((xx0+(c-1)*ww/11,yy0-hh*hits/len(ids)))
            if len(points)>1:d.line(points,fill=colors[arm],width=4)
            for x,yy in points:d.ellipse((x-5,yy-5,x+5,yy+5),fill=colors[arm])
        for c in range(1,13):text(d,(xx0+(c-1)*ww/11-8,yy0+16),c,22,MUTED)
        text(d,(2520,1530),'Green: full path   Amber: no retirement   Gray: frozen',25,MUTED)
        text(d,(2520,1575),'Keys 2/3: original connection vectors stay unchanged.',26,GREEN)
        text(d,(2520,1620),'Eight records stay resident · no weight repair or reset.',25,MUTED)
        return im

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    data=source();(a.out/'recorded-events.json').write_text(json.dumps(data,separators=(',',':'))+'\n');film=Film(data)
    for name,t in [('bootstrap',2.2),('correction',4.6),('retirement',4.95),('reused',13.0),('repeat',28.5)]:film.frame(t).save(a.out/f'{name}.png')
    if a.video:
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','h264_videotoolbox','-b:v','20M','-profile:v','main','-level:v','5.2','-bf','0','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(a.out/'native-memory-correction-4k60.mp4')]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);start=time.monotonic()
        try:
            for i in range(FPS*DURATION):
                proc.stdin.write(film.frame(i/FPS).tobytes())
                if i%300==0:print(json.dumps(dict(frame=i,total=FPS*DURATION,elapsed_s=round(time.monotonic()-start,1))),flush=True)
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(a.out/'native-memory-correction-4k60.mp4'),'-vf','scale=1920:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','19','-profile:v','main','-level:v','4.2','-refs','1','-bf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out/'native-memory-correction-1080p60.mp4')],check=True)
    (a.out/'SOURCE_BINDING.json').write_text(json.dumps(dict(archive_sha256=data['archive_sha256'],events_sha256=sha(a.out/'recorded-events.json'),renderer_sha256=sha(__file__),scope=data['scope'],outputs={p.name:sha(p) for p in a.out.iterdir() if p.suffix in ['.png','.mp4']}),indent=2)+'\n')
    print(json.dumps(dict(status='RENDERED',out=str(a.out))))
if __name__=='__main__':main()
