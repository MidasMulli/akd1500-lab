"""4K60 recorded stream with measured native scores and RAM edits, time expanded."""
from functools import lru_cache
from pathlib import Path
import argparse,hashlib,json,math,subprocess
import numpy as np
from PIL import Image,ImageDraw
from streaming.design import ROOT,sha,save
from streaming.archive import BASE
from streaming.report import main as report
from resident.or_archive import expand
from tools.visualize import font as existing_font

BG='#080f19';PANEL='#101e2d';INK='#edf3f9';MUTED='#a1b5c8';LINE='#30495f';BLUE='#73c9ee';GOLD='#efbd72';GREEN='#78d0b6';PURPLE='#b69bea';RED='#ed8c89'
COLORS=[BLUE,PURPLE,GOLD,GREEN]
@lru_cache(maxsize=40)
def font(size,bold=False):return existing_font(size,bold)

def load():
    meta=json.loads(BASE.with_suffix('.json').read_text());r=next(r for r in meta['runs'] if r['name']=='s105101_learn')
    archive=np.load(BASE.with_suffix('.npz'));prefix=r['name']+'__';z={k[len(prefix):]:archive[k] for k in archive.files if k.startswith(prefix)}
    states=expand(z['initial'],{k:z['delta.'+k] for k in ['offsets','indices','values','sha256']},r['native_forwards'])
    selected=[]
    for phase,count in [('baseline',2),('changed_before',2),('adapt',4),('changed_after',2),('retention',2)]:
        ids=[i for i,a in enumerate(r['rows']) if a['phase']==phase]
        selected.extend(ids[int(round(j*(len(ids)-1)/(count-1)))] for j in range(count))
    return meta,r,z,states,selected

def render(video=False):
    metrics=report();meta,r,z,states,selected=load();out=ROOT/'visual'
    base=Image.new('RGB',(3840,2160),BG);d=ImageDraw.Draw(base)
    def text(x,y,s,size=28,col=INK,bold=False,anchor=None):d.text((x,y),s,font=font(size,bold),fill=col,anchor=anchor)
    text(110,65,'STREAMING PREDICTION / NATIVE FEEDBACK',61,INK,True)
    text(110,155,'Frozen Nano T202 + one learning AKD FNP · predictions committed before the observed outcome',31,MUTED)
    d.rounded_rectangle((110,235,2375,1980),radius=24,fill=PANEL,outline=LINE,width=3)
    d.rounded_rectangle((2430,235,3730,1980),radius=24,fill=PANEL,outline=LINE,width=3)
    text(150,275,'CAPTURED STREAM / 224 × 224 SOURCE RGB',30,BLUE,True)
    for x,title in [(150,'CURRENT OBSERVATION'),(900,'NANO FORECAST / +5 STEPS'),(1650,'SUBSEQUENT OBSERVATION')]:text(x,340,title,24,MUTED,True)
    text(150,1135,'MATCHED PILOT / ALL CASES, NOT JUST THE DISPLAYED SAMPLES',28,INK,True)
    plot=Image.open(out/'streaming-prediction-results.png').convert('RGB').resize((2180,720),Image.Resampling.LANCZOS);base.paste(plot,(150,1190));d=ImageDraw.Draw(base)
    text(2470,275,f'NATIVE LEARNING FNP {tuple(r["learning_np"])}',31,BLUE,True)
    text(2470,330,'160 allocated records · 128 trainable / 32 query-protected',25,MUTED)
    text(2470,380,'Dots show measured overlap scores, not electrical spikes.',25,MUTED)
    text(2470,420,'128 trainable records shown; protected query class omitted.',23,MUTED)
    for j in range(4):
        cx=2470+(j%2)*625;cy=465+(j//2)*365
        d.rounded_rectangle((cx,cy,cx+570,cy+320),radius=18,fill='#14283b',outline=LINE,width=2)
        text(cx+22,cy+18,f'OUTPUT BAND {j+1} / 32 RECORDS',24,COLORS[j],True)
    text(2470,1230,'RECORDED NATIVE CONNECTION CHANGES',27,GOLD,True)
    text(2470,1280,'Each spoke: one stored input address; angle is schematic.',23,MUTED)
    text(2470,1650,'PILOT DISPOSITION / SKIPPING HELD',31,RED,True)
    text(2470,1710,'Changed stream: 49/69 learning · 27/69 frozen',28,INK,True)
    text(2470,1760,'CPU nearest neighbor: 61/69 · Nano T202: 41/69',26,MUTED)
    text(2470,1810,'Confidence rule: 35 eligible / 1 error. Zero skips.',26,MUTED)
    text(2470,1860,'75% quality floor + class-coverage gate failed.',26,RED)
    text(110,2020,'Recorded live Nano + AKD replies; playback slowed. Fixed phase-position samples, not success-selected. Full matched pilot: 533 inputs per arm.',26,MUTED)
    text(110,2070,'One chip / one initialization. Host tracks pixels and supplies delayed teachers. Bounded band prediction, not full RGB replacement. No CPU or energy win.',26,MUTED)
    text(110,2110,'All Nano forecasts retained · 3 device sessions · 2,148 native forwards and complete RAM snapshots · model weights unchanged',26,INK)
    cached={}
    for idx in selected:
        row=r['rows'][idx];a=[]
        for key in ['current','forecast','future']:
            im=Image.fromarray(z['frames.'+key][idx]).resize((670,670),Image.Resampling.LANCZOS);a.append(im)
        cached[idx]=a
    def frame(fi):
        im=base.copy();q=ImageDraw.Draw(im);at=min(11,fi//120);idx=selected[at];row=r['rows'][idx];phase=(fi%120)/120
        reveal=phase>=.48;teach=phase>=.72
        def tx(x,y,s,size=28,col=INK,bold=False,anchor=None):q.text((x,y),s,font=font(size,bold),fill=col,anchor=anchor)
        for j,x in enumerate([150,900,1650]):
            if j<2 or reveal:im.paste(cached[idx][j],(x,390))
            else:
                q.rounded_rectangle((x,390,x+670,1060),radius=12,fill='#0b1522',outline=LINE,width=3)
                tx(x+335,680,'OUTCOME NOT YET OBSERVED',27,MUTED,True,anchor='mm')
        for j,x in enumerate([150,900,1650]):
            if j==2 and not reveal:continue
            for k in range(4):
                yy=390+k*167.5
                if k:q.line((x,yy,x+670,yy),fill='#8093a3',width=2)
                q.rounded_rectangle((x+8,yy+65,x+58,yy+103),radius=6,fill='#101e2d')
                tx(x+33,yy+84,str(k+1),23,COLORS[k],True,anchor='mm')
        label=row['phase'].replace('_',' ').upper();tx(2330,280,f'{label} · SAMPLE {idx}',26,GOLD,True,anchor='ra')
        color=GREEN if row['prediction']==row['truth'] else RED
        tx(150,1070,f'AKD predicts band {row["prediction"]+1}',30,BLUE,True)
        tx(900,1070,f'Overlap {row["score"]}/64 · margin {row["margin"]}',29,MUTED)
        if reveal:tx(1650,1070,f'Observed band {row["truth"]+1} · {"CORRECT" if color==GREEN else "ERROR"}',28,color,True)
        scores=z['scores'][2*idx];after=states[2*idx+1];before=states[2*idx]
        all_changed=np.flatnonzero((before[2:].reshape(160,36)[:,4:]!=after[2:].reshape(160,36)[:,4:]).any(1));changed=all_changed if teach else []
        for group in range(4):
            cx=2470+(group%2)*625;cy=465+(group//2)*365
            for k in range(32):
                neuron=group*32+k;xx=cx+40+(k%8)*67;yy=cy+95+(k//8)*53;level=int(scores[neuron]);rad=10+level*7/64
                fill=tuple(int(a*(.18+.82*level/64)) for a in ImageColor(COLORS[group]))
                q.ellipse((xx-rad,yy-rad,xx+rad,yy+rad),fill=fill)
                if neuron in changed:q.ellipse((xx-23,yy-23,xx+23,yy+23),outline=GOLD,width=3)
            tx(cx+540,cy+270,f'MAX {max(scores[group*32:(group+1)*32])}/64',23,COLORS[group],True,anchor='ra')
        winner=int(all_changed[0]) if len(all_changed) else int(np.argmax(scores[:128]))
        bb=before[2:].reshape(160,36)[winner,4:];aa=after[2:].reshape(160,36)[winner,4:]
        cx,cy=2640,1470;current=aa if teach else bb
        old={k*32+b for k,w in enumerate(bb) for b in range(32) if int(w)>>b&1}
        conn={k*32+b for k,w in enumerate(current) for b in range(32) if int(w)>>b&1}
        for bit in sorted(conn):
            angle=2*math.pi*bit/1024;end=(cx+math.cos(angle)*115,cy+math.sin(angle)*115)
            q.line((cx,cy,*end),fill=GOLD if bit not in old else '#65829a',width=3)
        q.ellipse((cx-15,cy-15,cx+15,cy+15),fill=GOLD if teach and len(changed) else BLUE)
        tx(2810,1360,f'RECORD {winner}',30,INK,True)
        tx(2810,1415,'QUERY → OBSERVED OUTCOME',24,MUTED)
        tx(2810,1460,('→ PROTECTED QUERY' if row['teacher']==4 else '→ NATIVE TEACHING') if teach else '→ FEEDBACK PENDING',25,(BLUE if row['teacher']==4 else GOLD) if teach else MUTED,True)
        tx(2810,1510,f'{len(conn-old)} added / {len(old-conn)} removed' if teach else '64 stored connections',25,MUTED)
        tx(2810,1560,f'{len(changed)} record(s) changed' if teach else 'Query leaves RAM unchanged',24,MUTED)
        q.rectangle((110,2148,3730,2155),fill=LINE);q.rectangle((110,2148,110+3620*fi/1440,2155),fill=BLUE)
        return im
    frame(10*120+110).save(out/'streaming-prediction-4k.png');frame(6*120+110).save(out/'streaming-prediction-adapt-4k.png')
    if video:
        proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-c:v','h264_videotoolbox','-b:v','14M','-bf','0','-fps_mode','cfr','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'streaming-prediction-4k60.mp4')],stdin=subprocess.PIPE)
        try:
            for i in range(1440):proc.stdin.write(frame(i).tobytes())
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate()
    files=[BASE.with_suffix(s) for s in ['.json','.npz']]+[Path(__file__),ROOT/'streaming/report.py',ROOT/'streaming/archive.py',ROOT/'streaming/audit.py',ROOT/'streaming/design.py',ROOT/'resident/or_archive.py',ROOT/'pong/relearn/model.py']+[out/n for n in ['streaming-prediction-metrics.json','streaming-prediction-results.png','streaming-prediction-4k.png','streaming-prediction-adapt-4k.png']]
    if (out/'streaming-prediction-4k60.mp4').exists():files.append(out/'streaming-prediction-4k60.mp4')
    save(out/'STREAMING_PREDICTION_SOURCE_BINDING.json',dict(status='RECORDED_STREAM_NATIVE_RESPONSE_REPLAY',selected_samples=selected,record_selection='First record with a captured feedback connection edit; otherwise first maximum-score eligible record. Replay selection, not acting input.',selection='Two fixed endpoints per evaluation phase; four evenly spaced adaptation samples. No selection by correctness.',fps=60,seconds=24,hashes={str(p.relative_to(ROOT)):sha(p) for p in files}))

def ImageColor(c):return tuple(int(c[i:i+2],16) for i in (1,3,5))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',action='store_true');render(p.parse_args().video)
