"""4K evidence replay of selective native repair; illustrative flow is labeled."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
import sys
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw

P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
from relay.audit_public import audit
from relay.reference import Network
from relay.design import schedule,THRESHOLD
from visualize import font as load_font

@lru_cache(maxsize=64)
def font(size,bold=False):return load_font(size,bold)

BG='#080e17';PANEL='#101d2b';INK='#ecf1f7';MUTED='#9bafc3';LINE='#2c4258'
COLORS=['#73c9f2','#b19ae9','#edbc70','#78cfb4'];GOLD='#f4c771'


def records():
    audit();report=json.loads((P/'results/2026-10-23-signal-relay.json').read_text())
    with np.load(P/'results/2026-10-23-signal-relay.npz',allow_pickle=False) as raw:
        data={k:raw[k] for k in raw.files}
    run=next(r for r in report['runs'] if r['run']=='s97101_recognition');key=run['run']+'__'
    net=Network([data[key+f'bank{k}_initial'].tolist() for k in range(2)])
    inputs,rows=schedule('recognition_repair');result=[]
    for i,row in enumerate(rows):
        before=[b.ram[:] for b in net.banks]
        output,response=net.forward(inputs[i],row['label'],row['learning_mask'])
        assert output==data[key+'outputs'][i].tolist()
        banks=[]
        for k,b in enumerate(net.banks):
            after=data[key+f'bank{k}_snapshots'][i].tolist();assert after==b.ram
            changed=[];added=[];removed=[]
            for j in range(64):
                at=2+j*20+4
                if before[k][at:at+16]!=after[at:at+16]:
                    changed.append(j)
                    for a in range(512):
                        old=before[k][at+a//32]>>(a%32)&1;new=after[at+a//32]>>(a%32)&1
                        if old and not new:removed.append(a)
                        if new and not old:added.append(a)
            potentials=response['recognition_scores' if k==0 else 'routing_scores']
            j=changed[0] if changed else int(np.argmax(potentials));at=2+j*20+4
            selected=[a for a in range(512) if after[at+a//32]>>(a%32)&1]
            assert len(selected)==32
            banks.append(dict(changed_records=changed,added=added,removed=removed,
                              selected_neuron=j,selected_addresses=selected,
                              ram_sha256=hashlib.sha256(np.asarray(after,'<u4').tobytes()).hexdigest()))
        result.append(dict(**row,measured_output=output,reconstructed=response,banks=banks))
    provenance=dict(status='NATIVE_OUTPUT_AND_RAM_BOUND_REPLAY',seed=97101,
                    samples=result,placements=report['fixtures']['97101']['mapping'],
                    boundary='Output counts and RAM edits measured. Responses reconstructed. Geometry and flow motion schematic. Time expanded.')
    (P/'visual/signal-relay-replay.json').write_text(json.dumps(provenance,separators=(',',':'))+'\n')
    return report,data,inputs,result


def main(video=False):
    report,data,inputs,events=records();W,H=3840,2160
    base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(base)
    fonts={s:font(s) for s in [24,28,30,32,34,36,38,42,46,52,60]}
    def text(x,y,s,size=34,color=INK,bold=False,anchor=None):
        d.text((x,y),s,font=font(size,True) if bold else fonts.get(size,font(size)),fill=color,anchor=anchor)
    text(140,84,'AKD1500 / TWO CONNECTED LEARNING MEMORIES',32,MUTED)
    text(140,146,'Repair recognition. Keep the routing rule.',76,INK,True)
    text(140,254,'128 learning neurons  ·  five resident NPUs  ·  separate teachers  ·  native communication',38,MUTED)
    for box in [(140,450,900,1430),(1000,450,2000,1430),(2440,450,3440,1430),(140,1490,1940,1930),(2020,1490,3700,1930)]:
        d.rounded_rectangle(box,radius=28,fill=PANEL,outline=LINE,width=3)
    text(190,495,'SENSOR INPUT',32,MUTED);text(190,550,'A noisy beacon signature',38,INK,True)
    text(190,980,'512 binary features / 256 active',30,MUTED)
    text(1050,495,'RECOGNITION MEMORY',34,COLORS[0],True)
    text(1050,552,'NPU (1, 3, 0) / 64 native neuron records',30,MUTED)
    text(2490,495,'ROUTING MEMORY',34,COLORS[1],True)
    text(2490,552,'NPU (2, 3, 0) / 64 native neuron records',30,MUTED)
    text(2215,590,'NATIVE',28,MUTED,anchor='mm');text(2215,630,'MESSAGE',28,MUTED,anchor='mm')
    text(2215,1125,'Fixed comparator',24,MUTED,anchor='mm');text(2215,1160,'+ encoder',24,MUTED,anchor='mm')
    text(2215,1200,'(2, 1, 0)',24,MUTED,anchor='mm');text(2215,1234,'(2, 2, 0)',24,MUTED,anchor='mm')
    text(3500,550,'FIXED',28,MUTED);text(3500,590,'READOUT',28,MUTED)
    text(3500,640,'(3, 1, 0)',24,MUTED)
    for b,x in enumerate([1000,2440]):
        for group in range(4):
            text(x+54,660+group*115,f'{group}',28,COLORS[group])
        text(x+50,1220,'Filled dots: reconstructed threshold response',28,MUTED)
        text(x+50,1265,'White ring: measured connection record edited',28,MUTED)
    text(190,1525,'MATCHED HARDWARE CONTROLS / SAME SIGNATURE STREAM',30,MUTED)
    for j,label in enumerate(['Repair recognition','Freeze both memories','Teach routing instead']):
        text(200,1615+j*85,label,30,INK)
    text(2070,1525,'ON-CHIP CONNECTIONS / MEASURED RAM',30,MUTED)
    text(140,1980,'Native output + every bank RAM captured after every input. Three initialization seeds; one synthetic sensor world.',30,INK)
    text(140,2030,'Recorded replay, time expanded. Responses reconstructed; layout and flow motion schematic. No electrical spike or energy claim.',28,MUTED)
    text(140,2075,'Host supplies sensor events, teaching labels and learning-mode changes. Intermediate neural processing stays on the mapped device.',28,MUTED)
    phase_labels={'old_probe':'KNOWN SIGNALS / both memories fixed',
                  'changed_before':'SIGNATURES CHANGE / destination rules stay the same',
                  'repair':'RECOGNITION REPAIR / identity teacher enabled only upstream',
                  'changed_after':'NEW SIGNALS / both memories fixed again',
                  'old_retention':'OLD SIGNALS RETURN / no reminder training'}
    # Explicitly expanded time: more viewing time for the first repair updates.
    segments=[(512,576,4),(576,640,4),(640,672,6),(672,896,2),(896,960,4),(960,1024,4)]
    duration=sum(s[2] for s in segments)
    def position(t):
        elapsed=0
        for start,end,seconds in segments:
            if t<elapsed+seconds:return min(end-1,start+int((t-elapsed)/seconds*(end-start)))
            elapsed+=seconds
        return 1023
    def frame(fi):
        t=fi/60;idx=position(t);e=events[idx];phase=e['phase'];im=base.copy();q=ImageDraw.Draw(im)
        def tt(x,y,s,size=34,color=INK,bold=False,anchor=None):
            q.text((x,y),s,font=font(size,bold),fill=color,anchor=anchor)
        q.rounded_rectangle((140,345,3700,408),radius=18,fill='#182b3d')
        tt(170,357,phase_labels[phase],32,INK,True)
        tt(3655,357,f'EVENT {idx+1} / 1024',28,MUTED,anchor='ra')
        for j,on in enumerate(inputs[idx]):
            x=200+(j%32)*20;y=665+(j//32)*17
            q.rounded_rectangle((x,y,x+12,y+11),radius=2,fill=COLORS[0] if on else '#203247')
        tt(190,1050,'Input identity is NOT supplied to the chip',28,MUTED)
        tt(190,1120,'EXPECTED DESTINATION',26,MUTED)
        tt(190,1170,f'Dock {chr(65+e["destination"])}',52,COLORS[e['destination']],True)
        tt(190,1260,'Ground truth shown for evaluation',28,MUTED)
        rec=e['reconstructed'];identity=rec['identity'];destination=int(np.argmax(e['measured_output']))
        for group in range(4):
            y=720+group*100
            q.line((2010,y,2430,y),fill=COLORS[group] if identity==group else '#253c52',width=7 if identity==group else 3)
            tt(2215,y-32,f'ID {group}',26,COLORS[group] if identity==group else MUTED,anchor='mm')
            if identity==group:
                xx=2025+((t*.75)%1)*390;q.ellipse((xx-10,y-10,xx+10,y+10),fill=INK)
        support=rec['recognition_support']
        if support.count(max(support))>1:
            tt(2215,1310,f'Tie → ID {identity}',24,MUTED,anchor='mm')
        for k,xx in enumerate([1000,2440]):
            score=rec['recognition_scores' if k==0 else 'routing_scores'];bank=e['banks'][k]
            active=bool(e['learning_mask']&(1<<k));color=COLORS[k]
            for j,potential in enumerate(score):
                group=j//16;within=j%16;x=xx+170+(within%8)*96;y=655+group*115+(within//8)*52
                q.ellipse((x-14,y-14,x+14,y+14),fill=COLORS[group] if potential>THRESHOLD else '#23384c')
                if j in bank['changed_records']:
                    radius=24+3*math.sin(t*9);q.ellipse((x-radius,y-radius,x+radius,y+radius),outline=INK,width=4)
            state='LEARNING' if active else 'FROZEN'
            q.rounded_rectangle((xx+48,1085,xx+950,1170),radius=14,fill='#19384a' if active else '#182637')
            tt(xx+72,1101,state,36,color if active else MUTED,True)
            tt(xx+920,1106,f'{len(bank["added"])} connection swaps',28,INK,anchor='ra')
            tt(xx+50,1330,'RAM '+bank['ram_sha256'][:16],26,MUTED)
        for j,count in enumerate(e['measured_output']):
            y=750+j*110;tt(3490,y-30,chr(65+j),30,COLORS[j])
            q.rounded_rectangle((3545,y-25,3675,y+20),radius=6,fill='#23384c')
            if count:q.rounded_rectangle((3545,y-25,3545+min(130,count*32),y+20),radius=6,fill=COLORS[j])
            tt(3610,y+35,str(count),26,MUTED,anchor='mm')
        tt(3490,1250,'OUTPUT',26,MUTED);tt(3490,1300,'Dock '+chr(65+destination),34,COLORS[destination],True)
        # Controls are measured at the same original event indices, never borrowed from another phase.
        selected=[i for i in range(512,idx+1) if events[i]['phase']==phase]
        for j,(suffix,color) in enumerate([('recognition',COLORS[0]),('frozen','#8295a7'),('routing',COLORS[1])]):
            ys=data[f's97101_{suffix}__outputs'];correct=sum(int(np.argmax(ys[i]))==events[i]['destination'] for i in selected)
            value=correct/len(selected);yy=1620+j*85
            q.rounded_rectangle((700,yy,1710,yy+28),radius=7,fill='#24364a')
            if correct:q.rounded_rectangle((700,yy,700+1010*value,yy+28),radius=7,fill=color)
            tt(1850,yy-9,f'{correct}/{len(selected)}',32,INK,anchor='ra')
        tt(200,1870,'Correct destinations so far in the displayed phase',26,MUTED)
        a=e['banks'][0];tt(2070,1580,f'Recognition neuron {a["selected_neuron"]:02} / 32 selected input addresses',30,INK)
        for j,address in enumerate(a['selected_addresses']):
            x=2070+(j%8)*195;y=1645+(j//8)*55
            changed=address in a['added'];q.rounded_rectangle((x,y,x+165,y+40),radius=7,fill='#59482b' if changed else '#203247')
            tt(x+82,y+3,f'{address:03}',28,GOLD if changed else MUTED,anchor='ma')
        tt(2070,1880,'Gold: inserted connections. Routing RAM stays fixed during recognition repair.',26,MUTED)
        return im
    frame(13*60).save(P/'visual/signal-relay-4k.png')
    frame(19*60).save(P/'visual/signal-relay-result-4k.png')
    if video:
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-',
             '-an','-c:v','h264_videotoolbox','-b:v','16M','-pix_fmt','yuv420p','-movflags','+faststart',str(P/'visual/signal-relay-4k60.mp4')]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for fi in range(duration*60):proc.stdin.write(frame(fi).tobytes())
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
    files=[P/'results/2026-10-23-signal-relay.json',P/'results/2026-10-23-signal-relay.npz',P/'visual/signal-relay-replay.json']
    media=[P/'visual/signal-relay-4k.png',P/'visual/signal-relay-result-4k.png']
    if video:media.append(P/'visual/signal-relay-4k60.mp4')
    (P/'visual/SIGNAL_RELAY_SOURCE_BINDING.json').write_text(json.dumps(dict(status='SOURCE_BOUND',
        sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},duration_seconds=duration,
        artifacts_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in media},
        resolution=[3840,2160],fps=60,scope='Recorded measured outputs/RAM; derived responses; schematic flow and expanded time'),indent=2)+'\n')
    print(json.dumps(dict(status='RENDERED',seconds=duration,video=video)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',action='store_true');main(p.parse_args().video)
