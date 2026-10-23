"""4K60 replay of saved wall-paced Pong and actual physical AKD packets.

Balls and trails reconstruct captured simulator states. Landing markers and
neuron brightness use received native scores. Gold marks actual RAM differences,
not animation-generated growth, current flow or electrical spike timing.
"""
from pathlib import Path
import argparse
import bisect
import functools
import json
import math
import subprocess
import sys
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from visualize import BG,PANEL,INK,MUTED,BLUE,GOLD,font
from pong.experiment import N,U,CLASSES,PER_CLASS,decode,sha
from pong.audit_public import audit

GREEN,RED,PURPLE='#73d7b2','#ef8584','#b59bea'
COLORS=[BLUE,PURPLE,GREEN,GOLD]
F={n:font(n) for n in [23,25,27,29,31,34,38]};B={n:font(n,True) for n in [25,27,31,34,38,44,54,70]}


def available_index(times,wall):
    """Return -1 until a physical response has actually arrived."""
    return bisect.bisect_right(times,wall)-1


def render(video=False):
    audit();data=json.loads((ROOT/'visual/pong-physics-data.json').read_text())
    source=ROOT/'results/2026-10-23-pong-physics.npz';assert sha(source)==data['archive_sha256']
    raw=np.load(source,allow_pickle=False);arms={}
    for arm,c in data['captures'].items():
        game=np.load(ROOT/c['game_path'],allow_pickle=False)['frames'] # decompress once
        origin=c['receipt']['origin_monotonic_ns'];packets=[e for e in c['events'] if 'response' in e]
        key=c['raw_key'];arms[arm]=dict(c=c,game=game,origin=origin,packets=packets,
            times=[e['response']['received_monotonic_ns'] for e in packets],
            states=raw[key+'__snapshots'],initial=raw[f'fixture__{key}__ram'],
            inputs=raw[f'fixture__{key}__inputs'].reshape(-1,N))
    duration=max(a['game'][-1,0]/1e9 for a in arms.values())
    base=Image.new('RGB',(3840,2160),BG);d=ImageDraw.Draw(base)
    d.text((120,65),'AKD1500  /  CHANGING-PHYSICS PONG  /  MEASURED NATIVE LEARNING',font=F[34],fill=MUTED)
    d.text((120,135),'The physics changed. Did the prediction?',font=B[70],fill=INK)
    d.text((120,235),'Observed height + velocity → chip predicts a landing band → ball lands → native connection update.',font=F[34],fill=MUTED)
    d.text((120,305),'Matched serves · same trained prefix · sequential hardware captures aligned at start · fixed paddles, shadow predictions',font=F[29],fill=MUTED)
    for col,arm in enumerate(['full','frozen']):
        left=120+1810*col
        d.rounded_rectangle((left,400,left+1770,1850),radius=24,fill=PANEL,outline='#34465a',width=3)
        d.text((left+35,435),'CONTINUE NATIVE LEARNING' if arm=='full' else 'TRAINED, THEN FROZEN',font=B[44],fill=GREEN if arm=='full' else BLUE)
        d.text((left+35,500),'160 records · 4 prediction groups + 1 protected query group',font=F[27],fill=MUTED)
        d.text((left+35,565),'PONG / RECORDED SIMULATOR POSITIONS',font=B[27],fill=MUTED)
        d.text((left+1230,565),'NATIVE MATCH VALUES',font=B[27],fill=MUTED)
        for group in range(5):
            y=635+group*189;tone=COLORS[group] if group<4 else '#728499'
            label=f'LANDING BAND {group+1}' if group<4 else 'PROTECTED / NOT RANKED'
            d.text((left+1230,y),label,font=B[25],fill=tone)
            for k in range(32):
                x=left+1230+(k%8)*51;yy=y+42+(k//8)*32
                d.rounded_rectangle((x,yy,x+39,yy+24),radius=3,fill='#192b3b')
        d.text((left+35,1445),'RECORDED BINARY CONNECTION SELECTION',font=B[27],fill=MUTED)
        d.text((left+35,1750),'Each square = one input address · selected links are filled',font=F[25],fill=MUTED)
        d.text((left+35,1790),'Gold = inserted · coral = removed · no electrical spike timing measured',font=F[25],fill=MUTED)
    d.text((120,1910),'MATCHED TEST: continued learning 12/48 · trained–frozen 4/48 · majority constant 19/48',font=B[38],fill=INK)
    d.text((120,1980),'Useful prediction gate FAILED. More seeds and learned paddle control were not admitted.',font=B[34],fill=GOLD)
    d.text((120,2050),'One chip, one development seed. Match scores and RAM edits are measured; chip layout is schematic. Export: 3840 × 2160 / 60 fps.',font=F[27],fill=MUTED)

    @functools.lru_cache(maxsize=256)
    def weights(arm,index,unit):
        a=arms[arm];state=a['states'][index];before=a['initial'] if index==0 else a['states'][index-1]
        pos=2+unit*36+4
        def bits(r):return {k for k in range(N) if int(r[pos+k//32])>>(k%32)&1}
        now,old=bits(state),bits(before)
        return now,now-old,old-now

    def frame(t):
        image=base.copy();q=ImageDraw.Draw(image)
        for col,arm in enumerate(['full','frozen']):
            a=arms[arm];left=120+1810*col;game=a['game'];wall=a['origin']+min(int(t*1e9),int(game[-1,0]))
            fi=available_index(game[:,0].tolist(),wall-a['origin'])
            pi=available_index(a['times'],wall)
            if fi<0 or pi<0:
                q.text((left+70,850),'Waiting for physical reply…',font=B[38],fill=MUTED);continue
            f=game[fi];case=int(f[1]);pred=next(e for e in a['c']['events'] if e['kind']=='prediction' and e['case']==case)
            phase=pred['phase'];changed=pred['changed'];prediction=int(f[6]);x0,y0,w,h=left+60,655,1040,570
            q.rounded_rectangle((x0-20,y0-20,x0+w+120,y0+h+20),radius=12,fill=BG,outline='#34465a',width=2)
            for band in range(4):
                yy=y0+band*h/4
                if band:q.line((x0,yy,x0+w,yy),fill='#233447',width=2)
                q.text((x0+w+47,yy+48),str(band+1),font=B[34],fill=COLORS[band])
            for k in range(14):
                yy=y0+k*h/14;q.line((x0+w/2,yy,x0+w/2,yy+h/28),fill='#233447',width=3)
            # One band prediction; never invent a chip-predicted trajectory.
            py=y0+(prediction+.5)*h/4
            q.rectangle((x0+w-17,y0+prediction*h/4+3,x0+w+10,y0+(prediction+1)*h/4-3),fill=COLORS[prediction])
            q.line((x0+w-130,py,x0+w-28,py),fill=COLORS[prediction],width=6)
            # Fixed centre paddle is visibly separate from the prediction marker.
            q.rounded_rectangle((x0+w-45,y0+h*.37,x0+w-32,y0+h*.63),radius=5,fill=INK)
            q.rounded_rectangle((x0-5,y0+h*.37,x0+8,y0+h*.63),radius=5,fill='#5d718b')
            trail=game[max(0,fi-22):fi+1];trail=trail[trail[:,1]==case]
            if len(trail)>1:
                q.line([(x0+row[3]*w,y0+row[4]*h) for row in trail],fill='#485d75',width=5)
            bx,by=x0+f[3]*w,y0+f[4]*h
            q.ellipse((bx-11,by-11,bx+11,by+11),fill=INK)
            q.text((left+35,1280),f'{case+1:02d}/16  /  {phase.upper()}',font=B[27],fill=GOLD if changed else BLUE)
            landings=[e for e in a['c']['events'] if e['kind']=='landing' and e['case']==case and e['mono_ns']<=wall]
            actual=f'actual {landings[-1]["truth"]+1}' if landings else 'actual — ball in flight'
            q.text((left+35,1330),f'Chip predicts band {prediction+1}  /  {actual}',font=B[34],fill=INK)
            q.text((left+35,1385),'Marker = native prediction · white paddle = fixed host control',font=F[25],fill=MUTED)
            # Replies for a later serve cannot decorate this serve's held frame.
            packet_ids=[i for i,e in enumerate(a['packets'][:pi+1]) if e['case']==case]
            if not packet_ids:continue
            event=a['packets'][packet_ids[-1]];r=event['response']
            recent=[e for e in a['packets'][:pi+1] if e['kind']=='feedback']
            edits=[e for e in recent if e['response']['changed_records']]
            swaps=sum(e['response']['connection_bit_edits']//2 for e in recent)
            last_edit=edits[-1] if edits else None
            active=last_edit and wall-last_edit['response']['received_monotonic_ns']<350000000
            for j,v in enumerate(r['scores']):
                group,k=divmod(j,PER_CLASS);xx=left+1230+(k%8)*51;yy=677+group*189+(k//8)*32
                color=COLORS[group] if group<4 else '#728499'
                rgb=tuple(int(int(color[c:c+2],16)*(.12+.88*v/64)) for c in [1,3,5])
                edited=active and j in last_edit['response']['changed_records']
                q.rounded_rectangle((xx,yy,xx+39,yy+24),radius=3,fill=rgb,outline=GOLD if edited else '#34465a',width=3 if edited else 1)
            q.text((left+1230,1610),f'64 links / record',font=B[27],fill=INK)
            q.text((left+1230,1660),f'{swaps} live replacements',font=B[27],fill=GOLD if swaps else MUTED)
            q.text((left+1230,1710),f'#{r["id"]} / native {r["transaction_ns"]/1e6:.1f} ms',font=F[25],fill=MUTED)
            if last_edit:
                edit=last_edit['response'];unit=edit['changed_records'][0];idx=edit['index']
                note=f'Last edit: R{unit} / {edit["connection_bit_edits"]//2} replacements'
            else:
                idx=r['index'];unit=int(np.argmax(np.array(r['scores'][:128])));note=f'RAM record R{unit} / no live edits'
            selected,added,removed=weights(arm,idx,unit)
            q.text((left+35,1495),note,font=B[27],fill=GOLD if last_edit else INK)
            for k in range(N):
                xx=left+35+(k%64)*16;yy=1550+(k//64)*11
                tone=BLUE if k<512 else PURPLE
                q.rectangle((xx,yy,xx+12,yy+7),fill=GOLD if k in added else tone if k in selected else '#172535',
                    outline=RED if k in removed else None)
        return image
    preview=frame(13.3);preview.save(ROOT/'visual/pong-physics-4k.png')
    if video:
        target=ROOT/'visual/pong-physics-4k60.mp4'
        process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
            '-s','3840x2160','-r','60','-i','-','-an','-c:v','h264_videotoolbox','-b:v','14M','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
        try:
            for k in range(math.ceil(duration*60)):process.stdin.write(frame(k/60).tobytes())
            process.stdin.close();assert process.wait()==0
        finally:
            if process.poll() is None:process.terminate();process.wait()
        print(json.dumps(dict(status='RENDER_PASS',duration=duration,frames=math.ceil(duration*60),sha256=sha(target))))
    return frame


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',action='store_true');render(p.parse_args().video)
