"""Paired wall-time replay of continuous physical runs; logical cells are measured matches."""
from pathlib import Path
import argparse,bisect,functools,hashlib,json,math,subprocess
import numpy as np
from PIL import Image,ImageDraw
from visualize import BG,PANEL,INK,MUTED,BLUE,GOLD,font
ROOT=Path(__file__).resolve().parents[1]
GREEN,RED,PURPLE='#73d7b2','#ef8584','#b59bea'
F={n:font(n) for n in [23,25,27,29,31,34,38]};B={n:font(n,True) for n in [27,31,34,38,52,70]}


def arm_wall(start,end,elapsed):
    """An ended arm must never sample replies from its later physical episode."""
    return min(start+int(elapsed*1e9),end)


def render(video):
    data=json.loads((ROOT/'visual/mario-continuous-data.json').read_text());result=json.loads((ROOT/'results/2026-10-23-mario-continuous.json').read_text())
    raw_path=ROOT/'results/2026-10-23-mario-continuous.npz';assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==data['archive_sha256'];raw=np.load(raw_path,allow_pickle=False)
    arms={}
    for arm,c in data['captures'].items():
        p=ROOT/c['game_path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==c['game_sha256'];game=np.load(p,allow_pickle=False)
        events=c['events'];origin=next(e['mono_ns'] for e in events if e['kind']=='game_start');packets=[e for e in events if e['kind']=='forward'];times=[e['response']['received_monotonic_ns'] for e in packets]
        frames=[e for e in events if e['kind']=='frame'];by_episode=[[e for e in frames if e['episode']==j] for j in range(4)]
        ends=[next(e for e in events if e['kind']=='episode_end' and e['episode']==j) for j in range(4)]
        arms[arm]=dict(c=c,images=game['frames'],origin=origin,packets=packets,times=times,episodes=by_episode,ends=ends,frame_episode={e['frame_id']:e['episode'] for e in frames},
                       initial=raw[f'fixture__dev_{arm}01__ram'],states=raw[f'dev_{arm}01__snapshots'],inputs=raw[f'fixture__dev_{arm}01__inputs'].reshape(-1,1024))
    durations=[max((a['ends'][j]['mono_ns']-a['episodes'][j][0]['mono_ns'])/1e9 for a in arms.values())+.15 for j in range(4)]
    boundaries=np.cumsum([0]+durations).tolist();duration=boundaries[-1]
    base=Image.new('RGB',(3840,2160),BG);d=ImageDraw.Draw(base)
    d.text((130,55),'AKD1500 / CONTINUOUS DECISIONS / MATCHED FROZEN CONTROL',font=F[34],fill=MUTED)
    d.text((130,122),'Every 24 frames, the chip chooses again.',font=B[70],fill=INK)
    d.text((130,220),'Actual queries → short action → observed outcome → native update. Separate hardware runs aligned at episode starts.',font=F[34],fill=MUTED)
    for col,arm in enumerate(['full','frozen']):
        left=130+1810*col;d.rounded_rectangle((left,375,left+1770,1880),radius=24,fill=PANEL,outline='#34465a',width=3)
        d.text((left+35,410),'NATIVE LEARNING' if arm=='full' else 'FROZEN / IDENTICAL INITIAL RAM',font=B[38],fill=GREEN if arm=='full' else BLUE)
        d.text((left+35,470),'NPU (1, 3, 0) · 192 logical records · 32 connections per record',font=F[27],fill=MUTED)
        d.rounded_rectangle((left+25,590,left+1005,1510),radius=14,fill=BG)
        for group,(label,tone) in enumerate([('UNSAFE / R0–63',RED),('SAFE / R64–127',GREEN),('PROTECTED / NOT RANKED',BLUE)]):
            y=600+group*250;d.text((left+1050,y),label,font=B[27],fill=tone)
            for j in range(64):
                x=left+1050+(j%16)*40;yy=y+58+(j//16)*42
                d.rounded_rectangle((x,yy,x+29,yy+29),radius=4,fill='#1e2e40',outline='#34465a')
        d.text((left+35,1745),'32 selected links / blue=image · purple=time · gold=inserted address',font=F[25],fill=MUTED)
    d.text((130,1935),'FIRST MATCHED PAIR: learned 1,418 px · frozen 1,554 px · no completed levels · expansion STOPPED',font=B[34],fill=GOLD)
    d.text((130,2000),'Four development starts, one seed, one chip. Training contained only 4 unsafe outcomes out of 192; no reliable learning benefit.',font=F[29],fill=MUTED)
    d.text((130,2070),'Cells show measured match values (0–32); white=sampled rise, gold=RAM edit. Schematic records; electrical spike timing is not measured.',font=F[27],fill=MUTED)

    @functools.lru_cache(maxsize=16)
    def resized(arm,index):return Image.fromarray(arms[arm]['images'][index]).resize((960,900),Image.Resampling.NEAREST)

    def frame(t):
        episode=min(3,bisect.bisect_right(boundaries,t)-1);elapsed=min(durations[episode],t-boundaries[episode]);image=base.copy();q=ImageDraw.Draw(image)
        q.text((130,305),f'LEVEL {episode+1}-1 / CONTINUOUS EPISODE {episode+1}/4 / {elapsed:05.2f}s',font=B[31],fill=INK)
        for col,arm in enumerate(['full','frozen']):
            a=arms[arm];left=130+1810*col;es=a['episodes'][episode];start=es[0]['mono_ns'];view_wall=start+int(elapsed*1e9);wall=arm_wall(start,a['ends'][episode]['mono_ns'],elapsed)
            clocks=[e['mono_ns'] for e in es];fi=max(0,min(len(es)-1,bisect.bisect_right(clocks,wall)-1));f=es[fi]
            image.paste(resized(arm,f['frame_id']),(left+35,600));q=ImageDraw.Draw(image)
            decisions=[e for e in a['c']['events'] if e['kind']=='decision' and e['episode']==episode and e['mono_ns']<=wall]
            outcomes=[e for e in a['c']['events'] if e['kind']=='outcome' and e['episode']==episode and e['mono_ns']<=wall]
            current=decisions[-1] if decisions else None
            action='awaiting native query' if current is None else 'JUMP 8 / RIGHT 16' if current['action']==0 else 'RIGHT 24'
            q.text((left+35,535),f'Decision {len(decisions):02d} / {action}',font=B[31],fill=INK)
            if wall>=a['ends'][episode]['mono_ns']:
                overlay=Image.new('RGBA',(960,900),(0,0,0,125));image.paste(overlay,(left+35,600),overlay);q=ImageDraw.Draw(image)
                q.text((left+150,990),'DEATH / EPISODE ENDED',font=B[38],fill=RED)
            packet_index=bisect.bisect_right(a['times'],wall)-1
            if packet_index>=0 and a['frame_episode'][a['packets'][packet_index]['input_frame']]==episode:
                e=a['packets'][packet_index];r=e['response'];before=a['initial'] if r['index']==0 else a['states'][r['index']-1];after=a['states'][r['index']]
                old=a['packets'][packet_index-1]['response']['scores'] if packet_index else [0]*192
                for j,v in enumerate(r['scores']):
                    group=j//64;k=j%64;xx=left+1050+(k%16)*40;yy=658+group*250+(k//16)*42;tone=[RED,GREEN,BLUE][group]
                    rgb=tuple(int(int(tone[i:i+2],16)*(.18+.82*v/32)) for i in [1,3,5]);border=GOLD if j in r['changed_records'] and view_wall-r['received_monotonic_ns']<300000000 else INK if v>old[j] else '#34465a'
                    q.rounded_rectangle((xx,yy,xx+29,yy+29),radius=4,fill=rgb,outline=border,width=3 if j in r['changed_records'] else 1)
                label='TEACH PREVIOUS DECISION' if e['phase']=='observed_feedback' and r['label']!=2 else 'PROTECTED QUERY'
                q.text((left+1050,1390),label,font=B[27],fill=GOLD if r['changed_records'] else BLUE)
                q.text((left+1050,1435),f'Packet {r["id"]} · age {(view_wall-r["received_monotonic_ns"])/1e9:.2f}s',font=F[25],fill=MUTED)
                recent=[p['response'] for p in a['packets'][:packet_index+1] if p['phase']=='observed_feedback']
                swaps=sum(p['connection_bit_edits'] for p in recent)//2
                if current:q.text((left+1050,1480),f'Margin A0 {current["margins"][0]:+d} / A1 {current["margins"][1]:+d}',font=B[27],fill=INK)
                q.text((left+1050,1525),f'Native swaps: {swaps}',font=B[27],fill=GOLD if swaps else MUTED)
                edited=next((p for p in reversed(recent) if p['changed_records']),None)
                edit_active=edited is not None and view_wall-edited['received_monotonic_ns']<300000000
                if edit_active:
                    r=edited;before=a['initial'] if r['index']==0 else a['states'][r['index']-1];after=a['states'][r['index']]
                    for j in r['changed_records']:
                        group=j//64;k=j%64;xx=left+1050+(k%16)*40;yy=658+group*250+(k//16)*42
                        q.rounded_rectangle((xx,yy,xx+29,yy+29),radius=4,outline=GOLD,width=3)
                unit=r['changed_records'][0] if r['changed_records'] else int(np.argmax(r['scores'][:128]));state=after if r['changed_records'] else before
                words=state[2+unit*36+4:2+(unit+1)*36];previous=before[2+unit*36+4:2+(unit+1)*36]
                selected=[k for k in range(1024) if int(words[k//32])>>(k%32)&1];added={k for k in selected if not int(previous[k//32])>>(k%32)&1}
                label=f'Recorded edit R{unit} / age {(view_wall-r["received_monotonic_ns"])/1e9:.2f}s' if edit_active else f'RAM record R{unit} / {len(added)} inserted links in this reply'
                q.text((left+35,1690),label,font=B[27],fill=GOLD if added else MUTED)
                for j,k in enumerate(selected):
                    x=left+35+j*51;tone=BLUE if k%512<256 else PURPLE;active=a['inputs'][r['index'],k]
                    q.rounded_rectangle((x,1800,x+35,1834),radius=4,fill=tone if active else BG,outline=GOLD if k in added else tone,width=4 if k in added else 1)
            q.text((left+35,1545),f'{f["controlled"]} game frames · {len(outcomes)} actual feedback events',font=B[31],fill=INK)
            for j,choice in enumerate(decisions):
                x=left+35+j*30;tone=BLUE if choice['action']==0 else PURPLE
                q.rectangle((x,1615,x+23,1641),fill=tone)
                if j<len(outcomes):q.rectangle((x,1648,x+23,1654),fill=GREEN if outcomes[j]['success'] else RED)
            end=a['ends'][episode]
            if wall>=end['mono_ns']:q.text((left+1050,1575),f'Advance: {end["advance_evaluation_only"]} px',font=B[27],fill=INK)
        return image
    frame(boundaries[3]+2.4).save(ROOT/'visual/mario-continuous-4k.png')
    if video:
        child=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-c:v','h264_videotoolbox','-b:v','16M','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'visual/mario-continuous-4k60.mp4')],stdin=subprocess.PIPE)
        try:
            for k in range(math.ceil(duration*60)):child.stdin.write(frame(k/60).tobytes())
            child.stdin.close();assert child.wait()==0
        finally:
            if child.poll() is None:child.terminate();child.wait()
    print(json.dumps(dict(duration_seconds=duration,frames=math.ceil(duration*60),chapters=durations)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',action='store_true');a=p.parse_args();render(a.video)
