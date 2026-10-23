"""Measured connection/readout replay; fixed layout and expanded time."""
from pathlib import Path
import argparse,json,math,subprocess
from PIL import Image,ImageDraw
from visualize import font,BG,PANEL,INK,MUTED,BLUE,GOLD
P=Path(__file__).resolve().parents[1];V=P/'visual';RED='#ef8584';GREEN='#73d7b2'
STATE=['PNG verified','CPU buffer','GStreamer buffer','JPEG ready']

def render(video):
    z=json.loads((V/'recovery-replay.json').read_text());r=json.loads((P/'results/2026-10-23-recovery.json').read_text())
    data=z['samples'];rr=z['reactive_samples'];W,H=3840,2160
    source=Image.open(V/'recovery-input.jpg').convert('RGB');source.thumbnail((570,330))
    output=Image.open(V/'recovery-output.png').convert('RGB').resize((560,315))
    base=Image.new('RGB',(W,H),BG);q=ImageDraw.Draw(base)
    q.text((135,85),'AKD1500 / REAL NANO IMAGE WORK / NATIVE LEARNING',font=font(34),fill=MUTED)
    q.text((135,150),'Learning when to try again',font=font(88,True),fill=INK)
    q.text((135,275),'A fixed probe clock requests exploration. The chip chooses an action and learns its actual consequence.',font=font(38),fill=MUTED)
    boxes=[(135,420,825,1630),(915,420,2285,1630),(2375,420,3705,1630)]
    for box in boxes:q.rounded_rectangle(box,radius=30,fill=PANEL,outline='#34465a',width=3)
    q.text((180,465),'ACTUAL IMAGE PIPELINE',font=font(34,True),fill=INK)
    base.paste(source,(190,540));q.text((190,895),'Verified example source JPEG',font=font(28),fill=MUTED)
    base.paste(output,(190,1050));q.text((190,1395),'Validated 320 × 180 PNG',font=font(28),fill=MUTED)
    q.text((190,1480),'Workers run on the Nano.',font=font(28),fill=INK)
    q.text((190,1530),'AKD receives task/outcome codes.',font=font(27),fill=MUTED)
    q.text((965,465),'TWO NATIVE LEARNING MEMORIES',font=font(34,True),fill=BLUE)
    q.text((965,520),f'Transitions NPU {tuple(z["learning_npus"][0]["np"])} · history NPU {tuple(z["learning_npus"][1]["np"])}',font=font(29),fill=MUTED)
    q.text((965,585),'32 selected feature links per mutable record',font=font(29),fill=MUTED)
    q.text((965,1085),'RECENT ACTION / OUTCOME MEMORY',font=font(28),fill=GOLD)
    q.text((965,1370),'S0 PNG · S1 CPU buffer · S2 GStreamer buffer · S3 JPEG',font=font(26),fill=MUTED)
    q.text((965,1420),'A0 CPU worker · A1 GStreamer worker (this replay)',font=font(26),fill=MUTED)
    q.text((965,1500),'One dot per selected link; white = inserted feature address.',font=font(27),fill=MUTED)
    q.text((965,1550),'Logical placement; no electrical spike animation.',font=font(27),fill=MUTED)
    q.text((2430,465),'THE RECOVERY QUESTION',font=font(34,True),fill=INK)
    q.text((2430,530),'What happens if we retry the preferred decoder?',font=font(29),fill=MUTED)
    q.rounded_rectangle((2430,620,3650,870),radius=24,fill='#142c3b')
    q.text((2475,655),'NATIVE MEMORY + BOUNDED PROBING',font=font(30),fill=BLUE)
    q.rounded_rectangle((2430,920,3650,1170),radius=24,fill='#202a36')
    q.text((2475,955),'NATIVE MEMORY / REACTIVE CONTROL',font=font(30),fill=MUTED)
    q.text((2430,1250),'Host clock → native query → chosen operation',font=font(29),fill=INK)
    q.text((2430,1310),'→ checked image outcome → native connection update',font=font(28),fill=MUTED)
    q.text((2430,1430),'Six mapped NPUs / 12 plastic records configured',font=font(29),fill=INK)
    q.text((2430,1490),'Search and exploration are supplied by us.',font=font(28),fill=MUTED)
    q.text((2430,1550),'Only the experienced associations learn on-chip.',font=font(28),fill=MUTED)
    q.rounded_rectangle((135,1730,3705,1940),radius=24,fill=PANEL)
    q.text((180,1770),'THREE ENCODINGS / ONE PHYSICAL CHIP',font=font(30),fill=MUTED)
    q.text((180,1830),'Probing: 72/72 PNGs · 156 operations     Reactive: 72/72 · 154     Frozen transition memory: 53/72',font=font(38,True),fill=INK)
    q.text((135,1990),'Earlier recovery recognition, at two extra operations overall. Identical CPU algorithm matches native actions and outcomes.',font=font(31),fill=INK)
    q.text((135,2050),'Recorded RAM/output replay; expanded time and schematic geometry. No speed, energy, CPU advantage or power-loss persistence claim.',font=font(27),fill=MUTED)

    def jobtime(fi):
        for start,end,lo,hi in [(0,300,0,8),(300,660,8,12),(660,1140,12,20),(1140,1440,20,24)]:
            if fi<end:return lo+(fi-start)/(end-start)*(hi-lo)
        return 23.999
    def frame(fi):
        jt=jobtime(fi);job=min(23,int(jt));within=jt-job
        seq=[x for x in data if x['job']==job];e=seq[min(len(seq)-1,int(within*len(seq)))];idx=data.index(e)
        controls=[x for x in rr if x['job']<=job];control=controls[-1]['mapping_after']
        im=base.copy();d=ImageDraw.Draw(im)
        phase=0 if job<8 else 1 if job<12 else 2 if job<20 else 3
        names=['LEARN WORKING ROUTES','PREFERRED WORKERS STOPPED','WORKERS RESTORED / NOT ANNOUNCED','DETOUR WORKERS STOPPED']
        d.text((135,355),names[phase],font=font(32,True),fill=GOLD if phase in [1,3] else GREEN)
        d.text((3150,105),f'JOB {job+1:02}/24',font=font(34),fill=INK)
        d.text((180,970),('PROBE / ' if e['probe'] else 'QUERY / ')+e['stage'].upper()+' / '+e['backend'].upper(),font=font(26,True),fill=GOLD if e['probe'] else BLUE)
        for bank,top in [(0,655),(1,1140)]:
            for k,row in enumerate(e['banks'][bank]):
                y=top+k*48;label=(f'S{k//2} / A{k%2}' if bank==0 else f'S{k} / last')
                d.text((965,y-13),label,font=font(24),fill=INK)
                for bit,value in enumerate(row['selected_features']):
                    x=1180+bit*29;color=INK if value in row['added'] else BLUE if bank==0 else GOLD
                    d.rounded_rectangle((x,y-6,x+19,y+10),radius=3,fill=color)
                value=e['mapping_after'][k] if bank==0 else e['history_after'][k]
                label='?' if value is None else str(value) if bank==0 else f'{value[0]}>{value[1]}'
                d.text((2130,y-12),'→ '+label,font=font(24,True),fill=INK)
        native=e['mapping_after'][6];reactive=control[6]
        lookup=lambda value:'unknown' if value is None else 'FAIL / stays JPEG' if value==3 else STATE[value]
        d.text((2475,725),lookup(native),font=font(43,True),fill=GREEN if native in [1,2] else RED)
        d.text((2475,795),'Prediction from read-back transition links',font=font(28),fill=MUTED)
        d.text((2475,1025),lookup(reactive),font=font(43,True),fill=GREEN if reactive in [1,2] else RED)
        d.text((2475,1095),'No retry during the restored-worker window' if phase==2 else 'Same environment; exploration clock disabled',font=font(27),fill=MUTED)
        for k in range(24):
            x=150+k*148;fill=BLUE if k<=job else '#233344'
            d.rounded_rectangle((x,1660,x+130,1682),radius=5,fill=fill)
        return im
    frame(720).save(V/'recovery-replay-4k.png')
    if video:
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-c:v','h264_videotoolbox','-b:v','18M','-pix_fmt','yuv420p','-movflags','+faststart',str(V/'recovery-replay-4k60.mp4')]
        pr=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for k in range(1440):pr.stdin.write(frame(k).tobytes())
            pr.stdin.close();assert pr.wait()==0
        finally:
            if pr.poll() is None:pr.terminate();pr.wait()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',action='store_true');a=p.parse_args();render(a.video)
