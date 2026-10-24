"""Measured capability dashboard/replay. Logical geometry; no electrical spike trace."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import math
import subprocess
import numpy as np
from PIL import Image,ImageDraw
from capability import archive,contribution_archive
from capability.design import P
from capability.report import main as report
from tools.visualize import font as existing_font
BG='#080f19';PANEL='#101e2d';INK='#edf3f9';MUTED='#a1b5c8';LINE='#30495f';BLUE='#73c9ee';GOLD='#efbd72';GREEN='#78d0b6';PURPLE='#b69bea'

@lru_cache(maxsize=48)
def font(n,bold=False):return existing_font(n,bold)


def evidence():
    report();v=P/'visual';m=json.loads((v/'native-primitives-metrics.json').read_text());cm=json.loads(contribution_archive.BASE.with_suffix('.json').read_text());pm=json.loads(archive.BASE.with_suffix('.json').read_text());cases=[]
    with np.load(contribution_archive.BASE.with_suffix('.npz'),allow_pickle=False) as z:
        i=int(np.flatnonzero(z['baseline__inputs'].sum(1)==512)[0]);base=z['baseline__scores'][i].tolist()
        baseline=next(r for r in cm['runs'] if r['name']=='baseline')
        for n in range(11):
            r=next(r for r in cm['runs'] if r['name']==f'suppress{n}');cases.append(dict(name=r['name'],np=r['audit']['changed_packets'][0]['np'],block=r['audit']['suppressed_block'],measured_output=z[r['name']+'__scores'][i].tolist(),measured_drop=r['audit']['full_active_output_drop']))
    q=next(r for r in pm['runs'] if r['name']=='quiet_b32_s100_r0');key=q['name']+'__'
    with np.load(archive.BASE.with_suffix('.npz'),allow_pickle=False) as z:
        a=archive.decode(q,{k.split('__',1)[1]:z[k] for k in z.files if k.startswith(key)})
    states=a['bank0.snapshots'];bound=a['boundary_indices'];ids=[511,543,575,607,639];ss=[states[int(np.flatnonzero(bound==i)[0])] for i in ids];initial=a['bank0.initial'];learn=states[127]
    old=np.flatnonzero((initial[2:].reshape(512,20)[:,4:]!=learn[2:].reshape(512,20)[:,4:]).any(1)).tolist();new=np.flatnonzero((ss[0][2:].reshape(512,20)[:,4:]!=ss[-1][2:].reshape(512,20)[:,4:]).any(1)).tolist();assert len(old)==len(new)==4 and set(old).isdisjoint(new)
    neurons=[]
    for rid in old+new:
        records=[]
        for ram in ss:
            w=ram[2+rid*20+4:2+rid*20+20];conn=[j for j in range(512) if (int(w[j//32])>>(j%32))&1];assert len(conn)==32;records.append(conn)
        neurons.append(dict(record=rid,role='old' if rid in old else 'additional',connections_at_boundaries=records))
    replay=dict(status='MEASURED_OUTPUTS_AND_PARAMETER_CONNECTIONS',fixed_input='First frozen all512-active input in the40-case causal bank',baseline_output=base,causal_cases=cases,logical_mapping=baseline['fixture_info']['mapping'],quiet_capture_indices=ids,quiet_recognition_records=neurons,
        boundaries=['Causal fixed graph and learning graph are separate experiments.','Suppression changes thresholds, not power gating.','Connection spokes encode stored input addresses; geometric angles are schematic.','No per-neuron electrical activity or hardware completion timestamp is inferred.'])
    (v/'native-primitives-replay.json').write_text(json.dumps(replay,indent=2)+'\n');return m,replay


def main(video=False):
    m,r=evidence();v=P/'visual';W,H=3840,2160;base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(base)
    def text(x,y,s,size=32,col=INK,bold=False,anchor=None):d.text((x,y),s,font=font(size,bold),fill=col,anchor=anchor)
    text(110,65,'AKD1500 / NATIVE EXECUTION, MEASURED',32,MUTED)
    text(110,125,'Test the hardware. Keep the CPU comparison honest.',64,INK,True)
    text(110,230,'11 compute-NPU regions causally checked · queued native learning while the host worker blocks',33,MUTED)
    for box in [(110,335,1840,1180),(1940,335,3730,1180),(110,1240,1840,1870),(1940,1240,3730,1870)]:d.rounded_rectangle(box,radius=24,fill=PANEL,outline=LINE,width=3)
    text(150,370,'A / CAUSAL FIXED-GRAPH TEST',32,BLUE,True)
    text(150,425,'Logical resource map: 11 CNPs + one fixed FNP readout',29,MUTED)
    positions={};nodes=[(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)]
    text(190,510,'ALL 512 INPUT BITS ACTIVE',26,MUTED,True)
    text(190,555,'Same stimulus in each causal arm',23,MUTED)
    for col,row in nodes:
        x=185+(col-1)*530;y=495+(row-1)*135
        d.rounded_rectangle((x,y,x+470,y+120),radius=15,fill='#14283b',outline=LINE,width=2);text(x+15,y+7,f'NODE ({col}, {row})',21,MUTED)
        for nid in range(4):
            xx=x+18+nid*112;yy=y+40;positions[(col,row,nid)]=(xx,yy)
            active=(col,row,nid) in {tuple(t['np']) for t in r['causal_cases']};readout=(col,row,nid)==(3,3,0);color=BLUE if active else PURPLE if readout else '#25394b'
            d.rounded_rectangle((xx,yy,xx+94,yy+60),radius=8,fill=color)
            text(xx+47,yy+6,'FNP' if nid==0 else 'CNP',19,BG if active or readout else MUTED,True,anchor='ma');text(xx+47,yy+32,str(nid),22,BG if active or readout else MUTED,True,anchor='ma')
    text(155,1070,'Colored: mapped regions verified by RAM readback.',26,MUTED)
    text(155,1110,'One threshold region suppressed at a time; no power gating.',26,MUTED)
    text(1980,370,'A / MATCHED MATH, DIFFERENT NATIVE SCALING',32,BLUE,True)
    plot=Image.open(v/'native-primitives-scaling.png').convert('RGB').resize((1730,690),Image.Resampling.LANCZOS);base.paste(plot,(1970,420));d=ImageDraw.Draw(base)
    f=m['cnp_half_active']['service_cycle_factor'];text(1980,1120,f'128 → 2,048 detectors: 16× population / {f:.2f}× monitored service',28,INK,True)
    text(150,1275,'A / CPU CONTROL: SAME INPUT BYTES, SAME TWO COUNTS',30,GREEN,True)
    text(150,1330,'256 of 512 input bits active · resident setup excluded on both paths',27,MUTED)
    for x,label in [(165,'DETECTORS'),(595,'CPU NEON'),(1055,'NATIVE CNP'),(1490,'NATIVE FNP')]:text(x,1400,label,26,MUTED,True)
    for j,u in enumerate([128,512,2048]):
        y=1460+j*85;cpu=next(g['median_wall_ns']/1000 for a in m['cpu'] if a['units']==u for g in a['groups'] if g['density']==256 and g['method']=='packed_neon');native=next(g['wall_us'] for a in m['fixed'] if a['cell']==f'cnp{u}' for g in a['groups'] if g['density']==256);fnp=next((g['wall_us'] for a in m['fixed'] if a['cell']==f'fnp{u}' for g in a['groups'] if g['density']==256),None)
        for x,s,col in [(165,f'{u:,}',INK),(595,f'{cpu:.2f} µs',GREEN),(1055,f'{native:.2f} µs',BLUE),(1490,'—' if fnp is None else f'{fnp:.2f} µs',GOLD)]:text(x,y,s,36,col,True)
    text(150,1745,'CPU is faster here. Native figures include the serial host I/O path.',28,INK,True)
    text(150,1795,'Packing included on CPU. No silicon-only latency or energy claim.',26,MUTED)
    text(1980,1275,'B / NATIVE LEARNING WHILE THE HOST WORKER SLEEPS',30,GOLD,True)
    text(1980,1330,'4 × 32 teaching events · old/new 64/64 · downstream RAM fixed · no sleep polling',25,MUTED)
    text(1980,1660,'Each spoke is a measured stored input-address connection.',25,MUTED)
    text(1980,1710,'Matched 8-event releases: caller CPU across 128 teaching events',26,MUTED)
    busy=m['quiet']['polling8']['total_released_thread_cpu_ns']/1e6;quiet=m['quiet']['sleeping8']['total_released_thread_cpu_ns']/1e6
    text(1980,1755,f'Polling {busy:.3f} ms  →  sleeping {quiet:.3f} ms',35,INK,True)
    text(1980,1810,'Released interval only; preload/drain excluded. Sleeps add delay.',25,MUTED)
    text(110,1920,'A and B are separate experiments. Layout/connection angles schematic; replay time expanded. No electrical spike trace.',28,MUTED)
    text(110,1970,'43 owned device sessions · 9,040 verified forwards · 7,404 full learning-bank captures · CPU controls: 1.84M timed calls',28,INK)
    text(110,2020,'128-event learning stage failed before repair, followed by cleanup crash. Separate zero-forward cleanup passed; failure retained.',26,MUTED)
    text(110,2070,'One chip. Fixed paths: three temporal runs; learning: one initialization. No CPU speed or calibrated energy advantage established.',26,MUTED)
    def frame(fi):
        im=base.copy();q=ImageDraw.Draw(im);t=fi/60;case=min(10,int(max(0,t-2)/1.8));c=r['causal_cases'][case];coords=positions[tuple(c['np'])];xx,yy=coords;q.rounded_rectangle((xx-5,yy-5,xx+99,yy+65),radius=11,outline=GOLD,width=6)
        def tx(x,y,s,size=30,col=INK,bold=False,anchor=None):q.text((x,y),s,font=font(size,bold),fill=col,anchor=anchor)
        tx(150,920,f'SUPPRESS {case+1}/11 · NPU {tuple(c["np"])} · thresholds 15 → 32',29,GOLD,True)
        tx(150,975,f'Measured output {r["baseline_output"]} → {c["measured_output"]}',34,INK,True)
        tx(150,1025,f'Expected block [{c["block"][0]}, {c["block"][1]}) · measured drop {c["measured_drop"]}',26,MUTED)
        stage=min(4,int(t/4.8));tx(3700,1370,f'REPAIR EVENTS: {stage*32}/128',24,GOLD,True,anchor='ra')
        for j,n in enumerate(r['quiet_recognition_records']):
            cx=2110+(j%4)*410;cy=1435+(j//4)*150;before=set(n['connections_at_boundaries'][max(0,stage-1)]);conn=n['connections_at_boundaries'][stage];col=BLUE if n['role']=='old' else GOLD
            q.ellipse((cx-40,cy-40,cx+40,cy+40),outline=col,width=2)
            for bit in conn:
                angle=2*math.pi*bit/512;end=(cx+math.cos(angle)*40,cy+math.sin(angle)*40);q.line((cx,cy,*end),fill=col if bit not in before else '#65829a',width=2)
            tx(cx+60,cy-25,f'RECORD {n["record"]}',24,col,True);tx(cx+60,cy+10,'old / unchanged' if n['role']=='old' else f'{len(set(conn)-before)} added at boundary',21,MUTED)
        q.rectangle((110,2130,3730,2137),fill=LINE);q.rectangle((110,2130,110+3620*min(1,t/24),2137),fill=BLUE)
        return im
    frame(18*60).save(v/'native-primitives-4k.png');frame(5*60).save(v/'native-primitives-causal-4k.png')
    if video:
        p=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2160','-r','60','-i','-','-an','-c:v','h264_videotoolbox','-b:v','14M','-pix_fmt','yuv420p','-movflags','+faststart',str(v/'native-primitives-4k60.mp4')],stdin=subprocess.PIPE)
        try:
            for i in range(1440):p.stdin.write(frame(i).tobytes())
            p.stdin.close();assert p.wait()==0
        finally:
            if p.poll() is None:p.terminate()
    files=[archive.BASE.with_suffix(e) for e in ['.json','.npz']]+[contribution_archive.BASE.with_suffix(e) for e in ['.json','.npz']]+[Path(__file__),P/'capability/report.py',P/'capability/metrics.py']+[v/name for name in ['native-primitives-metrics.json','native-primitives-replay.json','native-primitives-scaling.png','native-primitives-cpu.png','native-primitives-4k.png','native-primitives-causal-4k.png']]
    if video:files.append(v/'native-primitives-4k60.mp4')
    (v/'NATIVE_PRIMITIVES_SOURCE_BINDING.json').write_text(json.dumps(dict(status='MEASURED_NATIVE_CAPABILITY_REPLAY',fps=60,seconds=24,hashes={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}),indent=2)+'\n')


if __name__=='__main__':
    import sys
    main('--video' in sys.argv)
